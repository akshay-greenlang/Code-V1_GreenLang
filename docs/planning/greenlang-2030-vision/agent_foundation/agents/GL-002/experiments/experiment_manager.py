# -*- coding: utf-8 -*-
"""
Experiment Manager for A/B Testing Framework

This module manages the lifecycle of A/B tests including creation,
execution, monitoring, and analysis.

Example:
    >>> manager = ExperimentManager(db_url="postgresql://...", redis_url="redis://...")
    >>> await manager.initialize()
    >>> experiment = await manager.create_experiment(
    ...     name="combustion_optimization_v2",
    ...     variants=[control_variant, treatment_variant],
    ...     primary_metric="energy_savings"
    ... )
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import asyncio
import asyncpg
import redis.asyncio as redis
import json

from greenlang.determinism import DeterministicClock
from .experiment_models import (
    Experiment,
    ExperimentVariant,
    ExperimentResult,
    ExperimentMetrics,
    StatisticalSignificance,
    ExperimentStatus,
    MetricType,
    ExperimentAssignment
)
from .traffic_router import TrafficRouter
from .statistical_analyzer import StatisticalAnalyzer

logger = logging.getLogger(__name__)


class ExperimentManager:
    """
    Manages A/B testing experiments lifecycle.

    This class provides methods to create, start, stop, and analyze experiments.
    Uses PostgreSQL for persistent storage and Redis for traffic routing.

    Attributes:
        db_pool: PostgreSQL connection pool
        redis_client: Redis client for traffic routing
        traffic_router: Traffic routing system
        analyzer: Statistical analysis engine
    """

    def __init__(
        self,
        db_url: str,
        redis_url: str,
        min_sample_size: int = 100,
        significance_level: float = 0.05
    ):
        """
        Initialize ExperimentManager.

        Args:
            db_url: PostgreSQL connection URL
            redis_url: Redis connection URL
            min_sample_size: Minimum samples per variant
            significance_level: Statistical significance threshold (alpha)
        """
        self.db_url = db_url
        self.redis_url = redis_url
        self.min_sample_size = min_sample_size
        self.significance_level = significance_level

        self.db_pool: Optional[asyncpg.Pool] = None
        self.redis_client: Optional[redis.Redis] = None
        self.traffic_router: Optional[TrafficRouter] = None
        self.analyzer = StatisticalAnalyzer(significance_level=significance_level)

        logger.info(
            f"ExperimentManager initialized: min_sample_size={min_sample_size}, "
            f"significance_level={significance_level}"
        )

    async def initialize(self) -> None:
        """Initialize database and Redis connections."""
        try:
            # Initialize PostgreSQL
            self.db_pool = await asyncpg.create_pool(
                self.db_url,
                min_size=2,
                max_size=10,
                command_timeout=60
            )

            # Initialize Redis
            self.redis_client = await redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )

            # Initialize traffic router
            self.traffic_router = TrafficRouter(self.redis_client)

            logger.info("ExperimentManager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize ExperimentManager: {e}", exc_info=True)
            raise

    async def close(self) -> None:
        """Close database and Redis connections."""
        if self.db_pool:
            await self.db_pool.close()
        if self.redis_client:
            await self.redis_client.close()
        logger.info("ExperimentManager closed")

    async def create_experiment(
        self,
        name: str,
        description: str,
        hypothesis: str,
        variants: List[ExperimentVariant],
        primary_metric: str,
        primary_metric_type: MetricType,
        secondary_metrics: Optional[List[str]] = None,
        duration_days: Optional[int] = None,
        created_by: str = "system",
        tags: Optional[List[str]] = None
    ) -> Experiment:
        """
        Create a new A/B test experiment.

        Args:
            name: Experiment name
            description: What this experiment tests
            hypothesis: Expected outcome
            variants: List of variants to test
            primary_metric: Primary success metric
            primary_metric_type: Type of primary metric
            secondary_metrics: Additional metrics to track
            duration_days: Planned duration
            created_by: User creating experiment
            tags: Organization tags

        Returns:
            Created experiment

        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If creation fails
        """
        if not self.db_pool:
            raise RuntimeError("Database pool not initialized")

        try:
            # Create experiment model
            experiment = Experiment(
                name=name,
                description=description,
                hypothesis=hypothesis,
                variants=variants,
                primary_metric=primary_metric,
                primary_metric_type=primary_metric_type,
                secondary_metrics=secondary_metrics or [],
                min_sample_size=self.min_sample_size,
                significance_level=self.significance_level,
                duration_days=duration_days,
                created_by=created_by,
                tags=tags or []
            )

            # Store in database
            async with self.db_pool.acquire() as conn:
                query = """
                    INSERT INTO experiments (
                        experiment_id, name, description, hypothesis,
                        variants, primary_metric, primary_metric_type,
                        secondary_metrics, min_sample_size, significance_level,
                        power, status, duration_days, created_by, created_at, tags
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                """

                await conn.execute(
                    query,
                    experiment.experiment_id,
                    experiment.name,
                    experiment.description,
                    experiment.hypothesis,
                    json.dumps([v.dict() for v in experiment.variants]),
                    experiment.primary_metric,
                    experiment.primary_metric_type.value,
                    experiment.secondary_metrics,
                    experiment.min_sample_size,
                    experiment.significance_level,
                    experiment.power,
                    experiment.status.value,
                    experiment.duration_days,
                    experiment.created_by,
                    experiment.created_at,
                    experiment.tags
                )

            logger.info(
                f"Experiment created: id={experiment.experiment_id}, name={name}, "
                f"variants={len(variants)}"
            )

            return experiment

        except asyncpg.UniqueViolationError:
            raise ValueError(f"Experiment with name '{name}' already exists")

        except Exception as e:
            logger.error(f"Failed to create experiment: {e}", exc_info=True)
            raise RuntimeError(f"Experiment creation failed: {str(e)}")

    async def start_experiment(self, experiment_id: str) -> Experiment:
        """
        Start running an experiment.

        Args:
            experiment_id: Experiment to start

        Returns:
            Updated experiment

        Raises:
            ValueError: If experiment not found or already running
            RuntimeError: If start fails
        """
        if not self.db_pool or not self.traffic_router:
            raise RuntimeError("Manager not initialized")

        try:
            async with self.db_pool.acquire() as conn:
                # Get experiment
                experiment = await self._get_experiment(conn, experiment_id)

                if experiment.status == ExperimentStatus.RUNNING:
                    raise ValueError("Experiment already running")

                # Update status
                query = """
                    UPDATE experiments
                    SET status = $1, start_date = $2
                    WHERE experiment_id = $3
                """

                start_date = DeterministicClock.utcnow()
                await conn.execute(
                    query,
                    ExperimentStatus.RUNNING.value,
                    start_date,
                    experiment_id
                )

                experiment.status = ExperimentStatus.RUNNING
                experiment.start_date = start_date

                # Configure traffic router
                await self.traffic_router.configure_experiment(
                    experiment_id=experiment_id,
                    variants=experiment.variants
                )

            logger.info(f"Experiment started: {experiment_id}")

            return experiment

        except Exception as e:
            logger.error(f"Failed to start experiment {experiment_id}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to start experiment: {str(e)}")

    async def stop_experiment(self, experiment_id: str) -> Experiment:
        """
        Stop a running experiment.

        Args:
            experiment_id: Experiment to stop

        Returns:
            Updated experiment
        """
        if not self.db_pool:
            raise RuntimeError("Database pool not initialized")

        try:
            async with self.db_pool.acquire() as conn:
                # Get experiment
                experiment = await self._get_experiment(conn, experiment_id)

                if experiment.status != ExperimentStatus.RUNNING:
                    raise ValueError("Experiment not running")

                # Update status
                query = """
                    UPDATE experiments
                    SET status = $1, end_date = $2
                    WHERE experiment_id = $3
                """

                end_date = DeterministicClock.utcnow()
                await conn.execute(
                    query,
                    ExperimentStatus.COMPLETED.value,
                    end_date,
                    experiment_id
                )

                experiment.status = ExperimentStatus.COMPLETED
                experiment.end_date = end_date

            logger.info(f"Experiment stopped: {experiment_id}")

            return experiment

        except Exception as e:
            logger.error(f"Failed to stop experiment {experiment_id}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to stop experiment: {str(e)}")

    async def record_metric(
        self,
        experiment_id: str,
        user_id: str,
        metric_name: str,
        metric_value: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a metric observation for a user.

        Args:
            experiment_id: Experiment ID
            user_id: User who triggered metric
            metric_name: Name of metric
            metric_value: Metric value
            metadata: Additional context
        """
        if not self.db_pool or not self.traffic_router:
            raise RuntimeError("Manager not initialized")

        try:
            # Get user's variant assignment
            variant = await self.traffic_router.get_variant(experiment_id, user_id)

            if not variant:
                logger.warning(f"User {user_id} not assigned to experiment {experiment_id}")
                return

            # Store metric
            async with self.db_pool.acquire() as conn:
                query = """
                    INSERT INTO experiment_metrics (
                        experiment_id, variant_name, user_id, metric_name,
                        metric_value, metadata, recorded_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                """

                await conn.execute(
                    query,
                    experiment_id,
                    variant,
                    user_id,
                    metric_name,
                    metric_value,
                    json.dumps(metadata or {}),
                    DeterministicClock.utcnow()
                )

            logger.debug(
                f"Metric recorded: experiment={experiment_id}, variant={variant}, "
                f"metric={metric_name}, value={metric_value}"
            )

        except Exception as e:
            logger.error(f"Failed to record metric: {e}", exc_info=True)
            # Don't raise - metrics should not break user experience

    async def analyze_experiment(self, experiment_id: str) -> ExperimentResult:
        """
        Analyze experiment results and determine winner.

        Args:
            experiment_id: Experiment to analyze

        Returns:
            Complete experiment results with statistical analysis

        Raises:
            RuntimeError: If analysis fails
        """
        if not self.db_pool:
            raise RuntimeError("Database pool not initialized")

        try:
            async with self.db_pool.acquire() as conn:
                # Get experiment
                experiment = await self._get_experiment(conn, experiment_id)

                # Get metrics for all variants
                variant_metrics = await self._get_variant_metrics(conn, experiment_id)

                # Perform statistical analysis
                significance_tests = await self._analyze_variants(
                    experiment,
                    variant_metrics
                )

                # Calculate days running
                days_running = None
                if experiment.start_date:
                    end_date = experiment.end_date or DeterministicClock.utcnow()
                    days_running = (end_date - experiment.start_date).days

                # Determine final recommendation
                final_recommendation = self._determine_recommendation(
                    experiment,
                    variant_metrics,
                    significance_tests
                )

                # Generate key insights
                key_insights = self._generate_insights(
                    experiment,
                    variant_metrics,
                    significance_tests
                )

                # Calculate total samples
                total_samples = sum(m.sample_size for m in variant_metrics)

                result = ExperimentResult(
                    experiment=experiment,
                    metrics=variant_metrics,
                    significance_tests=significance_tests,
                    total_samples=total_samples,
                    days_running=days_running,
                    final_recommendation=final_recommendation,
                    key_insights=key_insights
                )

                logger.info(
                    f"Experiment analyzed: {experiment_id}, winner={result.winner}, "
                    f"conclusive={result.is_conclusive}"
                )

                return result

        except Exception as e:
            logger.error(f"Failed to analyze experiment {experiment_id}: {e}", exc_info=True)
            raise RuntimeError(f"Experiment analysis failed: {str(e)}")

    async def _get_experiment(
        self,
        conn: asyncpg.Connection,
        experiment_id: str
    ) -> Experiment:
        """Get experiment from database."""
        query = """
            SELECT * FROM experiments WHERE experiment_id = $1
        """

        row = await conn.fetchrow(query, experiment_id)

        if not row:
            raise ValueError(f"Experiment not found: {experiment_id}")

        # Parse variants JSON
        variants = [
            ExperimentVariant(**v)
            for v in json.loads(row['variants'])
        ]

        return Experiment(
            experiment_id=row['experiment_id'],
            name=row['name'],
            description=row['description'],
            hypothesis=row['hypothesis'],
            variants=variants,
            primary_metric=row['primary_metric'],
            primary_metric_type=MetricType(row['primary_metric_type']),
            secondary_metrics=row['secondary_metrics'],
            min_sample_size=row['min_sample_size'],
            significance_level=row['significance_level'],
            power=row['power'],
            status=ExperimentStatus(row['status']),
            start_date=row['start_date'],
            end_date=row['end_date'],
            duration_days=row['duration_days'],
            created_by=row['created_by'],
            created_at=row['created_at'],
            tags=row['tags']
        )

    async def _get_variant_metrics(
        self,
        conn: asyncpg.Connection,
        experiment_id: str
    ) -> List[ExperimentMetrics]:
        """Get aggregated metrics for all variants."""
        query = """
            SELECT
                variant_name,
                COUNT(*) as sample_size,
                AVG(metric_value) as mean,
                STDDEV(metric_value) as std,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY metric_value) as median
            FROM experiment_metrics
            WHERE experiment_id = $1
            GROUP BY variant_name
        """

        rows = await conn.fetch(query, experiment_id)

        metrics_list = []
        for row in rows:
            # Calculate 95% confidence interval
            ci_lower, ci_upper = self.analyzer.calculate_confidence_interval(
                mean=row['mean'],
                std=row['std'] or 0,
                n=row['sample_size']
            )

            metrics = ExperimentMetrics(
                variant_name=row['variant_name'],
                experiment_id=experiment_id,
                sample_size=row['sample_size'],
                primary_metric_name="primary_metric",  # Would come from experiment config
                primary_metric_mean=row['mean'],
                primary_metric_std=row['std'] or 0,
                primary_metric_median=row['median'],
                ci_lower=ci_lower,
                ci_upper=ci_upper
            )
            metrics_list.append(metrics)

        return metrics_list

    async def _analyze_variants(
        self,
        experiment: Experiment,
        variant_metrics: List[ExperimentMetrics]
    ) -> List[StatisticalSignificance]:
        """Perform statistical analysis comparing variants."""
        significance_tests = []

        # Find control variant
        control_metrics = next(
            (m for m in variant_metrics
             if any(v.name == m.variant_name and v.is_control for v in experiment.variants)),
            None
        )

        if not control_metrics:
            logger.warning("No control variant found")
            return []

        # Compare each treatment to control
        for metrics in variant_metrics:
            if metrics.variant_name == control_metrics.variant_name:
                continue

            # Perform t-test or appropriate statistical test
            test_result = self.analyzer.compare_variants(
                control_mean=control_metrics.primary_metric_mean,
                control_std=control_metrics.primary_metric_std,
                control_n=control_metrics.sample_size,
                treatment_mean=metrics.primary_metric_mean,
                treatment_std=metrics.primary_metric_std,
                treatment_n=metrics.sample_size
            )

            significance = StatisticalSignificance(
                experiment_id=experiment.experiment_id,
                control_variant=control_metrics.variant_name,
                treatment_variant=metrics.variant_name,
                **test_result
            )

            significance_tests.append(significance)

        return significance_tests

    def _determine_recommendation(
        self,
        experiment: Experiment,
        variant_metrics: List[ExperimentMetrics],
        significance_tests: List[StatisticalSignificance]
    ) -> str:
        """Determine final recommendation based on results."""
        # Check if minimum sample size reached
        min_samples_reached = all(
            m.sample_size >= experiment.min_sample_size
            for m in variant_metrics
        )

        if not min_samples_reached:
            return "continue - need more data to reach minimum sample size"

        # Check if any variant is significantly better
        significant_improvements = [
            t for t in significance_tests
            if t.is_significant and t.relative_improvement > 0
        ]

        if significant_improvements:
            best = max(significant_improvements, key=lambda x: x.relative_improvement)
            return f"ship - deploy {best.treatment_variant} ({best.relative_improvement:.1f}% improvement)"

        # Check if significantly worse
        significant_regressions = [
            t for t in significance_tests
            if t.is_significant and t.relative_improvement < 0
        ]

        if significant_regressions:
            return "stop - no improvement detected, revert to control"

        # No significant difference
        return "iterate - no significant difference, try different approach"

    def _generate_insights(
        self,
        experiment: Experiment,
        variant_metrics: List[ExperimentMetrics],
        significance_tests: List[StatisticalSignificance]
    ) -> List[str]:
        """Generate key insights from experiment."""
        insights = []

        # Best performing variant
        best_variant = max(variant_metrics, key=lambda x: x.primary_metric_mean)
        insights.append(
            f"Best performing variant: {best_variant.variant_name} "
            f"(mean={best_variant.primary_metric_mean:.2f})"
        )

        # Statistical significance
        significant_count = sum(1 for t in significance_tests if t.is_significant)
        insights.append(
            f"{significant_count}/{len(significance_tests)} comparisons "
            f"showed statistical significance"
        )

        # Sample sizes
        total_samples = sum(m.sample_size for m in variant_metrics)
        insights.append(f"Total samples collected: {total_samples}")

        return insights
