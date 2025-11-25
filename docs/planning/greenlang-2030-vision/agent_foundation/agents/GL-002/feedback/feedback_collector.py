# -*- coding: utf-8 -*-
"""
Feedback Collector for GL-002 BoilerEfficiencyOptimizer

This module implements the feedback collection system that stores, retrieves,
and analyzes user feedback for continuous improvement.

Example:
    >>> collector = FeedbackCollector(db_url="postgresql://...")
    >>> feedback = OptimizationFeedback(
    ...     optimization_id="opt_123",
    ...     rating=5,
    ...     user_id="user_456"
    ... )
    >>> result = await collector.collect_feedback(feedback)
    >>> stats = await collector.get_stats(days=30)
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
from collections import Counter
import asyncio
import asyncpg
import hashlib

from greenlang.determinism import DeterministicClock
from .feedback_models import (
    OptimizationFeedback,
    FeedbackStats,
    SatisfactionTrend,
    FeedbackSummary,
    FeedbackAlert,
    FeedbackCategory
)

logger = logging.getLogger(__name__)


class FeedbackCollector:
    """
    Collects and manages user feedback for optimization recommendations.

    This class handles feedback storage, retrieval, aggregation, and analysis
    to enable data-driven continuous improvement of the optimizer.

    Attributes:
        db_pool: PostgreSQL connection pool
        alert_threshold_low_rating: Rating threshold for alerts (default: 2.5)
        alert_threshold_accuracy: Accuracy threshold for alerts (default: 50%)
    """

    def __init__(
        self,
        db_url: str,
        alert_threshold_low_rating: float = 2.5,
        alert_threshold_accuracy: float = 50.0
    ):
        """
        Initialize FeedbackCollector.

        Args:
            db_url: PostgreSQL connection URL
            alert_threshold_low_rating: Trigger alert if avg rating below this
            alert_threshold_accuracy: Trigger alert if accuracy below this %
        """
        self.db_url = db_url
        self.db_pool: Optional[asyncpg.Pool] = None
        self.alert_threshold_low_rating = alert_threshold_low_rating
        self.alert_threshold_accuracy = alert_threshold_accuracy

        logger.info(f"FeedbackCollector initialized with thresholds: rating={alert_threshold_low_rating}, accuracy={alert_threshold_accuracy}%")

    async def initialize(self) -> None:
        """Initialize database connection pool."""
        try:
            self.db_pool = await asyncpg.create_pool(
                self.db_url,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            logger.info("Database connection pool created successfully")
        except Exception as e:
            logger.error(f"Failed to create database pool: {e}", exc_info=True)
            raise

    async def close(self) -> None:
        """Close database connection pool."""
        if self.db_pool:
            await self.db_pool.close()
            logger.info("Database connection pool closed")

    async def collect_feedback(
        self,
        feedback: OptimizationFeedback
    ) -> Dict[str, Any]:
        """
        Store user feedback in database.

        Args:
            feedback: Validated feedback data

        Returns:
            Result dictionary with feedback_id and status

        Raises:
            ValueError: If feedback validation fails
            RuntimeError: If database operation fails
        """
        if not self.db_pool:
            raise RuntimeError("Database pool not initialized. Call initialize() first.")

        start_time = DeterministicClock.utcnow()

        try:
            async with self.db_pool.acquire() as conn:
                # Insert feedback into database
                query = """
                    INSERT INTO optimization_feedback (
                        optimization_id, rating, comment, actual_savings,
                        predicted_savings, category, user_id, timestamp,
                        metadata, savings_accuracy, provenance_hash
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    RETURNING id
                """

                feedback_id = await conn.fetchval(
                    query,
                    feedback.optimization_id,
                    feedback.rating,
                    feedback.comment,
                    feedback.actual_savings,
                    feedback.predicted_savings,
                    feedback.category.value,
                    feedback.user_id,
                    feedback.timestamp,
                    feedback.metadata,
                    feedback.savings_accuracy,
                    feedback.provenance_hash
                )

                # Update satisfaction trends table
                await self._update_satisfaction_trends(conn, feedback)

                # Check if alert should be triggered
                await self._check_and_create_alerts(conn, feedback)

                processing_time = (DeterministicClock.utcnow() - start_time).total_seconds() * 1000

                logger.info(
                    f"Feedback collected successfully: id={feedback_id}, "
                    f"optimization_id={feedback.optimization_id}, rating={feedback.rating}, "
                    f"processing_time={processing_time:.2f}ms"
                )

                return {
                    "feedback_id": feedback_id,
                    "status": "success",
                    "processing_time_ms": processing_time,
                    "provenance_hash": feedback.provenance_hash
                }

        except asyncpg.UniqueViolationError:
            logger.warning(f"Duplicate feedback for optimization {feedback.optimization_id}")
            raise ValueError("Feedback already submitted for this optimization")

        except Exception as e:
            logger.error(f"Failed to collect feedback: {e}", exc_info=True)
            raise RuntimeError(f"Feedback collection failed: {str(e)}")

    async def get_stats(
        self,
        days: int = 30,
        category: Optional[FeedbackCategory] = None
    ) -> FeedbackStats:
        """
        Get aggregated feedback statistics.

        Args:
            days: Number of days to include in statistics
            category: Optional category filter

        Returns:
            Aggregated feedback statistics

        Raises:
            RuntimeError: If database operation fails
        """
        if not self.db_pool:
            raise RuntimeError("Database pool not initialized")

        try:
            async with self.db_pool.acquire() as conn:
                period_start = DeterministicClock.utcnow() - timedelta(days=days)
                period_end = DeterministicClock.utcnow()

                # Build query with optional category filter
                where_clause = "WHERE timestamp >= $1 AND timestamp <= $2"
                params: List[Any] = [period_start, period_end]

                if category:
                    where_clause += " AND category = $3"
                    params.append(category.value)

                # Get basic statistics
                query = f"""
                    SELECT
                        COUNT(*) as total_count,
                        AVG(rating) as avg_rating,
                        AVG(savings_accuracy) as avg_accuracy
                    FROM optimization_feedback
                    {where_clause}
                """

                row = await conn.fetchrow(query, *params)

                total_count = row['total_count'] or 0
                avg_rating = float(row['avg_rating'] or 0)
                avg_accuracy = float(row['avg_accuracy'] or 0) if row['avg_accuracy'] else None

                # Get rating distribution
                dist_query = f"""
                    SELECT rating, COUNT(*) as count
                    FROM optimization_feedback
                    {where_clause}
                    GROUP BY rating
                    ORDER BY rating
                """

                rating_rows = await conn.fetch(dist_query, *params)
                rating_distribution = {row['rating']: row['count'] for row in rating_rows}

                # Ensure all ratings 1-5 are present
                for rating in range(1, 6):
                    if rating not in rating_distribution:
                        rating_distribution[rating] = 0

                # Get category distribution
                cat_query = f"""
                    SELECT category, COUNT(*) as count
                    FROM optimization_feedback
                    {where_clause}
                    GROUP BY category
                """

                cat_rows = await conn.fetch(cat_query, *params)
                category_distribution = {row['category']: row['count'] for row in cat_rows}

                stats = FeedbackStats(
                    total_feedback_count=total_count,
                    average_rating=round(avg_rating, 2) if total_count > 0 else 0.0,
                    rating_distribution=rating_distribution,
                    average_savings_accuracy=round(avg_accuracy, 2) if avg_accuracy else None,
                    period_start=period_start,
                    period_end=period_end,
                    category_distribution=category_distribution
                )

                logger.info(f"Statistics generated: {days} days, total_feedback={total_count}, avg_rating={stats.average_rating}")

                return stats

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}", exc_info=True)
            raise RuntimeError(f"Statistics retrieval failed: {str(e)}")

    async def get_recent_feedback(
        self,
        limit: int = 10,
        rating_filter: Optional[int] = None
    ) -> List[OptimizationFeedback]:
        """
        Get recent feedback submissions.

        Args:
            limit: Maximum number of feedback items to return
            rating_filter: Optional filter by rating (1-5)

        Returns:
            List of recent feedback items
        """
        if not self.db_pool:
            raise RuntimeError("Database pool not initialized")

        try:
            async with self.db_pool.acquire() as conn:
                where_clause = ""
                params: List[Any] = []

                if rating_filter is not None:
                    where_clause = "WHERE rating = $1"
                    params = [rating_filter, limit]
                else:
                    params = [limit]

                query = f"""
                    SELECT
                        optimization_id, rating, comment, actual_savings,
                        predicted_savings, category, user_id, timestamp,
                        metadata, savings_accuracy, provenance_hash
                    FROM optimization_feedback
                    {where_clause}
                    ORDER BY timestamp DESC
                    LIMIT ${len(params)}
                """

                rows = await conn.fetch(query, *params)

                feedback_list = []
                for row in rows:
                    feedback = OptimizationFeedback(
                        optimization_id=row['optimization_id'],
                        rating=row['rating'],
                        comment=row['comment'],
                        actual_savings=row['actual_savings'],
                        predicted_savings=row['predicted_savings'],
                        category=FeedbackCategory(row['category']),
                        user_id=row['user_id'],
                        timestamp=row['timestamp'],
                        metadata=row['metadata'] or {},
                        savings_accuracy=row['savings_accuracy']
                    )
                    feedback_list.append(feedback)

                logger.info(f"Retrieved {len(feedback_list)} recent feedback items")

                return feedback_list

        except Exception as e:
            logger.error(f"Failed to get recent feedback: {e}", exc_info=True)
            raise RuntimeError(f"Recent feedback retrieval failed: {str(e)}")

    async def get_satisfaction_trends(
        self,
        days: int = 90
    ) -> List[SatisfactionTrend]:
        """
        Get satisfaction trends over time.

        Args:
            days: Number of days to include

        Returns:
            List of daily satisfaction trends
        """
        if not self.db_pool:
            raise RuntimeError("Database pool not initialized")

        try:
            async with self.db_pool.acquire() as conn:
                period_start = DeterministicClock.utcnow() - timedelta(days=days)

                query = """
                    SELECT
                        date,
                        average_rating,
                        feedback_count,
                        nps_score,
                        ma_7day,
                        ma_30day,
                        is_anomaly,
                        anomaly_score
                    FROM satisfaction_trends
                    WHERE date >= $1
                    ORDER BY date ASC
                """

                rows = await conn.fetch(query, period_start)

                trends = [
                    SatisfactionTrend(
                        date=row['date'],
                        average_rating=row['average_rating'],
                        feedback_count=row['feedback_count'],
                        nps_score=row['nps_score'],
                        ma_7day=row['ma_7day'],
                        ma_30day=row['ma_30day'],
                        is_anomaly=row['is_anomaly'],
                        anomaly_score=row['anomaly_score']
                    )
                    for row in rows
                ]

                logger.info(f"Retrieved {len(trends)} satisfaction trend records")

                return trends

        except Exception as e:
            logger.error(f"Failed to get satisfaction trends: {e}", exc_info=True)
            raise RuntimeError(f"Trend retrieval failed: {str(e)}")

    async def _update_satisfaction_trends(
        self,
        conn: asyncpg.Connection,
        feedback: OptimizationFeedback
    ) -> None:
        """Update daily satisfaction trends table."""
        today = feedback.timestamp.date()

        # Aggregate today's feedback
        query = """
            INSERT INTO satisfaction_trends (date, average_rating, feedback_count, nps_score)
            VALUES ($1, $2, 1, 0)
            ON CONFLICT (date)
            DO UPDATE SET
                average_rating = (satisfaction_trends.average_rating * satisfaction_trends.feedback_count + $2) / (satisfaction_trends.feedback_count + 1),
                feedback_count = satisfaction_trends.feedback_count + 1,
                updated_at = NOW()
        """

        await conn.execute(query, today, feedback.rating)

        # Calculate moving averages (7-day and 30-day)
        await self._calculate_moving_averages(conn, today)

        # Detect anomalies
        await self._detect_anomalies(conn, today)

    async def _calculate_moving_averages(
        self,
        conn: asyncpg.Connection,
        current_date: datetime.date
    ) -> None:
        """Calculate 7-day and 30-day moving averages."""
        # 7-day MA
        query_7day = """
            UPDATE satisfaction_trends
            SET ma_7day = (
                SELECT AVG(average_rating)
                FROM satisfaction_trends
                WHERE date >= $1 AND date <= $2
            )
            WHERE date = $2
        """
        await conn.execute(query_7day, current_date - timedelta(days=6), current_date)

        # 30-day MA
        query_30day = """
            UPDATE satisfaction_trends
            SET ma_30day = (
                SELECT AVG(average_rating)
                FROM satisfaction_trends
                WHERE date >= $1 AND date <= $2
            )
            WHERE date = $2
        """
        await conn.execute(query_30day, current_date - timedelta(days=29), current_date)

    async def _detect_anomalies(
        self,
        conn: asyncpg.Connection,
        current_date: datetime.date
    ) -> None:
        """Detect statistical anomalies in satisfaction trends."""
        # Simple anomaly detection: if current rating deviates >2 std dev from 30-day MA
        query = """
            WITH stats AS (
                SELECT
                    AVG(average_rating) as mean,
                    STDDEV(average_rating) as stddev
                FROM satisfaction_trends
                WHERE date >= $1 AND date < $2
            )
            UPDATE satisfaction_trends
            SET
                is_anomaly = (ABS(average_rating - stats.mean) > 2 * stats.stddev),
                anomaly_score = ABS(average_rating - stats.mean) / NULLIF(stats.stddev, 0)
            FROM stats
            WHERE date = $2
        """

        await conn.execute(query, current_date - timedelta(days=30), current_date)

    async def _check_and_create_alerts(
        self,
        conn: asyncpg.Connection,
        feedback: OptimizationFeedback
    ) -> None:
        """Check if feedback should trigger alerts."""
        alerts_to_create = []

        # Alert: Very low rating (1-2 stars)
        if feedback.rating <= 2:
            alert = FeedbackAlert(
                alert_id=f"low_rating_{feedback.optimization_id}_{int(DeterministicClock.utcnow().timestamp())}",
                severity="warning",
                title=f"Low Rating Received: {feedback.rating} stars",
                description=f"Optimization {feedback.optimization_id} received a low rating of {feedback.rating} stars.",
                triggered_by="low_rating",
                threshold_value=2.0,
                actual_value=float(feedback.rating),
                affected_optimizations=[feedback.optimization_id]
            )
            alerts_to_create.append(alert)

        # Alert: Poor savings accuracy
        if feedback.savings_accuracy is not None and feedback.savings_accuracy < self.alert_threshold_accuracy:
            alert = FeedbackAlert(
                alert_id=f"low_accuracy_{feedback.optimization_id}_{int(DeterministicClock.utcnow().timestamp())}",
                severity="warning",
                title=f"Low Prediction Accuracy: {feedback.savings_accuracy:.1f}%",
                description=f"Optimization {feedback.optimization_id} achieved only {feedback.savings_accuracy:.1f}% accuracy.",
                triggered_by="low_accuracy",
                threshold_value=self.alert_threshold_accuracy,
                actual_value=feedback.savings_accuracy,
                affected_optimizations=[feedback.optimization_id]
            )
            alerts_to_create.append(alert)

        # Store alerts in database
        for alert in alerts_to_create:
            await self._store_alert(conn, alert)

    async def _store_alert(
        self,
        conn: asyncpg.Connection,
        alert: FeedbackAlert
    ) -> None:
        """Store alert in database."""
        query = """
            INSERT INTO feedback_alerts (
                alert_id, severity, title, description, triggered_by,
                threshold_value, actual_value, affected_optimizations,
                created_at, acknowledged
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (alert_id) DO NOTHING
        """

        await conn.execute(
            query,
            alert.alert_id,
            alert.severity,
            alert.title,
            alert.description,
            alert.triggered_by,
            alert.threshold_value,
            alert.actual_value,
            alert.affected_optimizations,
            alert.created_at,
            alert.acknowledged
        )

        logger.warning(f"Alert created: {alert.severity} - {alert.title}")
