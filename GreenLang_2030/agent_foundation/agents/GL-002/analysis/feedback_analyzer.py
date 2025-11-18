"""
Feedback Analyzer for Automated Analysis

This module analyzes user feedback to identify patterns, trends, and
opportunities for improvement.

Example:
    >>> analyzer = FeedbackAnalyzer(db_url="postgresql://...")
    >>> await analyzer.initialize()
    >>> insights = await analyzer.analyze_feedback_patterns(days=30)
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import asyncpg
from collections import Counter
import numpy as np

logger = logging.getLogger(__name__)


class FeedbackAnalyzer:
    """
    Analyzes user feedback patterns and trends.

    Identifies low-rated optimizations, systematic errors, and
    opportunities for improvement through automated analysis.

    Attributes:
        db_pool: PostgreSQL connection pool
    """

    def __init__(self, db_url: str):
        """
        Initialize FeedbackAnalyzer.

        Args:
            db_url: PostgreSQL connection URL
        """
        self.db_url = db_url
        self.db_pool: Optional[asyncpg.Pool] = None

        logger.info("FeedbackAnalyzer initialized")

    async def initialize(self) -> None:
        """Initialize database connection pool."""
        try:
            self.db_pool = await asyncpg.create_pool(
                self.db_url,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            logger.info("FeedbackAnalyzer database pool created")
        except Exception as e:
            logger.error(f"Failed to initialize FeedbackAnalyzer: {e}", exc_info=True)
            raise

    async def close(self) -> None:
        """Close database connection pool."""
        if self.db_pool:
            await self.db_pool.close()
            logger.info("FeedbackAnalyzer closed")

    async def analyze_feedback_patterns(
        self,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze feedback patterns over a time period.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with pattern analysis results
        """
        if not self.db_pool:
            raise RuntimeError("Analyzer not initialized")

        try:
            async with self.db_pool.acquire() as conn:
                period_start = datetime.utcnow() - timedelta(days=days)

                # Get feedback data
                query = """
                    SELECT
                        rating,
                        category,
                        savings_accuracy,
                        timestamp,
                        comment
                    FROM optimization_feedback
                    WHERE timestamp >= $1
                    ORDER BY timestamp DESC
                """

                rows = await conn.fetch(query, period_start)

                if not rows:
                    return {"message": "No feedback data available"}

                # Analyze patterns
                ratings = [row['rating'] for row in rows]
                accuracies = [row['savings_accuracy'] for row in rows if row['savings_accuracy'] is not None]
                categories = [row['category'] for row in rows]

                # Calculate statistics
                avg_rating = np.mean(ratings)
                rating_trend = self._calculate_trend([row['rating'] for row in rows])

                # Identify problem categories
                category_stats = await self._analyze_categories(conn, period_start)

                # Analyze comment sentiment
                negative_comments = [
                    row['comment'] for row in rows
                    if row['comment'] and row['rating'] <= 2
                ]

                # Identify systematic issues
                systematic_issues = await self._identify_systematic_issues(conn, period_start)

                results = {
                    "period_days": days,
                    "total_feedback": len(rows),
                    "average_rating": round(avg_rating, 2),
                    "rating_trend": rating_trend,
                    "average_accuracy": round(np.mean(accuracies), 2) if accuracies else None,
                    "category_performance": category_stats,
                    "systematic_issues": systematic_issues,
                    "negative_feedback_count": sum(1 for r in ratings if r <= 2),
                    "top_complaints": self._extract_top_complaints(negative_comments),
                    "analyzed_at": datetime.utcnow().isoformat()
                }

                logger.info(
                    f"Feedback analysis complete: {days} days, avg_rating={avg_rating:.2f}, "
                    f"trend={rating_trend}"
                )

                return results

        except Exception as e:
            logger.error(f"Failed to analyze feedback patterns: {e}", exc_info=True)
            raise

    async def identify_underperforming_optimizations(
        self,
        min_rating: float = 3.0,
        min_samples: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Identify optimizations with consistently low ratings.

        Args:
            min_rating: Threshold for low rating
            min_samples: Minimum number of feedback samples required

        Returns:
            List of underperforming optimization IDs with statistics
        """
        if not self.db_pool:
            raise RuntimeError("Analyzer not initialized")

        try:
            async with self.db_pool.acquire() as conn:
                query = """
                    SELECT
                        optimization_id,
                        COUNT(*) as feedback_count,
                        AVG(rating) as avg_rating,
                        AVG(savings_accuracy) as avg_accuracy,
                        STRING_AGG(DISTINCT category, ', ') as categories
                    FROM optimization_feedback
                    GROUP BY optimization_id
                    HAVING COUNT(*) >= $1 AND AVG(rating) < $2
                    ORDER BY AVG(rating) ASC
                    LIMIT 20
                """

                rows = await conn.fetch(query, min_samples, min_rating)

                underperforming = [
                    {
                        "optimization_id": row['optimization_id'],
                        "feedback_count": row['feedback_count'],
                        "avg_rating": round(row['avg_rating'], 2),
                        "avg_accuracy": round(row['avg_accuracy'], 2) if row['avg_accuracy'] else None,
                        "categories": row['categories'],
                        "severity": "high" if row['avg_rating'] < 2.0 else "medium"
                    }
                    for row in rows
                ]

                logger.info(f"Identified {len(underperforming)} underperforming optimizations")

                return underperforming

        except Exception as e:
            logger.error(f"Failed to identify underperforming optimizations: {e}", exc_info=True)
            raise

    async def detect_anomalies(self, days: int = 90) -> List[Dict[str, Any]]:
        """
        Detect anomalies in satisfaction trends.

        Args:
            days: Number of days to analyze

        Returns:
            List of detected anomalies
        """
        if not self.db_pool:
            raise RuntimeError("Analyzer not initialized")

        try:
            async with self.db_pool.acquire() as conn:
                period_start = datetime.utcnow() - timedelta(days=days)

                query = """
                    SELECT
                        date,
                        average_rating,
                        feedback_count,
                        is_anomaly,
                        anomaly_score
                    FROM satisfaction_trends
                    WHERE date >= $1 AND is_anomaly = TRUE
                    ORDER BY anomaly_score DESC
                    LIMIT 10
                """

                rows = await conn.fetch(query, period_start)

                anomalies = [
                    {
                        "date": row['date'].isoformat(),
                        "average_rating": round(row['average_rating'], 2),
                        "feedback_count": row['feedback_count'],
                        "anomaly_score": round(row['anomaly_score'], 2),
                        "severity": "high" if row['anomaly_score'] > 3 else "medium"
                    }
                    for row in rows
                ]

                logger.info(f"Detected {len(anomalies)} anomalies in satisfaction trends")

                return anomalies

        except Exception as e:
            logger.error(f"Failed to detect anomalies: {e}", exc_info=True)
            raise

    async def _analyze_categories(
        self,
        conn: asyncpg.Connection,
        period_start: datetime
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze performance by category."""
        query = """
            SELECT
                category,
                COUNT(*) as count,
                AVG(rating) as avg_rating,
                AVG(savings_accuracy) as avg_accuracy
            FROM optimization_feedback
            WHERE timestamp >= $1
            GROUP BY category
            ORDER BY avg_rating ASC
        """

        rows = await conn.fetch(query, period_start)

        return {
            row['category']: {
                "count": row['count'],
                "avg_rating": round(row['avg_rating'], 2),
                "avg_accuracy": round(row['avg_accuracy'], 2) if row['avg_accuracy'] else None,
                "status": "needs_attention" if row['avg_rating'] < 3.5 else "ok"
            }
            for row in rows
        }

    async def _identify_systematic_issues(
        self,
        conn: asyncpg.Connection,
        period_start: datetime
    ) -> List[Dict[str, Any]]:
        """Identify systematic issues from patterns."""
        # Look for patterns in low-accuracy predictions
        query = """
            SELECT
                COUNT(*) as count,
                AVG(savings_accuracy) as avg_accuracy,
                AVG(ABS(actual_savings - predicted_savings)) as avg_error
            FROM optimization_feedback
            WHERE timestamp >= $1
              AND savings_accuracy IS NOT NULL
              AND savings_accuracy < 50
        """

        row = await conn.fetchrow(query, period_start)

        issues = []

        if row and row['count'] > 0:
            issues.append({
                "type": "low_accuracy_predictions",
                "count": row['count'],
                "avg_accuracy": round(row['avg_accuracy'], 2),
                "avg_error_kwh": round(row['avg_error'], 2),
                "recommendation": "Review prediction models and calibration"
            })

        return issues

    def _calculate_trend(self, values: List[float]) -> str:
        """
        Calculate trend direction from time series.

        Args:
            values: List of values (newest last)

        Returns:
            Trend direction: "improving", "declining", "stable"
        """
        if len(values) < 10:
            return "insufficient_data"

        # Simple linear regression slope
        x = np.arange(len(values))
        y = np.array(values)

        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]

        # Determine trend
        if slope > 0.05:
            return "improving"
        elif slope < -0.05:
            return "declining"
        else:
            return "stable"

    def _extract_top_complaints(
        self,
        comments: List[str],
        top_n: int = 5
    ) -> List[str]:
        """
        Extract top complaints from negative comments.

        Uses simple keyword extraction. In production, could use NLP.

        Args:
            comments: List of negative comments
            top_n: Number of top complaints to return

        Returns:
            List of top complaint phrases
        """
        if not comments:
            return []

        # Simple keyword extraction (production would use NLP)
        keywords = []
        for comment in comments:
            if comment:
                # Extract words longer than 4 characters
                words = [
                    word.lower().strip('.,!?')
                    for word in comment.split()
                    if len(word) > 4
                ]
                keywords.extend(words)

        # Count frequency
        counter = Counter(keywords)

        # Return top N
        return [word for word, count in counter.most_common(top_n)]

    async def generate_weekly_insights(self) -> Dict[str, Any]:
        """
        Generate weekly insights summary.

        Returns:
            Dictionary with weekly insights and recommendations
        """
        try:
            # Analyze last 7 days
            patterns = await self.analyze_feedback_patterns(days=7)

            # Identify underperforming
            underperforming = await self.identify_underperforming_optimizations()

            # Detect anomalies
            anomalies = await self.detect_anomalies(days=7)

            insights = {
                "week_number": datetime.utcnow().isocalendar()[1],
                "year": datetime.utcnow().year,
                "summary": {
                    "total_feedback": patterns.get("total_feedback", 0),
                    "avg_rating": patterns.get("average_rating", 0),
                    "trend": patterns.get("rating_trend", "unknown")
                },
                "underperforming_optimizations": underperforming[:5],
                "anomalies_detected": len(anomalies),
                "top_issues": patterns.get("systematic_issues", []),
                "recommendations": self._generate_recommendations(
                    patterns, underperforming, anomalies
                ),
                "generated_at": datetime.utcnow().isoformat()
            }

            logger.info(f"Weekly insights generated for week {insights['week_number']}")

            return insights

        except Exception as e:
            logger.error(f"Failed to generate weekly insights: {e}", exc_info=True)
            raise

    def _generate_recommendations(
        self,
        patterns: Dict[str, Any],
        underperforming: List[Dict[str, Any]],
        anomalies: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Check rating trend
        if patterns.get("rating_trend") == "declining":
            recommendations.append(
                "URGENT: User satisfaction is declining. Review recent changes and gather more feedback."
            )

        # Check underperforming optimizations
        if len(underperforming) > 5:
            recommendations.append(
                f"Review {len(underperforming)} underperforming optimizations. "
                "Consider disabling or improving low-rated recommendations."
            )

        # Check anomalies
        if anomalies:
            recommendations.append(
                f"{len(anomalies)} anomalies detected in satisfaction trends. "
                "Investigate potential issues or external factors."
            )

        # Check accuracy
        avg_accuracy = patterns.get("average_accuracy")
        if avg_accuracy and avg_accuracy < 75:
            recommendations.append(
                f"Prediction accuracy is low ({avg_accuracy:.1f}%). "
                "Recalibrate prediction models with recent data."
            )

        # Default recommendation if all good
        if not recommendations:
            recommendations.append(
                "No critical issues detected. Continue monitoring feedback trends."
            )

        return recommendations
