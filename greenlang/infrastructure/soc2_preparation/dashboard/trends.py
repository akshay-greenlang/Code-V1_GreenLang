# -*- coding: utf-8 -*-
"""
Trend Analyzer Module - SEC-009 Phase 9

Provides historical trend analysis and prediction for SOC 2 compliance metrics.
Analyzes readiness score trends, finding resolution velocity, and evidence
collection progress over time.

Classes:
    - TrendPoint: Data point for time series
    - FindingTrend: Finding trend data
    - EvidenceTrend: Evidence collection trend data
    - TrendAnalyzer: Main trend analysis class

Example:
    >>> analyzer = TrendAnalyzer(metrics_history)
    >>> readiness_trend = await analyzer.analyze_readiness_trend(days=90)
    >>> predicted_completion = analyzer.predict_completion(
    ...     current_rate=2.5, remaining=25
    ... )

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-009 SOC 2 Type II Preparation
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class TrendPoint(BaseModel):
    """Data point for time series analysis.

    Attributes:
        date: Date of the data point.
        value: Metric value at this date.
        metadata: Additional metadata for the point.
    """

    model_config = ConfigDict(extra="forbid")

    date: date = Field(..., description="Date of the data point.")
    value: float = Field(..., description="Metric value at this date.")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata."
    )


class FindingTrend(BaseModel):
    """Finding trend data over time.

    Attributes:
        date: Date of the data point.
        total_open: Total open findings.
        critical: Critical severity count.
        high: High severity count.
        medium: Medium severity count.
        low: Low severity count.
        resolved_today: Findings resolved on this date.
        opened_today: Findings opened on this date.
    """

    model_config = ConfigDict(extra="forbid")

    date: date = Field(..., description="Date of the data point.")
    total_open: int = Field(default=0, description="Total open findings.")
    critical: int = Field(default=0, description="Critical severity count.")
    high: int = Field(default=0, description="High severity count.")
    medium: int = Field(default=0, description="Medium severity count.")
    low: int = Field(default=0, description="Low severity count.")
    resolved_today: int = Field(default=0, description="Findings resolved today.")
    opened_today: int = Field(default=0, description="Findings opened today.")


class EvidenceTrend(BaseModel):
    """Evidence collection trend data over time.

    Attributes:
        date: Date of the data point.
        collected: Total evidence collected.
        required: Total required evidence.
        percentage: Collection percentage.
        collected_today: Evidence collected on this date.
    """

    model_config = ConfigDict(extra="forbid")

    date: date = Field(..., description="Date of the data point.")
    collected: int = Field(default=0, description="Total evidence collected.")
    required: int = Field(default=0, description="Total required evidence.")
    percentage: float = Field(default=0.0, description="Collection percentage.")
    collected_today: int = Field(default=0, description="Evidence collected today.")


# ---------------------------------------------------------------------------
# Trend Analyzer
# ---------------------------------------------------------------------------


class TrendAnalyzer:
    """Historical trend analysis and prediction for compliance metrics.

    Analyzes time series data to identify trends, calculate velocities,
    and predict future completion dates based on historical performance.

    Attributes:
        metrics_history: Historical metrics data store.

    Example:
        >>> analyzer = TrendAnalyzer(history)
        >>> trend = await analyzer.analyze_readiness_trend(days=90)
        >>> completion = analyzer.predict_completion(rate=2.5, remaining=25)
    """

    def __init__(self, metrics_history: Any = None) -> None:
        """Initialize TrendAnalyzer.

        Args:
            metrics_history: Historical metrics data store.
        """
        self.metrics_history = metrics_history

        # Internal storage for demo/testing
        self._readiness_history: List[TrendPoint] = []
        self._finding_history: List[FindingTrend] = []
        self._evidence_history: List[EvidenceTrend] = []

        logger.info("TrendAnalyzer initialized")

    async def analyze_readiness_trend(
        self,
        days: int = 90,
    ) -> List[TrendPoint]:
        """Analyze readiness score trend over time.

        Args:
            days: Number of days to analyze.

        Returns:
            List of TrendPoint objects showing readiness over time.
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=days)

        # Filter history to date range
        trend = [
            point
            for point in self._readiness_history
            if start_date <= point.date <= end_date
        ]

        # Sort by date
        trend.sort(key=lambda x: x.date)

        # If no data, generate sample trend (for demo)
        if not trend:
            trend = self._generate_sample_readiness_trend(start_date, end_date)

        # Calculate trend direction and velocity
        if len(trend) >= 2:
            start_value = trend[0].value
            end_value = trend[-1].value
            change = end_value - start_value
            velocity = change / days if days > 0 else 0

            logger.debug(
                "Readiness trend analyzed: days=%d, start=%.1f, end=%.1f, "
                "change=%.1f, velocity=%.2f/day",
                days,
                start_value,
                end_value,
                change,
                velocity,
            )

        return trend

    async def analyze_finding_trend(
        self,
        days: int = 90,
    ) -> List[FindingTrend]:
        """Analyze finding count trend over time.

        Args:
            days: Number of days to analyze.

        Returns:
            List of FindingTrend objects showing finding counts over time.
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=days)

        # Filter history to date range
        trend = [
            point
            for point in self._finding_history
            if start_date <= point.date <= end_date
        ]

        # Sort by date
        trend.sort(key=lambda x: x.date)

        # If no data, generate sample trend (for demo)
        if not trend:
            trend = self._generate_sample_finding_trend(start_date, end_date)

        return trend

    async def analyze_evidence_trend(
        self,
        days: int = 30,
    ) -> List[EvidenceTrend]:
        """Analyze evidence collection trend over time.

        Args:
            days: Number of days to analyze.

        Returns:
            List of EvidenceTrend objects showing evidence progress over time.
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=days)

        # Filter history to date range
        trend = [
            point
            for point in self._evidence_history
            if start_date <= point.date <= end_date
        ]

        # Sort by date
        trend.sort(key=lambda x: x.date)

        # If no data, generate sample trend (for demo)
        if not trend:
            trend = self._generate_sample_evidence_trend(start_date, end_date)

        return trend

    def predict_completion(
        self,
        current_rate: float,
        remaining: int,
    ) -> date:
        """Predict completion date based on current velocity.

        Args:
            current_rate: Current completion rate (items per day).
            remaining: Number of items remaining.

        Returns:
            Predicted completion date.
        """
        if current_rate <= 0:
            # No progress - predict far future
            return date.today() + timedelta(days=365)

        days_to_complete = int(remaining / current_rate)
        predicted_date = date.today() + timedelta(days=days_to_complete)

        logger.debug(
            "Completion predicted: rate=%.2f/day, remaining=%d, "
            "days=%d, date=%s",
            current_rate,
            remaining,
            days_to_complete,
            predicted_date.isoformat(),
        )

        return predicted_date

    def calculate_velocity(
        self,
        trend: List[TrendPoint],
        window_days: int = 7,
    ) -> float:
        """Calculate recent velocity from trend data.

        Args:
            trend: List of TrendPoint objects.
            window_days: Number of days for velocity calculation.

        Returns:
            Velocity (change per day) over the window.
        """
        if len(trend) < 2:
            return 0.0

        # Get recent points within window
        cutoff = date.today() - timedelta(days=window_days)
        recent = [p for p in trend if p.date >= cutoff]

        if len(recent) < 2:
            # Fall back to all data
            recent = trend

        recent.sort(key=lambda x: x.date)

        start_value = recent[0].value
        end_value = recent[-1].value
        days = (recent[-1].date - recent[0].date).days

        if days <= 0:
            return 0.0

        return (end_value - start_value) / days

    def calculate_moving_average(
        self,
        trend: List[TrendPoint],
        window_days: int = 7,
    ) -> List[TrendPoint]:
        """Calculate moving average for trend smoothing.

        Args:
            trend: List of TrendPoint objects.
            window_days: Number of days for moving average.

        Returns:
            List of TrendPoint with smoothed values.
        """
        if len(trend) < window_days:
            return trend

        smoothed: List[TrendPoint] = []
        sorted_trend = sorted(trend, key=lambda x: x.date)

        for i in range(len(sorted_trend)):
            # Get points in window
            window_start = i - window_days + 1
            if window_start < 0:
                window_start = 0

            window = sorted_trend[window_start : i + 1]
            avg_value = sum(p.value for p in window) / len(window)

            smoothed.append(
                TrendPoint(
                    date=sorted_trend[i].date,
                    value=round(avg_value, 2),
                    metadata={"type": "moving_average", "window": window_days},
                )
            )

        return smoothed

    def detect_anomalies(
        self,
        trend: List[TrendPoint],
        threshold_std: float = 2.0,
    ) -> List[TrendPoint]:
        """Detect anomalous data points in the trend.

        Args:
            trend: List of TrendPoint objects.
            threshold_std: Number of standard deviations for anomaly threshold.

        Returns:
            List of TrendPoint objects that are anomalies.
        """
        if len(trend) < 3:
            return []

        values = [p.value for p in trend]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = variance ** 0.5

        if std == 0:
            return []

        anomalies = [
            p
            for p in trend
            if abs(p.value - mean) > threshold_std * std
        ]

        if anomalies:
            logger.warning(
                "Anomalies detected in trend: count=%d, threshold=%.1f std",
                len(anomalies),
                threshold_std,
            )

        return anomalies

    # -----------------------------------------------------------------------
    # Data Management (for testing/demo)
    # -----------------------------------------------------------------------

    def add_readiness_point(self, point: TrendPoint) -> None:
        """Add a readiness trend data point."""
        self._readiness_history.append(point)

    def add_finding_point(self, point: FindingTrend) -> None:
        """Add a finding trend data point."""
        self._finding_history.append(point)

    def add_evidence_point(self, point: EvidenceTrend) -> None:
        """Add an evidence trend data point."""
        self._evidence_history.append(point)

    # -----------------------------------------------------------------------
    # Sample Data Generation (for demo)
    # -----------------------------------------------------------------------

    def _generate_sample_readiness_trend(
        self,
        start_date: date,
        end_date: date,
    ) -> List[TrendPoint]:
        """Generate sample readiness trend data."""
        trend = []
        current_date = start_date
        value = 45.0  # Starting readiness

        while current_date <= end_date:
            # Simulate gradual improvement with some noise
            import random

            random.seed(current_date.toordinal())
            daily_change = random.uniform(-1, 2)  # Slight positive bias
            value = max(0, min(100, value + daily_change))

            trend.append(
                TrendPoint(
                    date=current_date,
                    value=round(value, 1),
                    metadata={"generated": True},
                )
            )

            current_date += timedelta(days=1)

        return trend

    def _generate_sample_finding_trend(
        self,
        start_date: date,
        end_date: date,
    ) -> List[FindingTrend]:
        """Generate sample finding trend data."""
        trend = []
        current_date = start_date
        open_findings = {"critical": 2, "high": 5, "medium": 10, "low": 15}

        while current_date <= end_date:
            import random

            random.seed(current_date.toordinal())

            # Simulate resolution and new findings
            resolved = random.randint(0, 2)
            opened = random.randint(0, 1)

            # Prioritize resolving high-severity first
            for sev in ["critical", "high", "medium", "low"]:
                if resolved > 0 and open_findings[sev] > 0:
                    open_findings[sev] -= 1
                    resolved -= 1

            # New findings are usually medium/low
            if opened > 0:
                sev = random.choice(["medium", "low"])
                open_findings[sev] += 1

            total_open = sum(open_findings.values())

            trend.append(
                FindingTrend(
                    date=current_date,
                    total_open=total_open,
                    critical=open_findings["critical"],
                    high=open_findings["high"],
                    medium=open_findings["medium"],
                    low=open_findings["low"],
                    resolved_today=random.randint(0, 2),
                    opened_today=opened,
                )
            )

            current_date += timedelta(days=1)

        return trend

    def _generate_sample_evidence_trend(
        self,
        start_date: date,
        end_date: date,
    ) -> List[EvidenceTrend]:
        """Generate sample evidence collection trend data."""
        trend = []
        current_date = start_date
        collected = 20
        required = 100

        while current_date <= end_date:
            import random

            random.seed(current_date.toordinal())

            # Simulate daily collection
            collected_today = random.randint(0, 3)
            collected = min(required, collected + collected_today)
            percentage = (collected / required * 100) if required > 0 else 0

            trend.append(
                EvidenceTrend(
                    date=current_date,
                    collected=collected,
                    required=required,
                    percentage=round(percentage, 1),
                    collected_today=collected_today,
                )
            )

            current_date += timedelta(days=1)

        return trend


__all__ = [
    "TrendPoint",
    "FindingTrend",
    "EvidenceTrend",
    "TrendAnalyzer",
]
