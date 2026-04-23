"""
Explainability Report Generator for GL-016 Waterguard

This module generates daily and periodic reports on model explanations,
feature importance trends, and explanation consistency metrics.

All reports are generated from structured data - NO generative AI.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .explanation_schemas import (
    ExplanationMethod,
    ExplanationPayload,
    FeatureContribution,
    GlobalExplanation,
    LocalExplanation,
    RecommendationType,
)

logger = logging.getLogger(__name__)


@dataclass
class FeatureImportanceTrend:
    """Trend data for a single feature's importance over time."""
    feature_name: str
    timestamps: List[datetime] = field(default_factory=list)
    importance_values: List[float] = field(default_factory=list)
    contribution_mean: float = 0.0
    contribution_std: float = 0.0
    trend_direction: str = "stable"  # increasing, decreasing, stable
    trend_magnitude: float = 0.0

    def add_observation(self, timestamp: datetime, importance: float) -> None:
        """Add a new observation to the trend."""
        self.timestamps.append(timestamp)
        self.importance_values.append(importance)
        self._update_statistics()

    def _update_statistics(self) -> None:
        """Update trend statistics."""
        if len(self.importance_values) < 2:
            return

        values = np.array(self.importance_values)
        self.contribution_mean = float(np.mean(values))
        self.contribution_std = float(np.std(values))

        # Simple linear trend
        if len(values) >= 3:
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            self.trend_magnitude = float(abs(slope))

            if slope > 0.01:
                self.trend_direction = "increasing"
            elif slope < -0.01:
                self.trend_direction = "decreasing"
            else:
                self.trend_direction = "stable"


@dataclass
class ExplanationConsistencyMetrics:
    """Metrics for explanation consistency over time."""
    period_start: datetime
    period_end: datetime
    total_explanations: int = 0
    reliable_explanations: int = 0
    unreliable_explanations: int = 0
    average_confidence: float = 0.0
    confidence_std: float = 0.0
    method_distribution: Dict[str, int] = field(default_factory=dict)
    top_feature_stability: float = 0.0  # How often same top feature appears
    recommendation_distribution: Dict[str, int] = field(default_factory=dict)

    @property
    def reliability_rate(self) -> float:
        """Calculate reliability rate."""
        if self.total_explanations == 0:
            return 0.0
        return self.reliable_explanations / self.total_explanations


@dataclass
class DailyExplainabilityReport:
    """Daily report on explainability metrics."""
    report_id: str
    report_date: datetime
    period_start: datetime
    period_end: datetime

    # Summary statistics
    total_recommendations: int = 0
    total_explanations: int = 0
    explanations_generated: int = 0

    # Feature importance
    feature_importance_summary: Dict[str, float] = field(default_factory=dict)
    top_features_today: List[str] = field(default_factory=list)
    feature_importance_changes: Dict[str, float] = field(default_factory=dict)

    # Consistency metrics
    consistency_metrics: Optional[ExplanationConsistencyMetrics] = None

    # Method usage
    shap_usage_count: int = 0
    lime_usage_count: int = 0
    combined_usage_count: int = 0

    # Reliability
    reliable_explanation_rate: float = 0.0
    average_confidence: float = 0.0
    ood_detection_count: int = 0

    # Trends
    feature_trends: List[FeatureImportanceTrend] = field(default_factory=list)

    # Drift indicators
    drift_detected: bool = False
    drift_features: List[str] = field(default_factory=list)
    drift_severity: str = "none"  # none, low, medium, high

    # Model info
    model_version: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for serialization."""
        return {
            'report_id': self.report_id,
            'report_date': self.report_date.isoformat(),
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'summary': {
                'total_recommendations': self.total_recommendations,
                'total_explanations': self.total_explanations,
                'reliable_rate': self.reliable_explanation_rate,
                'average_confidence': self.average_confidence,
            },
            'feature_importance': {
                'summary': self.feature_importance_summary,
                'top_features': self.top_features_today,
                'changes': self.feature_importance_changes,
            },
            'method_usage': {
                'shap': self.shap_usage_count,
                'lime': self.lime_usage_count,
                'combined': self.combined_usage_count,
            },
            'drift': {
                'detected': self.drift_detected,
                'features': self.drift_features,
                'severity': self.drift_severity,
            },
            'model_version': self.model_version,
        }

    def to_json(self) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class ExplainabilityReportGenerator:
    """
    Generator for explainability reports.

    Produces daily reports with:
    - Feature importance trends over time
    - Explanation consistency metrics
    - Model drift indicators from explanation changes
    - Reliability statistics
    """

    def __init__(
        self,
        history_days: int = 30,
        trend_window_days: int = 7,
        drift_threshold: float = 0.2
    ):
        """
        Initialize the report generator.

        Args:
            history_days: Days of history to maintain
            trend_window_days: Days for trend calculation
            drift_threshold: Threshold for drift detection
        """
        self.history_days = history_days
        self.trend_window_days = trend_window_days
        self.drift_threshold = drift_threshold

        # Storage for historical data
        self._explanation_history: List[LocalExplanation] = []
        self._global_explanations: List[GlobalExplanation] = []
        self._feature_trends: Dict[str, FeatureImportanceTrend] = {}
        self._daily_stats: Dict[str, Dict[str, Any]] = {}

    def add_explanation(self, explanation: LocalExplanation) -> None:
        """Add an explanation to history."""
        self._explanation_history.append(explanation)
        self._update_feature_trends(explanation)
        self._cleanup_old_history()

    def add_global_explanation(self, global_exp: GlobalExplanation) -> None:
        """Add a global explanation to history."""
        self._global_explanations.append(global_exp)
        self._cleanup_old_history()

    def _update_feature_trends(self, explanation: LocalExplanation) -> None:
        """Update feature trends from explanation."""
        for feature in explanation.features:
            if feature.feature_name not in self._feature_trends:
                self._feature_trends[feature.feature_name] = FeatureImportanceTrend(
                    feature_name=feature.feature_name
                )
            self._feature_trends[feature.feature_name].add_observation(
                timestamp=explanation.timestamp,
                importance=abs(feature.contribution)
            )

    def _cleanup_old_history(self) -> None:
        """Remove history older than retention period."""
        cutoff = datetime.utcnow() - timedelta(days=self.history_days)

        self._explanation_history = [
            e for e in self._explanation_history
            if e.timestamp > cutoff
        ]
        self._global_explanations = [
            g for g in self._global_explanations
            if g.timestamp > cutoff
        ]

    def generate_daily_report(
        self,
        recommendations: List[Dict[str, Any]],
        explanations: List[LocalExplanation],
        report_date: Optional[datetime] = None,
        model_version: str = "unknown"
    ) -> DailyExplainabilityReport:
        """
        Generate daily explainability report.

        Args:
            recommendations: List of recommendations made today
            explanations: List of explanations generated today
            report_date: Date for report (defaults to today)
            model_version: Model version string

        Returns:
            DailyExplainabilityReport with all metrics
        """
        import uuid

        report_date = report_date or datetime.utcnow()
        period_start = report_date.replace(hour=0, minute=0, second=0, microsecond=0)
        period_end = period_start + timedelta(days=1) - timedelta(microseconds=1)

        report = DailyExplainabilityReport(
            report_id=str(uuid.uuid4()),
            report_date=report_date,
            period_start=period_start,
            period_end=period_end,
            model_version=model_version,
            total_recommendations=len(recommendations),
            total_explanations=len(explanations)
        )

        if not explanations:
            return report

        # Add to history
        for exp in explanations:
            self.add_explanation(exp)

        # Calculate metrics
        self._calculate_method_usage(explanations, report)
        self._calculate_reliability_metrics(explanations, report)
        self._calculate_feature_importance(explanations, report)
        self._calculate_consistency_metrics(explanations, report)
        self._detect_drift(report)

        return report

    def _calculate_method_usage(
        self,
        explanations: List[LocalExplanation],
        report: DailyExplainabilityReport
    ) -> None:
        """Calculate method usage statistics."""
        for exp in explanations:
            if exp.method == ExplanationMethod.SHAP:
                report.shap_usage_count += 1
            elif exp.method == ExplanationMethod.LIME:
                report.lime_usage_count += 1
            else:
                report.combined_usage_count += 1

        report.explanations_generated = len(explanations)

    def _calculate_reliability_metrics(
        self,
        explanations: List[LocalExplanation],
        report: DailyExplainabilityReport
    ) -> None:
        """Calculate reliability metrics."""
        reliable_count = sum(1 for e in explanations if e.is_reliable)
        report.reliable_explanation_rate = (
            reliable_count / len(explanations) if explanations else 0.0
        )

        confidences = [e.confidence for e in explanations]
        report.average_confidence = float(np.mean(confidences)) if confidences else 0.0

        # Count OOD detections (from warnings)
        report.ood_detection_count = sum(
            1 for e in explanations
            if any('out-of-distribution' in w.lower() for w in e.warning_messages)
        )

    def _calculate_feature_importance(
        self,
        explanations: List[LocalExplanation],
        report: DailyExplainabilityReport
    ) -> None:
        """Calculate feature importance summary."""
        feature_contributions: Dict[str, List[float]] = defaultdict(list)

        for exp in explanations:
            for feat in exp.features:
                feature_contributions[feat.feature_name].append(
                    abs(feat.contribution)
                )

        # Calculate mean importance for each feature
        for name, contributions in feature_contributions.items():
            report.feature_importance_summary[name] = float(np.mean(contributions))

        # Get top features
        sorted_features = sorted(
            report.feature_importance_summary.items(),
            key=lambda x: x[1],
            reverse=True
        )
        report.top_features_today = [f[0] for f in sorted_features[:5]]

        # Calculate changes from historical baseline
        for name, current_importance in report.feature_importance_summary.items():
            trend = self._feature_trends.get(name)
            if trend and trend.contribution_mean > 0:
                change = (current_importance - trend.contribution_mean) / trend.contribution_mean
                report.feature_importance_changes[name] = float(change)

        # Copy feature trends
        report.feature_trends = list(self._feature_trends.values())

    def _calculate_consistency_metrics(
        self,
        explanations: List[LocalExplanation],
        report: DailyExplainabilityReport
    ) -> None:
        """Calculate explanation consistency metrics."""
        metrics = ExplanationConsistencyMetrics(
            period_start=report.period_start,
            period_end=report.period_end,
            total_explanations=len(explanations)
        )

        metrics.reliable_explanations = sum(1 for e in explanations if e.is_reliable)
        metrics.unreliable_explanations = len(explanations) - metrics.reliable_explanations

        confidences = [e.confidence for e in explanations]
        metrics.average_confidence = float(np.mean(confidences)) if confidences else 0.0
        metrics.confidence_std = float(np.std(confidences)) if confidences else 0.0

        # Method distribution
        method_counts: Dict[str, int] = defaultdict(int)
        for exp in explanations:
            method_counts[exp.method.value] += 1
        metrics.method_distribution = dict(method_counts)

        # Top feature stability
        if explanations:
            top_features = [
                exp.get_top_features(1)[0].feature_name
                if exp.features else None
                for exp in explanations
            ]
            top_features = [f for f in top_features if f is not None]
            if top_features:
                most_common = max(set(top_features), key=top_features.count)
                metrics.top_feature_stability = top_features.count(most_common) / len(top_features)

        report.consistency_metrics = metrics

    def _detect_drift(self, report: DailyExplainabilityReport) -> None:
        """Detect explanation drift from feature importance changes."""
        significant_changes = []

        for name, change in report.feature_importance_changes.items():
            if abs(change) > self.drift_threshold:
                significant_changes.append((name, change))

        if significant_changes:
            report.drift_detected = True
            report.drift_features = [name for name, _ in significant_changes]

            # Determine severity
            max_change = max(abs(c) for _, c in significant_changes)
            if max_change > 0.5:
                report.drift_severity = "high"
            elif max_change > 0.3:
                report.drift_severity = "medium"
            else:
                report.drift_severity = "low"

    def get_feature_trend(
        self,
        feature_name: str,
        days: Optional[int] = None
    ) -> Optional[FeatureImportanceTrend]:
        """Get trend data for a specific feature."""
        trend = self._feature_trends.get(feature_name)
        if trend is None:
            return None

        if days is None:
            return trend

        # Filter to requested days
        cutoff = datetime.utcnow() - timedelta(days=days)
        filtered_trend = FeatureImportanceTrend(feature_name=feature_name)

        for ts, val in zip(trend.timestamps, trend.importance_values):
            if ts > cutoff:
                filtered_trend.add_observation(ts, val)

        return filtered_trend

    def get_historical_summary(
        self,
        days: int = 7
    ) -> Dict[str, Any]:
        """Get summary of historical explanations."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        recent = [e for e in self._explanation_history if e.timestamp > cutoff]

        if not recent:
            return {
                'period_days': days,
                'total_explanations': 0,
                'message': 'No explanations in period'
            }

        return {
            'period_days': days,
            'total_explanations': len(recent),
            'average_confidence': float(np.mean([e.confidence for e in recent])),
            'reliability_rate': sum(1 for e in recent if e.is_reliable) / len(recent),
            'method_distribution': dict(
                defaultdict(int, [(e.method.value, 1) for e in recent])
            ),
        }

    def clear_history(self) -> None:
        """Clear all historical data."""
        self._explanation_history.clear()
        self._global_explanations.clear()
        self._feature_trends.clear()
        self._daily_stats.clear()
        logger.info("Explanation history cleared")
