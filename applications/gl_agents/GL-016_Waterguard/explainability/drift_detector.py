"""
Explanation Drift Detection for GL-016 Waterguard

This module detects drift in model explanations over time, which can indicate
model degradation, data distribution shift, or concept drift.

All detection is based on structured statistical analysis - NO generative AI.
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .explanation_schemas import (
    ExplanationStabilityMetrics,
    FeatureContribution,
    LocalExplanation,
)

logger = logging.getLogger(__name__)


class DriftSeverity(str, Enum):
    """Severity levels for detected drift."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DriftType(str, Enum):
    """Types of drift that can be detected."""
    FEATURE_IMPORTANCE = "feature_importance"
    EXPLANATION_INSTABILITY = "explanation_instability"
    DISTRIBUTION_SHIFT = "distribution_shift"
    CONCEPT_DRIFT = "concept_drift"


@dataclass
class DriftAlert:
    """Alert generated when drift is detected."""
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    drift_type: DriftType = DriftType.FEATURE_IMPORTANCE
    severity: DriftSeverity = DriftSeverity.LOW
    affected_features: List[str] = field(default_factory=list)
    description: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    recommended_action: str = ""
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None

    def acknowledge(self, user: str) -> None:
        """Mark alert as acknowledged."""
        self.acknowledged = True
        self.acknowledged_by = user
        self.acknowledged_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'alert_id': self.alert_id,
            'timestamp': self.timestamp.isoformat(),
            'drift_type': self.drift_type.value,
            'severity': self.severity.value,
            'affected_features': self.affected_features,
            'description': self.description,
            'metrics': self.metrics,
            'recommended_action': self.recommended_action,
            'acknowledged': self.acknowledged,
        }


@dataclass
class DriftDetectionResult:
    """Result of drift detection analysis."""
    detection_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    drift_detected: bool = False
    overall_severity: DriftSeverity = DriftSeverity.NONE
    alerts: List[DriftAlert] = field(default_factory=list)

    # Feature-level drift metrics
    feature_drift_scores: Dict[str, float] = field(default_factory=dict)
    drifted_features: List[str] = field(default_factory=list)

    # Instability metrics
    instability_score: float = 0.0
    instability_details: Dict[str, Any] = field(default_factory=dict)

    # Analysis window
    baseline_period: Optional[Tuple[datetime, datetime]] = None
    analysis_period: Optional[Tuple[datetime, datetime]] = None

    # Model info
    model_version: str = ""

    def add_alert(self, alert: DriftAlert) -> None:
        """Add an alert to the result."""
        self.alerts.append(alert)
        self.drift_detected = True
        if alert.severity.value > self.overall_severity.value:
            self.overall_severity = alert.severity


@dataclass
class BaselineStatistics:
    """Baseline statistics for drift comparison."""
    model_version: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    sample_count: int = 0

    # Feature importance baseline
    feature_importance_mean: Dict[str, float] = field(default_factory=dict)
    feature_importance_std: Dict[str, float] = field(default_factory=dict)

    # Top feature ranking
    top_feature_ranking: List[str] = field(default_factory=list)

    # Confidence baseline
    confidence_mean: float = 0.0
    confidence_std: float = 0.0

    # Direction distribution
    direction_distribution: Dict[str, Dict[str, float]] = field(default_factory=dict)


class ExplanationDriftDetector:
    """
    Detector for explanation drift and instability.

    Monitors explanations over time to detect:
    - Feature importance drift (changing which features matter)
    - Explanation instability (inconsistent explanations for similar inputs)
    - Distribution shift (changes in feature value distributions)
    - Concept drift (changes in model behavior patterns)

    All detection is deterministic and based on statistical analysis.
    """

    def __init__(
        self,
        baseline_window_days: int = 14,
        analysis_window_days: int = 1,
        feature_drift_threshold: float = 0.3,
        instability_threshold: float = 0.4,
        ranking_drift_threshold: int = 2,
        alert_cooldown_minutes: int = 60,
        max_alerts_per_day: int = 10
    ):
        """
        Initialize the drift detector.

        Args:
            baseline_window_days: Days to use for baseline
            analysis_window_days: Days for analysis window
            feature_drift_threshold: Threshold for feature importance drift
            instability_threshold: Threshold for explanation instability
            ranking_drift_threshold: Max rank change before alert
            alert_cooldown_minutes: Minimum time between similar alerts
            max_alerts_per_day: Maximum alerts per day
        """
        self.baseline_window_days = baseline_window_days
        self.analysis_window_days = analysis_window_days
        self.feature_drift_threshold = feature_drift_threshold
        self.instability_threshold = instability_threshold
        self.ranking_drift_threshold = ranking_drift_threshold
        self.alert_cooldown_minutes = alert_cooldown_minutes
        self.max_alerts_per_day = max_alerts_per_day

        # Storage
        self._baselines: Dict[str, BaselineStatistics] = {}
        self._explanation_history: List[LocalExplanation] = []
        self._alerts: List[DriftAlert] = []
        self._alert_handlers: List[Callable[[DriftAlert], None]] = []

        # Daily alert count
        self._daily_alert_count = 0
        self._last_alert_date: Optional[datetime] = None

    def set_baseline(
        self,
        explanations: List[LocalExplanation],
        model_version: str
    ) -> BaselineStatistics:
        """
        Set baseline statistics from a set of explanations.

        Args:
            explanations: List of explanations for baseline
            model_version: Model version string

        Returns:
            BaselineStatistics computed from explanations
        """
        baseline = BaselineStatistics(
            model_version=model_version,
            sample_count=len(explanations)
        )

        if not explanations:
            self._baselines[model_version] = baseline
            return baseline

        # Compute feature importance statistics
        feature_contributions: Dict[str, List[float]] = defaultdict(list)
        confidences = []

        for exp in explanations:
            confidences.append(exp.confidence)
            for feat in exp.features:
                feature_contributions[feat.feature_name].append(
                    abs(feat.contribution)
                )

        for name, contributions in feature_contributions.items():
            baseline.feature_importance_mean[name] = float(np.mean(contributions))
            baseline.feature_importance_std[name] = float(np.std(contributions))

        # Compute top feature ranking
        sorted_features = sorted(
            baseline.feature_importance_mean.items(),
            key=lambda x: x[1],
            reverse=True
        )
        baseline.top_feature_ranking = [f[0] for f in sorted_features[:10]]

        # Confidence statistics
        baseline.confidence_mean = float(np.mean(confidences))
        baseline.confidence_std = float(np.std(confidences))

        # Direction distribution per feature
        for name in feature_contributions.keys():
            directions = defaultdict(int)
            for exp in explanations:
                for feat in exp.features:
                    if feat.feature_name == name:
                        directions[feat.direction.value] += 1
            total = sum(directions.values())
            if total > 0:
                baseline.direction_distribution[name] = {
                    d: c / total for d, c in directions.items()
                }

        self._baselines[model_version] = baseline
        logger.info(f"Baseline set for model {model_version} with {len(explanations)} samples")
        return baseline

    def detect_feature_drift(
        self,
        recent_explanations: List[LocalExplanation],
        baseline: Optional[BaselineStatistics] = None,
        model_version: Optional[str] = None
    ) -> DriftDetectionResult:
        """
        Detect drift in feature importance compared to baseline.

        Args:
            recent_explanations: Recent explanations to analyze
            baseline: Baseline to compare against (uses stored if None)
            model_version: Model version (uses first explanation's if None)

        Returns:
            DriftDetectionResult with drift analysis
        """
        result = DriftDetectionResult()

        if not recent_explanations:
            return result

        # Get model version and baseline
        if model_version is None:
            model_version = recent_explanations[0].model_version
        result.model_version = model_version

        if baseline is None:
            baseline = self._baselines.get(model_version)

        if baseline is None:
            logger.warning(f"No baseline found for model {model_version}")
            return result

        # Set analysis period
        result.analysis_period = (
            min(e.timestamp for e in recent_explanations),
            max(e.timestamp for e in recent_explanations)
        )

        # Compute current feature importance
        current_importance: Dict[str, List[float]] = defaultdict(list)
        for exp in recent_explanations:
            for feat in exp.features:
                current_importance[feat.feature_name].append(abs(feat.contribution))

        current_means = {
            name: float(np.mean(vals))
            for name, vals in current_importance.items()
        }

        # Detect drift per feature
        drifted_features = []
        for name, current_mean in current_means.items():
            baseline_mean = baseline.feature_importance_mean.get(name, 0)
            baseline_std = baseline.feature_importance_std.get(name, 0.1)

            if baseline_mean > 0:
                # Z-score based drift detection
                z_score = abs(current_mean - baseline_mean) / max(baseline_std, 0.01)
                drift_score = min(z_score / 3.0, 1.0)  # Normalize to 0-1
            else:
                drift_score = 1.0 if current_mean > 0.1 else 0.0

            result.feature_drift_scores[name] = drift_score

            if drift_score > self.feature_drift_threshold:
                drifted_features.append(name)

        result.drifted_features = drifted_features

        # Check ranking drift
        current_ranking = sorted(
            current_means.items(), key=lambda x: x[1], reverse=True
        )
        current_top = [f[0] for f in current_ranking[:5]]

        ranking_changes = []
        for i, feature in enumerate(current_top):
            if feature in baseline.top_feature_ranking:
                baseline_rank = baseline.top_feature_ranking.index(feature)
                if abs(i - baseline_rank) > self.ranking_drift_threshold:
                    ranking_changes.append(feature)

        # Generate alerts
        if drifted_features:
            severity = self._calculate_severity(
                len(drifted_features),
                max(result.feature_drift_scores.values())
            )

            alert = DriftAlert(
                drift_type=DriftType.FEATURE_IMPORTANCE,
                severity=severity,
                affected_features=drifted_features,
                description=f"Feature importance drift detected in {len(drifted_features)} features",
                metrics={
                    'drifted_count': len(drifted_features),
                    'max_drift_score': max(result.feature_drift_scores.values()),
                },
                recommended_action="Review model performance and consider retraining"
            )
            result.add_alert(alert)
            self._emit_alert(alert)

        if ranking_changes:
            alert = DriftAlert(
                drift_type=DriftType.CONCEPT_DRIFT,
                severity=DriftSeverity.MEDIUM,
                affected_features=ranking_changes,
                description=f"Feature ranking changed significantly for {len(ranking_changes)} features",
                metrics={'ranking_changes': len(ranking_changes)},
                recommended_action="Investigate changes in data patterns"
            )
            result.add_alert(alert)
            self._emit_alert(alert)

        return result

    def detect_explanation_instability(
        self,
        repeated_explanations: List[LocalExplanation]
    ) -> DriftDetectionResult:
        """
        Detect instability in explanations for similar inputs.

        This checks if explanations are consistent when they should be
        (e.g., same or very similar inputs producing different explanations).

        Args:
            repeated_explanations: Explanations that should be similar

        Returns:
            DriftDetectionResult with instability analysis
        """
        result = DriftDetectionResult()

        if len(repeated_explanations) < 2:
            return result

        result.model_version = repeated_explanations[0].model_version

        # Group by recommendation_id to find repeated explanations
        groups: Dict[str, List[LocalExplanation]] = defaultdict(list)
        for exp in repeated_explanations:
            groups[exp.recommendation_id].append(exp)

        # Analyze stability within each group
        instability_scores = []
        unstable_features = set()

        for rec_id, group in groups.items():
            if len(group) < 2:
                continue

            # Compare feature contributions across group
            feature_variations: Dict[str, List[float]] = defaultdict(list)
            for exp in group:
                for feat in exp.features:
                    feature_variations[feat.feature_name].append(feat.contribution)

            for name, contributions in feature_variations.items():
                if len(contributions) > 1:
                    cv = np.std(contributions) / (np.mean(np.abs(contributions)) + 1e-10)
                    if cv > self.instability_threshold:
                        unstable_features.add(name)
                        instability_scores.append(cv)

        if instability_scores:
            result.instability_score = float(np.mean(instability_scores))
            result.instability_details = {
                'unstable_feature_count': len(unstable_features),
                'unstable_features': list(unstable_features),
                'max_variation': float(max(instability_scores)),
            }

            if result.instability_score > self.instability_threshold:
                severity = self._calculate_severity(
                    len(unstable_features),
                    result.instability_score
                )

                alert = DriftAlert(
                    drift_type=DriftType.EXPLANATION_INSTABILITY,
                    severity=severity,
                    affected_features=list(unstable_features),
                    description=f"Explanation instability detected (score: {result.instability_score:.2f})",
                    metrics={
                        'instability_score': result.instability_score,
                        'affected_features': len(unstable_features),
                    },
                    recommended_action="Review model stability and input preprocessing"
                )
                result.add_alert(alert)
                self._emit_alert(alert)

        return result

    def _calculate_severity(
        self,
        affected_count: int,
        drift_score: float
    ) -> DriftSeverity:
        """Calculate severity level from metrics."""
        if drift_score > 0.8 or affected_count > 5:
            return DriftSeverity.CRITICAL
        elif drift_score > 0.6 or affected_count > 3:
            return DriftSeverity.HIGH
        elif drift_score > 0.4 or affected_count > 2:
            return DriftSeverity.MEDIUM
        elif drift_score > 0.2 or affected_count > 0:
            return DriftSeverity.LOW
        else:
            return DriftSeverity.NONE

    def _emit_alert(self, alert: DriftAlert) -> None:
        """Emit an alert to registered handlers."""
        # Check daily limit
        today = datetime.utcnow().date()
        if self._last_alert_date != today:
            self._daily_alert_count = 0
            self._last_alert_date = today

        if self._daily_alert_count >= self.max_alerts_per_day:
            logger.warning("Daily alert limit reached, suppressing alert")
            return

        # Check cooldown
        recent_similar = [
            a for a in self._alerts
            if a.drift_type == alert.drift_type
            and (datetime.utcnow() - a.timestamp).total_seconds() < self.alert_cooldown_minutes * 60
        ]
        if recent_similar:
            logger.debug("Alert suppressed due to cooldown")
            return

        # Store and emit
        self._alerts.append(alert)
        self._daily_alert_count += 1

        for handler in self._alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

        logger.info(f"Drift alert emitted: {alert.drift_type.value} - {alert.severity.value}")

    def register_alert_handler(
        self,
        handler: Callable[[DriftAlert], None]
    ) -> None:
        """Register a handler for drift alerts."""
        self._alert_handlers.append(handler)

    def get_recent_alerts(
        self,
        hours: int = 24,
        severity_filter: Optional[DriftSeverity] = None
    ) -> List[DriftAlert]:
        """Get recent alerts."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        alerts = [a for a in self._alerts if a.timestamp > cutoff]

        if severity_filter:
            alerts = [a for a in alerts if a.severity == severity_filter]

        return alerts

    def get_baseline(self, model_version: str) -> Optional[BaselineStatistics]:
        """Get stored baseline for a model version."""
        return self._baselines.get(model_version)

    def log_for_governance(
        self,
        result: DriftDetectionResult,
        logger_func: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Create governance log entry for drift detection.

        Args:
            result: Drift detection result
            logger_func: Optional custom logging function

        Returns:
            Dictionary with governance log data
        """
        log_entry = {
            'detection_id': result.detection_id,
            'timestamp': result.timestamp.isoformat(),
            'model_version': result.model_version,
            'drift_detected': result.drift_detected,
            'severity': result.overall_severity.value,
            'drifted_features': result.drifted_features,
            'feature_drift_scores': result.feature_drift_scores,
            'instability_score': result.instability_score,
            'alert_count': len(result.alerts),
            'alerts': [a.to_dict() for a in result.alerts],
        }

        if logger_func:
            logger_func(log_entry)
        else:
            logger.info(f"Governance log: {log_entry}")

        return log_entry

    def clear_history(self) -> None:
        """Clear historical data."""
        self._explanation_history.clear()
        self._alerts.clear()
        logger.info("Drift detector history cleared")
