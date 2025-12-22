"""
GL-004 BURNMASTER Drift Monitor Module

This module provides comprehensive drift monitoring for combustion optimization
operations, including data drift detection, model drift monitoring, and
process drift analysis with recalibration recommendations.

Example:
    >>> monitor = DriftMonitor()
    >>> data_drift = monitor.monitor_data_drift(current_data)
    >>> model_drift = monitor.monitor_model_drift(predictions, actuals)
    >>> if data_drift.is_drifted:
    ...     recommendation = monitor.recommend_recalibration(data_drift)
"""

from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging
import statistics
import uuid

from pydantic import BaseModel, Field, validator

# Try to import pandas, fallback to dict-based approach if not available
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class DriftType(str, Enum):
    """Type of drift detected."""
    DATA_DRIFT = "DATA_DRIFT"
    MODEL_DRIFT = "MODEL_DRIFT"
    PROCESS_DRIFT = "PROCESS_DRIFT"
    CONCEPT_DRIFT = "CONCEPT_DRIFT"


class DriftSeverity(str, Enum):
    """Severity of drift."""
    NONE = "NONE"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RecalibrationUrgency(str, Enum):
    """Urgency level for recalibration."""
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    IMMEDIATE = "IMMEDIATE"


# =============================================================================
# DATA MODELS
# =============================================================================

class FeatureDrift(BaseModel):
    """Drift information for a single feature."""

    feature_name: str = Field(..., description="Feature name")
    drift_detected: bool = Field(default=False, description="Drift detected")
    drift_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Drift score 0-1"
    )

    # Statistical changes
    baseline_mean: Optional[float] = Field(None, description="Baseline mean")
    current_mean: Optional[float] = Field(None, description="Current mean")
    mean_shift: Optional[float] = Field(None, description="Mean shift")

    baseline_std: Optional[float] = Field(None, description="Baseline std dev")
    current_std: Optional[float] = Field(None, description="Current std dev")
    std_ratio: Optional[float] = Field(None, description="Std dev ratio")

    # Distribution change
    psi_score: Optional[float] = Field(
        None, description="Population Stability Index"
    )
    ks_statistic: Optional[float] = Field(
        None, description="Kolmogorov-Smirnov statistic"
    )


class DriftStatus(BaseModel):
    """Status of drift detection analysis."""

    drift_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Drift analysis identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis timestamp"
    )

    # Overall assessment
    drift_type: DriftType = Field(..., description="Type of drift")
    is_drifted: bool = Field(default=False, description="Overall drift detected")
    severity: DriftSeverity = Field(
        default=DriftSeverity.NONE, description="Drift severity"
    )
    overall_drift_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Overall drift score"
    )

    # Feature-level drift
    drifted_features: List[str] = Field(
        default_factory=list, description="Features with detected drift"
    )
    feature_drift_details: List[FeatureDrift] = Field(
        default_factory=list, description="Per-feature drift details"
    )

    # Statistics
    features_analyzed: int = Field(default=0, ge=0, description="Features analyzed")
    features_drifted: int = Field(default=0, ge=0, description="Features drifted")
    drift_percentage: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Percentage of features drifted"
    )

    # Time context
    baseline_period: Optional[str] = Field(None, description="Baseline period")
    current_period: Optional[str] = Field(None, description="Current period")
    samples_analyzed: int = Field(default=0, ge=0, description="Samples analyzed")

    # Recommendations
    requires_action: bool = Field(
        default=False, description="Action required"
    )
    recommended_action: Optional[str] = Field(
        None, description="Recommended action"
    )

    # Audit
    provenance_hash: str = Field(default="", description="SHA-256 audit hash")


class AlertResult(BaseModel):
    """Result of sending a drift alert."""

    success: bool = Field(..., description="Alert sent successfully")
    drift_id: str = Field(..., description="Drift analysis ID")
    alert_type: str = Field(..., description="Type of alert")
    severity: DriftSeverity = Field(..., description="Alert severity")
    recipients: List[str] = Field(
        default_factory=list, description="Alert recipients"
    )
    sent_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Alert timestamp"
    )
    message: str = Field(default="", description="Alert message")
    error: Optional[str] = Field(None, description="Error if failed")


class RecalibrationRecommendation(BaseModel):
    """Recommendation for model/system recalibration."""

    recommendation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Recommendation identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Recommendation timestamp"
    )

    # Urgency and priority
    urgency: RecalibrationUrgency = Field(
        ..., description="Recalibration urgency"
    )
    priority_score: float = Field(
        ..., ge=0.0, le=100.0, description="Priority score 0-100"
    )

    # Drift context
    drift_status: DriftStatus = Field(..., description="Associated drift status")

    # Recommendations
    recommended_actions: List[str] = Field(
        default_factory=list, description="Recommended actions"
    )
    affected_components: List[str] = Field(
        default_factory=list, description="Components needing recalibration"
    )

    # Timeline
    recommended_deadline: Optional[datetime] = Field(
        None, description="Recommended deadline"
    )
    estimated_effort_hours: Optional[float] = Field(
        None, description="Estimated effort in hours"
    )

    # Impact assessment
    risk_if_not_addressed: str = Field(
        default="UNKNOWN", description="Risk if not addressed"
    )
    expected_improvement: Optional[str] = Field(
        None, description="Expected improvement after recalibration"
    )

    # Audit
    provenance_hash: str = Field(default="", description="SHA-256 audit hash")


# =============================================================================
# DRIFT MONITOR
# =============================================================================

class DriftMonitor:
    """
    Comprehensive drift monitoring for combustion optimization.

    Monitors data drift, model drift, and process drift to ensure
    model and system reliability. Provides alerts and recalibration
    recommendations when significant drift is detected.

    Attributes:
        drift_threshold: Threshold for considering drift significant

    Example:
        >>> monitor = DriftMonitor()
        >>> data_drift = monitor.monitor_data_drift(current_data)
        >>> if data_drift.is_drifted:
        ...     alert = monitor.alert_on_drift(data_drift)
        ...     recommendation = monitor.recommend_recalibration(data_drift)
    """

    # Default thresholds
    DRIFT_THRESHOLD = 0.15  # 15% drift score triggers alert
    PSI_THRESHOLD = 0.2  # Population Stability Index threshold
    KS_THRESHOLD = 0.1  # Kolmogorov-Smirnov threshold
    MEAN_SHIFT_THRESHOLD = 2.0  # Standard deviations for mean shift

    def __init__(
        self,
        drift_threshold: float = 0.15,
        alert_handlers: Optional[List[Any]] = None
    ):
        """
        Initialize the DriftMonitor.

        Args:
            drift_threshold: Threshold for significant drift (0-1)
            alert_handlers: Optional alert handler callbacks
        """
        self.drift_threshold = drift_threshold
        self._alert_handlers = alert_handlers or []
        self._baseline_data: Dict[str, Any] = {}
        self._drift_history: List[DriftStatus] = []

        logger.info(
            f"DriftMonitor initialized with drift_threshold={drift_threshold}"
        )

    def set_baseline(self, feature_name: str, data: List[float]) -> None:
        """
        Set baseline data for a feature.

        Args:
            feature_name: Feature name
            data: Baseline data values
        """
        if not data:
            return

        self._baseline_data[feature_name] = {
            'values': data,
            'mean': statistics.mean(data),
            'std': statistics.stdev(data) if len(data) > 1 else 0,
            'min': min(data),
            'max': max(data),
            'timestamp': datetime.now(timezone.utc),
        }
        logger.info(f"Set baseline for feature: {feature_name}")

    def monitor_data_drift(
        self,
        current: Any,  # pd.DataFrame or Dict[str, List[float]]
        baseline: Optional[Any] = None
    ) -> DriftStatus:
        """
        Monitor data drift between current and baseline data.

        Args:
            current: Current data (DataFrame or dict of feature -> values)
            baseline: Optional baseline data (uses stored baseline if None)

        Returns:
            DriftStatus with drift analysis results
        """
        feature_drifts = []
        drifted_features = []

        # Convert DataFrame to dict if needed
        if PANDAS_AVAILABLE and hasattr(current, 'columns'):
            current_dict = {col: current[col].tolist() for col in current.columns}
        elif isinstance(current, dict):
            current_dict = current
        else:
            logger.error("Invalid data format for drift monitoring")
            return DriftStatus(
                drift_type=DriftType.DATA_DRIFT,
                is_drifted=False,
                severity=DriftSeverity.NONE,
            )

        # Use provided baseline or stored baseline
        if baseline is not None:
            if PANDAS_AVAILABLE and hasattr(baseline, 'columns'):
                baseline_dict = {col: baseline[col].tolist() for col in baseline.columns}
            else:
                baseline_dict = baseline
        else:
            baseline_dict = {
                k: v['values'] for k, v in self._baseline_data.items()
            }

        # Analyze each feature
        for feature_name, current_values in current_dict.items():
            if not current_values or not isinstance(current_values, list):
                continue

            baseline_values = baseline_dict.get(feature_name)
            if not baseline_values:
                # Check stored baseline
                if feature_name in self._baseline_data:
                    baseline_values = self._baseline_data[feature_name]['values']
                else:
                    continue

            # Calculate drift metrics
            drift_info = self._calculate_feature_drift(
                feature_name, baseline_values, current_values
            )
            feature_drifts.append(drift_info)

            if drift_info.drift_detected:
                drifted_features.append(feature_name)

        # Calculate overall drift
        if feature_drifts:
            overall_score = statistics.mean([f.drift_score for f in feature_drifts])
            drift_percentage = (len(drifted_features) / len(feature_drifts)) * 100
        else:
            overall_score = 0.0
            drift_percentage = 0.0

        # Determine severity
        is_drifted = overall_score >= self.drift_threshold
        severity = self._determine_severity(overall_score, drift_percentage)

        # Determine if action required
        requires_action = severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]
        recommended_action = None
        if requires_action:
            recommended_action = "Review data pipeline and consider model retraining"
        elif severity == DriftSeverity.MODERATE:
            recommended_action = "Monitor closely, prepare for potential retraining"

        status = DriftStatus(
            drift_type=DriftType.DATA_DRIFT,
            is_drifted=is_drifted,
            severity=severity,
            overall_drift_score=overall_score,
            drifted_features=drifted_features,
            feature_drift_details=feature_drifts,
            features_analyzed=len(feature_drifts),
            features_drifted=len(drifted_features),
            drift_percentage=drift_percentage,
            samples_analyzed=len(current_values) if current_dict else 0,
            requires_action=requires_action,
            recommended_action=recommended_action,
        )

        # Compute provenance hash
        status.provenance_hash = self._compute_provenance(status)

        # Store in history
        self._drift_history.append(status)

        logger.info(
            f"Data drift analysis: score={overall_score:.3f}, "
            f"severity={severity.value}, drifted={len(drifted_features)}/{len(feature_drifts)}"
        )

        return status

    def _calculate_feature_drift(
        self,
        feature_name: str,
        baseline: List[float],
        current: List[float]
    ) -> FeatureDrift:
        """Calculate drift metrics for a single feature."""
        # Calculate basic statistics
        baseline_mean = statistics.mean(baseline)
        baseline_std = statistics.stdev(baseline) if len(baseline) > 1 else 1e-6
        current_mean = statistics.mean(current)
        current_std = statistics.stdev(current) if len(current) > 1 else 1e-6

        # Mean shift (in standard deviations)
        mean_shift = abs(current_mean - baseline_mean) / (baseline_std + 1e-10)

        # Standard deviation ratio
        std_ratio = current_std / (baseline_std + 1e-10)

        # Calculate PSI (Population Stability Index) - simplified
        psi = self._calculate_psi(baseline, current)

        # Calculate KS statistic - simplified
        ks_stat = self._calculate_ks_statistic(baseline, current)

        # Determine drift score (weighted combination)
        drift_score = (
            min(1.0, mean_shift / 5) * 0.3 +  # Mean shift contribution
            min(1.0, abs(std_ratio - 1)) * 0.2 +  # Std change contribution
            min(1.0, psi / 0.5) * 0.25 +  # PSI contribution
            min(1.0, ks_stat / 0.3) * 0.25  # KS contribution
        )

        # Determine if drifted
        drift_detected = (
            drift_score >= self.drift_threshold or
            psi >= self.PSI_THRESHOLD or
            ks_stat >= self.KS_THRESHOLD or
            mean_shift >= self.MEAN_SHIFT_THRESHOLD
        )

        return FeatureDrift(
            feature_name=feature_name,
            drift_detected=drift_detected,
            drift_score=drift_score,
            baseline_mean=baseline_mean,
            current_mean=current_mean,
            mean_shift=mean_shift,
            baseline_std=baseline_std,
            current_std=current_std,
            std_ratio=std_ratio,
            psi_score=psi,
            ks_statistic=ks_stat,
        )

    def _calculate_psi(
        self,
        baseline: List[float],
        current: List[float],
        num_bins: int = 10
    ) -> float:
        """Calculate Population Stability Index."""
        if not baseline or not current:
            return 0.0

        # Create bins from baseline
        min_val = min(min(baseline), min(current))
        max_val = max(max(baseline), max(current))
        bin_width = (max_val - min_val + 1e-10) / num_bins

        # Count baseline distribution
        baseline_counts = [0] * num_bins
        for val in baseline:
            bin_idx = min(int((val - min_val) / bin_width), num_bins - 1)
            baseline_counts[bin_idx] += 1

        # Count current distribution
        current_counts = [0] * num_bins
        for val in current:
            bin_idx = min(int((val - min_val) / bin_width), num_bins - 1)
            current_counts[bin_idx] += 1

        # Convert to percentages
        baseline_pcts = [c / (len(baseline) + 1e-10) for c in baseline_counts]
        current_pcts = [c / (len(current) + 1e-10) for c in current_counts]

        # Calculate PSI
        psi = 0.0
        for bp, cp in zip(baseline_pcts, current_pcts):
            # Add small epsilon to avoid log(0)
            bp = max(bp, 0.001)
            cp = max(cp, 0.001)
            psi += (cp - bp) * (statistics.fabs(cp / bp) if bp > 0 else 0)

        return abs(psi)

    def _calculate_ks_statistic(
        self,
        baseline: List[float],
        current: List[float]
    ) -> float:
        """Calculate Kolmogorov-Smirnov statistic."""
        if not baseline or not current:
            return 0.0

        # Sort both datasets
        baseline_sorted = sorted(baseline)
        current_sorted = sorted(current)

        # Create combined sorted values
        combined = sorted(set(baseline + current))

        # Calculate CDFs
        max_diff = 0.0
        for val in combined:
            # Baseline CDF
            baseline_cdf = sum(1 for x in baseline_sorted if x <= val) / len(baseline)
            # Current CDF
            current_cdf = sum(1 for x in current_sorted if x <= val) / len(current)
            # Track maximum difference
            diff = abs(baseline_cdf - current_cdf)
            max_diff = max(max_diff, diff)

        return max_diff

    def _determine_severity(
        self,
        drift_score: float,
        drift_percentage: float
    ) -> DriftSeverity:
        """Determine drift severity based on score and percentage."""
        if drift_score >= 0.5 or drift_percentage >= 50:
            return DriftSeverity.CRITICAL
        elif drift_score >= 0.35 or drift_percentage >= 35:
            return DriftSeverity.HIGH
        elif drift_score >= 0.2 or drift_percentage >= 20:
            return DriftSeverity.MODERATE
        elif drift_score >= 0.1 or drift_percentage >= 10:
            return DriftSeverity.LOW
        else:
            return DriftSeverity.NONE

    def monitor_model_drift(
        self,
        predictions: List[float],
        actuals: List[float]
    ) -> DriftStatus:
        """
        Monitor model drift based on prediction accuracy.

        Args:
            predictions: Model predictions
            actuals: Actual observed values

        Returns:
            DriftStatus with model drift analysis
        """
        if len(predictions) != len(actuals) or not predictions:
            logger.warning("Invalid predictions/actuals for model drift monitoring")
            return DriftStatus(
                drift_type=DriftType.MODEL_DRIFT,
                is_drifted=False,
                severity=DriftSeverity.NONE,
            )

        # Calculate prediction errors
        errors = [abs(p - a) for p, a in zip(predictions, actuals)]
        relative_errors = [
            abs(p - a) / (abs(a) + 1e-10)
            for p, a in zip(predictions, actuals)
        ]

        # Calculate error statistics
        mae = statistics.mean(errors)
        mape = statistics.mean(relative_errors) * 100
        rmse = (statistics.mean([e**2 for e in errors])) ** 0.5

        # Check for systematic bias
        residuals = [p - a for p, a in zip(predictions, actuals)]
        mean_residual = statistics.mean(residuals)
        std_residual = statistics.stdev(residuals) if len(residuals) > 1 else 0

        # Calculate drift score based on error metrics
        # Compare to expected baseline performance
        baseline_mape = 5.0  # Assume 5% baseline MAPE
        drift_score = min(1.0, mape / (baseline_mape * 5))  # Significant if 5x worse

        # Check for bias drift
        bias_drift = abs(mean_residual) / (std_residual + 1e-10) > 2

        is_drifted = drift_score >= self.drift_threshold or bias_drift
        severity = self._determine_severity(drift_score, 0)

        feature_drifts = [
            FeatureDrift(
                feature_name="prediction_error",
                drift_detected=is_drifted,
                drift_score=drift_score,
                baseline_mean=baseline_mape,
                current_mean=mape,
                mean_shift=abs(mape - baseline_mape) / baseline_mape,
            ),
            FeatureDrift(
                feature_name="prediction_bias",
                drift_detected=bias_drift,
                drift_score=abs(mean_residual) / (std_residual + 1e-10),
                current_mean=mean_residual,
                current_std=std_residual,
            ),
        ]

        drifted_features = [f.feature_name for f in feature_drifts if f.drift_detected]

        status = DriftStatus(
            drift_type=DriftType.MODEL_DRIFT,
            is_drifted=is_drifted,
            severity=severity,
            overall_drift_score=drift_score,
            drifted_features=drifted_features,
            feature_drift_details=feature_drifts,
            features_analyzed=2,
            features_drifted=len(drifted_features),
            samples_analyzed=len(predictions),
            requires_action=severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL],
            recommended_action=(
                "Model retraining recommended" if is_drifted else None
            ),
        )

        status.provenance_hash = self._compute_provenance(status)
        self._drift_history.append(status)

        logger.info(
            f"Model drift analysis: score={drift_score:.3f}, "
            f"MAPE={mape:.1f}%, severity={severity.value}"
        )

        return status

    def monitor_process_drift(self, kpis: Dict[str, float]) -> DriftStatus:
        """
        Monitor process drift based on KPI changes.

        Args:
            kpis: Dictionary of KPI name to current value

        Returns:
            DriftStatus with process drift analysis
        """
        feature_drifts = []
        drifted_features = []

        for kpi_name, current_value in kpis.items():
            # Get baseline if available
            if kpi_name in self._baseline_data:
                baseline = self._baseline_data[kpi_name]
                baseline_mean = baseline['mean']
                baseline_std = baseline['std']

                # Calculate z-score
                if baseline_std > 0:
                    z_score = abs(current_value - baseline_mean) / baseline_std
                else:
                    z_score = 0 if current_value == baseline_mean else 10

                drift_score = min(1.0, z_score / 5)
                drift_detected = z_score > 3  # 3 sigma rule

                feature_drifts.append(FeatureDrift(
                    feature_name=kpi_name,
                    drift_detected=drift_detected,
                    drift_score=drift_score,
                    baseline_mean=baseline_mean,
                    current_mean=current_value,
                    mean_shift=z_score,
                    baseline_std=baseline_std,
                ))

                if drift_detected:
                    drifted_features.append(kpi_name)

        # Calculate overall drift
        if feature_drifts:
            overall_score = statistics.mean([f.drift_score for f in feature_drifts])
            drift_percentage = (len(drifted_features) / len(feature_drifts)) * 100
        else:
            overall_score = 0.0
            drift_percentage = 0.0

        is_drifted = overall_score >= self.drift_threshold
        severity = self._determine_severity(overall_score, drift_percentage)

        status = DriftStatus(
            drift_type=DriftType.PROCESS_DRIFT,
            is_drifted=is_drifted,
            severity=severity,
            overall_drift_score=overall_score,
            drifted_features=drifted_features,
            feature_drift_details=feature_drifts,
            features_analyzed=len(feature_drifts),
            features_drifted=len(drifted_features),
            drift_percentage=drift_percentage,
            requires_action=severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL],
            recommended_action=(
                "Investigate process changes" if is_drifted else None
            ),
        )

        status.provenance_hash = self._compute_provenance(status)
        self._drift_history.append(status)

        logger.info(
            f"Process drift analysis: score={overall_score:.3f}, "
            f"severity={severity.value}"
        )

        return status

    def alert_on_drift(
        self,
        drift: DriftStatus,
        recipients: Optional[List[str]] = None
    ) -> AlertResult:
        """
        Send alert based on drift status.

        Args:
            drift: Drift status to alert on
            recipients: Optional list of recipients

        Returns:
            AlertResult with alert details
        """
        recipients = recipients or ["operations@example.com"]

        # Build alert message
        message = (
            f"DRIFT ALERT [{drift.severity.value}]: "
            f"{drift.drift_type.value} detected\n"
            f"Score: {drift.overall_drift_score:.3f}\n"
            f"Drifted features: {', '.join(drift.drifted_features)}\n"
        )

        if drift.recommended_action:
            message += f"Recommended action: {drift.recommended_action}"

        # Call alert handlers
        success = True
        for handler in self._alert_handlers:
            try:
                handler(drift, message)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
                success = False

        # Log alert
        if drift.severity in [DriftSeverity.CRITICAL, DriftSeverity.HIGH]:
            logger.warning(message)
        else:
            logger.info(f"Drift alert sent: {drift.drift_type.value}")

        return AlertResult(
            success=success,
            drift_id=drift.drift_id,
            alert_type=drift.drift_type.value,
            severity=drift.severity,
            recipients=recipients,
            message=message,
        )

    def recommend_recalibration(
        self,
        drift: DriftStatus
    ) -> RecalibrationRecommendation:
        """
        Generate recalibration recommendation based on drift status.

        Args:
            drift: Drift status to base recommendation on

        Returns:
            RecalibrationRecommendation with actions
        """
        # Determine urgency
        if drift.severity == DriftSeverity.CRITICAL:
            urgency = RecalibrationUrgency.IMMEDIATE
            priority_score = 100.0
            deadline = datetime.now(timezone.utc) + timedelta(hours=4)
            effort_hours = 8.0
            risk = "HIGH - Model predictions may be unreliable"
        elif drift.severity == DriftSeverity.HIGH:
            urgency = RecalibrationUrgency.HIGH
            priority_score = 80.0
            deadline = datetime.now(timezone.utc) + timedelta(days=1)
            effort_hours = 16.0
            risk = "MEDIUM - Degraded performance expected"
        elif drift.severity == DriftSeverity.MODERATE:
            urgency = RecalibrationUrgency.MEDIUM
            priority_score = 50.0
            deadline = datetime.now(timezone.utc) + timedelta(days=7)
            effort_hours = 8.0
            risk = "LOW - Minor performance impact"
        else:
            urgency = RecalibrationUrgency.LOW
            priority_score = 20.0
            deadline = datetime.now(timezone.utc) + timedelta(days=30)
            effort_hours = 4.0
            risk = "MINIMAL - Monitoring recommended"

        # Generate recommended actions
        actions = []
        affected_components = []

        if drift.drift_type == DriftType.DATA_DRIFT:
            actions.append("Collect new training data from recent time period")
            actions.append("Review data preprocessing pipeline for changes")
            actions.append("Validate sensor calibrations")
            affected_components = ["data_pipeline", "feature_engineering"]

        elif drift.drift_type == DriftType.MODEL_DRIFT:
            actions.append("Retrain model with recent data")
            actions.append("Evaluate model architecture for improvements")
            actions.append("Update hyperparameters based on new data distribution")
            affected_components = ["prediction_model", "model_serving"]

        elif drift.drift_type == DriftType.PROCESS_DRIFT:
            actions.append("Review operational changes that may have caused drift")
            actions.append("Update KPI baselines if process change is permanent")
            actions.append("Investigate root cause of KPI changes")
            affected_components = ["process_monitoring", "baseline_definitions"]

        # Add feature-specific actions
        for feature in drift.drifted_features[:3]:  # Top 3 drifted features
            actions.append(f"Investigate drift in feature: {feature}")

        recommendation = RecalibrationRecommendation(
            urgency=urgency,
            priority_score=priority_score,
            drift_status=drift,
            recommended_actions=actions,
            affected_components=affected_components,
            recommended_deadline=deadline,
            estimated_effort_hours=effort_hours,
            risk_if_not_addressed=risk,
            expected_improvement=(
                "Restore model accuracy to baseline levels"
                if drift.drift_type == DriftType.MODEL_DRIFT
                else "Align system behavior with expected parameters"
            ),
        )

        # Compute provenance hash
        recommendation.provenance_hash = self._compute_provenance(recommendation)

        logger.info(
            f"Recalibration recommendation generated: urgency={urgency.value}, "
            f"priority={priority_score}"
        )

        return recommendation

    def _compute_provenance(self, obj: BaseModel) -> str:
        """Compute SHA-256 provenance hash for audit."""
        content = obj.json(exclude={'provenance_hash'})
        return hashlib.sha256(content.encode()).hexdigest()

    def get_drift_history(
        self,
        drift_type: Optional[DriftType] = None,
        limit: int = 100
    ) -> List[DriftStatus]:
        """Get drift detection history."""
        history = self._drift_history
        if drift_type:
            history = [d for d in history if d.drift_type == drift_type]
        return history[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get drift monitor statistics."""
        return {
            'total_analyses': len(self._drift_history),
            'baselines_set': len(self._baseline_data),
            'drift_detected_count': sum(
                1 for d in self._drift_history if d.is_drifted
            ),
            'critical_drift_count': sum(
                1 for d in self._drift_history
                if d.severity == DriftSeverity.CRITICAL
            ),
        }
