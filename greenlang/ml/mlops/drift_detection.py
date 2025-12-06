"""
Drift Detection - Data and Concept Drift Detection for ML Models.

This module provides comprehensive drift detection capabilities for GreenLang
Process Heat agents, enabling proactive monitoring of model performance degradation.

Supports statistical tests (KS, Chi-squared, PSI, KL divergence) with
feature-level analysis and actionable recommendations.

Example:
    >>> from greenlang.ml.mlops.drift_detection import DriftDetector
    >>> detector = DriftDetector(reference_data=X_train)
    >>> report = detector.detect_data_drift(reference_data, current_data)
    >>> if report.drift_detected:
    ...     print(f"Drift severity: {report.severity}")
"""

import hashlib
import json
import logging
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .config import MLOpsConfig, get_config
from .schemas import (
    DriftReport,
    DriftSeverity,
    DriftType,
    FeatureDriftResult,
    ConceptDriftMetrics,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Statistical Utilities
# =============================================================================

def _calculate_histogram(
    data: np.ndarray, bins: int = 10, range_vals: Optional[Tuple[float, float]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate histogram with specified bins."""
    if range_vals is None:
        range_vals = (float(np.min(data)), float(np.max(data)))
    counts, bin_edges = np.histogram(data, bins=bins, range=range_vals)
    return counts, bin_edges


def _safe_log(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Safe logarithm to avoid log(0)."""
    return np.log(np.clip(x, eps, None))


def _safe_divide(a: np.ndarray, b: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Safe division to avoid division by zero."""
    return a / np.clip(b, eps, None)


# =============================================================================
# Drift Detector Implementation
# =============================================================================

class DriftDetector:
    """
    Detect data and concept drift in production ML models.

    This class provides comprehensive drift detection capabilities including:
    - Data drift: Changes in input feature distributions
    - Concept drift: Changes in relationship between features and target
    - Statistical tests: KS test, Chi-squared, PSI, KL divergence
    - Feature-level analysis
    - Alert thresholds and recommendations

    Attributes:
        config: MLOps configuration
        reference_data: Reference (training) data distribution
        reference_stats: Precomputed statistics for reference data

    Example:
        >>> detector = DriftDetector()
        >>> detector.set_reference_data(X_train)
        >>> report = detector.detect_data_drift(X_train, X_new)
        >>> print(f"Drift detected: {report.drift_detected}")
    """

    def __init__(self, config: Optional[MLOpsConfig] = None):
        """
        Initialize DriftDetector.

        Args:
            config: MLOps configuration. If None, uses default configuration.
        """
        self.config = config or get_config()
        self.storage_path = Path(self.config.drift_detection.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._reference_data: Optional[np.ndarray] = None
        self._reference_stats: Dict[str, Any] = {}
        self._feature_names: List[str] = []

        logger.info("DriftDetector initialized")

    def set_reference_data(
        self,
        data: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """
        Set reference data distribution for drift comparison.

        Args:
            data: Reference data (typically training data).
            feature_names: Names of features.
        """
        with self._lock:
            self._reference_data = np.array(data)
            self._feature_names = feature_names or [
                f"feature_{i}" for i in range(data.shape[1])
            ]
            self._compute_reference_stats()

        logger.info(f"Reference data set: shape={data.shape}")

    def _compute_reference_stats(self) -> None:
        """Compute and cache statistics for reference data."""
        if self._reference_data is None:
            return

        n_features = self._reference_data.shape[1]
        self._reference_stats = {
            "n_samples": len(self._reference_data),
            "n_features": n_features,
            "features": {},
        }

        for i in range(n_features):
            feature_data = self._reference_data[:, i]
            self._reference_stats["features"][i] = {
                "mean": float(np.mean(feature_data)),
                "std": float(np.std(feature_data)),
                "min": float(np.min(feature_data)),
                "max": float(np.max(feature_data)),
                "median": float(np.median(feature_data)),
                "q1": float(np.percentile(feature_data, 25)),
                "q3": float(np.percentile(feature_data, 75)),
                "histogram": _calculate_histogram(
                    feature_data, bins=self.config.drift_detection.num_bins_psi
                ),
            }

    def _calculate_data_hash(self, data: np.ndarray) -> str:
        """Calculate SHA-256 hash of data array."""
        return hashlib.sha256(data.tobytes()).hexdigest()

    def _generate_report_id(self) -> str:
        """Generate unique report identifier."""
        timestamp = datetime.utcnow().isoformat()
        return hashlib.sha256(timestamp.encode()).hexdigest()[:16]

    def detect_data_drift(
        self,
        reference_data: np.ndarray,
        current_data: np.ndarray,
        model_name: str = "unknown",
        model_version: str = "unknown",
        feature_names: Optional[List[str]] = None,
    ) -> DriftReport:
        """
        Detect data drift between reference and current data.

        Args:
            reference_data: Reference (baseline) data.
            current_data: Current (production) data.
            model_name: Name of the model being monitored.
            model_version: Version of the model.
            feature_names: Names of features.

        Returns:
            DriftReport with comprehensive drift analysis.

        Raises:
            ValueError: If data shapes don't match.
        """
        start_time = datetime.utcnow()
        logger.info(f"Detecting data drift for {model_name} v{model_version}")

        # Validate inputs
        reference_data = np.array(reference_data)
        current_data = np.array(current_data)

        if reference_data.shape[1] != current_data.shape[1]:
            raise ValueError(
                f"Feature count mismatch: reference has {reference_data.shape[1]} features, "
                f"current has {current_data.shape[1]} features"
            )

        n_features = reference_data.shape[1]
        feature_names = feature_names or [f"feature_{i}" for i in range(n_features)]

        # Analyze each feature
        feature_results: List[FeatureDriftResult] = []
        drifted_features: List[str] = []

        for i in range(n_features):
            ref_feature = reference_data[:, i]
            cur_feature = current_data[:, i]

            # Calculate statistics
            ks_stat, ks_pvalue = self._ks_test(ref_feature, cur_feature)
            psi_value = self.calculate_psi(ref_feature, cur_feature)
            kl_div = self.calculate_kl_divergence(ref_feature, cur_feature)

            # Determine drift
            drift_detected = (
                ks_pvalue < self.config.drift_detection.ks_test_threshold
                or psi_value > self.config.drift_detection.psi_threshold
            )

            # Calculate drift score (weighted combination)
            drift_score = min(1.0, (1 - ks_pvalue) * 0.5 + psi_value * 0.5)

            feature_result = FeatureDriftResult(
                feature_name=feature_names[i],
                drift_detected=drift_detected,
                drift_score=drift_score,
                statistic=ks_stat,
                p_value=ks_pvalue,
                test_used="ks_test",
                reference_mean=float(np.mean(ref_feature)),
                current_mean=float(np.mean(cur_feature)),
                reference_std=float(np.std(ref_feature)),
                current_std=float(np.std(cur_feature)),
                psi=psi_value,
                kl_divergence=kl_div,
            )
            feature_results.append(feature_result)

            if drift_detected:
                drifted_features.append(feature_names[i])

        # Calculate overall drift score
        overall_drift_score = float(np.mean([f.drift_score for f in feature_results]))
        overall_drift_detected = len(drifted_features) > 0

        # Determine severity
        severity = self._determine_severity(overall_drift_score, len(drifted_features), n_features)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            severity, drifted_features, feature_results
        )

        # Create report
        report = DriftReport(
            report_id=self._generate_report_id(),
            model_name=model_name,
            model_version=model_version,
            drift_type=DriftType.DATA_DRIFT,
            drift_detected=overall_drift_detected,
            overall_drift_score=overall_drift_score,
            severity=severity,
            feature_results=feature_results,
            drifted_features=drifted_features,
            reference_data_size=len(reference_data),
            current_data_size=len(current_data),
            reference_data_hash=self._calculate_data_hash(reference_data),
            current_data_hash=self._calculate_data_hash(current_data),
            analysis_timestamp=start_time,
            recommendations=recommendations,
        )

        # Save report
        self._save_report(report)

        logger.info(
            f"Data drift analysis complete: detected={overall_drift_detected}, "
            f"severity={severity.value}, drifted_features={len(drifted_features)}/{n_features}"
        )

        return report

    def detect_concept_drift(
        self,
        model: Any,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "unknown",
        model_version: str = "unknown",
        baseline_metrics: Optional[Dict[str, float]] = None,
    ) -> DriftReport:
        """
        Detect concept drift by analyzing prediction errors.

        Args:
            model: The model being monitored.
            X: Feature data.
            y_true: True target values.
            y_pred: Predicted values.
            model_name: Name of the model.
            model_version: Version of the model.
            baseline_metrics: Baseline performance metrics for comparison.

        Returns:
            DriftReport with concept drift analysis.
        """
        start_time = datetime.utcnow()
        logger.info(f"Detecting concept drift for {model_name} v{model_version}")

        X = np.array(X)
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        # Calculate error metrics
        errors = y_true - y_pred
        abs_errors = np.abs(errors)

        mae = float(np.mean(abs_errors))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mape = float(np.mean(np.abs(errors / np.clip(y_true, 1e-10, None))) * 100)

        # Calculate error distribution statistics
        error_mean = float(np.mean(errors))
        error_std = float(np.std(errors))

        # Compare to baseline if provided
        drift_detected = False
        severity = DriftSeverity.NONE
        recommendations = []

        if baseline_metrics:
            baseline_mae = baseline_metrics.get("mae", mae)
            baseline_rmse = baseline_metrics.get("rmse", rmse)

            mae_degradation = (mae - baseline_mae) / max(baseline_mae, 1e-10)
            rmse_degradation = (rmse - baseline_rmse) / max(baseline_rmse, 1e-10)

            # Check for significant degradation
            if mae_degradation > self.config.monitoring.mae_degradation_threshold:
                drift_detected = True
                recommendations.append(
                    f"MAE increased by {mae_degradation*100:.1f}% from baseline"
                )

            if rmse_degradation > self.config.monitoring.rmse_degradation_threshold:
                drift_detected = True
                recommendations.append(
                    f"RMSE increased by {rmse_degradation*100:.1f}% from baseline"
                )

            # Calculate overall concept drift score
            drift_score = max(mae_degradation, rmse_degradation, 0)
        else:
            # No baseline - just report current metrics
            drift_score = 0.0
            recommendations.append(
                "No baseline metrics provided. Current metrics recorded for future comparison."
            )

        # Determine severity based on drift score
        severity = self._determine_severity(drift_score, int(drift_detected), 1)

        # Analyze residuals for patterns
        residual_analysis = self._analyze_residuals(errors, X)
        if residual_analysis.get("pattern_detected"):
            drift_detected = True
            recommendations.append(
                f"Residual pattern detected: {residual_analysis.get('pattern_description')}"
            )

        # Create concept drift metrics
        concept_metrics = ConceptDriftMetrics(
            accuracy_degradation=drift_score,
            mae_degradation=mae_degradation if baseline_metrics else 0.0,
            rmse_degradation=rmse_degradation if baseline_metrics else 0.0,
            error_distribution_shift=abs(error_mean),
            residual_correlation=residual_analysis.get("max_correlation", 0.0),
        )

        # Create report
        report = DriftReport(
            report_id=self._generate_report_id(),
            model_name=model_name,
            model_version=model_version,
            drift_type=DriftType.CONCEPT_DRIFT,
            drift_detected=drift_detected,
            overall_drift_score=drift_score,
            severity=severity,
            feature_results=[],  # Not applicable for concept drift
            drifted_features=[],
            reference_data_size=0,
            current_data_size=len(X),
            reference_data_hash="",
            current_data_hash=self._calculate_data_hash(X),
            analysis_timestamp=start_time,
            recommendations=recommendations,
        )

        # Save report
        self._save_report(report)

        logger.info(
            f"Concept drift analysis complete: detected={drift_detected}, "
            f"MAE={mae:.4f}, RMSE={rmse:.4f}"
        )

        return report

    def _ks_test(
        self, reference: np.ndarray, current: np.ndarray
    ) -> Tuple[float, float]:
        """
        Perform Kolmogorov-Smirnov two-sample test.

        Args:
            reference: Reference distribution samples.
            current: Current distribution samples.

        Returns:
            Tuple of (statistic, p-value).
        """
        try:
            from scipy import stats
            statistic, p_value = stats.ks_2samp(reference, current)
            return float(statistic), float(p_value)
        except ImportError:
            # Manual KS test implementation
            n1, n2 = len(reference), len(current)

            # Sort both samples
            ref_sorted = np.sort(reference)
            cur_sorted = np.sort(current)

            # Compute ECDF values at all points
            all_values = np.sort(np.concatenate([reference, current]))

            # Calculate ECDFs
            ref_ecdf = np.searchsorted(ref_sorted, all_values, side="right") / n1
            cur_ecdf = np.searchsorted(cur_sorted, all_values, side="right") / n2

            # KS statistic is max difference
            statistic = float(np.max(np.abs(ref_ecdf - cur_ecdf)))

            # Approximate p-value using asymptotic distribution
            en = np.sqrt(n1 * n2 / (n1 + n2))
            p_value = 2 * np.exp(-2 * (en * statistic) ** 2)

            return statistic, float(min(p_value, 1.0))

    def calculate_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: Optional[int] = None,
    ) -> float:
        """
        Calculate Population Stability Index (PSI).

        PSI measures how much a distribution has shifted.
        - PSI < 0.1: No significant change
        - 0.1 <= PSI < 0.2: Moderate change
        - PSI >= 0.2: Significant change

        Args:
            reference: Reference distribution samples.
            current: Current distribution samples.
            n_bins: Number of bins for histogram.

        Returns:
            PSI value.
        """
        n_bins = n_bins or self.config.drift_detection.num_bins_psi

        # Determine bin edges from combined data
        min_val = min(float(np.min(reference)), float(np.min(current)))
        max_val = max(float(np.max(reference)), float(np.max(current)))

        # Handle edge case where all values are the same
        if min_val == max_val:
            return 0.0

        bin_edges = np.linspace(min_val - 1e-10, max_val + 1e-10, n_bins + 1)

        # Calculate proportions
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)

        # Add small constant to avoid division by zero
        eps = 1e-10
        ref_props = (ref_counts + eps) / (len(reference) + n_bins * eps)
        cur_props = (cur_counts + eps) / (len(current) + n_bins * eps)

        # PSI formula: sum((cur - ref) * ln(cur/ref))
        psi = float(np.sum((cur_props - ref_props) * np.log(cur_props / ref_props)))

        return max(0.0, psi)  # PSI should be non-negative

    def calculate_kl_divergence(
        self,
        p: np.ndarray,
        q: np.ndarray,
        n_bins: Optional[int] = None,
    ) -> float:
        """
        Calculate Kullback-Leibler divergence between two distributions.

        KL divergence measures how much distribution Q differs from distribution P.

        Args:
            p: Reference (true) distribution samples.
            q: Comparison distribution samples.
            n_bins: Number of bins for histogram.

        Returns:
            KL divergence value.
        """
        n_bins = n_bins or self.config.drift_detection.num_bins_psi

        # Determine bin edges from combined data
        min_val = min(float(np.min(p)), float(np.min(q)))
        max_val = max(float(np.max(p)), float(np.max(q)))

        if min_val == max_val:
            return 0.0

        bin_edges = np.linspace(min_val - 1e-10, max_val + 1e-10, n_bins + 1)

        # Calculate probability distributions
        p_counts, _ = np.histogram(p, bins=bin_edges)
        q_counts, _ = np.histogram(q, bins=bin_edges)

        # Convert to probabilities with smoothing
        eps = 1e-10
        p_probs = (p_counts + eps) / (len(p) + n_bins * eps)
        q_probs = (q_counts + eps) / (len(q) + n_bins * eps)

        # KL divergence: sum(p * log(p/q))
        kl_div = float(np.sum(p_probs * np.log(p_probs / q_probs)))

        return max(0.0, kl_div)

    def _chi_squared_test(
        self, reference: np.ndarray, current: np.ndarray, n_bins: int = 10
    ) -> Tuple[float, float]:
        """
        Perform Chi-squared test for categorical/binned data.

        Args:
            reference: Reference distribution samples.
            current: Current distribution samples.
            n_bins: Number of bins.

        Returns:
            Tuple of (statistic, p-value).
        """
        # Bin the data
        min_val = min(float(np.min(reference)), float(np.min(current)))
        max_val = max(float(np.max(reference)), float(np.max(current)))
        bin_edges = np.linspace(min_val, max_val, n_bins + 1)

        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)

        # Normalize to expected counts
        total_ref = len(reference)
        total_cur = len(current)
        expected = ref_counts * (total_cur / total_ref)

        # Avoid division by zero
        expected = np.clip(expected, 1e-10, None)

        # Chi-squared statistic
        chi2 = float(np.sum((cur_counts - expected) ** 2 / expected))

        # Degrees of freedom
        df = n_bins - 1

        # Approximate p-value using chi-squared distribution
        try:
            from scipy import stats
            p_value = 1 - stats.chi2.cdf(chi2, df)
        except ImportError:
            # Simple approximation
            p_value = np.exp(-chi2 / 2)

        return chi2, float(p_value)

    def _determine_severity(
        self, drift_score: float, n_drifted: int, n_total: int
    ) -> DriftSeverity:
        """Determine drift severity based on scores and thresholds."""
        drift_ratio = n_drifted / max(n_total, 1)

        thresholds = self.config.drift_detection

        if drift_score < thresholds.low_drift_threshold and drift_ratio < 0.1:
            return DriftSeverity.NONE
        elif drift_score < thresholds.medium_drift_threshold and drift_ratio < 0.2:
            return DriftSeverity.LOW
        elif drift_score < thresholds.high_drift_threshold and drift_ratio < 0.3:
            return DriftSeverity.MEDIUM
        elif drift_score < thresholds.critical_drift_threshold and drift_ratio < 0.5:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL

    def _generate_recommendations(
        self,
        severity: DriftSeverity,
        drifted_features: List[str],
        feature_results: List[FeatureDriftResult],
    ) -> List[str]:
        """Generate actionable recommendations based on drift analysis."""
        recommendations = []

        if severity == DriftSeverity.CRITICAL:
            recommendations.append(
                "CRITICAL: Immediate model retraining required. Production predictions may be unreliable."
            )
            recommendations.append(
                "Investigate data pipeline for potential data quality issues."
            )
        elif severity == DriftSeverity.HIGH:
            recommendations.append(
                "HIGH: Schedule model retraining within 24-48 hours."
            )
            recommendations.append(
                "Monitor production metrics closely for degradation."
            )
        elif severity == DriftSeverity.MEDIUM:
            recommendations.append(
                "MEDIUM: Plan model retraining within 1 week."
            )
            recommendations.append(
                "Increase monitoring frequency for affected features."
            )
        elif severity == DriftSeverity.LOW:
            recommendations.append(
                "LOW: Minor drift detected. Continue monitoring."
            )
        else:
            recommendations.append(
                "No significant drift detected. Model performance stable."
            )

        # Feature-specific recommendations
        if drifted_features:
            top_drifted = drifted_features[:5]
            recommendations.append(
                f"Top drifted features: {', '.join(top_drifted)}"
            )

            # Analyze drift types
            high_psi_features = [
                f.feature_name for f in feature_results
                if f.psi and f.psi > self.config.drift_detection.psi_threshold * 1.5
            ]
            if high_psi_features:
                recommendations.append(
                    f"Features with high PSI (distribution shift): {', '.join(high_psi_features[:3])}"
                )

        return recommendations

    def _analyze_residuals(
        self, errors: np.ndarray, X: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze residuals for patterns indicating concept drift."""
        result = {
            "pattern_detected": False,
            "pattern_description": "",
            "max_correlation": 0.0,
        }

        # Check for correlation between errors and features
        n_features = X.shape[1]
        correlations = []

        for i in range(n_features):
            corr = np.corrcoef(errors, X[:, i])[0, 1]
            if np.isnan(corr):
                corr = 0.0
            correlations.append(abs(corr))

        max_corr = max(correlations) if correlations else 0.0
        result["max_correlation"] = float(max_corr)

        # Significant correlation indicates potential concept drift
        if max_corr > 0.3:
            result["pattern_detected"] = True
            result["pattern_description"] = (
                f"Errors correlated with feature {correlations.index(max_corr)} "
                f"(correlation: {max_corr:.3f})"
            )

        # Check for trend in errors
        if len(errors) > 100:
            # Simple linear trend check
            x_trend = np.arange(len(errors))
            trend_corr = np.corrcoef(x_trend, errors)[0, 1]
            if not np.isnan(trend_corr) and abs(trend_corr) > 0.2:
                result["pattern_detected"] = True
                trend_dir = "increasing" if trend_corr > 0 else "decreasing"
                result["pattern_description"] += (
                    f" Temporal trend detected ({trend_dir}, r={trend_corr:.3f})."
                )

        return result

    def _save_report(self, report: DriftReport) -> None:
        """Save drift report to storage."""
        report_path = self.storage_path / f"drift_report_{report.report_id}.json"
        with open(report_path, "w") as f:
            f.write(report.json(indent=2))

    def load_report(self, report_id: str) -> Optional[DriftReport]:
        """Load a drift report by ID."""
        report_path = self.storage_path / f"drift_report_{report_id}.json"
        if not report_path.exists():
            return None

        with open(report_path, "r") as f:
            return DriftReport(**json.loads(f.read()))

    def list_reports(
        self,
        model_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[DriftReport]:
        """List drift reports, optionally filtered by model name."""
        reports = []
        report_files = sorted(
            self.storage_path.glob("drift_report_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        for report_file in report_files[:limit * 2]:  # Load extra to filter
            try:
                with open(report_file, "r") as f:
                    report = DriftReport(**json.loads(f.read()))
                    if model_name is None or report.model_name == model_name:
                        reports.append(report)
                        if len(reports) >= limit:
                            break
            except Exception as e:
                logger.warning(f"Failed to load report {report_file}: {e}")

        return reports

    def get_drift_summary(
        self, model_name: str, window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get summary of drift reports for a model within time window.

        Args:
            model_name: Name of the model.
            window_hours: Time window in hours.

        Returns:
            Summary dictionary with drift statistics.
        """
        reports = self.list_reports(model_name=model_name)

        cutoff = datetime.utcnow().timestamp() - (window_hours * 3600)
        recent_reports = [
            r for r in reports
            if r.analysis_timestamp.timestamp() > cutoff
        ]

        if not recent_reports:
            return {
                "model_name": model_name,
                "window_hours": window_hours,
                "report_count": 0,
                "drift_detected_count": 0,
                "average_drift_score": 0.0,
                "severity_distribution": {},
                "most_drifted_features": [],
            }

        # Calculate statistics
        drift_count = sum(1 for r in recent_reports if r.drift_detected)
        avg_score = float(np.mean([r.overall_drift_score for r in recent_reports]))

        # Severity distribution
        severity_dist = {}
        for severity in DriftSeverity:
            severity_dist[severity.value] = sum(
                1 for r in recent_reports if r.severity == severity
            )

        # Most commonly drifted features
        feature_drift_counts: Dict[str, int] = {}
        for report in recent_reports:
            for feature in report.drifted_features:
                feature_drift_counts[feature] = feature_drift_counts.get(feature, 0) + 1

        most_drifted = sorted(
            feature_drift_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]

        return {
            "model_name": model_name,
            "window_hours": window_hours,
            "report_count": len(recent_reports),
            "drift_detected_count": drift_count,
            "average_drift_score": avg_score,
            "severity_distribution": severity_dist,
            "most_drifted_features": most_drifted,
        }
