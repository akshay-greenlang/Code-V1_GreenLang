"""
Evidently Monitor - Drift Detection for GreenLang Process Heat Agents.

This module provides comprehensive drift detection capabilities using Evidently AI
for monitoring data drift, prediction drift, and concept drift in production
ML models powering the GreenLang Process Heat agent pipeline (GL-001 through GL-020).

Supports:
    - Data drift detection for input features
    - Prediction drift detection for model outputs
    - Concept drift detection for relationship changes
    - Agent-specific drift profiles
    - Prometheus metrics export
    - SHA-256 provenance tracking for audit trails

Example:
    >>> from greenlang.ml.drift_detection import ProcessHeatDriftMonitor
    >>> monitor = ProcessHeatDriftMonitor()
    >>> report = monitor.detect_data_drift(
    ...     reference_data=X_train,
    ...     current_data=X_prod,
    ...     agent_id="GL-001"
    ... )
    >>> if report.drift_detected:
    ...     print(f"Drift severity: {report.severity}")
    ...     monitor.generate_drift_report(report, output_path="./reports")
"""

import hashlib
import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Models
# =============================================================================

class EvidentlyDriftConfig(BaseModel):
    """Configuration for Evidently drift detection."""

    # Statistical test settings
    significance_level: float = Field(
        default=0.05,
        ge=0.001,
        le=0.1,
        description="Statistical significance level for drift tests"
    )

    # Data drift thresholds
    data_drift_share_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Share of drifted features to trigger overall drift"
    )

    # PSI thresholds
    psi_threshold_low: float = Field(
        default=0.1,
        description="PSI threshold for low drift"
    )
    psi_threshold_medium: float = Field(
        default=0.2,
        description="PSI threshold for medium drift"
    )
    psi_threshold_high: float = Field(
        default=0.3,
        description="PSI threshold for high drift"
    )

    # Jensen-Shannon divergence thresholds
    js_divergence_threshold: float = Field(
        default=0.1,
        description="Jensen-Shannon divergence threshold"
    )

    # Wasserstein distance settings
    wasserstein_threshold: float = Field(
        default=0.1,
        description="Wasserstein distance threshold for drift detection"
    )

    # Concept drift settings
    performance_degradation_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Performance degradation threshold for concept drift"
    )

    # Windowing settings
    reference_window_size: int = Field(
        default=10000,
        ge=100,
        description="Number of samples in reference window"
    )
    current_window_size: int = Field(
        default=1000,
        ge=100,
        description="Number of samples in current window"
    )

    # Storage settings
    storage_path: str = Field(
        default="./mlops_data/evidently_reports",
        description="Path for storing drift reports"
    )

    # Prometheus settings
    prometheus_enabled: bool = Field(
        default=True,
        description="Enable Prometheus metrics export"
    )
    prometheus_prefix: str = Field(
        default="greenlang_drift",
        description="Prometheus metric name prefix"
    )

    @validator("storage_path")
    def validate_storage_path(cls, v: str) -> str:
        """Ensure storage path exists."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return str(path.absolute())


class FeatureDriftInfo(BaseModel):
    """Information about drift for a single feature."""

    feature_name: str = Field(..., description="Name of the feature")
    drift_detected: bool = Field(..., description="Whether drift was detected")
    drift_score: float = Field(..., ge=0.0, le=1.0, description="Drift score (0-1)")
    statistic_name: str = Field(..., description="Name of statistical test used")
    statistic_value: float = Field(..., description="Test statistic value")
    p_value: Optional[float] = Field(None, description="P-value if applicable")
    threshold: float = Field(..., description="Threshold used for detection")
    reference_distribution: Optional[Dict[str, Any]] = Field(
        None, description="Summary of reference distribution"
    )
    current_distribution: Optional[Dict[str, Any]] = Field(
        None, description="Summary of current distribution"
    )


class DriftAnalysisResult(BaseModel):
    """Result of drift analysis."""

    # Identification
    report_id: str = Field(..., description="Unique report identifier")
    agent_id: str = Field(..., description="Agent ID (GL-001 through GL-020)")
    model_name: str = Field(..., description="Model name being monitored")
    model_version: str = Field(..., description="Model version")

    # Drift detection results
    drift_type: str = Field(..., description="Type of drift (data, prediction, concept)")
    drift_detected: bool = Field(..., description="Whether drift was detected")
    overall_drift_score: float = Field(..., ge=0.0, le=1.0, description="Overall drift score")
    severity: str = Field(..., description="Drift severity (none, low, medium, high, critical)")

    # Feature-level results
    feature_drift_results: List[FeatureDriftInfo] = Field(
        default_factory=list, description="Per-feature drift analysis"
    )
    drifted_features: List[str] = Field(
        default_factory=list, description="List of features with detected drift"
    )
    drift_share: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Share of drifted features"
    )

    # Data statistics
    reference_data_size: int = Field(..., description="Reference dataset size")
    current_data_size: int = Field(..., description="Current dataset size")

    # Provenance tracking
    reference_data_hash: str = Field(..., description="SHA-256 hash of reference data")
    current_data_hash: str = Field(..., description="SHA-256 hash of current data")
    report_hash: str = Field(default="", description="SHA-256 hash of this report")

    # Timestamps
    analysis_timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When analysis was performed"
    )
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

    # Recommendations
    recommendations: List[str] = Field(
        default_factory=list, description="Recommended actions"
    )

    # Evidently report path
    evidently_report_path: Optional[str] = Field(
        None, description="Path to full Evidently HTML report"
    )

    class Config:
        """Pydantic configuration."""
        json_encoders = {datetime: lambda v: v.isoformat()}

    def calculate_report_hash(self) -> str:
        """Calculate SHA-256 hash of this report for provenance."""
        report_dict = self.model_dump(exclude={"report_hash"})
        report_str = json.dumps(report_dict, sort_keys=True, default=str)
        return hashlib.sha256(report_str.encode()).hexdigest()


# =============================================================================
# Prometheus Metrics Models
# =============================================================================

class PrometheusMetricExport(BaseModel):
    """Prometheus metric for drift monitoring."""

    name: str = Field(..., description="Metric name")
    help_text: str = Field(..., description="Metric description")
    metric_type: str = Field(..., description="gauge, counter, histogram")
    labels: Dict[str, str] = Field(default_factory=dict, description="Metric labels")
    value: float = Field(..., description="Metric value")
    timestamp_ms: Optional[int] = Field(None, description="Unix timestamp in ms")


# =============================================================================
# Process Heat Drift Monitor
# =============================================================================

class ProcessHeatDriftMonitor:
    """
    Drift monitor for GreenLang Process Heat agents using Evidently AI.

    This class provides comprehensive drift detection capabilities for
    monitoring data drift, prediction drift, and concept drift in production
    ML models powering agents GL-001 through GL-020.

    Attributes:
        config: Evidently drift detection configuration
        _lock: Thread lock for concurrent access safety
        _reference_data: Cached reference data per agent
        _drift_profiles: Agent-specific drift profiles

    Example:
        >>> monitor = ProcessHeatDriftMonitor()
        >>> report = monitor.detect_data_drift(
        ...     reference_data=X_train,
        ...     current_data=X_prod,
        ...     agent_id="GL-001",
        ...     feature_names=["temperature", "pressure", "flow_rate"]
        ... )
        >>> print(f"Drift detected: {report.drift_detected}")
        >>> print(f"Severity: {report.severity}")
        >>> metrics = monitor.get_drift_metrics("GL-001")
        >>> print(f"Overall drift score: {metrics['overall_drift_score']}")
    """

    # Supported agents
    SUPPORTED_AGENTS = [f"GL-{str(i).zfill(3)}" for i in range(1, 21)]

    def __init__(self, config: Optional[EvidentlyDriftConfig] = None):
        """
        Initialize ProcessHeatDriftMonitor.

        Args:
            config: Evidently drift detection configuration. If None, uses defaults.
        """
        self.config = config or EvidentlyDriftConfig()
        self._lock = threading.RLock()

        # Storage path setup
        self.storage_path = Path(self.config.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Reference data cache per agent
        self._reference_data: Dict[str, np.ndarray] = {}
        self._reference_stats: Dict[str, Dict[str, Any]] = {}
        self._feature_names: Dict[str, List[str]] = {}

        # Drift profiles cache
        self._drift_profiles: Dict[str, Any] = {}

        # Recent analysis results for metrics
        self._recent_results: Dict[str, List[DriftAnalysisResult]] = {}
        self._max_cached_results = 100

        logger.info(
            f"ProcessHeatDriftMonitor initialized with storage at {self.storage_path}"
        )

    def _validate_agent_id(self, agent_id: str) -> None:
        """Validate agent ID is in supported range GL-001 to GL-020."""
        if agent_id not in self.SUPPORTED_AGENTS:
            raise ValueError(
                f"Invalid agent ID: {agent_id}. "
                f"Supported agents: GL-001 through GL-020"
            )

    def _calculate_data_hash(self, data: np.ndarray) -> str:
        """Calculate SHA-256 hash of data array for provenance tracking."""
        return hashlib.sha256(data.tobytes()).hexdigest()

    def _generate_report_id(self, agent_id: str) -> str:
        """Generate unique report identifier with timestamp, agent ID, and random suffix."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        random_suffix = np.random.randint(0, 1000000)
        unique_str = f"{agent_id}_{timestamp}_{random_suffix}"
        return hashlib.sha256(unique_str.encode()).hexdigest()[:16]

    def set_reference_data(
        self,
        agent_id: str,
        data: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """
        Set reference data for an agent.

        Args:
            agent_id: Agent identifier (GL-001 through GL-020).
            data: Reference data array (n_samples, n_features).
            feature_names: Optional list of feature names.
        """
        self._validate_agent_id(agent_id)

        with self._lock:
            self._reference_data[agent_id] = np.array(data)
            n_features = data.shape[1] if len(data.shape) > 1 else 1

            self._feature_names[agent_id] = feature_names or [
                f"feature_{i}" for i in range(n_features)
            ]

            # Compute reference statistics
            self._reference_stats[agent_id] = self._compute_distribution_stats(
                data, self._feature_names[agent_id]
            )

        logger.info(
            f"Reference data set for {agent_id}: shape={data.shape}, "
            f"features={len(self._feature_names[agent_id])}"
        )

    def _compute_distribution_stats(
        self, data: np.ndarray, feature_names: List[str]
    ) -> Dict[str, Any]:
        """Compute distribution statistics for reference data."""
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        stats = {
            "n_samples": len(data),
            "n_features": data.shape[1],
            "computed_at": datetime.utcnow().isoformat(),
            "features": {},
        }

        for i, name in enumerate(feature_names):
            feature_data = data[:, i]
            stats["features"][name] = {
                "mean": float(np.mean(feature_data)),
                "std": float(np.std(feature_data)),
                "min": float(np.min(feature_data)),
                "max": float(np.max(feature_data)),
                "median": float(np.median(feature_data)),
                "q1": float(np.percentile(feature_data, 25)),
                "q3": float(np.percentile(feature_data, 75)),
                "skewness": float(self._calculate_skewness(feature_data)),
                "kurtosis": float(self._calculate_kurtosis(feature_data)),
            }

        return stats

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of distribution."""
        n = len(data)
        if n < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 3))

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of distribution."""
        n = len(data)
        if n < 4:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 4) - 3)

    def detect_data_drift(
        self,
        reference_data: np.ndarray,
        current_data: np.ndarray,
        agent_id: str,
        model_name: str = "unknown",
        model_version: str = "1.0.0",
        feature_names: Optional[List[str]] = None,
    ) -> DriftAnalysisResult:
        """
        Detect data drift between reference and current data.

        Uses multiple statistical tests including:
        - Kolmogorov-Smirnov test for continuous features
        - Population Stability Index (PSI)
        - Jensen-Shannon divergence
        - Wasserstein distance

        Args:
            reference_data: Reference (baseline) data.
            current_data: Current (production) data.
            agent_id: Agent identifier (GL-001 through GL-020).
            model_name: Name of the model being monitored.
            model_version: Version of the model.
            feature_names: Names of features.

        Returns:
            DriftAnalysisResult with comprehensive drift analysis.

        Raises:
            ValueError: If data shapes don't match or agent ID is invalid.
        """
        start_time = datetime.utcnow()
        self._validate_agent_id(agent_id)

        logger.info(f"Detecting data drift for {agent_id} ({model_name} v{model_version})")

        # Validate and prepare data
        reference_data = np.array(reference_data)
        current_data = np.array(current_data)

        if len(reference_data.shape) == 1:
            reference_data = reference_data.reshape(-1, 1)
        if len(current_data.shape) == 1:
            current_data = current_data.reshape(-1, 1)

        if reference_data.shape[1] != current_data.shape[1]:
            raise ValueError(
                f"Feature count mismatch: reference has {reference_data.shape[1]} features, "
                f"current has {current_data.shape[1]} features"
            )

        n_features = reference_data.shape[1]
        feature_names = feature_names or [f"feature_{i}" for i in range(n_features)]

        # Try to use Evidently if available
        feature_results = []
        drifted_features = []

        try:
            feature_results, drifted_features = self._run_evidently_analysis(
                reference_data, current_data, feature_names
            )
        except ImportError:
            logger.warning(
                "Evidently not available, falling back to statistical tests"
            )
            feature_results, drifted_features = self._run_statistical_tests(
                reference_data, current_data, feature_names
            )

        # Calculate overall drift score
        drift_scores = [f.drift_score for f in feature_results]
        overall_drift_score = float(np.mean(drift_scores)) if drift_scores else 0.0
        drift_share = len(drifted_features) / max(n_features, 1)
        drift_detected = drift_share > self.config.data_drift_share_threshold

        # Determine severity
        severity = self._determine_severity(overall_drift_score, drift_share)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            severity, drifted_features, feature_results
        )

        # Calculate processing time
        processing_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        # Create result
        result = DriftAnalysisResult(
            report_id=self._generate_report_id(agent_id),
            agent_id=agent_id,
            model_name=model_name,
            model_version=model_version,
            drift_type="data",
            drift_detected=drift_detected,
            overall_drift_score=overall_drift_score,
            severity=severity,
            feature_drift_results=feature_results,
            drifted_features=drifted_features,
            drift_share=drift_share,
            reference_data_size=len(reference_data),
            current_data_size=len(current_data),
            reference_data_hash=self._calculate_data_hash(reference_data),
            current_data_hash=self._calculate_data_hash(current_data),
            analysis_timestamp=start_time,
            processing_time_ms=processing_time_ms,
            recommendations=recommendations,
        )

        # Calculate report hash for provenance
        result.report_hash = result.calculate_report_hash()

        # Save report
        self._save_report(result)

        # Cache result
        self._cache_result(agent_id, result)

        logger.info(
            f"Data drift analysis for {agent_id} complete: "
            f"drift_detected={drift_detected}, severity={severity}, "
            f"drifted_features={len(drifted_features)}/{n_features}"
        )

        return result

    def _run_evidently_analysis(
        self,
        reference_data: np.ndarray,
        current_data: np.ndarray,
        feature_names: List[str],
    ) -> Tuple[List[FeatureDriftInfo], List[str]]:
        """Run drift analysis using Evidently AI library."""
        try:
            import pandas as pd
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset
            from evidently.metrics import (
                DataDriftTable,
                DatasetDriftMetric,
            )
        except ImportError:
            raise ImportError("Evidently library not installed")

        # Create DataFrames
        ref_df = pd.DataFrame(reference_data, columns=feature_names)
        cur_df = pd.DataFrame(current_data, columns=feature_names)

        # Run Evidently analysis
        report = Report(metrics=[
            DataDriftPreset(stattest_threshold=self.config.significance_level),
        ])
        report.run(reference_data=ref_df, current_data=cur_df)

        # Extract results
        feature_results = []
        drifted_features = []

        # Get the drift results from Evidently
        report_dict = report.as_dict()

        # Parse Evidently output
        if "metrics" in report_dict:
            for metric in report_dict["metrics"]:
                if "result" in metric:
                    result = metric["result"]

                    # Dataset drift metric
                    if "drift_by_columns" in result:
                        for col_name, col_data in result["drift_by_columns"].items():
                            drift_detected = col_data.get("drift_detected", False)
                            drift_score = col_data.get("drift_score", 0.0)

                            feature_info = FeatureDriftInfo(
                                feature_name=col_name,
                                drift_detected=drift_detected,
                                drift_score=min(1.0, max(0.0, float(drift_score))),
                                statistic_name=col_data.get("stattest_name", "unknown"),
                                statistic_value=float(col_data.get("stattest_value", 0.0)),
                                p_value=col_data.get("p_value"),
                                threshold=float(col_data.get("threshold", self.config.significance_level)),
                            )
                            feature_results.append(feature_info)

                            if drift_detected:
                                drifted_features.append(col_name)

        return feature_results, drifted_features

    def _run_statistical_tests(
        self,
        reference_data: np.ndarray,
        current_data: np.ndarray,
        feature_names: List[str],
    ) -> Tuple[List[FeatureDriftInfo], List[str]]:
        """Run statistical drift tests without Evidently."""
        feature_results = []
        drifted_features = []

        for i, name in enumerate(feature_names):
            ref_feature = reference_data[:, i]
            cur_feature = current_data[:, i]

            # Run multiple tests
            ks_stat, ks_pvalue = self._ks_test(ref_feature, cur_feature)
            psi_value = self._calculate_psi(ref_feature, cur_feature)
            js_divergence = self._calculate_js_divergence(ref_feature, cur_feature)
            wasserstein = self._calculate_wasserstein(ref_feature, cur_feature)

            # Determine drift based on multiple criteria
            drift_detected = (
                ks_pvalue < self.config.significance_level or
                psi_value > self.config.psi_threshold_medium or
                js_divergence > self.config.js_divergence_threshold
            )

            # Combined drift score
            drift_score = min(1.0, (
                (1 - ks_pvalue) * 0.3 +
                min(psi_value / self.config.psi_threshold_high, 1.0) * 0.4 +
                min(js_divergence / 0.5, 1.0) * 0.3
            ))

            feature_info = FeatureDriftInfo(
                feature_name=name,
                drift_detected=drift_detected,
                drift_score=drift_score,
                statistic_name="ks_test",
                statistic_value=ks_stat,
                p_value=ks_pvalue,
                threshold=self.config.significance_level,
                reference_distribution={
                    "mean": float(np.mean(ref_feature)),
                    "std": float(np.std(ref_feature)),
                    "psi": psi_value,
                    "js_divergence": js_divergence,
                    "wasserstein": wasserstein,
                },
                current_distribution={
                    "mean": float(np.mean(cur_feature)),
                    "std": float(np.std(cur_feature)),
                },
            )
            feature_results.append(feature_info)

            if drift_detected:
                drifted_features.append(name)

        return feature_results, drifted_features

    def _ks_test(
        self, reference: np.ndarray, current: np.ndarray
    ) -> Tuple[float, float]:
        """Perform Kolmogorov-Smirnov two-sample test."""
        try:
            from scipy import stats
            statistic, p_value = stats.ks_2samp(reference, current)
            return float(statistic), float(p_value)
        except ImportError:
            # Manual KS test
            n1, n2 = len(reference), len(current)
            all_values = np.sort(np.concatenate([reference, current]))
            ref_ecdf = np.searchsorted(np.sort(reference), all_values, side="right") / n1
            cur_ecdf = np.searchsorted(np.sort(current), all_values, side="right") / n2
            statistic = float(np.max(np.abs(ref_ecdf - cur_ecdf)))
            en = np.sqrt(n1 * n2 / (n1 + n2))
            p_value = 2 * np.exp(-2 * (en * statistic) ** 2)
            return statistic, float(min(p_value, 1.0))

    def _calculate_psi(
        self, reference: np.ndarray, current: np.ndarray, n_bins: int = 10
    ) -> float:
        """Calculate Population Stability Index (PSI)."""
        min_val = min(float(np.min(reference)), float(np.min(current)))
        max_val = max(float(np.max(reference)), float(np.max(current)))

        if min_val == max_val:
            return 0.0

        bin_edges = np.linspace(min_val - 1e-10, max_val + 1e-10, n_bins + 1)
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)

        eps = 1e-10
        ref_props = (ref_counts + eps) / (len(reference) + n_bins * eps)
        cur_props = (cur_counts + eps) / (len(current) + n_bins * eps)

        psi = float(np.sum((cur_props - ref_props) * np.log(cur_props / ref_props)))
        return max(0.0, psi)

    def _calculate_js_divergence(
        self, reference: np.ndarray, current: np.ndarray, n_bins: int = 10
    ) -> float:
        """Calculate Jensen-Shannon divergence."""
        min_val = min(float(np.min(reference)), float(np.min(current)))
        max_val = max(float(np.max(reference)), float(np.max(current)))

        if min_val == max_val:
            return 0.0

        bin_edges = np.linspace(min_val - 1e-10, max_val + 1e-10, n_bins + 1)
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)

        eps = 1e-10
        p = (ref_counts + eps) / (len(reference) + n_bins * eps)
        q = (cur_counts + eps) / (len(current) + n_bins * eps)
        m = 0.5 * (p + q)

        kl_pm = np.sum(p * np.log(p / m))
        kl_qm = np.sum(q * np.log(q / m))

        return float(0.5 * (kl_pm + kl_qm))

    def _calculate_wasserstein(
        self, reference: np.ndarray, current: np.ndarray
    ) -> float:
        """Calculate Wasserstein distance (Earth Mover's Distance)."""
        try:
            from scipy import stats
            return float(stats.wasserstein_distance(reference, current))
        except ImportError:
            # Simplified Wasserstein approximation
            ref_sorted = np.sort(reference)
            cur_sorted = np.sort(current)

            # Resample to same size if needed
            n = max(len(ref_sorted), len(cur_sorted))
            ref_interp = np.interp(
                np.linspace(0, 1, n),
                np.linspace(0, 1, len(ref_sorted)),
                ref_sorted
            )
            cur_interp = np.interp(
                np.linspace(0, 1, n),
                np.linspace(0, 1, len(cur_sorted)),
                cur_sorted
            )

            return float(np.mean(np.abs(ref_interp - cur_interp)))

    def detect_prediction_drift(
        self,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray,
        agent_id: str,
        model_name: str = "unknown",
        model_version: str = "1.0.0",
    ) -> DriftAnalysisResult:
        """
        Detect prediction drift in model outputs.

        Args:
            reference_predictions: Reference (baseline) predictions.
            current_predictions: Current (production) predictions.
            agent_id: Agent identifier (GL-001 through GL-020).
            model_name: Name of the model being monitored.
            model_version: Version of the model.

        Returns:
            DriftAnalysisResult with prediction drift analysis.
        """
        start_time = datetime.utcnow()
        self._validate_agent_id(agent_id)

        logger.info(
            f"Detecting prediction drift for {agent_id} ({model_name} v{model_version})"
        )

        reference_predictions = np.array(reference_predictions).flatten()
        current_predictions = np.array(current_predictions).flatten()

        # Run drift analysis on predictions
        ks_stat, ks_pvalue = self._ks_test(reference_predictions, current_predictions)
        psi_value = self._calculate_psi(reference_predictions, current_predictions)
        js_divergence = self._calculate_js_divergence(
            reference_predictions, current_predictions
        )

        drift_detected = (
            ks_pvalue < self.config.significance_level or
            psi_value > self.config.psi_threshold_medium
        )

        drift_score = min(1.0, (
            (1 - ks_pvalue) * 0.4 +
            min(psi_value / self.config.psi_threshold_high, 1.0) * 0.3 +
            min(js_divergence / 0.5, 1.0) * 0.3
        ))

        severity = self._determine_severity(drift_score, 1.0 if drift_detected else 0.0)

        # Create feature result for predictions
        prediction_info = FeatureDriftInfo(
            feature_name="predictions",
            drift_detected=drift_detected,
            drift_score=drift_score,
            statistic_name="ks_test",
            statistic_value=ks_stat,
            p_value=ks_pvalue,
            threshold=self.config.significance_level,
            reference_distribution={
                "mean": float(np.mean(reference_predictions)),
                "std": float(np.std(reference_predictions)),
                "min": float(np.min(reference_predictions)),
                "max": float(np.max(reference_predictions)),
                "psi": psi_value,
            },
            current_distribution={
                "mean": float(np.mean(current_predictions)),
                "std": float(np.std(current_predictions)),
                "min": float(np.min(current_predictions)),
                "max": float(np.max(current_predictions)),
            },
        )

        processing_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        recommendations = []
        if drift_detected:
            recommendations.append(
                f"Prediction distribution has shifted. PSI={psi_value:.3f}, "
                f"p-value={ks_pvalue:.4f}"
            )
            if severity in ["high", "critical"]:
                recommendations.append(
                    "Consider investigating model performance and data quality."
                )

        result = DriftAnalysisResult(
            report_id=self._generate_report_id(agent_id),
            agent_id=agent_id,
            model_name=model_name,
            model_version=model_version,
            drift_type="prediction",
            drift_detected=drift_detected,
            overall_drift_score=drift_score,
            severity=severity,
            feature_drift_results=[prediction_info],
            drifted_features=["predictions"] if drift_detected else [],
            drift_share=1.0 if drift_detected else 0.0,
            reference_data_size=len(reference_predictions),
            current_data_size=len(current_predictions),
            reference_data_hash=self._calculate_data_hash(reference_predictions),
            current_data_hash=self._calculate_data_hash(current_predictions),
            analysis_timestamp=start_time,
            processing_time_ms=processing_time_ms,
            recommendations=recommendations,
        )

        result.report_hash = result.calculate_report_hash()
        self._save_report(result)
        self._cache_result(agent_id, result)

        logger.info(
            f"Prediction drift analysis for {agent_id} complete: "
            f"drift_detected={drift_detected}, severity={severity}"
        )

        return result

    def detect_concept_drift(
        self,
        reference_predictions: np.ndarray,
        reference_actuals: np.ndarray,
        current_predictions: np.ndarray,
        current_actuals: np.ndarray,
        agent_id: str,
        model_name: str = "unknown",
        model_version: str = "1.0.0",
    ) -> DriftAnalysisResult:
        """
        Detect concept drift by analyzing prediction error distributions.

        Concept drift occurs when the relationship between features and
        target changes over time, even if feature distributions remain stable.

        Args:
            reference_predictions: Reference (baseline) predictions.
            reference_actuals: Reference (baseline) actual values.
            current_predictions: Current (production) predictions.
            current_actuals: Current (production) actual values.
            agent_id: Agent identifier (GL-001 through GL-020).
            model_name: Name of the model being monitored.
            model_version: Version of the model.

        Returns:
            DriftAnalysisResult with concept drift analysis.
        """
        start_time = datetime.utcnow()
        self._validate_agent_id(agent_id)

        logger.info(
            f"Detecting concept drift for {agent_id} ({model_name} v{model_version})"
        )

        # Calculate residuals/errors
        reference_errors = np.array(reference_actuals) - np.array(reference_predictions)
        current_errors = np.array(current_actuals) - np.array(current_predictions)

        # Analyze error distribution shift
        ks_stat, ks_pvalue = self._ks_test(reference_errors, current_errors)
        psi_value = self._calculate_psi(reference_errors, current_errors)

        # Calculate performance metrics
        ref_mae = float(np.mean(np.abs(reference_errors)))
        cur_mae = float(np.mean(np.abs(current_errors)))
        ref_rmse = float(np.sqrt(np.mean(reference_errors ** 2)))
        cur_rmse = float(np.sqrt(np.mean(current_errors ** 2)))

        # Performance degradation
        mae_degradation = (cur_mae - ref_mae) / max(ref_mae, 1e-10)
        rmse_degradation = (cur_rmse - ref_rmse) / max(ref_rmse, 1e-10)

        performance_degradation = max(mae_degradation, rmse_degradation, 0)

        drift_detected = (
            performance_degradation > self.config.performance_degradation_threshold or
            psi_value > self.config.psi_threshold_medium or
            ks_pvalue < self.config.significance_level
        )

        drift_score = min(1.0, (
            min(performance_degradation, 1.0) * 0.5 +
            min(psi_value / self.config.psi_threshold_high, 1.0) * 0.3 +
            (1 - ks_pvalue) * 0.2
        ))

        severity = self._determine_severity(drift_score, 1.0 if drift_detected else 0.0)

        error_info = FeatureDriftInfo(
            feature_name="prediction_errors",
            drift_detected=drift_detected,
            drift_score=drift_score,
            statistic_name="ks_test",
            statistic_value=ks_stat,
            p_value=ks_pvalue,
            threshold=self.config.significance_level,
            reference_distribution={
                "mae": ref_mae,
                "rmse": ref_rmse,
                "error_mean": float(np.mean(reference_errors)),
                "error_std": float(np.std(reference_errors)),
            },
            current_distribution={
                "mae": cur_mae,
                "rmse": cur_rmse,
                "error_mean": float(np.mean(current_errors)),
                "error_std": float(np.std(current_errors)),
                "mae_degradation": mae_degradation,
                "rmse_degradation": rmse_degradation,
            },
        )

        processing_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        recommendations = []
        if drift_detected:
            recommendations.append(
                f"Concept drift detected. MAE degradation: {mae_degradation*100:.1f}%, "
                f"RMSE degradation: {rmse_degradation*100:.1f}%"
            )
            if severity in ["high", "critical"]:
                recommendations.append(
                    "Model retraining recommended due to significant concept drift."
                )

        result = DriftAnalysisResult(
            report_id=self._generate_report_id(agent_id),
            agent_id=agent_id,
            model_name=model_name,
            model_version=model_version,
            drift_type="concept",
            drift_detected=drift_detected,
            overall_drift_score=drift_score,
            severity=severity,
            feature_drift_results=[error_info],
            drifted_features=["prediction_errors"] if drift_detected else [],
            drift_share=1.0 if drift_detected else 0.0,
            reference_data_size=len(reference_predictions),
            current_data_size=len(current_predictions),
            reference_data_hash=self._calculate_data_hash(reference_predictions),
            current_data_hash=self._calculate_data_hash(current_predictions),
            analysis_timestamp=start_time,
            processing_time_ms=processing_time_ms,
            recommendations=recommendations,
        )

        result.report_hash = result.calculate_report_hash()
        self._save_report(result)
        self._cache_result(agent_id, result)

        logger.info(
            f"Concept drift analysis for {agent_id} complete: "
            f"drift_detected={drift_detected}, severity={severity}, "
            f"mae_degradation={mae_degradation:.2%}"
        )

        return result

    def _determine_severity(self, drift_score: float, drift_share: float) -> str:
        """Determine drift severity based on score and share."""
        if drift_score < 0.1 and drift_share < 0.1:
            return "none"
        elif drift_score < 0.2 and drift_share < 0.2:
            return "low"
        elif drift_score < 0.3 and drift_share < 0.3:
            return "medium"
        elif drift_score < 0.5 and drift_share < 0.5:
            return "high"
        else:
            return "critical"

    def _generate_recommendations(
        self,
        severity: str,
        drifted_features: List[str],
        feature_results: List[FeatureDriftInfo],
    ) -> List[str]:
        """Generate actionable recommendations based on drift analysis."""
        recommendations = []

        if severity == "critical":
            recommendations.append(
                "CRITICAL: Immediate investigation required. "
                "Model predictions may be unreliable."
            )
            recommendations.append(
                "Consider triggering automatic model rollback or retraining."
            )
        elif severity == "high":
            recommendations.append(
                "HIGH: Schedule model retraining within 24-48 hours."
            )
            recommendations.append(
                "Investigate data pipeline for potential quality issues."
            )
        elif severity == "medium":
            recommendations.append(
                "MEDIUM: Plan model retraining within 1 week."
            )
            recommendations.append(
                "Increase monitoring frequency for affected features."
            )
        elif severity == "low":
            recommendations.append(
                "LOW: Minor drift detected. Continue monitoring."
            )
        else:
            recommendations.append(
                "No significant drift detected. Model performance stable."
            )

        if drifted_features:
            recommendations.append(
                f"Top drifted features: {', '.join(drifted_features[:5])}"
            )

            # High PSI features
            high_psi = [
                f.feature_name for f in feature_results
                if f.reference_distribution and
                f.reference_distribution.get("psi", 0) > self.config.psi_threshold_high
            ]
            if high_psi:
                recommendations.append(
                    f"Features with high PSI: {', '.join(high_psi[:3])}"
                )

        return recommendations

    def generate_drift_report(
        self,
        result: DriftAnalysisResult,
        output_path: Optional[str] = None,
        format: str = "html",
    ) -> str:
        """
        Generate a drift report file.

        Args:
            result: DriftAnalysisResult from drift detection.
            output_path: Path to save report. Defaults to config storage path.
            format: Report format ("html", "json", "both").

        Returns:
            Path to the generated report.
        """
        output_path = output_path or str(self.storage_path)
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = result.analysis_timestamp.strftime("%Y%m%d_%H%M%S")
        base_name = f"drift_report_{result.agent_id}_{timestamp}"

        generated_paths = []

        if format in ["json", "both"]:
            json_path = output_dir / f"{base_name}.json"
            with open(json_path, "w") as f:
                f.write(result.model_dump_json(indent=2))
            generated_paths.append(str(json_path))

        if format in ["html", "both"]:
            html_path = output_dir / f"{base_name}.html"
            html_content = self._generate_html_report(result)
            with open(html_path, "w") as f:
                f.write(html_content)
            generated_paths.append(str(html_path))

        logger.info(f"Generated drift report(s): {', '.join(generated_paths)}")

        return generated_paths[0] if generated_paths else ""

    def _generate_html_report(self, result: DriftAnalysisResult) -> str:
        """Generate HTML report content."""
        severity_colors = {
            "none": "#28a745",
            "low": "#17a2b8",
            "medium": "#ffc107",
            "high": "#fd7e14",
            "critical": "#dc3545",
        }

        feature_rows = ""
        for f in result.feature_drift_results:
            drift_badge = (
                '<span style="color: #dc3545;">DRIFT</span>'
                if f.drift_detected else
                '<span style="color: #28a745;">OK</span>'
            )
            p_value_str = f"{f.p_value:.4f}" if f.p_value is not None else "N/A"
            feature_rows += f"""
            <tr>
                <td>{f.feature_name}</td>
                <td>{drift_badge}</td>
                <td>{f.drift_score:.3f}</td>
                <td>{f.statistic_name}</td>
                <td>{f.statistic_value:.4f}</td>
                <td>{p_value_str}</td>
            </tr>
            """

        recommendations_html = "".join(
            f"<li>{rec}</li>" for rec in result.recommendations
        )

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Drift Report - {result.agent_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #343a40; color: white; padding: 20px; border-radius: 8px; }}
        .severity {{
            background: {severity_colors.get(result.severity, '#6c757d')};
            color: white;
            padding: 5px 15px;
            border-radius: 4px;
            font-weight: bold;
        }}
        .section {{ margin: 20px 0; padding: 20px; background: #f8f9fa; border-radius: 8px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        th {{ background: #e9ecef; }}
        .metric {{ display: inline-block; margin: 10px; padding: 15px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric-value {{ font-size: 24px; font-weight: bold; }}
        .metric-label {{ font-size: 12px; color: #6c757d; }}
        .hash {{ font-family: monospace; font-size: 10px; color: #6c757d; word-break: break-all; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Drift Detection Report</h1>
        <p>Agent: {result.agent_id} | Model: {result.model_name} v{result.model_version}</p>
        <p>Generated: {result.analysis_timestamp.isoformat()}</p>
    </div>

    <div class="section">
        <h2>Summary</h2>
        <div class="metric">
            <div class="metric-value">{result.drift_type.upper()}</div>
            <div class="metric-label">Drift Type</div>
        </div>
        <div class="metric">
            <div class="metric-value"><span class="severity">{result.severity.upper()}</span></div>
            <div class="metric-label">Severity</div>
        </div>
        <div class="metric">
            <div class="metric-value">{result.overall_drift_score:.3f}</div>
            <div class="metric-label">Overall Score</div>
        </div>
        <div class="metric">
            <div class="metric-value">{len(result.drifted_features)}/{len(result.feature_drift_results)}</div>
            <div class="metric-label">Drifted Features</div>
        </div>
        <div class="metric">
            <div class="metric-value">{result.processing_time_ms:.1f}ms</div>
            <div class="metric-label">Processing Time</div>
        </div>
    </div>

    <div class="section">
        <h2>Feature Analysis</h2>
        <table>
            <tr>
                <th>Feature</th>
                <th>Status</th>
                <th>Drift Score</th>
                <th>Test</th>
                <th>Statistic</th>
                <th>P-Value</th>
            </tr>
            {feature_rows}
        </table>
    </div>

    <div class="section">
        <h2>Recommendations</h2>
        <ul>{recommendations_html}</ul>
    </div>

    <div class="section">
        <h2>Provenance</h2>
        <p><strong>Report ID:</strong> {result.report_id}</p>
        <p><strong>Reference Data Hash:</strong></p>
        <p class="hash">{result.reference_data_hash}</p>
        <p><strong>Current Data Hash:</strong></p>
        <p class="hash">{result.current_data_hash}</p>
        <p><strong>Report Hash:</strong></p>
        <p class="hash">{result.report_hash}</p>
    </div>
</body>
</html>
        """
        return html

    def _save_report(self, result: DriftAnalysisResult) -> None:
        """Save drift report to storage."""
        report_path = self.storage_path / f"drift_report_{result.report_id}.json"
        with open(report_path, "w") as f:
            f.write(result.model_dump_json(indent=2))

    def _cache_result(self, agent_id: str, result: DriftAnalysisResult) -> None:
        """Cache result for metrics aggregation."""
        with self._lock:
            if agent_id not in self._recent_results:
                self._recent_results[agent_id] = []

            self._recent_results[agent_id].append(result)

            # Limit cache size
            if len(self._recent_results[agent_id]) > self._max_cached_results:
                self._recent_results[agent_id] = (
                    self._recent_results[agent_id][-self._max_cached_results:]
                )

    def get_drift_metrics(
        self,
        agent_id: str,
        window_hours: int = 24,
    ) -> Dict[str, Any]:
        """
        Get aggregated drift metrics for an agent.

        Args:
            agent_id: Agent identifier (GL-001 through GL-020).
            window_hours: Time window in hours for aggregation.

        Returns:
            Dictionary with drift metrics.
        """
        self._validate_agent_id(agent_id)

        cutoff = datetime.utcnow().timestamp() - (window_hours * 3600)

        with self._lock:
            results = [
                r for r in self._recent_results.get(agent_id, [])
                if r.analysis_timestamp.timestamp() > cutoff
            ]

        if not results:
            return {
                "agent_id": agent_id,
                "window_hours": window_hours,
                "analysis_count": 0,
                "drift_detected_count": 0,
                "overall_drift_score": 0.0,
                "severity_distribution": {},
                "most_drifted_features": [],
            }

        drift_count = sum(1 for r in results if r.drift_detected)
        avg_score = float(np.mean([r.overall_drift_score for r in results]))

        severity_dist = {}
        for severity in ["none", "low", "medium", "high", "critical"]:
            severity_dist[severity] = sum(1 for r in results if r.severity == severity)

        feature_drift_counts: Dict[str, int] = {}
        for result in results:
            for feature in result.drifted_features:
                feature_drift_counts[feature] = feature_drift_counts.get(feature, 0) + 1

        most_drifted = sorted(
            feature_drift_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]

        return {
            "agent_id": agent_id,
            "window_hours": window_hours,
            "analysis_count": len(results),
            "drift_detected_count": drift_count,
            "drift_rate": drift_count / len(results) if results else 0.0,
            "overall_drift_score": avg_score,
            "severity_distribution": severity_dist,
            "most_drifted_features": most_drifted,
        }

    def export_prometheus_metrics(
        self, agent_id: str
    ) -> List[PrometheusMetricExport]:
        """
        Export drift metrics in Prometheus format.

        Args:
            agent_id: Agent identifier (GL-001 through GL-020).

        Returns:
            List of PrometheusMetricExport objects.
        """
        if not self.config.prometheus_enabled:
            return []

        metrics = self.get_drift_metrics(agent_id, window_hours=1)
        prefix = self.config.prometheus_prefix
        timestamp_ms = int(datetime.utcnow().timestamp() * 1000)

        prometheus_metrics = []

        # Drift analysis count
        prometheus_metrics.append(PrometheusMetricExport(
            name=f"{prefix}_analysis_total",
            help_text="Total number of drift analyses performed",
            metric_type="counter",
            labels={"agent": agent_id},
            value=float(metrics["analysis_count"]),
            timestamp_ms=timestamp_ms,
        ))

        # Drift detection count
        prometheus_metrics.append(PrometheusMetricExport(
            name=f"{prefix}_detected_total",
            help_text="Total number of drift detections",
            metric_type="counter",
            labels={"agent": agent_id},
            value=float(metrics["drift_detected_count"]),
            timestamp_ms=timestamp_ms,
        ))

        # Drift rate
        prometheus_metrics.append(PrometheusMetricExport(
            name=f"{prefix}_rate",
            help_text="Drift detection rate (0-1)",
            metric_type="gauge",
            labels={"agent": agent_id},
            value=float(metrics.get("drift_rate", 0.0)),
            timestamp_ms=timestamp_ms,
        ))

        # Overall drift score
        prometheus_metrics.append(PrometheusMetricExport(
            name=f"{prefix}_score",
            help_text="Overall drift score (0-1)",
            metric_type="gauge",
            labels={"agent": agent_id},
            value=float(metrics["overall_drift_score"]),
            timestamp_ms=timestamp_ms,
        ))

        # Severity counts
        for severity, count in metrics.get("severity_distribution", {}).items():
            prometheus_metrics.append(PrometheusMetricExport(
                name=f"{prefix}_severity",
                help_text=f"Count of {severity} severity drift events",
                metric_type="gauge",
                labels={"agent": agent_id, "severity": severity},
                value=float(count),
                timestamp_ms=timestamp_ms,
            ))

        return prometheus_metrics

    def get_prometheus_text(self, agent_id: str) -> str:
        """
        Get Prometheus metrics in text exposition format.

        Args:
            agent_id: Agent identifier.

        Returns:
            Prometheus text format string.
        """
        metrics = self.export_prometheus_metrics(agent_id)
        lines = []

        for metric in metrics:
            lines.append(f"# HELP {metric.name} {metric.help_text}")
            lines.append(f"# TYPE {metric.name} {metric.metric_type}")
            labels_str = ",".join(f'{k}="{v}"' for k, v in metric.labels.items())
            if metric.timestamp_ms:
                lines.append(
                    f"{metric.name}{{{labels_str}}} {metric.value} {metric.timestamp_ms}"
                )
            else:
                lines.append(f"{metric.name}{{{labels_str}}} {metric.value}")

        return "\n".join(lines)
