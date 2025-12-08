# -*- coding: utf-8 -*-
"""
Uncertainty Visualization Module

This module provides data generators for uncertainty visualization in
GreenLang ML models, creating JSON-serializable data for heatmaps,
confidence bands, calibration plots, and feature-uncertainty correlations.

The visualization module produces data structures suitable for frontend
rendering, enabling interactive uncertainty exploration for regulatory
compliance and decision-making.

Example:
    >>> from greenlang.ml.uncertainty.visualization import UncertaintyVisualizer
    >>> visualizer = UncertaintyVisualizer()
    >>> heatmap_data = visualizer.generate_heatmap_data(predictions, uncertainties)
    >>> calibration_data = visualizer.generate_calibration_plot(probas, labels)
"""

from typing import Any, Dict, List, Optional, Union, Tuple
from pydantic import BaseModel, Field
import numpy as np
import hashlib
import logging
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models for Visualization
# ============================================================================

class ColorScale(str, Enum):
    """Color scales for visualizations."""
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    INFERNO = "inferno"
    MAGMA = "magma"
    COOLWARM = "coolwarm"
    RD_YL_GN = "RdYlGn"


class HeatmapCell(BaseModel):
    """Single cell in uncertainty heatmap."""

    x: Union[int, float, str] = Field(
        ...,
        description="X coordinate or label"
    )
    y: Union[int, float, str] = Field(
        ...,
        description="Y coordinate or label"
    )
    value: float = Field(
        ...,
        description="Cell value (uncertainty)"
    )
    prediction: Optional[float] = Field(
        default=None,
        description="Associated prediction"
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Confidence level"
    )


class HeatmapData(BaseModel):
    """Data for uncertainty heatmap."""

    cells: List[HeatmapCell] = Field(
        ...,
        description="Heatmap cells"
    )
    x_labels: List[str] = Field(
        ...,
        description="X-axis labels"
    )
    y_labels: List[str] = Field(
        ...,
        description="Y-axis labels"
    )
    min_value: float = Field(
        ...,
        description="Minimum value"
    )
    max_value: float = Field(
        ...,
        description="Maximum value"
    )
    color_scale: str = Field(
        default="viridis",
        description="Color scale to use"
    )
    title: str = Field(
        default="Uncertainty Heatmap",
        description="Chart title"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash"
    )


class ConfidenceBandPoint(BaseModel):
    """Single point in confidence band."""

    x: float = Field(
        ...,
        description="X value (typically time or index)"
    )
    prediction: float = Field(
        ...,
        description="Point prediction"
    )
    lower: float = Field(
        ...,
        description="Lower bound"
    )
    upper: float = Field(
        ...,
        description="Upper bound"
    )
    uncertainty: Optional[float] = Field(
        default=None,
        description="Uncertainty value"
    )


class ConfidenceBandData(BaseModel):
    """Data for confidence band visualization."""

    points: List[ConfidenceBandPoint] = Field(
        ...,
        description="Band points"
    )
    confidence_level: float = Field(
        ...,
        description="Confidence level"
    )
    x_label: str = Field(
        default="Time",
        description="X-axis label"
    )
    y_label: str = Field(
        default="Value",
        description="Y-axis label"
    )
    title: str = Field(
        default="Prediction with Confidence Bands",
        description="Chart title"
    )
    actual_values: Optional[List[Optional[float]]] = Field(
        default=None,
        description="Actual values if available"
    )
    coverage_rate: Optional[float] = Field(
        default=None,
        description="Empirical coverage rate"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash"
    )


class CalibrationPlotBin(BaseModel):
    """Single bin in calibration plot."""

    bin_center: float = Field(
        ...,
        description="Center of confidence bin"
    )
    accuracy: float = Field(
        ...,
        description="Accuracy in bin"
    )
    confidence: float = Field(
        ...,
        description="Mean confidence in bin"
    )
    count: int = Field(
        ...,
        description="Number of samples in bin"
    )
    gap: float = Field(
        ...,
        description="Gap between accuracy and confidence"
    )


class CalibrationPlotData(BaseModel):
    """Data for calibration/reliability diagram."""

    bins: List[CalibrationPlotBin] = Field(
        ...,
        description="Calibration bins"
    )
    perfect_calibration: List[Dict[str, float]] = Field(
        ...,
        description="Perfect calibration line"
    )
    ece: float = Field(
        ...,
        description="Expected Calibration Error"
    )
    mce: float = Field(
        ...,
        description="Maximum Calibration Error"
    )
    n_samples: int = Field(
        ...,
        description="Total samples"
    )
    is_calibrated: bool = Field(
        ...,
        description="Whether well-calibrated"
    )
    title: str = Field(
        default="Reliability Diagram",
        description="Chart title"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash"
    )


class FeatureCorrelationPoint(BaseModel):
    """Feature-uncertainty correlation point."""

    feature_value: float = Field(
        ...,
        description="Feature value"
    )
    uncertainty: float = Field(
        ...,
        description="Uncertainty value"
    )
    prediction: Optional[float] = Field(
        default=None,
        description="Prediction value"
    )


class FeatureCorrelationData(BaseModel):
    """Data for feature-uncertainty correlation."""

    feature_name: str = Field(
        ...,
        description="Feature name"
    )
    points: List[FeatureCorrelationPoint] = Field(
        ...,
        description="Correlation points"
    )
    correlation: float = Field(
        ...,
        description="Pearson correlation"
    )
    spearman_correlation: float = Field(
        ...,
        description="Spearman correlation"
    )
    trend_line: Optional[Dict[str, List[float]]] = Field(
        default=None,
        description="Trend line x and y values"
    )
    title: str = Field(
        default="Feature-Uncertainty Correlation",
        description="Chart title"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash"
    )


class UncertaintyDistributionData(BaseModel):
    """Data for uncertainty distribution histogram."""

    bins: List[float] = Field(
        ...,
        description="Bin edges"
    )
    counts: List[int] = Field(
        ...,
        description="Counts per bin"
    )
    density: List[float] = Field(
        ...,
        description="Density per bin"
    )
    mean: float = Field(
        ...,
        description="Mean uncertainty"
    )
    median: float = Field(
        ...,
        description="Median uncertainty"
    )
    std: float = Field(
        ...,
        description="Standard deviation"
    )
    percentiles: Dict[str, float] = Field(
        ...,
        description="Percentiles (25, 50, 75, 90, 95, 99)"
    )
    title: str = Field(
        default="Uncertainty Distribution",
        description="Chart title"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash"
    )


class UncertaintyDecompositionData(BaseModel):
    """Data for epistemic/aleatoric decomposition visualization."""

    sample_indices: List[int] = Field(
        ...,
        description="Sample indices"
    )
    epistemic: List[float] = Field(
        ...,
        description="Epistemic uncertainty per sample"
    )
    aleatoric: List[float] = Field(
        ...,
        description="Aleatoric uncertainty per sample"
    )
    total: List[float] = Field(
        ...,
        description="Total uncertainty per sample"
    )
    epistemic_ratio: List[float] = Field(
        ...,
        description="Epistemic ratio per sample"
    )
    mean_epistemic: float = Field(
        ...,
        description="Mean epistemic uncertainty"
    )
    mean_aleatoric: float = Field(
        ...,
        description="Mean aleatoric uncertainty"
    )
    title: str = Field(
        default="Uncertainty Decomposition",
        description="Chart title"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash"
    )


# ============================================================================
# Visualizer Class
# ============================================================================

class UncertaintyVisualizer:
    """
    Uncertainty Visualizer for generating visualization data.

    This class generates JSON-serializable data structures for
    uncertainty visualization, suitable for frontend rendering
    in web applications.

    Key capabilities:
    - Uncertainty heatmap data generators
    - Confidence band data for time series
    - Calibration plot data
    - Feature-uncertainty correlation
    - Uncertainty distribution histograms
    - Epistemic/aleatoric decomposition

    Attributes:
        config: Visualizer configuration
        color_scale: Default color scale

    Example:
        >>> visualizer = UncertaintyVisualizer()
        >>> heatmap = visualizer.generate_heatmap_data(preds, uncerts, shape=(10, 10))
        >>> calibration = visualizer.generate_calibration_plot(probas, labels)
        >>> bands = visualizer.generate_confidence_bands(time_idx, preds, lower, upper)
    """

    def __init__(
        self,
        color_scale: ColorScale = ColorScale.VIRIDIS,
        n_bins: int = 15
    ):
        """
        Initialize uncertainty visualizer.

        Args:
            color_scale: Default color scale
            n_bins: Default number of bins
        """
        self.color_scale = color_scale
        self.n_bins = n_bins

        logger.info("UncertaintyVisualizer initialized")

    def _calculate_provenance(self, data_sum: float, n_points: int) -> str:
        """Calculate SHA-256 provenance hash."""
        combined = f"{data_sum:.8f}|{n_points}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def generate_heatmap_data(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        shape: Optional[Tuple[int, int]] = None,
        x_labels: Optional[List[str]] = None,
        y_labels: Optional[List[str]] = None,
        title: str = "Uncertainty Heatmap"
    ) -> HeatmapData:
        """
        Generate heatmap data for uncertainty visualization.

        Args:
            predictions: Predictions array
            uncertainties: Uncertainty values
            shape: Reshape to (rows, cols) if provided
            x_labels: Custom X-axis labels
            y_labels: Custom Y-axis labels
            title: Chart title

        Returns:
            HeatmapData for rendering

        Example:
            >>> heatmap = visualizer.generate_heatmap_data(
            ...     predictions, uncertainties, shape=(10, 10)
            ... )
        """
        if shape is not None:
            uncerts = uncertainties.reshape(shape)
            preds = predictions.reshape(shape) if predictions.shape == uncertainties.shape else None
        else:
            if len(uncertainties.shape) == 1:
                # Reshape to 2D
                size = len(uncertainties)
                rows = int(np.sqrt(size))
                cols = size // rows
                uncerts = uncertainties[:rows * cols].reshape(rows, cols)
                preds = predictions[:rows * cols].reshape(rows, cols) if predictions is not None else None
            else:
                uncerts = uncertainties
                preds = predictions

        rows, cols = uncerts.shape

        cells = []
        for i in range(rows):
            for j in range(cols):
                cell = HeatmapCell(
                    x=j,
                    y=i,
                    value=float(uncerts[i, j]),
                    prediction=float(preds[i, j]) if preds is not None else None
                )
                cells.append(cell)

        x_labels = x_labels or [str(j) for j in range(cols)]
        y_labels = y_labels or [str(i) for i in range(rows)]

        provenance = self._calculate_provenance(
            float(np.sum(uncerts)),
            len(cells)
        )

        return HeatmapData(
            cells=cells,
            x_labels=x_labels,
            y_labels=y_labels,
            min_value=float(np.min(uncerts)),
            max_value=float(np.max(uncerts)),
            color_scale=self.color_scale.value,
            title=title,
            provenance_hash=provenance
        )

    def generate_confidence_bands(
        self,
        x_values: np.ndarray,
        predictions: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        actual_values: Optional[np.ndarray] = None,
        confidence_level: float = 0.95,
        x_label: str = "Time",
        y_label: str = "Value",
        title: str = "Prediction with Confidence Bands"
    ) -> ConfidenceBandData:
        """
        Generate confidence band data for time series visualization.

        Args:
            x_values: X-axis values (time indices)
            predictions: Point predictions
            lower_bounds: Lower confidence bounds
            upper_bounds: Upper confidence bounds
            actual_values: Optional actual values
            confidence_level: Confidence level
            x_label: X-axis label
            y_label: Y-axis label
            title: Chart title

        Returns:
            ConfidenceBandData for rendering

        Example:
            >>> bands = visualizer.generate_confidence_bands(
            ...     time_idx, predictions, lower, upper, confidence=0.95
            ... )
        """
        points = []
        coverage_count = 0

        for i in range(len(x_values)):
            point = ConfidenceBandPoint(
                x=float(x_values[i]),
                prediction=float(predictions[i]),
                lower=float(lower_bounds[i]),
                upper=float(upper_bounds[i]),
                uncertainty=float((upper_bounds[i] - lower_bounds[i]) / 2)
            )
            points.append(point)

            if actual_values is not None and actual_values[i] is not None:
                if lower_bounds[i] <= actual_values[i] <= upper_bounds[i]:
                    coverage_count += 1

        coverage_rate = None
        if actual_values is not None:
            valid_count = sum(1 for v in actual_values if v is not None)
            if valid_count > 0:
                coverage_rate = coverage_count / valid_count

        provenance = self._calculate_provenance(
            float(np.sum(predictions)),
            len(points)
        )

        return ConfidenceBandData(
            points=points,
            confidence_level=confidence_level,
            x_label=x_label,
            y_label=y_label,
            title=title,
            actual_values=[float(v) if v is not None else None for v in actual_values] if actual_values is not None else None,
            coverage_rate=coverage_rate,
            provenance_hash=provenance
        )

    def generate_calibration_plot(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        n_bins: Optional[int] = None,
        title: str = "Reliability Diagram"
    ) -> CalibrationPlotData:
        """
        Generate calibration/reliability diagram data.

        Args:
            probabilities: Predicted probabilities
            labels: True labels
            n_bins: Number of bins
            title: Chart title

        Returns:
            CalibrationPlotData for rendering

        Example:
            >>> calibration = visualizer.generate_calibration_plot(
            ...     probabilities, labels, n_bins=10
            ... )
        """
        n_bins = n_bins or self.n_bins

        # Handle multi-class vs binary
        if len(probabilities.shape) > 1:
            confidences = np.max(probabilities, axis=1)
            predictions = np.argmax(probabilities, axis=1)
        else:
            confidences = np.maximum(probabilities, 1 - probabilities)
            predictions = (probabilities > 0.5).astype(int)

        accuracies = (predictions == labels).astype(float)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bins = []
        ece = 0.0
        mce = 0.0

        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            bin_center = (bin_boundaries[i] + bin_boundaries[i + 1]) / 2

            if np.sum(in_bin) > 0:
                bin_acc = float(np.mean(accuracies[in_bin]))
                bin_conf = float(np.mean(confidences[in_bin]))
                bin_count = int(np.sum(in_bin))
                gap = abs(bin_acc - bin_conf)

                ece += gap * bin_count / len(probabilities)
                mce = max(mce, gap)

                bins.append(CalibrationPlotBin(
                    bin_center=bin_center,
                    accuracy=bin_acc,
                    confidence=bin_conf,
                    count=bin_count,
                    gap=float(bin_acc - bin_conf)
                ))
            else:
                bins.append(CalibrationPlotBin(
                    bin_center=bin_center,
                    accuracy=bin_center,
                    confidence=bin_center,
                    count=0,
                    gap=0.0
                ))

        # Perfect calibration line
        perfect_calibration = [
            {"x": 0.0, "y": 0.0},
            {"x": 1.0, "y": 1.0}
        ]

        provenance = self._calculate_provenance(float(ece), len(probabilities))

        return CalibrationPlotData(
            bins=bins,
            perfect_calibration=perfect_calibration,
            ece=float(ece),
            mce=float(mce),
            n_samples=len(probabilities),
            is_calibrated=ece < 0.05,
            title=title,
            provenance_hash=provenance
        )

    def generate_feature_correlation(
        self,
        feature_values: np.ndarray,
        uncertainties: np.ndarray,
        feature_name: str,
        predictions: Optional[np.ndarray] = None,
        include_trend: bool = True,
        title: Optional[str] = None
    ) -> FeatureCorrelationData:
        """
        Generate feature-uncertainty correlation data.

        Args:
            feature_values: Feature values
            uncertainties: Uncertainty values
            feature_name: Feature name
            predictions: Optional predictions
            include_trend: Include trend line
            title: Chart title

        Returns:
            FeatureCorrelationData for rendering

        Example:
            >>> correlation = visualizer.generate_feature_correlation(
            ...     feature_values, uncertainties, "temperature"
            ... )
        """
        points = []
        for i in range(len(feature_values)):
            point = FeatureCorrelationPoint(
                feature_value=float(feature_values[i]),
                uncertainty=float(uncertainties[i]),
                prediction=float(predictions[i]) if predictions is not None else None
            )
            points.append(point)

        # Calculate correlations
        correlation = float(np.corrcoef(feature_values, uncertainties)[0, 1])

        # Spearman correlation
        from scipy import stats
        spearman_corr, _ = stats.spearmanr(feature_values, uncertainties)

        # Trend line
        trend_line = None
        if include_trend:
            # Linear regression
            z = np.polyfit(feature_values, uncertainties, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(np.min(feature_values), np.max(feature_values), 50)
            y_trend = p(x_trend)
            trend_line = {
                "x": x_trend.tolist(),
                "y": y_trend.tolist()
            }

        title = title or f"Feature-Uncertainty Correlation: {feature_name}"

        provenance = self._calculate_provenance(
            float(np.sum(uncertainties)),
            len(points)
        )

        return FeatureCorrelationData(
            feature_name=feature_name,
            points=points,
            correlation=correlation,
            spearman_correlation=float(spearman_corr),
            trend_line=trend_line,
            title=title,
            provenance_hash=provenance
        )

    def generate_uncertainty_distribution(
        self,
        uncertainties: np.ndarray,
        n_bins: Optional[int] = None,
        title: str = "Uncertainty Distribution"
    ) -> UncertaintyDistributionData:
        """
        Generate uncertainty distribution histogram data.

        Args:
            uncertainties: Uncertainty values
            n_bins: Number of bins
            title: Chart title

        Returns:
            UncertaintyDistributionData for rendering

        Example:
            >>> dist = visualizer.generate_uncertainty_distribution(uncertainties)
        """
        n_bins = n_bins or 30

        counts, bin_edges = np.histogram(uncertainties, bins=n_bins)
        density = counts / (len(uncertainties) * (bin_edges[1] - bin_edges[0]))

        percentiles = {
            "p25": float(np.percentile(uncertainties, 25)),
            "p50": float(np.percentile(uncertainties, 50)),
            "p75": float(np.percentile(uncertainties, 75)),
            "p90": float(np.percentile(uncertainties, 90)),
            "p95": float(np.percentile(uncertainties, 95)),
            "p99": float(np.percentile(uncertainties, 99))
        }

        provenance = self._calculate_provenance(
            float(np.sum(uncertainties)),
            len(uncertainties)
        )

        return UncertaintyDistributionData(
            bins=bin_edges.tolist(),
            counts=counts.tolist(),
            density=density.tolist(),
            mean=float(np.mean(uncertainties)),
            median=float(np.median(uncertainties)),
            std=float(np.std(uncertainties)),
            percentiles=percentiles,
            title=title,
            provenance_hash=provenance
        )

    def generate_decomposition_plot(
        self,
        epistemic: np.ndarray,
        aleatoric: np.ndarray,
        sample_indices: Optional[np.ndarray] = None,
        title: str = "Uncertainty Decomposition"
    ) -> UncertaintyDecompositionData:
        """
        Generate epistemic/aleatoric decomposition visualization data.

        Args:
            epistemic: Epistemic uncertainty values
            aleatoric: Aleatoric uncertainty values
            sample_indices: Optional sample indices
            title: Chart title

        Returns:
            UncertaintyDecompositionData for rendering

        Example:
            >>> decomposition = visualizer.generate_decomposition_plot(
            ...     epistemic, aleatoric
            ... )
        """
        total = epistemic + aleatoric
        epistemic_ratio = epistemic / (total + 1e-10)

        if sample_indices is None:
            sample_indices = np.arange(len(epistemic))

        provenance = self._calculate_provenance(
            float(np.sum(total)),
            len(epistemic)
        )

        return UncertaintyDecompositionData(
            sample_indices=sample_indices.tolist(),
            epistemic=epistemic.tolist(),
            aleatoric=aleatoric.tolist(),
            total=total.tolist(),
            epistemic_ratio=epistemic_ratio.tolist(),
            mean_epistemic=float(np.mean(epistemic)),
            mean_aleatoric=float(np.mean(aleatoric)),
            title=title,
            provenance_hash=provenance
        )

    def generate_multi_feature_correlation(
        self,
        features: np.ndarray,
        uncertainties: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, FeatureCorrelationData]:
        """
        Generate correlation data for multiple features.

        Args:
            features: Feature matrix (n_samples x n_features)
            uncertainties: Uncertainty values
            feature_names: List of feature names

        Returns:
            Dictionary of feature name to correlation data
        """
        results = {}

        for i, name in enumerate(feature_names):
            results[name] = self.generate_feature_correlation(
                features[:, i],
                uncertainties,
                name
            )

        return results

    def to_json(self, data: BaseModel) -> str:
        """
        Convert visualization data to JSON string.

        Args:
            data: Pydantic model

        Returns:
            JSON string
        """
        return data.model_dump_json()


# ============================================================================
# Unit Tests
# ============================================================================

class TestUncertaintyVisualizer:
    """Unit tests for UncertaintyVisualizer."""

    def test_init(self):
        """Test initialization."""
        visualizer = UncertaintyVisualizer()
        assert visualizer.n_bins == 15

    def test_heatmap_data(self):
        """Test heatmap data generation."""
        visualizer = UncertaintyVisualizer()

        predictions = np.random.randn(100)
        uncertainties = np.abs(np.random.randn(100))

        heatmap = visualizer.generate_heatmap_data(
            predictions, uncertainties, shape=(10, 10)
        )

        assert len(heatmap.cells) == 100
        assert heatmap.min_value <= heatmap.max_value

    def test_confidence_bands(self):
        """Test confidence band generation."""
        visualizer = UncertaintyVisualizer()

        x = np.arange(50)
        predictions = np.sin(x / 5) + 5
        lower = predictions - 0.5
        upper = predictions + 0.5

        bands = visualizer.generate_confidence_bands(
            x, predictions, lower, upper
        )

        assert len(bands.points) == 50
        assert bands.confidence_level == 0.95

    def test_calibration_plot(self):
        """Test calibration plot generation."""
        visualizer = UncertaintyVisualizer()

        probabilities = np.random.rand(200)
        labels = (probabilities > 0.5).astype(int)

        calibration = visualizer.generate_calibration_plot(
            probabilities, labels
        )

        assert calibration.ece >= 0
        assert len(calibration.bins) == visualizer.n_bins

    def test_feature_correlation(self):
        """Test feature correlation generation."""
        visualizer = UncertaintyVisualizer()

        feature_values = np.random.randn(100)
        uncertainties = np.abs(feature_values) * 0.5 + 0.1

        correlation = visualizer.generate_feature_correlation(
            feature_values, uncertainties, "test_feature"
        )

        assert correlation.feature_name == "test_feature"
        assert -1 <= correlation.correlation <= 1

    def test_uncertainty_distribution(self):
        """Test uncertainty distribution generation."""
        visualizer = UncertaintyVisualizer()

        uncertainties = np.abs(np.random.randn(200))

        dist = visualizer.generate_uncertainty_distribution(uncertainties)

        assert dist.mean > 0
        assert len(dist.bins) > 0

    def test_decomposition_plot(self):
        """Test decomposition plot generation."""
        visualizer = UncertaintyVisualizer()

        epistemic = np.abs(np.random.randn(50))
        aleatoric = np.abs(np.random.randn(50))

        decomposition = visualizer.generate_decomposition_plot(
            epistemic, aleatoric
        )

        assert len(decomposition.epistemic) == 50
        assert all(r >= 0 and r <= 1 for r in decomposition.epistemic_ratio)

    def test_provenance_deterministic(self):
        """Test provenance hash is deterministic."""
        visualizer = UncertaintyVisualizer()

        hash1 = visualizer._calculate_provenance(10.5, 100)
        hash2 = visualizer._calculate_provenance(10.5, 100)

        assert hash1 == hash2
