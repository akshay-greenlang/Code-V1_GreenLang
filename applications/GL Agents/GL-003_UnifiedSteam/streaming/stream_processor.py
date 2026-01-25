"""
GL-003 UNIFIEDSTEAM - Stream Processor

Real-time stream processing for steam system data:
- Signal validation and transformation
- Feature extraction (statistical, spectral)
- Windowed aggregations (tumbling, sliding, session)
- Real-time anomaly detection
- Trigger-based computation dispatch

Processing Pipeline:
    Raw Signal -> Validation -> Feature Extraction -> Anomaly Detection -> Output
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Deque
from collections import deque
import asyncio
import logging
import math
import statistics

logger = logging.getLogger(__name__)


class WindowType(Enum):
    """Window types for stream aggregations."""
    TUMBLING = "tumbling"  # Non-overlapping fixed windows
    SLIDING = "sliding"  # Overlapping windows
    SESSION = "session"  # Gap-based windows
    COUNT = "count"  # Count-based windows


class AnomalyType(Enum):
    """Types of anomalies detected."""
    SPIKE = "spike"  # Sudden increase
    DROP = "drop"  # Sudden decrease
    DRIFT = "drift"  # Gradual change
    OSCILLATION = "oscillation"  # Excessive variance
    FLATLINE = "flatline"  # No change (stuck sensor)
    OUT_OF_RANGE = "out_of_range"  # Beyond expected limits
    RATE_VIOLATION = "rate_violation"  # Exceeds rate of change


class ComputationType(Enum):
    """Types of triggered computations."""
    THERMODYNAMIC = "thermodynamic"  # Enthalpy, entropy calculations
    EFFICIENCY = "efficiency"  # Boiler/turbine efficiency
    OPTIMIZATION = "optimization"  # Setpoint optimization
    PREDICTION = "prediction"  # Predictive calculations
    DIAGNOSTIC = "diagnostic"  # Equipment health assessment


@dataclass
class WindowConfig:
    """Configuration for windowed processing."""
    window_type: WindowType = WindowType.TUMBLING
    window_size_s: int = 60  # Window size in seconds
    slide_size_s: int = 30  # Slide size for sliding windows
    session_gap_s: int = 300  # Gap for session windows
    min_samples: int = 10  # Minimum samples for valid window


@dataclass
class ProcessingConfig:
    """Stream processing configuration."""
    # Validation
    enable_validation: bool = True
    validation_threshold: float = 0.95

    # Feature extraction
    enable_features: bool = True
    feature_window: WindowConfig = field(default_factory=WindowConfig)

    # Anomaly detection
    enable_anomaly_detection: bool = True
    anomaly_sensitivity: float = 3.0  # Standard deviations
    spike_threshold: float = 5.0  # Percent change for spike
    drift_window_s: int = 3600  # Window for drift detection
    flatline_tolerance: float = 0.001  # Tolerance for flatline
    flatline_count: int = 10  # Samples for flatline detection

    # Triggered computations
    enable_triggers: bool = True
    trigger_debounce_s: float = 1.0  # Minimum time between triggers


@dataclass
class ValidatedSignal:
    """Validated and processed signal."""
    tag: str
    value: float
    unit: str
    quality_code: int
    timestamp: datetime

    # Validation
    is_valid: bool = True
    validation_issues: List[str] = field(default_factory=list)

    # Transformation
    raw_value: Optional[float] = None
    calibration_applied: bool = False

    # Processing metadata
    processing_latency_ms: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "tag": self.tag,
            "value": self.value,
            "unit": self.unit,
            "quality_code": self.quality_code,
            "timestamp": self.timestamp.isoformat(),
            "is_valid": self.is_valid,
            "validation_issues": self.validation_issues,
        }


@dataclass
class FeatureSet:
    """Extracted statistical features from signal window."""
    tag: str
    window_start: datetime
    window_end: datetime
    sample_count: int

    # Central tendency
    mean: float = 0.0
    median: float = 0.0
    mode: Optional[float] = None

    # Dispersion
    std_dev: float = 0.0
    variance: float = 0.0
    iqr: float = 0.0  # Interquartile range
    range: float = 0.0

    # Shape
    skewness: float = 0.0
    kurtosis: float = 0.0

    # Extremes
    min_value: float = 0.0
    max_value: float = 0.0
    peak_to_peak: float = 0.0

    # Rates
    rate_of_change: float = 0.0  # Average rate
    max_rate_of_change: float = 0.0

    # Energy
    rms: float = 0.0  # Root mean square
    energy: float = 0.0

    # Quality
    good_quality_pct: float = 100.0

    def to_dict(self) -> Dict:
        return {
            "tag": self.tag,
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "sample_count": self.sample_count,
            "mean": round(self.mean, 6),
            "median": round(self.median, 6),
            "std_dev": round(self.std_dev, 6),
            "min_value": round(self.min_value, 6),
            "max_value": round(self.max_value, 6),
            "range": round(self.range, 6),
            "rate_of_change": round(self.rate_of_change, 6),
            "rms": round(self.rms, 6),
            "good_quality_pct": round(self.good_quality_pct, 2),
        }


@dataclass
class Anomaly:
    """Detected anomaly in signal stream."""
    anomaly_id: str
    anomaly_type: AnomalyType
    tag: str
    timestamp: datetime
    severity: str  # low, medium, high, critical

    # Values
    observed_value: float
    expected_value: float
    deviation: float  # Standard deviations or percent

    # Context
    window_mean: float = 0.0
    window_std: float = 0.0
    confidence: float = 0.0

    # Description
    description: str = ""

    def to_dict(self) -> Dict:
        return {
            "anomaly_id": self.anomaly_id,
            "anomaly_type": self.anomaly_type.value,
            "tag": self.tag,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity,
            "observed_value": self.observed_value,
            "expected_value": self.expected_value,
            "deviation": round(self.deviation, 4),
            "confidence": round(self.confidence, 4),
            "description": self.description,
        }


@dataclass
class ComputationTrigger:
    """Trigger for downstream computations."""
    trigger_id: str
    computation_type: ComputationType
    timestamp: datetime

    # Context
    trigger_reason: str
    source_tags: List[str]
    input_values: Dict[str, float]

    # Priority
    priority: str = "normal"  # low, normal, high, critical

    def to_dict(self) -> Dict:
        return {
            "trigger_id": self.trigger_id,
            "computation_type": self.computation_type.value,
            "timestamp": self.timestamp.isoformat(),
            "trigger_reason": self.trigger_reason,
            "source_tags": self.source_tags,
            "priority": self.priority,
        }


class SignalWindow:
    """Sliding window buffer for signal values."""

    def __init__(
        self,
        tag: str,
        window_size_s: int = 60,
        max_samples: int = 10000,
    ) -> None:
        self.tag = tag
        self.window_size_s = window_size_s
        self.max_samples = max_samples

        self._values: Deque[Tuple[datetime, float, int]] = deque(maxlen=max_samples)
        self._lock = asyncio.Lock()

    async def add(
        self,
        timestamp: datetime,
        value: float,
        quality: int = 0,
    ) -> None:
        """Add value to window."""
        async with self._lock:
            self._values.append((timestamp, value, quality))
            self._cleanup()

    def _cleanup(self) -> None:
        """Remove old values outside window."""
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self.window_size_s)
        while self._values and self._values[0][0] < cutoff:
            self._values.popleft()

    def get_values(self) -> List[Tuple[datetime, float, int]]:
        """Get all values in window."""
        self._cleanup()
        return list(self._values)

    def get_numeric_values(self) -> List[float]:
        """Get only numeric values (good quality)."""
        return [v for ts, v, q in self._values if q == 0]

    @property
    def count(self) -> int:
        """Get number of values in window."""
        return len(self._values)

    @property
    def is_empty(self) -> bool:
        """Check if window is empty."""
        return len(self._values) == 0


class StreamProcessor:
    """
    Real-time stream processor for steam system signals.

    Performs validation, feature extraction, and anomaly detection
    on streaming sensor data.

    Example:
        config = ProcessingConfig(
            enable_validation=True,
            enable_features=True,
            enable_anomaly_detection=True,
        )

        processor = StreamProcessor(config)

        # Process raw signal
        validated = await processor.validate_signal(raw_signal)

        # Extract features from window
        features = await processor.compute_features(validated_signals)

        # Detect anomalies
        anomalies = await processor.detect_anomalies(feature_window)

        # Trigger computations
        await processor.trigger_computation(ComputationType.EFFICIENCY, data)
    """

    def __init__(self, config: Optional[ProcessingConfig] = None) -> None:
        """
        Initialize stream processor.

        Args:
            config: Processing configuration
        """
        self.config = config or ProcessingConfig()

        # Signal windows for each tag
        self._windows: Dict[str, SignalWindow] = {}

        # Baseline statistics for anomaly detection
        self._baselines: Dict[str, Dict[str, float]] = {}

        # Computation trigger state
        self._last_triggers: Dict[str, datetime] = {}

        # Callbacks
        self._on_anomaly: Optional[Callable[[Anomaly], None]] = None
        self._on_trigger: Optional[Callable[[ComputationTrigger], None]] = None

        # Statistics
        self._stats = {
            "signals_processed": 0,
            "signals_validated": 0,
            "signals_invalid": 0,
            "features_computed": 0,
            "anomalies_detected": 0,
            "triggers_fired": 0,
        }

        logger.info("StreamProcessor initialized")

    def set_anomaly_callback(self, callback: Callable[[Anomaly], None]) -> None:
        """Set callback for anomaly detection."""
        self._on_anomaly = callback

    def set_trigger_callback(self, callback: Callable[[ComputationTrigger], None]) -> None:
        """Set callback for computation triggers."""
        self._on_trigger = callback

    def _get_or_create_window(self, tag: str) -> SignalWindow:
        """Get or create signal window for tag."""
        if tag not in self._windows:
            self._windows[tag] = SignalWindow(
                tag=tag,
                window_size_s=self.config.feature_window.window_size_s,
            )
        return self._windows[tag]

    async def validate_signal(
        self,
        raw_signal: Dict[str, Any],
    ) -> ValidatedSignal:
        """
        Validate raw signal from OPC-UA/SCADA.

        Args:
            raw_signal: Raw signal dictionary with tag, value, quality, timestamp

        Returns:
            ValidatedSignal with validation results
        """
        import time
        start_time = time.perf_counter()

        self._stats["signals_processed"] += 1

        tag = raw_signal.get("tag", "")
        value = raw_signal.get("value", 0.0)
        quality = raw_signal.get("quality", "good")
        timestamp_str = raw_signal.get("timestamp")
        unit = raw_signal.get("unit", "")

        # Parse timestamp
        if isinstance(timestamp_str, str):
            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        elif isinstance(timestamp_str, datetime):
            timestamp = timestamp_str
        else:
            timestamp = datetime.now(timezone.utc)

        # Convert quality to code
        quality_code = 0 if quality == "good" else (0x40000000 if quality == "uncertain" else 0x80000000)

        issues: List[str] = []
        is_valid = True

        if self.config.enable_validation:
            # Check value is numeric
            if not isinstance(value, (int, float)):
                issues.append("Non-numeric value")
                is_valid = False
                value = 0.0

            # Check for NaN/Inf
            if math.isnan(value) or math.isinf(value):
                issues.append("NaN or Inf value")
                is_valid = False
                value = 0.0

            # Check quality
            if quality != "good":
                issues.append(f"Quality: {quality}")
                if quality == "bad":
                    is_valid = False

            # Check timestamp freshness
            age = (datetime.now(timezone.utc) - timestamp).total_seconds()
            if age > 60:
                issues.append(f"Stale data: {age:.1f}s old")
            if age < -10:
                issues.append(f"Future timestamp: {-age:.1f}s ahead")
                is_valid = False

        # Add to window for feature extraction
        window = self._get_or_create_window(tag)
        await window.add(timestamp, value, quality_code)

        processing_latency = (time.perf_counter() - start_time) * 1000

        if is_valid:
            self._stats["signals_validated"] += 1
        else:
            self._stats["signals_invalid"] += 1

        return ValidatedSignal(
            tag=tag,
            value=value,
            unit=unit,
            quality_code=quality_code,
            timestamp=timestamp,
            is_valid=is_valid,
            validation_issues=issues,
            raw_value=raw_signal.get("raw_value"),
            calibration_applied=raw_signal.get("calibration_applied", False),
            processing_latency_ms=processing_latency,
        )

    async def compute_features(
        self,
        validated_signals: List[ValidatedSignal],
    ) -> Dict[str, FeatureSet]:
        """
        Compute statistical features from validated signals.

        Args:
            validated_signals: List of validated signals

        Returns:
            Dict of tag -> FeatureSet
        """
        if not self.config.enable_features:
            return {}

        features: Dict[str, FeatureSet] = {}

        # Group signals by tag
        signals_by_tag: Dict[str, List[ValidatedSignal]] = {}
        for signal in validated_signals:
            if signal.tag not in signals_by_tag:
                signals_by_tag[signal.tag] = []
            signals_by_tag[signal.tag].append(signal)

        # Compute features for each tag
        for tag, signals in signals_by_tag.items():
            window = self._get_or_create_window(tag)
            values = window.get_numeric_values()

            if len(values) < self.config.feature_window.min_samples:
                continue

            feature_set = self._compute_statistics(tag, values, window)
            features[tag] = feature_set
            self._stats["features_computed"] += 1

        return features

    def _compute_statistics(
        self,
        tag: str,
        values: List[float],
        window: SignalWindow,
    ) -> FeatureSet:
        """Compute statistical features from values."""
        window_data = window.get_values()

        # Time range
        timestamps = [ts for ts, v, q in window_data]
        window_start = min(timestamps) if timestamps else datetime.now(timezone.utc)
        window_end = max(timestamps) if timestamps else datetime.now(timezone.utc)

        # Central tendency
        mean = statistics.mean(values)
        median = statistics.median(values)
        try:
            mode = statistics.mode(values)
        except statistics.StatisticsError:
            mode = None

        # Dispersion
        if len(values) > 1:
            std_dev = statistics.stdev(values)
            variance = statistics.variance(values)
        else:
            std_dev = 0.0
            variance = 0.0

        min_val = min(values)
        max_val = max(values)
        value_range = max_val - min_val

        # Quartiles and IQR
        sorted_values = sorted(values)
        n = len(sorted_values)
        q1 = sorted_values[n // 4]
        q3 = sorted_values[3 * n // 4]
        iqr = q3 - q1

        # Shape (skewness and kurtosis)
        if std_dev > 0 and len(values) > 2:
            skewness = sum((v - mean) ** 3 for v in values) / (n * std_dev ** 3)
            kurtosis = sum((v - mean) ** 4 for v in values) / (n * std_dev ** 4) - 3
        else:
            skewness = 0.0
            kurtosis = 0.0

        # Rates of change
        rates = []
        prev_val = None
        prev_ts = None
        for ts, val, q in window_data:
            if prev_val is not None and prev_ts is not None:
                dt = (ts - prev_ts).total_seconds()
                if dt > 0:
                    rates.append(abs(val - prev_val) / dt)
            prev_val = val
            prev_ts = ts

        rate_of_change = statistics.mean(rates) if rates else 0.0
        max_rate_of_change = max(rates) if rates else 0.0

        # Energy metrics
        rms = math.sqrt(sum(v ** 2 for v in values) / n)
        energy = sum(v ** 2 for v in values)

        # Quality metrics
        good_count = sum(1 for ts, v, q in window_data if q == 0)
        good_quality_pct = (good_count / len(window_data) * 100) if window_data else 0.0

        return FeatureSet(
            tag=tag,
            window_start=window_start,
            window_end=window_end,
            sample_count=len(values),
            mean=mean,
            median=median,
            mode=mode,
            std_dev=std_dev,
            variance=variance,
            iqr=iqr,
            range=value_range,
            skewness=skewness,
            kurtosis=kurtosis,
            min_value=min_val,
            max_value=max_val,
            peak_to_peak=value_range,
            rate_of_change=rate_of_change,
            max_rate_of_change=max_rate_of_change,
            rms=rms,
            energy=energy,
            good_quality_pct=good_quality_pct,
        )

    async def detect_anomalies(
        self,
        feature_window: Dict[str, FeatureSet],
    ) -> List[Anomaly]:
        """
        Detect anomalies in feature window.

        Args:
            feature_window: Dict of tag -> FeatureSet

        Returns:
            List of detected anomalies
        """
        if not self.config.enable_anomaly_detection:
            return []

        anomalies: List[Anomaly] = []
        timestamp = datetime.now(timezone.utc)

        for tag, features in feature_window.items():
            # Get or initialize baseline
            if tag not in self._baselines:
                self._baselines[tag] = {
                    "mean": features.mean,
                    "std_dev": features.std_dev,
                    "samples": features.sample_count,
                }
                continue

            baseline = self._baselines[tag]

            # Detect spike/drop
            if baseline["std_dev"] > 0:
                deviation = (features.mean - baseline["mean"]) / baseline["std_dev"]

                if abs(deviation) > self.config.anomaly_sensitivity:
                    anomaly_type = AnomalyType.SPIKE if deviation > 0 else AnomalyType.DROP
                    severity = "high" if abs(deviation) > 5 else ("medium" if abs(deviation) > 3 else "low")

                    anomalies.append(Anomaly(
                        anomaly_id=f"{tag}_{timestamp.timestamp()}",
                        anomaly_type=anomaly_type,
                        tag=tag,
                        timestamp=timestamp,
                        severity=severity,
                        observed_value=features.mean,
                        expected_value=baseline["mean"],
                        deviation=deviation,
                        window_mean=baseline["mean"],
                        window_std=baseline["std_dev"],
                        confidence=min(1.0, abs(deviation) / 10),
                        description=f"{anomaly_type.value.title()}: {deviation:.2f} std devs from baseline",
                    ))

            # Detect oscillation
            if features.std_dev > baseline["std_dev"] * 2:
                anomalies.append(Anomaly(
                    anomaly_id=f"{tag}_osc_{timestamp.timestamp()}",
                    anomaly_type=AnomalyType.OSCILLATION,
                    tag=tag,
                    timestamp=timestamp,
                    severity="medium",
                    observed_value=features.std_dev,
                    expected_value=baseline["std_dev"],
                    deviation=features.std_dev / baseline["std_dev"] if baseline["std_dev"] > 0 else 0,
                    confidence=0.7,
                    description=f"Excessive variance: {features.std_dev:.4f} vs baseline {baseline['std_dev']:.4f}",
                ))

            # Detect flatline
            if features.range < self.config.flatline_tolerance * abs(features.mean or 1):
                if features.sample_count >= self.config.flatline_count:
                    anomalies.append(Anomaly(
                        anomaly_id=f"{tag}_flat_{timestamp.timestamp()}",
                        anomaly_type=AnomalyType.FLATLINE,
                        tag=tag,
                        timestamp=timestamp,
                        severity="high",
                        observed_value=features.range,
                        expected_value=baseline["std_dev"],
                        deviation=0,
                        confidence=0.9,
                        description=f"Flatline detected: range={features.range:.6f} over {features.sample_count} samples",
                    ))

            # Update baseline (exponential moving average)
            alpha = 0.1
            baseline["mean"] = alpha * features.mean + (1 - alpha) * baseline["mean"]
            baseline["std_dev"] = alpha * features.std_dev + (1 - alpha) * baseline["std_dev"]
            baseline["samples"] += features.sample_count

        # Callback for detected anomalies
        for anomaly in anomalies:
            self._stats["anomalies_detected"] += 1
            if self._on_anomaly:
                self._on_anomaly(anomaly)

        return anomalies

    async def trigger_computation(
        self,
        event_type: ComputationType,
        data: Dict[str, Any],
    ) -> Optional[ComputationTrigger]:
        """
        Trigger downstream computation.

        Args:
            event_type: Type of computation to trigger
            data: Input data for computation

        Returns:
            ComputationTrigger or None if debounced
        """
        if not self.config.enable_triggers:
            return None

        # Check debounce
        trigger_key = f"{event_type.value}"
        if trigger_key in self._last_triggers:
            elapsed = (datetime.now(timezone.utc) - self._last_triggers[trigger_key]).total_seconds()
            if elapsed < self.config.trigger_debounce_s:
                return None

        # Create trigger
        trigger = ComputationTrigger(
            trigger_id=f"{event_type.value}_{datetime.now(timezone.utc).timestamp()}",
            computation_type=event_type,
            timestamp=datetime.now(timezone.utc),
            trigger_reason=data.get("reason", "signal_update"),
            source_tags=data.get("source_tags", []),
            input_values=data.get("values", {}),
            priority=data.get("priority", "normal"),
        )

        self._last_triggers[trigger_key] = trigger.timestamp
        self._stats["triggers_fired"] += 1

        # Callback
        if self._on_trigger:
            self._on_trigger(trigger)

        logger.debug(f"Triggered computation: {event_type.value}")
        return trigger

    def get_statistics(self) -> Dict:
        """Get processor statistics."""
        return {
            **self._stats,
            "active_windows": len(self._windows),
            "tracked_baselines": len(self._baselines),
        }

    def get_window_summary(self) -> Dict[str, Dict]:
        """Get summary of all signal windows."""
        return {
            tag: {
                "count": window.count,
                "window_size_s": window.window_size_s,
            }
            for tag, window in self._windows.items()
        }

    def reset_baseline(self, tag: Optional[str] = None) -> None:
        """Reset anomaly detection baseline."""
        if tag:
            self._baselines.pop(tag, None)
        else:
            self._baselines.clear()


def create_steam_processor() -> StreamProcessor:
    """Create processor configured for steam system monitoring."""
    config = ProcessingConfig(
        enable_validation=True,
        enable_features=True,
        enable_anomaly_detection=True,
        anomaly_sensitivity=3.0,
        feature_window=WindowConfig(
            window_type=WindowType.SLIDING,
            window_size_s=60,
            slide_size_s=10,
            min_samples=10,
        ),
        trigger_debounce_s=5.0,
    )

    return StreamProcessor(config)
