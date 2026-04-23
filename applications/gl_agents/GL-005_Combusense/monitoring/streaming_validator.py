# -*- coding: utf-8 -*-
"""
Streaming Data Validation for GL-005 CombustionSense
====================================================

Provides real-time validation of streaming sensor data including:
    - Data quality checks
    - Range validation
    - Rate-of-change limits
    - Temporal consistency
    - Gap detection

Design Principles:
    - Low latency processing
    - Memory-efficient windowing
    - Complete data provenance
    - Deterministic validation

Author: GL-DataEngineer
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
import statistics
import hashlib
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class ValidationResult(Enum):
    """Result of data validation."""
    VALID = "valid"
    INVALID = "invalid"
    SUSPECT = "suspect"
    MISSING = "missing"


class QualityFlag(Enum):
    """Data quality flags."""
    GOOD = "good"
    UNCERTAIN = "uncertain"
    BAD = "bad"
    STALE = "stale"
    INTERPOLATED = "interpolated"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DataPoint:
    """Single data point from stream."""
    parameter: str
    value: float
    timestamp: datetime
    source_id: str
    raw_value: Optional[float] = None
    quality: QualityFlag = QualityFlag.GOOD


@dataclass
class ValidationSpec:
    """Specification for validating a parameter."""
    parameter: str
    range_min: float
    range_max: float
    max_rate_of_change: float   # Per second
    max_gap_seconds: float = 10.0
    spike_threshold: float = 3.0  # Standard deviations
    frozen_threshold_seconds: float = 30.0


@dataclass
class ValidationReport:
    """Report from data validation."""
    data_point: DataPoint
    result: ValidationResult
    quality_assigned: QualityFlag
    issues: List[str]
    processing_time_ms: float
    provenance_hash: str


# =============================================================================
# STREAMING VALIDATOR
# =============================================================================

class StreamingDataValidator:
    """
    Real-time streaming data validator.

    Features:
        - Sliding window statistics
        - Rate-of-change monitoring
        - Gap detection
        - Spike detection
        - Frozen value detection
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.specs: Dict[str, ValidationSpec] = {}
        self.windows: Dict[str, deque] = {}
        self.last_values: Dict[str, DataPoint] = {}
        self.validation_callbacks: List[Callable] = []
        self.stats_cache: Dict[str, Dict[str, float]] = {}

    def register_parameter(self, spec: ValidationSpec) -> None:
        """
        Register a parameter for validation.

        Args:
            spec: Validation specification
        """
        self.specs[spec.parameter] = spec
        self.windows[spec.parameter] = deque(maxlen=self.window_size)

    def validate(self, data_point: DataPoint) -> ValidationReport:
        """
        Validate a single data point.

        Args:
            data_point: Data point to validate

        Returns:
            ValidationReport with validation results
        """
        start_time = datetime.now()
        issues = []

        param = data_point.parameter

        if param not in self.specs:
            return ValidationReport(
                data_point=data_point,
                result=ValidationResult.INVALID,
                quality_assigned=QualityFlag.BAD,
                issues=["Unknown parameter"],
                processing_time_ms=0.0,
                provenance_hash=self._calculate_hash(data_point),
            )

        spec = self.specs[param]

        # Range check
        if not self._check_range(data_point.value, spec):
            issues.append(f"Out of range: {data_point.value} not in [{spec.range_min}, {spec.range_max}]")

        # Rate of change check
        if param in self.last_values:
            roc_ok, roc_issue = self._check_rate_of_change(data_point, spec)
            if not roc_ok:
                issues.append(roc_issue)

        # Gap check
        if param in self.last_values:
            gap_ok, gap_issue = self._check_gap(data_point, spec)
            if not gap_ok:
                issues.append(gap_issue)

        # Spike check
        if len(self.windows[param]) >= 10:
            spike_ok, spike_issue = self._check_spike(data_point, spec)
            if not spike_ok:
                issues.append(spike_issue)

        # Frozen check
        frozen_ok, frozen_issue = self._check_frozen(data_point, spec)
        if not frozen_ok:
            issues.append(frozen_issue)

        # Update window and last value
        self.windows[param].append(data_point)
        self.last_values[param] = data_point

        # Determine result and quality
        result, quality = self._determine_result(issues)

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        report = ValidationReport(
            data_point=data_point,
            result=result,
            quality_assigned=quality,
            issues=issues,
            processing_time_ms=processing_time,
            provenance_hash=self._calculate_hash(data_point),
        )

        # Execute callbacks
        for callback in self.validation_callbacks:
            try:
                callback(report)
            except Exception as e:
                logger.error(f"Validation callback error: {e}")

        return report

    def validate_batch(self, data_points: List[DataPoint]) -> List[ValidationReport]:
        """
        Validate a batch of data points.

        Args:
            data_points: List of data points to validate

        Returns:
            List of validation reports
        """
        return [self.validate(dp) for dp in data_points]

    def _check_range(self, value: float, spec: ValidationSpec) -> bool:
        """Check if value is within range."""
        return spec.range_min <= value <= spec.range_max

    def _check_rate_of_change(
        self,
        data_point: DataPoint,
        spec: ValidationSpec
    ) -> Tuple[bool, Optional[str]]:
        """Check rate of change limit."""
        last = self.last_values.get(data_point.parameter)
        if not last:
            return True, None

        dt = (data_point.timestamp - last.timestamp).total_seconds()
        if dt <= 0:
            return False, "Non-positive time delta"

        rate = abs(data_point.value - last.value) / dt

        if rate > spec.max_rate_of_change:
            return False, f"Rate of change {rate:.2f}/s exceeds limit {spec.max_rate_of_change}/s"

        return True, None

    def _check_gap(
        self,
        data_point: DataPoint,
        spec: ValidationSpec
    ) -> Tuple[bool, Optional[str]]:
        """Check for data gap."""
        last = self.last_values.get(data_point.parameter)
        if not last:
            return True, None

        gap = (data_point.timestamp - last.timestamp).total_seconds()

        if gap > spec.max_gap_seconds:
            return False, f"Data gap of {gap:.1f}s exceeds limit {spec.max_gap_seconds}s"

        return True, None

    def _check_spike(
        self,
        data_point: DataPoint,
        spec: ValidationSpec
    ) -> Tuple[bool, Optional[str]]:
        """Check for spike (outlier)."""
        window = self.windows[data_point.parameter]
        if len(window) < 10:
            return True, None

        values = [dp.value for dp in window]
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0

        if std == 0:
            return True, None

        z_score = abs(data_point.value - mean) / std

        if z_score > spec.spike_threshold:
            return False, f"Spike detected: z-score {z_score:.2f} > {spec.spike_threshold}"

        return True, None

    def _check_frozen(
        self,
        data_point: DataPoint,
        spec: ValidationSpec
    ) -> Tuple[bool, Optional[str]]:
        """Check for frozen value."""
        window = self.windows[data_point.parameter]
        if len(window) < 10:
            return True, None

        recent = list(window)[-10:]
        values = [dp.value for dp in recent]

        # Check if all values are identical
        if len(set(values)) == 1 and all(v == data_point.value for v in values):
            time_span = (data_point.timestamp - recent[0].timestamp).total_seconds()
            if time_span > spec.frozen_threshold_seconds:
                return False, f"Frozen value for {time_span:.1f}s"

        return True, None

    def _determine_result(
        self,
        issues: List[str]
    ) -> Tuple[ValidationResult, QualityFlag]:
        """Determine validation result and quality from issues."""
        if not issues:
            return ValidationResult.VALID, QualityFlag.GOOD

        # Count severity
        critical_keywords = ["out of range", "rate of change", "frozen"]
        warning_keywords = ["gap", "spike"]

        critical_count = sum(
            1 for issue in issues
            if any(kw in issue.lower() for kw in critical_keywords)
        )

        if critical_count > 0:
            return ValidationResult.INVALID, QualityFlag.BAD
        else:
            return ValidationResult.SUSPECT, QualityFlag.UNCERTAIN

    def get_window_statistics(self, parameter: str) -> Optional[Dict[str, float]]:
        """
        Get statistics for current window.

        Args:
            parameter: Parameter name

        Returns:
            Dictionary of statistics or None
        """
        if parameter not in self.windows:
            return None

        window = self.windows[parameter]
        if len(window) < 2:
            return None

        values = [dp.value for dp in window]

        return {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values),
            "min": min(values),
            "max": max(values),
            "count": len(values),
        }

    def register_callback(self, callback: Callable[[ValidationReport], None]) -> None:
        """Register callback for validation events."""
        self.validation_callbacks.append(callback)

    def _calculate_hash(self, data_point: DataPoint) -> str:
        """Calculate provenance hash."""
        data = f"{data_point.parameter}:{data_point.value}:{data_point.timestamp.isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_combustion_validation_specs() -> List[ValidationSpec]:
    """Create validation specs for combustion parameters."""
    return [
        ValidationSpec(
            parameter="O2",
            range_min=0.0,
            range_max=25.0,
            max_rate_of_change=2.0,  # % per second
            max_gap_seconds=5.0,
            spike_threshold=4.0,
        ),
        ValidationSpec(
            parameter="CO",
            range_min=0.0,
            range_max=5000.0,
            max_rate_of_change=100.0,  # ppm per second
            max_gap_seconds=5.0,
            spike_threshold=4.0,
        ),
        ValidationSpec(
            parameter="flame_signal",
            range_min=0.0,
            range_max=20.0,
            max_rate_of_change=5.0,  # mA per second
            max_gap_seconds=1.0,  # Faster for flame
            spike_threshold=3.0,
        ),
        ValidationSpec(
            parameter="furnace_pressure",
            range_min=-5.0,
            range_max=5.0,
            max_rate_of_change=1.0,  # inwc per second
            max_gap_seconds=2.0,
            spike_threshold=3.0,
        ),
        ValidationSpec(
            parameter="stack_temperature",
            range_min=50.0,
            range_max=500.0,
            max_rate_of_change=5.0,  # C per second
            max_gap_seconds=10.0,
            spike_threshold=3.0,
        ),
    ]


if __name__ == "__main__":
    # Example usage
    validator = StreamingDataValidator()

    for spec in create_combustion_validation_specs():
        validator.register_parameter(spec)

    # Simulate data stream
    base_time = datetime.now()

    for i in range(20):
        dp = DataPoint(
            parameter="O2",
            value=3.5 + (i * 0.1),
            timestamp=base_time + timedelta(seconds=i),
            source_id="O2-PRIMARY",
        )

        report = validator.validate(dp)
        print(f"O2={dp.value:.1f}%: {report.result.value}, Quality: {report.quality_assigned.value}")
