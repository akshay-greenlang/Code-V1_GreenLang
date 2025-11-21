# -*- coding: utf-8 -*-
"""
Data Transformation Module for GL-002 BoilerEfficiencyOptimizer

Provides comprehensive data normalization, validation, quality scoring,
and transformation utilities for multi-source industrial data integration.

Features:
- Unit conversion (Imperial/Metric/SI)
- Data quality scoring (0-100)
- Outlier detection and handling
- Missing data imputation
- Time-series alignment
- Sensor fusion
- Data validation
"""

import logging
import statistics
import math
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import numpy as np
from scipy import interpolate, signal
from greenlang.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class UnitSystem(Enum):
    """Supported unit systems."""
    SI = "si"           # International System
    METRIC = "metric"   # Metric system
    IMPERIAL = "imperial"  # Imperial/US system
    CUSTOM = "custom"   # Custom units


class DataQualityIssue(Enum):
    """Types of data quality issues."""
    MISSING = "missing"
    OUT_OF_RANGE = "out_of_range"
    SPIKE = "spike"
    FLATLINE = "flatline"
    NOISE = "excessive_noise"
    DRIFT = "sensor_drift"
    INCONSISTENT = "inconsistent"
    STALE = "stale_data"


@dataclass
class UnitConversion:
    """Unit conversion configuration."""
    from_unit: str
    to_unit: str
    factor: float
    offset: float = 0.0
    formula: Optional[Callable] = None


@dataclass
class DataPoint:
    """Single data point with metadata."""
    timestamp: datetime
    value: Any
    unit: str
    quality: float = 100.0  # 0-100 quality score
    source: str = ""
    validated: bool = False
    original_value: Optional[Any] = None
    issues: List[DataQualityIssue] = field(default_factory=list)


@dataclass
class SensorConfig:
    """Sensor configuration for validation."""
    sensor_id: str
    parameter: str
    unit: str
    min_valid: float
    max_valid: float
    rate_of_change_limit: Optional[float] = None  # Max change per minute
    deadband: float = 0.0
    expected_noise_level: float = 0.0
    calibration_factor: float = 1.0
    calibration_offset: float = 0.0


class UnitConverter:
    """
    Comprehensive unit conversion system.

    Handles all common engineering units for boiler systems.
    """

    def __init__(self):
        """Initialize unit converter with conversion database."""
        self.conversions = self._build_conversion_database()

    def _build_conversion_database(self) -> Dict[Tuple[str, str], UnitConversion]:
        """Build comprehensive unit conversion database."""
        conversions = {}

        # Temperature conversions
        conversions[('C', 'F')] = UnitConversion('C', 'F', 1.8, 32)
        conversions[('F', 'C')] = UnitConversion('F', 'C', 5/9, -32*5/9)
        conversions[('C', 'K')] = UnitConversion('C', 'K', 1.0, 273.15)
        conversions[('K', 'C')] = UnitConversion('K', 'C', 1.0, -273.15)
        conversions[('F', 'K')] = UnitConversion('F', 'K', 5/9, 255.372)
        conversions[('K', 'F')] = UnitConversion('K', 'F', 1.8, -459.67)

        # Pressure conversions
        conversions[('bar', 'psi')] = UnitConversion('bar', 'psi', 14.5038)
        conversions[('psi', 'bar')] = UnitConversion('psi', 'bar', 0.0689476)
        conversions[('bar', 'kPa')] = UnitConversion('bar', 'kPa', 100)
        conversions[('kPa', 'bar')] = UnitConversion('kPa', 'bar', 0.01)
        conversions[('psi', 'kPa')] = UnitConversion('psi', 'kPa', 6.89476)
        conversions[('kPa', 'psi')] = UnitConversion('kPa', 'psi', 0.145038)
        conversions[('bar', 'MPa')] = UnitConversion('bar', 'MPa', 0.1)
        conversions[('MPa', 'bar')] = UnitConversion('MPa', 'bar', 10)
        conversions[('atm', 'bar')] = UnitConversion('atm', 'bar', 1.01325)
        conversions[('bar', 'atm')] = UnitConversion('bar', 'atm', 0.986923)
        conversions[('mmHg', 'kPa')] = UnitConversion('mmHg', 'kPa', 0.133322)
        conversions[('kPa', 'mmHg')] = UnitConversion('kPa', 'mmHg', 7.50062)

        # Flow rate conversions
        conversions[('kg/hr', 'lb/hr')] = UnitConversion('kg/hr', 'lb/hr', 2.20462)
        conversions[('lb/hr', 'kg/hr')] = UnitConversion('lb/hr', 'kg/hr', 0.453592)
        conversions[('t/hr', 'kg/hr')] = UnitConversion('t/hr', 'kg/hr', 1000)
        conversions[('kg/hr', 't/hr')] = UnitConversion('kg/hr', 't/hr', 0.001)
        conversions[('m3/hr', 'ft3/hr')] = UnitConversion('m3/hr', 'ft3/hr', 35.3147)
        conversions[('ft3/hr', 'm3/hr')] = UnitConversion('ft3/hr', 'm3/hr', 0.0283168)
        conversions[('m3/hr', 'gpm')] = UnitConversion('m3/hr', 'gpm', 4.40287)
        conversions[('gpm', 'm3/hr')] = UnitConversion('gpm', 'm3/hr', 0.227125)
        conversions[('L/min', 'm3/hr')] = UnitConversion('L/min', 'm3/hr', 0.06)
        conversions[('m3/hr', 'L/min')] = UnitConversion('m3/hr', 'L/min', 16.6667)

        # Energy conversions
        conversions[('MJ', 'MMBtu')] = UnitConversion('MJ', 'MMBtu', 0.000947817)
        conversions[('MMBtu', 'MJ')] = UnitConversion('MMBtu', 'MJ', 1055.06)
        conversions[('kW', 'hp')] = UnitConversion('kW', 'hp', 1.34102)
        conversions[('hp', 'kW')] = UnitConversion('hp', 'kW', 0.745700)
        conversions[('MW', 'kW')] = UnitConversion('MW', 'kW', 1000)
        conversions[('kW', 'MW')] = UnitConversion('kW', 'MW', 0.001)
        conversions[('kcal/hr', 'kW')] = UnitConversion('kcal/hr', 'kW', 0.001163)
        conversions[('kW', 'kcal/hr')] = UnitConversion('kW', 'kcal/hr', 860.421)

        # Length conversions
        conversions[('m', 'ft')] = UnitConversion('m', 'ft', 3.28084)
        conversions[('ft', 'm')] = UnitConversion('ft', 'm', 0.3048)
        conversions[('mm', 'in')] = UnitConversion('mm', 'in', 0.0393701)
        conversions[('in', 'mm')] = UnitConversion('in', 'mm', 25.4)

        # Mass conversions
        conversions[('kg', 'lb')] = UnitConversion('kg', 'lb', 2.20462)
        conversions[('lb', 'kg')] = UnitConversion('lb', 'kg', 0.453592)
        conversions[('ton', 'kg')] = UnitConversion('ton', 'kg', 1000)
        conversions[('kg', 'ton')] = UnitConversion('kg', 'ton', 0.001)

        # Concentration conversions
        conversions[('ppm', 'mg/m3')] = UnitConversion('ppm', 'mg/m3', 1.0)  # Depends on MW
        conversions[('mg/m3', 'ppm')] = UnitConversion('mg/m3', 'ppm', 1.0)
        conversions[('%', 'ppm')] = UnitConversion('%', 'ppm', 10000)
        conversions[('ppm', '%')] = UnitConversion('ppm', '%', 0.0001)

        return conversions

    def convert(self, value: float, from_unit: str, to_unit: str) -> float:
        """
        Convert value between units.

        Args:
            value: Value to convert
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted value
        """
        if from_unit == to_unit:
            return value

        # Direct conversion
        key = (from_unit, to_unit)
        if key in self.conversions:
            conv = self.conversions[key]
            if conv.formula:
                return conv.formula(value)
            return value * conv.factor + conv.offset

        # Try reverse conversion
        reverse_key = (to_unit, from_unit)
        if reverse_key in self.conversions:
            conv = self.conversions[reverse_key]
            if conv.offset != 0:
                # Handle offset in reverse
                return (value - conv.offset) / conv.factor
            return value / conv.factor

        # Multi-step conversion through common unit
        # Try to find a path through SI units
        common_units = ['C', 'K', 'bar', 'kPa', 'kg/hr', 'm3/hr', 'kW', 'MJ']

        for common in common_units:
            if (from_unit, common) in self.conversions and (common, to_unit) in self.conversions:
                intermediate = self.convert(value, from_unit, common)
                return self.convert(intermediate, common, to_unit)

        logger.warning(f"No conversion found from {from_unit} to {to_unit}")
        return value


class DataValidator:
    """
    Validate and score data quality.

    Implements comprehensive validation rules for industrial data.
    """

    def __init__(self):
        """Initialize data validator."""
        self.validation_history = deque(maxlen=1000)
        self.sensor_configs: Dict[str, SensorConfig] = {}

    def configure_sensor(self, config: SensorConfig):
        """Add sensor configuration for validation."""
        self.sensor_configs[config.sensor_id] = config

    def validate_point(
        self,
        point: DataPoint,
        sensor_id: Optional[str] = None,
        previous_points: Optional[List[DataPoint]] = None
    ) -> Tuple[bool, float, List[DataQualityIssue]]:
        """
        Validate a single data point.

        Args:
            point: Data point to validate
            sensor_id: Sensor identifier
            previous_points: Historical points for comparison

        Returns:
            Tuple of (is_valid, quality_score, issues)
        """
        issues = []
        quality_score = 100.0

        # Check for missing value
        if point.value is None:
            issues.append(DataQualityIssue.MISSING)
            return False, 0.0, issues

        # Check sensor configuration if available
        if sensor_id and sensor_id in self.sensor_configs:
            config = self.sensor_configs[sensor_id]

            # Range check
            if isinstance(point.value, (int, float)):
                if point.value < config.min_valid or point.value > config.max_valid:
                    issues.append(DataQualityIssue.OUT_OF_RANGE)
                    quality_score -= 50

                # Rate of change check
                if previous_points and config.rate_of_change_limit:
                    last_point = previous_points[-1]
                    time_diff = (point.timestamp - last_point.timestamp).total_seconds() / 60
                    if time_diff > 0:
                        rate = abs(point.value - last_point.value) / time_diff
                        if rate > config.rate_of_change_limit:
                            issues.append(DataQualityIssue.SPIKE)
                            quality_score -= 30

        # Statistical checks if we have history
        if previous_points and len(previous_points) >= 10:
            values = [p.value for p in previous_points[-10:] if isinstance(p.value, (int, float))]

            if values and isinstance(point.value, (int, float)):
                mean = statistics.mean(values)
                stdev = statistics.stdev(values) if len(values) > 1 else 0

                # Outlier detection (3-sigma rule)
                if stdev > 0:
                    z_score = abs(point.value - mean) / stdev
                    if z_score > 3:
                        issues.append(DataQualityIssue.SPIKE)
                        quality_score -= 20

                # Flatline detection
                if stdev < 0.001 * abs(mean) if mean != 0 else stdev == 0:
                    if all(abs(v - point.value) < 0.001 for v in values[-5:]):
                        issues.append(DataQualityIssue.FLATLINE)
                        quality_score -= 15

        # Staleness check
        if previous_points:
            last_update = previous_points[-1].timestamp
            if (point.timestamp - last_update).total_seconds() > 3600:  # 1 hour
                issues.append(DataQualityIssue.STALE)
                quality_score -= 10

        # Update point quality
        point.quality = max(0, quality_score)
        point.issues = issues
        point.validated = True

        is_valid = quality_score >= 50 and DataQualityIssue.MISSING not in issues

        # Record validation
        self.validation_history.append({
            'timestamp': point.timestamp,
            'sensor_id': sensor_id,
            'quality_score': quality_score,
            'issues': [i.value for i in issues],
            'valid': is_valid
        })

        return is_valid, quality_score, issues

    def validate_dataset(
        self,
        data: List[DataPoint],
        sensor_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate entire dataset and provide statistics.

        Args:
            data: List of data points
            sensor_id: Sensor identifier

        Returns:
            Validation statistics
        """
        if not data:
            return {'valid': False, 'error': 'No data provided'}

        valid_count = 0
        total_quality = 0
        all_issues = defaultdict(int)

        for i, point in enumerate(data):
            previous = data[:i] if i > 0 else None
            is_valid, quality, issues = self.validate_point(point, sensor_id, previous)

            if is_valid:
                valid_count += 1
            total_quality += quality

            for issue in issues:
                all_issues[issue.value] += 1

        return {
            'total_points': len(data),
            'valid_points': valid_count,
            'validity_rate': valid_count / len(data) * 100,
            'average_quality': total_quality / len(data),
            'issues': dict(all_issues),
            'sensor_id': sensor_id
        }


class OutlierDetector:
    """
    Advanced outlier detection for time-series data.

    Implements multiple detection algorithms.
    """

    def __init__(self):
        """Initialize outlier detector."""
        self.methods = {
            'zscore': self._zscore_detection,
            'iqr': self._iqr_detection,
            'isolation': self._isolation_detection,
            'mad': self._mad_detection
        }

    def detect_outliers(
        self,
        data: List[float],
        method: str = 'zscore',
        threshold: float = 3.0
    ) -> List[int]:
        """
        Detect outliers in data.

        Args:
            data: List of values
            method: Detection method
            threshold: Detection threshold

        Returns:
            List of outlier indices
        """
        if method not in self.methods:
            method = 'zscore'

        return self.methods[method](data, threshold)

    def _zscore_detection(self, data: List[float], threshold: float) -> List[int]:
        """Z-score based outlier detection."""
        if len(data) < 3:
            return []

        mean = statistics.mean(data)
        stdev = statistics.stdev(data)

        if stdev == 0:
            return []

        outliers = []
        for i, value in enumerate(data):
            z_score = abs(value - mean) / stdev
            if z_score > threshold:
                outliers.append(i)

        return outliers

    def _iqr_detection(self, data: List[float], threshold: float) -> List[int]:
        """Interquartile range based detection."""
        if len(data) < 4:
            return []

        sorted_data = sorted(data)
        n = len(sorted_data)

        q1 = sorted_data[n // 4]
        q3 = sorted_data[3 * n // 4]
        iqr = q3 - q1

        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        outliers = []
        for i, value in enumerate(data):
            if value < lower_bound or value > upper_bound:
                outliers.append(i)

        return outliers

    def _mad_detection(self, data: List[float], threshold: float) -> List[int]:
        """Median Absolute Deviation based detection."""
        if len(data) < 3:
            return []

        median = statistics.median(data)
        mad = statistics.median([abs(x - median) for x in data])

        if mad == 0:
            return []

        outliers = []
        for i, value in enumerate(data):
            deviation = abs(value - median) / mad
            if deviation > threshold:
                outliers.append(i)

        return outliers

    def _isolation_detection(self, data: List[float], threshold: float) -> List[int]:
        """Simplified isolation forest detection."""
        # Simplified version - in production would use sklearn
        return self._zscore_detection(data, threshold)

    def remove_outliers(
        self,
        data: List[DataPoint],
        method: str = 'zscore',
        threshold: float = 3.0,
        replace_with: str = 'interpolate'
    ) -> List[DataPoint]:
        """
        Remove or replace outliers in dataset.

        Args:
            data: List of data points
            method: Detection method
            threshold: Detection threshold
            replace_with: 'interpolate', 'mean', 'median', or 'remove'

        Returns:
            Cleaned data
        """
        values = [p.value for p in data if isinstance(p.value, (int, float))]

        if not values:
            return data

        outlier_indices = self.detect_outliers(values, method, threshold)

        if not outlier_indices:
            return data

        cleaned_data = data.copy()

        if replace_with == 'remove':
            # Remove outliers
            for idx in sorted(outlier_indices, reverse=True):
                del cleaned_data[idx]

        elif replace_with == 'mean':
            # Replace with mean
            clean_values = [v for i, v in enumerate(values) if i not in outlier_indices]
            replacement = statistics.mean(clean_values) if clean_values else 0

            for idx in outlier_indices:
                cleaned_data[idx].value = replacement
                cleaned_data[idx].original_value = values[idx]
                cleaned_data[idx].issues.append(DataQualityIssue.SPIKE)

        elif replace_with == 'median':
            # Replace with median
            clean_values = [v for i, v in enumerate(values) if i not in outlier_indices]
            replacement = statistics.median(clean_values) if clean_values else 0

            for idx in outlier_indices:
                cleaned_data[idx].value = replacement
                cleaned_data[idx].original_value = values[idx]
                cleaned_data[idx].issues.append(DataQualityIssue.SPIKE)

        elif replace_with == 'interpolate':
            # Linear interpolation
            for idx in outlier_indices:
                # Find nearest valid points
                before = idx - 1
                while before >= 0 and before in outlier_indices:
                    before -= 1

                after = idx + 1
                while after < len(values) and after in outlier_indices:
                    after += 1

                if before >= 0 and after < len(values):
                    # Interpolate between valid points
                    t_before = cleaned_data[before].timestamp
                    t_after = cleaned_data[after].timestamp
                    t_current = cleaned_data[idx].timestamp

                    if t_after != t_before:
                        ratio = (t_current - t_before).total_seconds() / (t_after - t_before).total_seconds()
                        interpolated = values[before] + ratio * (values[after] - values[before])

                        cleaned_data[idx].value = interpolated
                        cleaned_data[idx].original_value = values[idx]
                        cleaned_data[idx].issues.append(DataQualityIssue.SPIKE)

        return cleaned_data


class DataImputer:
    """
    Handle missing data through various imputation methods.
    """

    def __init__(self):
        """Initialize data imputer."""
        self.methods = {
            'forward_fill': self._forward_fill,
            'backward_fill': self._backward_fill,
            'linear': self._linear_interpolation,
            'polynomial': self._polynomial_interpolation,
            'mean': self._mean_imputation,
            'median': self._median_imputation,
            'seasonal': self._seasonal_imputation
        }

    def impute_missing(
        self,
        data: List[DataPoint],
        method: str = 'linear',
        max_gap: int = 10
    ) -> List[DataPoint]:
        """
        Impute missing values in dataset.

        Args:
            data: List of data points
            method: Imputation method
            max_gap: Maximum consecutive missing values to impute

        Returns:
            Imputed dataset
        """
        if method not in self.methods:
            method = 'linear'

        return self.methods[method](data, max_gap)

    def _forward_fill(self, data: List[DataPoint], max_gap: int) -> List[DataPoint]:
        """Forward fill missing values."""
        imputed = data.copy()
        last_valid = None
        gap_count = 0

        for i, point in enumerate(imputed):
            if point.value is None:
                gap_count += 1
                if last_valid is not None and gap_count <= max_gap:
                    imputed[i].value = last_valid
                    imputed[i].issues.append(DataQualityIssue.MISSING)
            else:
                last_valid = point.value
                gap_count = 0

        return imputed

    def _backward_fill(self, data: List[DataPoint], max_gap: int) -> List[DataPoint]:
        """Backward fill missing values."""
        imputed = data.copy()

        for i in range(len(imputed) - 2, -1, -1):
            if imputed[i].value is None and imputed[i + 1].value is not None:
                # Check gap size
                gap_count = 1
                j = i - 1
                while j >= 0 and imputed[j].value is None:
                    gap_count += 1
                    j -= 1

                if gap_count <= max_gap:
                    imputed[i].value = imputed[i + 1].value
                    imputed[i].issues.append(DataQualityIssue.MISSING)

        return imputed

    def _linear_interpolation(self, data: List[DataPoint], max_gap: int) -> List[DataPoint]:
        """Linear interpolation for missing values."""
        imputed = data.copy()

        # Find missing value indices
        missing_indices = [i for i, p in enumerate(imputed) if p.value is None]

        for idx in missing_indices:
            # Find surrounding valid points
            before = idx - 1
            while before >= 0 and imputed[before].value is None:
                before -= 1

            after = idx + 1
            while after < len(imputed) and imputed[after].value is None:
                after += 1

            # Check gap size
            gap_size = after - before - 1
            if gap_size > max_gap:
                continue

            if before >= 0 and after < len(imputed):
                # Interpolate
                v_before = imputed[before].value
                v_after = imputed[after].value

                if isinstance(v_before, (int, float)) and isinstance(v_after, (int, float)):
                    t_before = imputed[before].timestamp
                    t_after = imputed[after].timestamp
                    t_current = imputed[idx].timestamp

                    if t_after != t_before:
                        ratio = (t_current - t_before).total_seconds() / (t_after - t_before).total_seconds()
                        interpolated = v_before + ratio * (v_after - v_before)

                        imputed[idx].value = interpolated
                        imputed[idx].issues.append(DataQualityIssue.MISSING)

        return imputed

    def _polynomial_interpolation(self, data: List[DataPoint], max_gap: int) -> List[DataPoint]:
        """Polynomial interpolation for missing values."""
        # For simplicity, use linear for now
        return self._linear_interpolation(data, max_gap)

    def _mean_imputation(self, data: List[DataPoint], max_gap: int) -> List[DataPoint]:
        """Mean value imputation."""
        values = [p.value for p in data if p.value is not None and isinstance(p.value, (int, float))]

        if not values:
            return data

        mean_value = statistics.mean(values)
        imputed = data.copy()

        for i, point in enumerate(imputed):
            if point.value is None:
                imputed[i].value = mean_value
                imputed[i].issues.append(DataQualityIssue.MISSING)

        return imputed

    def _median_imputation(self, data: List[DataPoint], max_gap: int) -> List[DataPoint]:
        """Median value imputation."""
        values = [p.value for p in data if p.value is not None and isinstance(p.value, (int, float))]

        if not values:
            return data

        median_value = statistics.median(values)
        imputed = data.copy()

        for i, point in enumerate(imputed):
            if point.value is None:
                imputed[i].value = median_value
                imputed[i].issues.append(DataQualityIssue.MISSING)

        return imputed

    def _seasonal_imputation(self, data: List[DataPoint], max_gap: int) -> List[DataPoint]:
        """Seasonal pattern based imputation."""
        # Simplified version - in production would use proper seasonal decomposition
        return self._linear_interpolation(data, max_gap)


class TimeSeriesAligner:
    """
    Align time-series data from multiple sources.
    """

    def __init__(self):
        """Initialize time series aligner."""
        self.alignment_methods = {
            'nearest': self._nearest_alignment,
            'linear': self._linear_alignment,
            'previous': self._previous_alignment,
            'next': self._next_alignment
        }

    def align_series(
        self,
        series: Dict[str, List[DataPoint]],
        target_timestamps: List[datetime],
        method: str = 'linear'
    ) -> Dict[str, List[DataPoint]]:
        """
        Align multiple time series to common timestamps.

        Args:
            series: Dict of series_name -> data points
            target_timestamps: Target alignment timestamps
            method: Alignment method

        Returns:
            Aligned time series
        """
        if method not in self.alignment_methods:
            method = 'linear'

        aligned = {}

        for name, data in series.items():
            aligned[name] = self.alignment_methods[method](
                data,
                target_timestamps
            )

        return aligned

    def _nearest_alignment(
        self,
        data: List[DataPoint],
        targets: List[datetime]
    ) -> List[DataPoint]:
        """Align to nearest timestamp."""
        aligned = []

        for target_time in targets:
            # Find nearest point
            nearest = min(
                data,
                key=lambda p: abs((p.timestamp - target_time).total_seconds())
            )

            aligned_point = DataPoint(
                timestamp=target_time,
                value=nearest.value,
                unit=nearest.unit,
                quality=nearest.quality * 0.95,  # Slight quality reduction
                source=nearest.source
            )
            aligned.append(aligned_point)

        return aligned

    def _linear_alignment(
        self,
        data: List[DataPoint],
        targets: List[datetime]
    ) -> List[DataPoint]:
        """Linear interpolation alignment."""
        aligned = []

        # Sort data by timestamp
        sorted_data = sorted(data, key=lambda p: p.timestamp)

        for target_time in targets:
            # Find surrounding points
            before = None
            after = None

            for i, point in enumerate(sorted_data):
                if point.timestamp <= target_time:
                    before = point
                elif point.timestamp > target_time and after is None:
                    after = point
                    break

            if before and after and isinstance(before.value, (int, float)):
                # Interpolate
                t_ratio = (target_time - before.timestamp).total_seconds() / \
                          (after.timestamp - before.timestamp).total_seconds()
                interpolated = before.value + t_ratio * (after.value - before.value)

                aligned_point = DataPoint(
                    timestamp=target_time,
                    value=interpolated,
                    unit=before.unit,
                    quality=min(before.quality, after.quality) * 0.9,
                    source=before.source
                )
            elif before:
                aligned_point = DataPoint(
                    timestamp=target_time,
                    value=before.value,
                    unit=before.unit,
                    quality=before.quality * 0.85,
                    source=before.source
                )
            else:
                aligned_point = DataPoint(
                    timestamp=target_time,
                    value=None,
                    unit='',
                    quality=0,
                    source='missing'
                )

            aligned.append(aligned_point)

        return aligned

    def _previous_alignment(
        self,
        data: List[DataPoint],
        targets: List[datetime]
    ) -> List[DataPoint]:
        """Use previous value alignment."""
        aligned = []
        sorted_data = sorted(data, key=lambda p: p.timestamp)

        for target_time in targets:
            previous = None
            for point in sorted_data:
                if point.timestamp <= target_time:
                    previous = point
                else:
                    break

            if previous:
                aligned_point = DataPoint(
                    timestamp=target_time,
                    value=previous.value,
                    unit=previous.unit,
                    quality=previous.quality * 0.9,
                    source=previous.source
                )
            else:
                aligned_point = DataPoint(
                    timestamp=target_time,
                    value=None,
                    unit='',
                    quality=0,
                    source='missing'
                )

            aligned.append(aligned_point)

        return aligned

    def _next_alignment(
        self,
        data: List[DataPoint],
        targets: List[datetime]
    ) -> List[DataPoint]:
        """Use next value alignment."""
        aligned = []
        sorted_data = sorted(data, key=lambda p: p.timestamp)

        for target_time in targets:
            next_point = None
            for point in sorted_data:
                if point.timestamp >= target_time:
                    next_point = point
                    break

            if next_point:
                aligned_point = DataPoint(
                    timestamp=target_time,
                    value=next_point.value,
                    unit=next_point.unit,
                    quality=next_point.quality * 0.9,
                    source=next_point.source
                )
            else:
                aligned_point = DataPoint(
                    timestamp=target_time,
                    value=None,
                    unit='',
                    quality=0,
                    source='missing'
                )

            aligned.append(aligned_point)

        return aligned

    def resample(
        self,
        data: List[DataPoint],
        interval: timedelta,
        aggregation: str = 'mean'
    ) -> List[DataPoint]:
        """
        Resample time series to different interval.

        Args:
            data: Original data points
            interval: New sampling interval
            aggregation: Aggregation method (mean, median, max, min, sum)

        Returns:
            Resampled data
        """
        if not data:
            return []

        # Sort by timestamp
        sorted_data = sorted(data, key=lambda p: p.timestamp)

        # Determine time range
        start_time = sorted_data[0].timestamp
        end_time = sorted_data[-1].timestamp

        # Generate new timestamps
        resampled = []
        current_time = start_time

        while current_time <= end_time:
            # Find points in current interval
            interval_end = current_time + interval
            interval_points = [
                p for p in sorted_data
                if current_time <= p.timestamp < interval_end
                and isinstance(p.value, (int, float))
            ]

            if interval_points:
                values = [p.value for p in interval_points]

                if aggregation == 'mean':
                    aggregated = statistics.mean(values)
                elif aggregation == 'median':
                    aggregated = statistics.median(values)
                elif aggregation == 'max':
                    aggregated = max(values)
                elif aggregation == 'min':
                    aggregated = min(values)
                elif aggregation == 'sum':
                    aggregated = sum(values)
                else:
                    aggregated = statistics.mean(values)

                quality = statistics.mean([p.quality for p in interval_points])

                resampled.append(DataPoint(
                    timestamp=current_time + interval / 2,
                    value=aggregated,
                    unit=interval_points[0].unit,
                    quality=quality,
                    source='resampled'
                ))

            current_time = interval_end

        return resampled


class SensorFusion:
    """
    Fuse data from multiple sensors for improved accuracy.
    """

    def __init__(self):
        """Initialize sensor fusion."""
        self.fusion_methods = {
            'weighted_average': self._weighted_average,
            'kalman': self._simplified_kalman,
            'voting': self._majority_voting,
            'best_quality': self._best_quality
        }

    def fuse_sensors(
        self,
        sensor_data: Dict[str, DataPoint],
        method: str = 'weighted_average',
        weights: Optional[Dict[str, float]] = None
    ) -> DataPoint:
        """
        Fuse multiple sensor readings.

        Args:
            sensor_data: Dict of sensor_id -> data point
            method: Fusion method
            weights: Optional sensor weights

        Returns:
            Fused data point
        """
        if method not in self.fusion_methods:
            method = 'weighted_average'

        return self.fusion_methods[method](sensor_data, weights)

    def _weighted_average(
        self,
        sensor_data: Dict[str, DataPoint],
        weights: Optional[Dict[str, float]]
    ) -> DataPoint:
        """Weighted average fusion."""
        if not sensor_data:
            return None

        # Use quality scores as weights if not provided
        if weights is None:
            weights = {
                sid: point.quality / 100.0
                for sid, point in sensor_data.items()
            }

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight == 0:
            return None

        normalized = {k: v / total_weight for k, v in weights.items()}

        # Calculate weighted average
        fused_value = 0
        fused_quality = 0
        timestamp = None
        unit = None

        for sid, point in sensor_data.items():
            if isinstance(point.value, (int, float)):
                fused_value += point.value * normalized.get(sid, 0)
                fused_quality += point.quality * normalized.get(sid, 0)

                if timestamp is None:
                    timestamp = point.timestamp
                    unit = point.unit

        return DataPoint(
            timestamp=timestamp,
            value=fused_value,
            unit=unit,
            quality=fused_quality,
            source='fused'
        )

    def _simplified_kalman(
        self,
        sensor_data: Dict[str, DataPoint],
        weights: Optional[Dict[str, float]]
    ) -> DataPoint:
        """Simplified Kalman filter fusion."""
        # For simplicity, use weighted average
        # In production would implement proper Kalman filter
        return self._weighted_average(sensor_data, weights)

    def _majority_voting(
        self,
        sensor_data: Dict[str, DataPoint],
        weights: Optional[Dict[str, float]]
    ) -> DataPoint:
        """Majority voting for discrete values."""
        if not sensor_data:
            return None

        # Count votes
        votes = defaultdict(int)
        for point in sensor_data.values():
            votes[point.value] += 1

        # Find majority
        majority_value = max(votes, key=votes.get)
        majority_count = votes[majority_value]

        # Calculate confidence
        total_sensors = len(sensor_data)
        confidence = (majority_count / total_sensors) * 100

        # Use first timestamp and unit
        first_point = next(iter(sensor_data.values()))

        return DataPoint(
            timestamp=first_point.timestamp,
            value=majority_value,
            unit=first_point.unit,
            quality=confidence,
            source='voted'
        )

    def _best_quality(
        self,
        sensor_data: Dict[str, DataPoint],
        weights: Optional[Dict[str, float]]
    ) -> DataPoint:
        """Select sensor with best quality."""
        if not sensor_data:
            return None

        best_point = max(sensor_data.values(), key=lambda p: p.quality)

        return DataPoint(
            timestamp=best_point.timestamp,
            value=best_point.value,
            unit=best_point.unit,
            quality=best_point.quality,
            source=f'best_{best_point.source}'
        )


class DataTransformationPipeline:
    """
    Complete data transformation pipeline.
    """

    def __init__(self):
        """Initialize transformation pipeline."""
        self.converter = UnitConverter()
        self.validator = DataValidator()
        self.outlier_detector = OutlierDetector()
        self.imputer = DataImputer()
        self.aligner = TimeSeriesAligner()
        self.fusion = SensorFusion()

    def process_data(
        self,
        raw_data: List[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process raw data through complete pipeline.

        Args:
            raw_data: Raw input data
            config: Processing configuration

        Returns:
            Processed data with statistics
        """
        # Convert to DataPoint objects
        data_points = []
        for item in raw_data:
            point = DataPoint(
                timestamp=item.get('timestamp', DeterministicClock.utcnow()),
                value=item.get('value'),
                unit=item.get('unit', ''),
                source=item.get('source', 'unknown')
            )
            data_points.append(point)

        # Unit conversion
        if config.get('target_unit'):
            for point in data_points:
                if point.unit and point.unit != config['target_unit']:
                    point.value = self.converter.convert(
                        point.value,
                        point.unit,
                        config['target_unit']
                    )
                    point.unit = config['target_unit']

        # Validation
        validation_result = self.validator.validate_dataset(
            data_points,
            config.get('sensor_id')
        )

        # Outlier removal
        if config.get('remove_outliers', True):
            data_points = self.outlier_detector.remove_outliers(
                data_points,
                method=config.get('outlier_method', 'zscore'),
                threshold=config.get('outlier_threshold', 3.0),
                replace_with=config.get('outlier_replace', 'interpolate')
            )

        # Missing data imputation
        if config.get('impute_missing', True):
            data_points = self.imputer.impute_missing(
                data_points,
                method=config.get('impute_method', 'linear'),
                max_gap=config.get('max_impute_gap', 10)
            )

        # Resampling if requested
        if config.get('resample_interval'):
            interval = timedelta(seconds=config['resample_interval'])
            data_points = self.aligner.resample(
                data_points,
                interval,
                config.get('resample_aggregation', 'mean')
            )

        # Calculate statistics
        values = [p.value for p in data_points if isinstance(p.value, (int, float))]

        statistics_result = {
            'count': len(data_points),
            'valid_count': len([p for p in data_points if p.validated and p.quality >= 50]),
            'min': min(values) if values else None,
            'max': max(values) if values else None,
            'mean': statistics.mean(values) if values else None,
            'median': statistics.median(values) if values else None,
            'stdev': statistics.stdev(values) if len(values) > 1 else None,
            'quality_score': statistics.mean([p.quality for p in data_points])
        }

        return {
            'processed_data': [
                {
                    'timestamp': p.timestamp.isoformat(),
                    'value': p.value,
                    'unit': p.unit,
                    'quality': p.quality,
                    'source': p.source,
                    'issues': [i.value for i in p.issues]
                }
                for p in data_points
            ],
            'validation': validation_result,
            'statistics': statistics_result,
            'processing_config': config
        }


# Example usage
def main():
    """Example usage of data transformers."""

    # Initialize pipeline
    pipeline = DataTransformationPipeline()

    # Sample raw data
    raw_data = [
        {'timestamp': DeterministicClock.utcnow() - timedelta(minutes=10), 'value': 100, 'unit': 'psi', 'source': 'sensor1'},
        {'timestamp': DeterministicClock.utcnow() - timedelta(minutes=9), 'value': 102, 'unit': 'psi', 'source': 'sensor1'},
        {'timestamp': DeterministicClock.utcnow() - timedelta(minutes=8), 'value': 500, 'unit': 'psi', 'source': 'sensor1'},  # Outlier
        {'timestamp': DeterministicClock.utcnow() - timedelta(minutes=7), 'value': None, 'unit': 'psi', 'source': 'sensor1'},  # Missing
        {'timestamp': DeterministicClock.utcnow() - timedelta(minutes=6), 'value': 105, 'unit': 'psi', 'source': 'sensor1'},
        {'timestamp': DeterministicClock.utcnow() - timedelta(minutes=5), 'value': 103, 'unit': 'psi', 'source': 'sensor1'},
        {'timestamp': DeterministicClock.utcnow() - timedelta(minutes=4), 'value': 104, 'unit': 'psi', 'source': 'sensor1'},
        {'timestamp': DeterministicClock.utcnow() - timedelta(minutes=3), 'value': 106, 'unit': 'psi', 'source': 'sensor1'},
        {'timestamp': DeterministicClock.utcnow() - timedelta(minutes=2), 'value': 104, 'unit': 'psi', 'source': 'sensor1'},
        {'timestamp': DeterministicClock.utcnow() - timedelta(minutes=1), 'value': 105, 'unit': 'psi', 'source': 'sensor1'},
    ]

    # Processing configuration
    config = {
        'target_unit': 'bar',  # Convert to bar
        'sensor_id': 'pressure_sensor_1',
        'remove_outliers': True,
        'outlier_method': 'zscore',
        'outlier_threshold': 2.5,
        'outlier_replace': 'interpolate',
        'impute_missing': True,
        'impute_method': 'linear',
        'max_impute_gap': 5,
        'resample_interval': 120,  # 2 minutes
        'resample_aggregation': 'mean'
    }

    # Process data
    result = pipeline.process_data(raw_data, config)

    print(f"Processing Results:")
    print(f"  Original count: {len(raw_data)}")
    print(f"  Processed count: {result['statistics']['count']}")
    print(f"  Valid count: {result['statistics']['valid_count']}")
    print(f"  Quality score: {result['statistics']['quality_score']:.1f}")
    print(f"  Mean: {result['statistics']['mean']:.2f} bar")
    print(f"  Stdev: {result['statistics']['stdev']:.2f} bar")
    print(f"  Validation: {result['validation']['validity_rate']:.1f}%")

    # Show processed data
    print("\nProcessed Data:")
    for item in result['processed_data']:
        print(f"  {item['timestamp']}: {item['value']:.2f} {item['unit']} (quality: {item['quality']:.0f})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()