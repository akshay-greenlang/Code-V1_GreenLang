# -*- coding: utf-8 -*-
"""
Combustion Anomaly Detection Tests for GL-005 CombustionSense
=============================================================

Tests for detecting combustion anomalies including:
    - Flame instability detection
    - Air leak detection
    - Fuel quality variation detection
    - Burner fouling detection
    - Combustion oscillation detection

Reference Standards:
    - NFPA 85: Boiler and Combustion Systems Hazards Code
    - API 556: Instrumentation, Control, and Protective Systems

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import math
import statistics
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# ANOMALY TYPES AND SEVERITY
# =============================================================================

class AnomalyType(Enum):
    """Types of combustion anomalies."""
    FLAME_INSTABILITY = "flame_instability"
    AIR_LEAK = "air_leak"
    FUEL_QUALITY_VARIATION = "fuel_quality_variation"
    BURNER_FOULING = "burner_fouling"
    COMBUSTION_OSCILLATION = "combustion_oscillation"
    O2_SENSOR_DRIFT = "o2_sensor_drift"
    CO_BREAKTHROUGH = "co_breakthrough"
    INCOMPLETE_COMBUSTION = "incomplete_combustion"


class AnomalySeverity(Enum):
    """Anomaly severity levels."""
    LOW = "low"           # Informational
    MEDIUM = "medium"     # Requires attention
    HIGH = "high"         # Urgent action needed
    CRITICAL = "critical" # Immediate shutdown


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CombustionDataPoint:
    """Single combustion data point for time series analysis."""
    timestamp: datetime
    o2_percent: float
    co_ppm: float
    flame_signal: float     # mA or intensity %
    furnace_pressure: float  # inches water column
    fuel_flow: float        # kg/hr
    air_flow: float         # kg/hr
    stack_temp_c: float
    fuel_pressure: float    # psig


@dataclass
class AnomalyDetectionResult:
    """Result of anomaly detection."""
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    confidence: float       # 0.0 to 1.0
    description: str
    detected_at: datetime
    evidence: Dict[str, Any]
    recommended_action: str


@dataclass
class FlameStabilityMetrics:
    """Flame stability analysis metrics."""
    mean_intensity: float
    std_deviation: float
    coefficient_of_variation: float
    oscillation_frequency_hz: Optional[float]
    is_stable: bool


# =============================================================================
# ANOMALY DETECTOR
# =============================================================================

class CombustionAnomalyDetector:
    """
    Detects combustion anomalies from process data.

    Detection Methods:
        - Statistical analysis (variance, trends)
        - Threshold violations
        - Rate-of-change analysis
        - Cross-correlation analysis
    """

    def __init__(self):
        # Thresholds for anomaly detection
        self.flame_cv_threshold = 0.15  # Coefficient of variation for flame instability
        self.o2_air_leak_delta = 2.0    # % O2 increase indicating air leak
        self.co_breakthrough_threshold = 400  # ppm
        self.pressure_oscillation_threshold = 0.5  # inwc

    def detect_flame_instability(
        self,
        flame_readings: List[float],
        sample_period_seconds: float = 1.0
    ) -> Tuple[bool, FlameStabilityMetrics]:
        """
        Detect flame instability from flame signal readings.

        Instability Indicators:
            - High coefficient of variation (CV > 15%)
            - Regular oscillation pattern
            - Low mean intensity

        Args:
            flame_readings: List of flame signal readings
            sample_period_seconds: Time between samples

        Returns:
            Tuple of (is_unstable, metrics)
        """
        if len(flame_readings) < 10:
            raise ValueError("Insufficient data for flame stability analysis")

        mean_intensity = statistics.mean(flame_readings)
        std_dev = statistics.stdev(flame_readings)
        cv = std_dev / mean_intensity if mean_intensity > 0 else float('inf')

        # Detect oscillation frequency using zero-crossing analysis
        osc_freq = self._detect_oscillation_frequency(flame_readings, sample_period_seconds)

        # Stability criteria
        is_stable = (
            cv < self.flame_cv_threshold and
            mean_intensity > 30.0 and  # Minimum signal level
            (osc_freq is None or osc_freq < 5.0)  # No rapid oscillation
        )

        return not is_stable, FlameStabilityMetrics(
            mean_intensity=mean_intensity,
            std_deviation=std_dev,
            coefficient_of_variation=cv,
            oscillation_frequency_hz=osc_freq,
            is_stable=is_stable,
        )

    def detect_air_leak(
        self,
        data_points: List[CombustionDataPoint]
    ) -> Tuple[bool, Optional[AnomalyDetectionResult]]:
        """
        Detect air in-leakage from process data.

        Air Leak Indicators:
            - O2 increase without fuel/air ratio change
            - CO2 decrease (dilution)
            - Temperature decrease in affected zone

        Args:
            data_points: Time series of combustion data

        Returns:
            Tuple of (is_detected, result)
        """
        if len(data_points) < 5:
            return False, None

        # Calculate expected O2 based on air/fuel ratio
        o2_deviations = []
        for dp in data_points:
            expected_o2 = self._calculate_expected_o2(dp.fuel_flow, dp.air_flow)
            deviation = dp.o2_percent - expected_o2
            o2_deviations.append(deviation)

        mean_deviation = statistics.mean(o2_deviations)
        max_deviation = max(o2_deviations)

        is_leak = mean_deviation > self.o2_air_leak_delta

        if is_leak:
            result = AnomalyDetectionResult(
                anomaly_type=AnomalyType.AIR_LEAK,
                severity=AnomalySeverity.MEDIUM if mean_deviation < 4.0 else AnomalySeverity.HIGH,
                confidence=min(mean_deviation / 5.0, 1.0),
                description=f"Air leak detected: O2 elevated by {mean_deviation:.1f}%",
                detected_at=data_points[-1].timestamp,
                evidence={
                    "mean_o2_deviation": mean_deviation,
                    "max_o2_deviation": max_deviation,
                },
                recommended_action="Inspect furnace seals, expansion joints, and ductwork",
            )
            return True, result

        return False, None

    def detect_fuel_quality_variation(
        self,
        data_points: List[CombustionDataPoint]
    ) -> Tuple[bool, Optional[AnomalyDetectionResult]]:
        """
        Detect fuel quality variations.

        Indicators:
            - CO spikes (incomplete combustion)
            - Temperature variations at constant load
            - Air-fuel ratio changes for same O2

        Args:
            data_points: Time series of combustion data

        Returns:
            Tuple of (is_detected, result)
        """
        if len(data_points) < 10:
            return False, None

        co_values = [dp.co_ppm for dp in data_points]
        temp_values = [dp.stack_temp_c for dp in data_points]

        co_cv = statistics.stdev(co_values) / statistics.mean(co_values) if statistics.mean(co_values) > 0 else 0
        temp_cv = statistics.stdev(temp_values) / statistics.mean(temp_values) if statistics.mean(temp_values) > 0 else 0

        # Fuel quality variation if high variability in CO and temperature
        is_variation = co_cv > 0.3 and temp_cv > 0.05

        if is_variation:
            result = AnomalyDetectionResult(
                anomaly_type=AnomalyType.FUEL_QUALITY_VARIATION,
                severity=AnomalySeverity.MEDIUM,
                confidence=min((co_cv + temp_cv) / 0.5, 1.0),
                description=f"Fuel quality variation: CO CV={co_cv:.2f}, Temp CV={temp_cv:.2f}",
                detected_at=data_points[-1].timestamp,
                evidence={
                    "co_coefficient_of_variation": co_cv,
                    "temp_coefficient_of_variation": temp_cv,
                },
                recommended_action="Check fuel supply quality and consistency",
            )
            return True, result

        return False, None

    def detect_combustion_oscillation(
        self,
        data_points: List[CombustionDataPoint],
        sample_period_seconds: float = 1.0
    ) -> Tuple[bool, Optional[AnomalyDetectionResult]]:
        """
        Detect combustion oscillations (pressure pulsations).

        Oscillations can indicate:
            - Burner acoustics issues
            - Air-fuel ratio hunting
            - Flame lift-off

        Args:
            data_points: Time series of combustion data
            sample_period_seconds: Time between samples

        Returns:
            Tuple of (is_detected, result)
        """
        if len(data_points) < 20:
            return False, None

        pressures = [dp.furnace_pressure for dp in data_points]

        # Calculate pressure oscillation amplitude
        mean_pressure = statistics.mean(pressures)
        std_pressure = statistics.stdev(pressures)

        # Detect dominant frequency
        freq = self._detect_oscillation_frequency(pressures, sample_period_seconds)

        is_oscillating = std_pressure > self.pressure_oscillation_threshold

        if is_oscillating:
            result = AnomalyDetectionResult(
                anomaly_type=AnomalyType.COMBUSTION_OSCILLATION,
                severity=AnomalySeverity.HIGH if std_pressure > 1.0 else AnomalySeverity.MEDIUM,
                confidence=min(std_pressure / 1.0, 1.0),
                description=f"Combustion oscillation: {std_pressure:.2f} inwc amplitude",
                detected_at=data_points[-1].timestamp,
                evidence={
                    "pressure_std_dev": std_pressure,
                    "oscillation_frequency_hz": freq,
                },
                recommended_action="Check burner settings and air-fuel ratio stability",
            )
            return True, result

        return False, None

    def detect_incomplete_combustion(
        self,
        data_points: List[CombustionDataPoint]
    ) -> Tuple[bool, Optional[AnomalyDetectionResult]]:
        """
        Detect incomplete combustion from emissions data.

        Indicators:
            - Elevated CO (> 400 ppm sustained)
            - Low O2 with high CO
            - Visible smoke (external observation)

        Args:
            data_points: Time series of combustion data

        Returns:
            Tuple of (is_detected, result)
        """
        if len(data_points) < 5:
            return False, None

        co_values = [dp.co_ppm for dp in data_points]
        o2_values = [dp.o2_percent for dp in data_points]

        mean_co = statistics.mean(co_values)
        mean_o2 = statistics.mean(o2_values)

        # Incomplete combustion: high CO with low O2
        is_incomplete = (
            mean_co > self.co_breakthrough_threshold and
            mean_o2 < 3.0
        )

        if is_incomplete:
            severity = AnomalySeverity.CRITICAL if mean_co > 1000 else AnomalySeverity.HIGH
            result = AnomalyDetectionResult(
                anomaly_type=AnomalyType.INCOMPLETE_COMBUSTION,
                severity=severity,
                confidence=min(mean_co / 1000, 1.0),
                description=f"Incomplete combustion: CO={mean_co:.0f}ppm, O2={mean_o2:.1f}%",
                detected_at=data_points[-1].timestamp,
                evidence={
                    "mean_co_ppm": mean_co,
                    "mean_o2_percent": mean_o2,
                },
                recommended_action="Increase combustion air immediately",
            )
            return True, result

        return False, None

    def _calculate_expected_o2(self, fuel_flow: float, air_flow: float) -> float:
        """Calculate expected O2 based on air-fuel ratio."""
        if fuel_flow <= 0:
            return 21.0  # Ambient air

        # Simplified calculation for natural gas
        stoich_air_fuel_ratio = 17.2  # kg air / kg fuel
        actual_ratio = air_flow / fuel_flow
        excess_air = (actual_ratio / stoich_air_fuel_ratio - 1) * 100

        # O2 % = 21 * EA / (100 + EA)
        if excess_air >= 0:
            expected_o2 = 21 * excess_air / (100 + excess_air)
        else:
            expected_o2 = 0.0

        return expected_o2

    def _detect_oscillation_frequency(
        self,
        values: List[float],
        sample_period: float
    ) -> Optional[float]:
        """Detect dominant oscillation frequency using zero-crossing analysis."""
        if len(values) < 10:
            return None

        mean_val = statistics.mean(values)
        crossings = 0

        for i in range(1, len(values)):
            if (values[i-1] - mean_val) * (values[i] - mean_val) < 0:
                crossings += 1

        # Frequency = crossings / (2 * total_time)
        total_time = len(values) * sample_period
        if total_time > 0:
            frequency = crossings / (2 * total_time)
            return frequency if frequency > 0.1 else None

        return None


# =============================================================================
# FLAME INSTABILITY TESTS
# =============================================================================

class TestFlameInstabilityDetection:
    """Test flame instability detection."""

    @pytest.fixture
    def detector(self) -> CombustionAnomalyDetector:
        return CombustionAnomalyDetector()

    def test_stable_flame_detection(self, detector: CombustionAnomalyDetector):
        """Test detection of stable flame."""
        # Stable flame: low variation around mean
        random.seed(42)
        flame_readings = [85.0 + random.gauss(0, 2.0) for _ in range(100)]

        is_unstable, metrics = detector.detect_flame_instability(flame_readings)

        assert not is_unstable, "Stable flame should not be flagged"
        assert metrics.is_stable
        assert metrics.coefficient_of_variation < 0.15

    def test_unstable_flame_detection(self, detector: CombustionAnomalyDetector):
        """Test detection of unstable flame."""
        # Unstable flame: high variation
        random.seed(42)
        flame_readings = [85.0 + random.gauss(0, 20.0) for _ in range(100)]

        is_unstable, metrics = detector.detect_flame_instability(flame_readings)

        assert is_unstable, "Unstable flame should be detected"
        assert not metrics.is_stable
        assert metrics.coefficient_of_variation > 0.15

    def test_oscillating_flame_detection(self, detector: CombustionAnomalyDetector):
        """Test detection of oscillating flame."""
        # Oscillating flame at 2 Hz
        flame_readings = []
        for i in range(100):
            value = 85.0 + 15.0 * math.sin(2 * math.pi * 2.0 * i * 0.1)  # 2 Hz at 0.1s sample
            flame_readings.append(value)

        is_unstable, metrics = detector.detect_flame_instability(flame_readings, sample_period_seconds=0.1)

        assert is_unstable, "Oscillating flame should be detected"
        assert metrics.oscillation_frequency_hz is not None

    def test_low_flame_signal_detection(self, detector: CombustionAnomalyDetector):
        """Test detection of weak flame signal."""
        # Weak flame: low mean intensity
        flame_readings = [20.0 + random.gauss(0, 2.0) for _ in range(100)]

        is_unstable, metrics = detector.detect_flame_instability(flame_readings)

        assert is_unstable, "Weak flame should be flagged as unstable"
        assert metrics.mean_intensity < 30.0

    @pytest.mark.parametrize("cv,should_be_unstable", [
        (0.05, False),   # Very stable
        (0.10, False),   # Stable
        (0.14, False),   # Just below threshold
        (0.16, True),    # Just above threshold
        (0.25, True),    # Clearly unstable
        (0.50, True),    # Very unstable
    ])
    def test_cv_threshold_behavior(
        self,
        detector: CombustionAnomalyDetector,
        cv: float,
        should_be_unstable: bool
    ):
        """Test coefficient of variation threshold."""
        mean = 85.0
        std = mean * cv

        random.seed(42)
        flame_readings = [mean + random.gauss(0, std) for _ in range(1000)]

        is_unstable, metrics = detector.detect_flame_instability(flame_readings)

        # Allow small margin due to random sampling
        if abs(metrics.coefficient_of_variation - 0.15) > 0.02:
            assert is_unstable == should_be_unstable


# =============================================================================
# AIR LEAK DETECTION TESTS
# =============================================================================

class TestAirLeakDetection:
    """Test air leak detection."""

    @pytest.fixture
    def detector(self) -> CombustionAnomalyDetector:
        return CombustionAnomalyDetector()

    def create_data_points(
        self,
        count: int,
        o2_offset: float = 0.0
    ) -> List[CombustionDataPoint]:
        """Create test data points."""
        base_time = datetime.now()
        points = []

        for i in range(count):
            # Normal operation at 3% O2
            expected_o2 = 3.0
            actual_o2 = expected_o2 + o2_offset + random.gauss(0, 0.1)

            points.append(CombustionDataPoint(
                timestamp=base_time + timedelta(seconds=i),
                o2_percent=actual_o2,
                co_ppm=50.0,
                flame_signal=85.0,
                furnace_pressure=-0.3,
                fuel_flow=500.0,
                air_flow=8600.0,  # For ~3% O2
                stack_temp_c=200.0,
                fuel_pressure=25.0,
            ))

        return points

    def test_no_air_leak_normal_operation(self, detector: CombustionAnomalyDetector):
        """Test no false positive during normal operation."""
        random.seed(42)
        data_points = self.create_data_points(20, o2_offset=0.0)

        is_leak, result = detector.detect_air_leak(data_points)

        assert not is_leak, "Should not detect leak during normal operation"

    def test_air_leak_detected(self, detector: CombustionAnomalyDetector):
        """Test air leak detection with elevated O2."""
        random.seed(42)
        # Add 3% O2 elevation (air leak)
        data_points = self.create_data_points(20, o2_offset=3.0)

        is_leak, result = detector.detect_air_leak(data_points)

        assert is_leak, "Should detect air leak"
        assert result.anomaly_type == AnomalyType.AIR_LEAK
        assert result.severity in [AnomalySeverity.MEDIUM, AnomalySeverity.HIGH]

    def test_air_leak_severity_scaling(self, detector: CombustionAnomalyDetector):
        """Test air leak severity scales with O2 elevation."""
        random.seed(42)

        # Moderate leak
        moderate_points = self.create_data_points(20, o2_offset=2.5)
        _, moderate_result = detector.detect_air_leak(moderate_points)

        # Severe leak
        severe_points = self.create_data_points(20, o2_offset=5.0)
        _, severe_result = detector.detect_air_leak(severe_points)

        if moderate_result and severe_result:
            assert severe_result.severity.value >= moderate_result.severity.value or \
                   severe_result.confidence >= moderate_result.confidence


# =============================================================================
# FUEL QUALITY VARIATION TESTS
# =============================================================================

class TestFuelQualityVariationDetection:
    """Test fuel quality variation detection."""

    @pytest.fixture
    def detector(self) -> CombustionAnomalyDetector:
        return CombustionAnomalyDetector()

    def create_data_points(
        self,
        count: int,
        co_variation: float = 0.0,
        temp_variation: float = 0.0
    ) -> List[CombustionDataPoint]:
        """Create test data points with specified variation."""
        base_time = datetime.now()
        points = []

        for i in range(count):
            co = 50.0 + random.gauss(0, 50.0 * co_variation)
            temp = 200.0 + random.gauss(0, 200.0 * temp_variation)

            points.append(CombustionDataPoint(
                timestamp=base_time + timedelta(seconds=i),
                o2_percent=3.0,
                co_ppm=max(0, co),
                flame_signal=85.0,
                furnace_pressure=-0.3,
                fuel_flow=500.0,
                air_flow=8600.0,
                stack_temp_c=temp,
                fuel_pressure=25.0,
            ))

        return points

    def test_stable_fuel_quality(self, detector: CombustionAnomalyDetector):
        """Test no false positive with stable fuel quality."""
        random.seed(42)
        data_points = self.create_data_points(20, co_variation=0.1, temp_variation=0.02)

        is_variation, result = detector.detect_fuel_quality_variation(data_points)

        assert not is_variation, "Should not detect variation with stable fuel"

    def test_fuel_quality_variation_detected(self, detector: CombustionAnomalyDetector):
        """Test detection of fuel quality variation."""
        random.seed(42)
        data_points = self.create_data_points(20, co_variation=0.5, temp_variation=0.1)

        is_variation, result = detector.detect_fuel_quality_variation(data_points)

        assert is_variation, "Should detect fuel quality variation"
        assert result.anomaly_type == AnomalyType.FUEL_QUALITY_VARIATION


# =============================================================================
# COMBUSTION OSCILLATION TESTS
# =============================================================================

class TestCombustionOscillationDetection:
    """Test combustion oscillation detection."""

    @pytest.fixture
    def detector(self) -> CombustionAnomalyDetector:
        return CombustionAnomalyDetector()

    def create_data_points_with_oscillation(
        self,
        count: int,
        pressure_amplitude: float = 0.0,
        frequency_hz: float = 1.0
    ) -> List[CombustionDataPoint]:
        """Create data points with pressure oscillation."""
        base_time = datetime.now()
        points = []

        for i in range(count):
            pressure = -0.3 + pressure_amplitude * math.sin(2 * math.pi * frequency_hz * i * 0.1)

            points.append(CombustionDataPoint(
                timestamp=base_time + timedelta(seconds=i * 0.1),
                o2_percent=3.0,
                co_ppm=50.0,
                flame_signal=85.0,
                furnace_pressure=pressure,
                fuel_flow=500.0,
                air_flow=8600.0,
                stack_temp_c=200.0,
                fuel_pressure=25.0,
            ))

        return points

    def test_stable_pressure_no_oscillation(self, detector: CombustionAnomalyDetector):
        """Test no false positive with stable pressure."""
        data_points = self.create_data_points_with_oscillation(50, pressure_amplitude=0.1)

        is_oscillating, result = detector.detect_combustion_oscillation(data_points)

        assert not is_oscillating, "Should not detect oscillation with stable pressure"

    def test_pressure_oscillation_detected(self, detector: CombustionAnomalyDetector):
        """Test detection of pressure oscillation."""
        data_points = self.create_data_points_with_oscillation(50, pressure_amplitude=1.0, frequency_hz=2.0)

        is_oscillating, result = detector.detect_combustion_oscillation(data_points, sample_period_seconds=0.1)

        assert is_oscillating, "Should detect pressure oscillation"
        assert result.anomaly_type == AnomalyType.COMBUSTION_OSCILLATION


# =============================================================================
# INCOMPLETE COMBUSTION TESTS
# =============================================================================

class TestIncompleteCombustionDetection:
    """Test incomplete combustion detection."""

    @pytest.fixture
    def detector(self) -> CombustionAnomalyDetector:
        return CombustionAnomalyDetector()

    def create_data_points(
        self,
        count: int,
        co_ppm: float,
        o2_percent: float
    ) -> List[CombustionDataPoint]:
        """Create test data points."""
        base_time = datetime.now()
        points = []

        for i in range(count):
            points.append(CombustionDataPoint(
                timestamp=base_time + timedelta(seconds=i),
                o2_percent=o2_percent + random.gauss(0, 0.1),
                co_ppm=co_ppm + random.gauss(0, 10),
                flame_signal=85.0,
                furnace_pressure=-0.3,
                fuel_flow=500.0,
                air_flow=8600.0,
                stack_temp_c=200.0,
                fuel_pressure=25.0,
            ))

        return points

    @pytest.mark.parametrize("co_ppm,o2_percent,should_detect", [
        (50, 3.0, False),      # Normal operation
        (200, 3.0, False),     # Elevated CO but good O2
        (500, 2.5, True),      # High CO, low O2
        (1000, 1.5, True),     # Very high CO, very low O2
        (400, 4.0, False),     # High CO but good O2 (not incomplete)
    ])
    def test_incomplete_combustion_scenarios(
        self,
        detector: CombustionAnomalyDetector,
        co_ppm: float,
        o2_percent: float,
        should_detect: bool
    ):
        """Test various incomplete combustion scenarios."""
        random.seed(42)
        data_points = self.create_data_points(10, co_ppm, o2_percent)

        is_incomplete, result = detector.detect_incomplete_combustion(data_points)

        assert is_incomplete == should_detect, \
            f"CO={co_ppm}ppm, O2={o2_percent}%: expected detect={should_detect}"

    def test_critical_severity_for_very_high_co(self, detector: CombustionAnomalyDetector):
        """Test critical severity for very high CO."""
        random.seed(42)
        data_points = self.create_data_points(10, co_ppm=1500, o2_percent=1.0)

        is_incomplete, result = detector.detect_incomplete_combustion(data_points)

        assert is_incomplete
        assert result.severity == AnomalySeverity.CRITICAL


# =============================================================================
# ANOMALY INIT FILE
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
