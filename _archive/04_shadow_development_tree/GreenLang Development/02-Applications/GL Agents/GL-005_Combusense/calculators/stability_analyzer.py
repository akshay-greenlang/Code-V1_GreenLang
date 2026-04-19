# -*- coding: utf-8 -*-
"""
Stability Analyzer for GL-005 CombustionControlAgent

Analyzes combustion stability through time-series analysis, FFT-based oscillation
detection, and multi-variable stability scoring. Zero-hallucination design using
signal processing and control theory.

Reference Standards:
- NFPA 85: Boiler and Combustion Systems Hazards Code
- ISA-5.1: Instrumentation Symbols and Identification
- IEEE 421.5: Excitation System Models for Power System Stability
- Ljung: System Identification - Time Series Analysis

Mathematical Formulas:
- Stability Index: SI = 1 / (1 + σ²/μ² + |max_deviation|/range)
- Variance: σ² = Σ(x_i - μ)² / (n-1)
- FFT: X(k) = Σ x(n) * e^(-j2πkn/N)
- Peak Detection: local_max where x[i] > x[i-1] and x[i] > x[i+1]
- Oscillation Score: OS = amplitude * frequency_factor * persistence_factor
"""

from typing import Dict, List, Optional, Tuple
from decimal import Decimal, ROUND_HALF_UP
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import math
import logging
import hashlib
import statistics
from dataclasses import dataclass
from collections import deque
import cmath

logger = logging.getLogger(__name__)


class StabilityLevel(str, Enum):
    """Stability classification levels"""
    EXCELLENT = "excellent"  # SI > 90
    GOOD = "good"  # SI 70-90
    FAIR = "fair"  # SI 50-70
    POOR = "poor"  # SI 30-50
    UNSTABLE = "unstable"  # SI < 30


class OscillationType(str, Enum):
    """Types of oscillations detected"""
    NONE = "none"
    LOW_FREQUENCY = "low_frequency"  # < 0.1 Hz (slow hunting)
    MEDIUM_FREQUENCY = "medium_frequency"  # 0.1 - 1 Hz (control loop)
    HIGH_FREQUENCY = "high_frequency"  # > 1 Hz (actuator, sensor noise)
    HARMONIC = "harmonic"  # Multiple frequency components


@dataclass
class OscillationPattern:
    """Detected oscillation pattern"""
    type: OscillationType
    frequency_hz: float
    amplitude: float
    phase_deg: float
    persistence_percent: float  # % of time oscillating
    severity_score: float  # 0-100 (100 = severe)


@dataclass
class VariableStats:
    """Statistical properties of a variable"""
    mean: float
    std_dev: float
    variance: float
    min_value: float
    max_value: float
    range: float
    coefficient_of_variation: float  # CV = σ/μ


class StabilityInput(BaseModel):
    """Input for stability analysis"""

    # Time series data (last N samples)
    heat_output_history_kw: List[float] = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="Heat output time series (kW)"
    )
    fuel_flow_history_kg_per_hr: List[float] = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="Fuel flow time series"
    )
    o2_percent_history: List[float] = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="O2 percentage time series"
    )
    temperature_history_c: Optional[List[float]] = Field(
        None,
        min_length=10,
        max_length=1000,
        description="Combustion temperature time series"
    )

    # Sampling information
    sample_time_sec: float = Field(
        ...,
        gt=0,
        le=60,
        description="Time between samples in seconds"
    )
    total_duration_sec: float = Field(
        ...,
        gt=0,
        description="Total duration of data collection"
    )

    # Operating conditions
    target_heat_output_kw: float = Field(
        ...,
        gt=0,
        description="Target heat output setpoint"
    )
    target_o2_percent: float = Field(
        default=3.0,
        ge=0,
        le=21,
        description="Target O2 percentage"
    )

    # Analysis parameters
    enable_fft_analysis: bool = Field(
        default=True,
        description="Enable FFT-based frequency analysis"
    )
    oscillation_threshold_percent: float = Field(
        default=5.0,
        ge=0,
        le=50,
        description="Oscillation detection threshold (% of setpoint)"
    )
    min_oscillation_cycles: int = Field(
        default=3,
        ge=2,
        le=100,
        description="Minimum cycles to confirm oscillation"
    )

    # Weighting factors for stability score
    heat_output_weight: float = Field(default=0.4, ge=0, le=1)
    fuel_flow_weight: float = Field(default=0.3, ge=0, le=1)
    o2_weight: float = Field(default=0.3, ge=0, le=1)

    @field_validator('heat_output_history_kw', 'fuel_flow_history_kg_per_hr', 'o2_percent_history')
    @classmethod
    def validate_history_lengths(cls, v, info):
        """Ensure all histories have same length"""
        if 'heat_output_history_kw' in info.data:
            if len(v) != len(info.data['heat_output_history_kw']):
                raise ValueError("All history arrays must have same length")
        return v


class StabilityResult(BaseModel):
    """Stability analysis results"""

    # Overall stability metrics
    overall_stability_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Overall stability score (0-100)"
    )
    stability_level: StabilityLevel

    # Variable-specific scores
    heat_output_stability_score: float = Field(..., ge=0, le=100)
    fuel_flow_stability_score: float = Field(..., ge=0, le=100)
    o2_stability_score: float = Field(..., ge=0, le=100)

    # Statistical properties
    heat_output_stats: Dict[str, float] = Field(
        ...,
        description="Heat output statistics"
    )
    fuel_flow_stats: Dict[str, float] = Field(
        ...,
        description="Fuel flow statistics"
    )
    o2_stats: Dict[str, float] = Field(
        ...,
        description="O2 statistics"
    )

    # Oscillation detection
    oscillations_detected: bool
    oscillation_patterns: List[Dict[str, float]] = Field(
        default_factory=list,
        description="Detected oscillation patterns"
    )
    dominant_frequency_hz: Optional[float] = None
    dominant_amplitude: Optional[float] = None

    # Deviation metrics
    heat_output_rmse_kw: float = Field(
        ...,
        description="RMSE from setpoint"
    )
    heat_output_max_deviation_percent: float = Field(
        ...,
        description="Maximum deviation from setpoint (%)"
    )
    o2_max_deviation_percent: float = Field(
        ...,
        description="Maximum O2 deviation from target (%)"
    )

    # Trend analysis
    heat_output_trend: str = Field(
        ...,
        description="Trend: increasing, decreasing, or stable"
    )
    stability_improving: bool = Field(
        ...,
        description="Whether stability is improving over time"
    )

    # Recommendations
    recommendations: List[str] = Field(
        default_factory=list,
        description="Stability improvement recommendations"
    )
    issues_detected: List[str] = Field(
        default_factory=list,
        description="Detected stability issues"
    )

    # Provenance
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash for calculation provenance"
    )


class StabilityAnalyzer:
    """
    Stability analyzer for combustion control systems.

    Analyzes time-series data from combustion process to detect:
    - Process oscillations (hunting, limit cycles)
    - Excessive variability
    - Trends and drift
    - Control loop performance issues

    Analysis Methods:
        1. Statistical Analysis: Mean, variance, CV
        2. FFT Analysis: Frequency domain analysis
        3. Peak Detection: Oscillation identification
        4. Trend Analysis: Linear regression
        5. Stability Scoring: Multi-variable weighted score

    Stability Score Calculation:
        SI = w1*SI_heat + w2*SI_fuel + w3*SI_o2

        Where SI_var = 100 * (1 - CV) * (1 - max_dev/range) * (1 - osc_factor)

        CV = coefficient of variation (σ/μ)
        max_dev = maximum deviation from setpoint
        osc_factor = oscillation severity factor
    """

    def __init__(self):
        """Initialize stability analyzer"""
        self.logger = logging.getLogger(__name__)

        # FFT cache
        self.fft_cache = {}

    def analyze_stability(
        self,
        stability_input: StabilityInput
    ) -> StabilityResult:
        """
        Perform comprehensive stability analysis.

        Algorithm:
            1. Calculate statistical properties for each variable
            2. Perform FFT analysis to detect oscillations
            3. Calculate variable-specific stability scores
            4. Calculate overall weighted stability score
            5. Detect trends and issues
            6. Generate recommendations

        Args:
            stability_input: Stability analysis input

        Returns:
            StabilityResult with complete stability assessment
        """
        self.logger.info("Starting stability analysis")

        # Step 1: Calculate statistics for each variable
        heat_stats = self._calculate_statistics(
            stability_input.heat_output_history_kw,
            stability_input.target_heat_output_kw
        )

        fuel_stats = self._calculate_statistics(
            stability_input.fuel_flow_history_kg_per_hr,
            None  # No setpoint for fuel flow
        )

        o2_stats = self._calculate_statistics(
            stability_input.o2_percent_history,
            stability_input.target_o2_percent
        )

        # Step 2: Perform FFT analysis for oscillation detection
        oscillations_detected = False
        oscillation_patterns = []
        dominant_frequency = None
        dominant_amplitude = None

        if stability_input.enable_fft_analysis:
            # Analyze heat output oscillations
            heat_oscillations = self._detect_oscillations_fft(
                stability_input.heat_output_history_kw,
                stability_input.sample_time_sec,
                stability_input.oscillation_threshold_percent,
                stability_input.target_heat_output_kw
            )

            if heat_oscillations:
                oscillations_detected = True
                oscillation_patterns.extend(heat_oscillations)

                # Get dominant oscillation
                if heat_oscillations:
                    dominant_pattern = max(heat_oscillations, key=lambda x: x['severity_score'])
                    dominant_frequency = dominant_pattern['frequency_hz']
                    dominant_amplitude = dominant_pattern['amplitude']

        # Step 3: Calculate deviation metrics
        heat_rmse = self._calculate_rmse(
            stability_input.heat_output_history_kw,
            stability_input.target_heat_output_kw
        )

        heat_max_dev = (
            abs(heat_stats.max_value - stability_input.target_heat_output_kw) /
            stability_input.target_heat_output_kw * 100
        )

        o2_max_dev = (
            abs(o2_stats.max_value - stability_input.target_o2_percent) /
            stability_input.target_o2_percent * 100
            if stability_input.target_o2_percent > 0 else 0
        )

        # Step 4: Calculate variable-specific stability scores
        heat_stability_score = self._calculate_variable_stability_score(
            heat_stats,
            stability_input.target_heat_output_kw,
            oscillation_patterns,
            'heat'
        )

        fuel_stability_score = self._calculate_variable_stability_score(
            fuel_stats,
            fuel_stats.mean,  # Use mean as "target"
            [],
            'fuel'
        )

        o2_stability_score = self._calculate_variable_stability_score(
            o2_stats,
            stability_input.target_o2_percent,
            [],
            'o2'
        )

        # Step 5: Calculate overall weighted stability score
        overall_score = (
            stability_input.heat_output_weight * heat_stability_score +
            stability_input.fuel_flow_weight * fuel_stability_score +
            stability_input.o2_weight * o2_stability_score
        )

        # Step 6: Classify stability level
        stability_level = self._classify_stability_level(overall_score)

        # Step 7: Trend analysis
        heat_trend = self._analyze_trend(stability_input.heat_output_history_kw)

        # Check if stability is improving (second half better than first half)
        mid_point = len(stability_input.heat_output_history_kw) // 2
        first_half_cv = self._calculate_statistics(
            stability_input.heat_output_history_kw[:mid_point],
            stability_input.target_heat_output_kw
        ).coefficient_of_variation

        second_half_cv = self._calculate_statistics(
            stability_input.heat_output_history_kw[mid_point:],
            stability_input.target_heat_output_kw
        ).coefficient_of_variation

        stability_improving = second_half_cv < first_half_cv

        # Step 8: Detect issues and generate recommendations
        issues = self._detect_stability_issues(
            heat_stats,
            fuel_stats,
            o2_stats,
            oscillations_detected,
            heat_max_dev,
            o2_max_dev
        )

        recommendations = self._generate_recommendations(
            stability_level,
            oscillations_detected,
            oscillation_patterns,
            heat_trend,
            issues
        )

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance(
            stability_input,
            overall_score,
            heat_stats,
            fuel_stats,
            o2_stats
        )

        return StabilityResult(
            overall_stability_score=self._round_decimal(overall_score, 2),
            stability_level=stability_level,
            heat_output_stability_score=self._round_decimal(heat_stability_score, 2),
            fuel_flow_stability_score=self._round_decimal(fuel_stability_score, 2),
            o2_stability_score=self._round_decimal(o2_stability_score, 2),
            heat_output_stats=self._stats_to_dict(heat_stats),
            fuel_flow_stats=self._stats_to_dict(fuel_stats),
            o2_stats=self._stats_to_dict(o2_stats),
            oscillations_detected=oscillations_detected,
            oscillation_patterns=oscillation_patterns,
            dominant_frequency_hz=dominant_frequency,
            dominant_amplitude=dominant_amplitude,
            heat_output_rmse_kw=self._round_decimal(heat_rmse, 2),
            heat_output_max_deviation_percent=self._round_decimal(heat_max_dev, 2),
            o2_max_deviation_percent=self._round_decimal(o2_max_dev, 2),
            heat_output_trend=heat_trend,
            stability_improving=stability_improving,
            recommendations=recommendations,
            issues_detected=issues,
            provenance_hash=provenance_hash
        )

    def _calculate_statistics(
        self,
        data: List[float],
        target: Optional[float]
    ) -> VariableStats:
        """
        Calculate statistical properties of time series.

        Args:
            data: Time series data
            target: Target/setpoint value (optional)

        Returns:
            VariableStats with statistical properties
        """
        if not data:
            return VariableStats(0, 0, 0, 0, 0, 0, 0)

        mean = statistics.mean(data)
        std_dev = statistics.stdev(data) if len(data) > 1 else 0
        variance = std_dev ** 2
        min_val = min(data)
        max_val = max(data)
        range_val = max_val - min_val

        # Coefficient of variation
        cv = (std_dev / mean) if mean != 0 else 0

        return VariableStats(
            mean=mean,
            std_dev=std_dev,
            variance=variance,
            min_value=min_val,
            max_value=max_val,
            range=range_val,
            coefficient_of_variation=cv
        )

    def _detect_oscillations_fft(
        self,
        data: List[float],
        sample_time: float,
        threshold_percent: float,
        setpoint: float
    ) -> List[Dict[str, float]]:
        """
        Detect oscillations using FFT (Fast Fourier Transform).

        Args:
            data: Time series data
            sample_time: Sampling period (seconds)
            threshold_percent: Detection threshold (% of setpoint)
            setpoint: Setpoint value

        Returns:
            List of detected oscillation patterns
        """
        n = len(data)
        if n < 4:
            return []

        # Remove DC component (mean)
        mean = sum(data) / n
        data_ac = [x - mean for x in data]

        # Compute FFT (using DFT algorithm)
        fft_result = self._compute_dft(data_ac)

        # Calculate sampling frequency
        fs = 1.0 / sample_time  # Hz

        # Frequency resolution
        freq_resolution = fs / n

        # Find peaks in frequency domain
        oscillations = []
        threshold_amplitude = setpoint * (threshold_percent / 100)

        # Only analyze positive frequencies (up to Nyquist)
        for k in range(1, n // 2):
            # Frequency
            freq_hz = k * freq_resolution

            # Magnitude (normalize by n/2)
            magnitude = abs(fft_result[k]) * 2 / n

            # Check if peak
            if magnitude > threshold_amplitude:
                # Phase angle
                phase_deg = math.degrees(cmath.phase(fft_result[k]))

                # Classify oscillation type
                if freq_hz < 0.1:
                    osc_type = OscillationType.LOW_FREQUENCY
                elif freq_hz < 1.0:
                    osc_type = OscillationType.MEDIUM_FREQUENCY
                else:
                    osc_type = OscillationType.HIGH_FREQUENCY

                # Calculate severity score (0-100)
                severity = min(100, (magnitude / setpoint) * 100 * freq_hz)

                oscillations.append({
                    'type': osc_type.value,
                    'frequency_hz': self._round_decimal(freq_hz, 4),
                    'amplitude': self._round_decimal(magnitude, 2),
                    'phase_deg': self._round_decimal(phase_deg, 2),
                    'persistence_percent': 100.0,  # Simplified
                    'severity_score': self._round_decimal(severity, 2)
                })

        return oscillations

    def _compute_dft(self, data: List[float]) -> List[complex]:
        """
        Compute Discrete Fourier Transform.

        DFT Formula:
            X[k] = Σ(n=0 to N-1) x[n] * e^(-j*2π*k*n/N)

        Args:
            data: Input time series

        Returns:
            Complex frequency domain representation
        """
        N = len(data)
        result = []

        for k in range(N):
            real_sum = 0.0
            imag_sum = 0.0

            for n in range(N):
                angle = -2 * math.pi * k * n / N
                real_sum += data[n] * math.cos(angle)
                imag_sum += data[n] * math.sin(angle)

            result.append(complex(real_sum, imag_sum))

        return result

    def _calculate_variable_stability_score(
        self,
        stats: VariableStats,
        target: float,
        oscillations: List[Dict[str, float]],
        variable_name: str
    ) -> float:
        """
        Calculate stability score for individual variable.

        Formula:
            SI = 100 * (1 - CV) * (1 - max_dev_factor) * (1 - osc_factor)

        Where:
            CV = coefficient of variation (capped at 0.5)
            max_dev_factor = (max_dev / target) capped at 1.0
            osc_factor = maximum oscillation severity / 100

        Args:
            stats: Variable statistics
            target: Target/setpoint value
            oscillations: Detected oscillations
            variable_name: Variable name for logging

        Returns:
            Stability score (0-100)
        """
        # CV factor (0-1, lower is better)
        cv_capped = min(stats.coefficient_of_variation, 0.5)
        cv_factor = 1 - (cv_capped / 0.5)

        # Deviation factor
        if target != 0:
            max_deviation = max(
                abs(stats.max_value - target),
                abs(stats.min_value - target)
            )
            dev_factor = 1 - min(1.0, max_deviation / abs(target))
        else:
            dev_factor = 1.0

        # Oscillation factor
        osc_factor = 0.0
        if oscillations:
            max_severity = max(osc['severity_score'] for osc in oscillations)
            osc_factor = max_severity / 100

        # Combined score
        stability_score = 100 * cv_factor * dev_factor * (1 - osc_factor)

        return max(0, min(100, stability_score))

    def _classify_stability_level(self, score: float) -> StabilityLevel:
        """Classify stability level based on score"""
        if score >= 90:
            return StabilityLevel.EXCELLENT
        elif score >= 70:
            return StabilityLevel.GOOD
        elif score >= 50:
            return StabilityLevel.FAIR
        elif score >= 30:
            return StabilityLevel.POOR
        else:
            return StabilityLevel.UNSTABLE

    def _calculate_rmse(self, data: List[float], target: float) -> float:
        """
        Calculate Root Mean Square Error from target.

        Formula:
            RMSE = sqrt(Σ(x_i - target)² / n)
        """
        if not data:
            return 0.0

        squared_errors = [(x - target) ** 2 for x in data]
        mse = sum(squared_errors) / len(data)
        rmse = math.sqrt(mse)

        return rmse

    def _analyze_trend(self, data: List[float]) -> str:
        """
        Analyze trend in time series using linear regression.

        Returns:
            "increasing", "decreasing", or "stable"
        """
        if len(data) < 3:
            return "stable"

        n = len(data)
        x = list(range(n))

        # Simple linear regression
        x_mean = sum(x) / n
        y_mean = sum(data) / n

        numerator = sum((x[i] - x_mean) * (data[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return "stable"

        slope = numerator / denominator

        # Classify based on slope magnitude
        threshold = y_mean * 0.01 / n  # 1% change over period

        if slope > threshold:
            return "increasing"
        elif slope < -threshold:
            return "decreasing"
        else:
            return "stable"

    def _detect_stability_issues(
        self,
        heat_stats: VariableStats,
        fuel_stats: VariableStats,
        o2_stats: VariableStats,
        oscillations_detected: bool,
        heat_max_dev: float,
        o2_max_dev: float
    ) -> List[str]:
        """Detect specific stability issues"""
        issues = []

        # High variability
        if heat_stats.coefficient_of_variation > 0.1:
            issues.append("High heat output variability (CV > 10%)")

        if fuel_stats.coefficient_of_variation > 0.15:
            issues.append("High fuel flow variability (CV > 15%)")

        if o2_stats.coefficient_of_variation > 0.2:
            issues.append("High O2 variability (CV > 20%)")

        # Large deviations
        if heat_max_dev > 10:
            issues.append(f"Large heat output deviation ({heat_max_dev:.1f}%)")

        if o2_max_dev > 20:
            issues.append(f"Large O2 deviation ({o2_max_dev:.1f}%)")

        # Oscillations
        if oscillations_detected:
            issues.append("Process oscillations detected - possible control loop tuning issue")

        return issues

    def _generate_recommendations(
        self,
        stability_level: StabilityLevel,
        oscillations_detected: bool,
        oscillation_patterns: List[Dict[str, float]],
        trend: str,
        issues: List[str]
    ) -> List[str]:
        """Generate stability improvement recommendations"""
        recommendations = []

        # Based on stability level
        if stability_level == StabilityLevel.UNSTABLE:
            recommendations.append("CRITICAL: System unstable - check control loop tuning immediately")
            recommendations.append("Consider reducing controller gains or increasing derivative time")

        elif stability_level == StabilityLevel.POOR:
            recommendations.append("System stability poor - tune PID controller")
            recommendations.append("Check for sensor noise or measurement issues")

        elif stability_level == StabilityLevel.FAIR:
            recommendations.append("System stability acceptable but could be improved")
            recommendations.append("Consider auto-tuning PID controller")

        # Based on oscillations
        if oscillations_detected and oscillation_patterns:
            dominant = max(oscillation_patterns, key=lambda x: x['severity_score'])
            freq = dominant['frequency_hz']

            if freq < 0.1:
                recommendations.append(
                    "Low-frequency oscillation detected - reduce integral gain (Ki)"
                )
            elif freq < 1.0:
                recommendations.append(
                    "Medium-frequency oscillation detected - reduce proportional gain (Kp)"
                )
            else:
                recommendations.append(
                    "High-frequency oscillation detected - check sensor noise or reduce derivative gain (Kd)"
                )

        # Based on trend
        if trend == "increasing" or trend == "decreasing":
            recommendations.append(
                f"Heat output trending {trend} - check for load changes or fuel quality variations"
            )

        # If stable
        if stability_level in [StabilityLevel.EXCELLENT, StabilityLevel.GOOD] and not issues:
            recommendations.append("System operating stably - maintain current settings")

        return recommendations

    def _stats_to_dict(self, stats: VariableStats) -> Dict[str, float]:
        """Convert VariableStats to dictionary"""
        return {
            'mean': self._round_decimal(stats.mean, 4),
            'std_dev': self._round_decimal(stats.std_dev, 4),
            'variance': self._round_decimal(stats.variance, 4),
            'min_value': self._round_decimal(stats.min_value, 4),
            'max_value': self._round_decimal(stats.max_value, 4),
            'range': self._round_decimal(stats.range, 4),
            'coefficient_of_variation': self._round_decimal(stats.coefficient_of_variation, 4)
        }

    def _calculate_provenance(
        self,
        stability_input: StabilityInput,
        overall_score: float,
        heat_stats: VariableStats,
        fuel_stats: VariableStats,
        o2_stats: VariableStats
    ) -> str:
        """Calculate SHA-256 hash for complete audit trail"""
        provenance_data = {
            'target_heat_output': stability_input.target_heat_output_kw,
            'sample_time': stability_input.sample_time_sec,
            'num_samples': len(stability_input.heat_output_history_kw),
            'overall_score': overall_score,
            'heat_stats': self._stats_to_dict(heat_stats),
            'fuel_stats': self._stats_to_dict(fuel_stats),
            'o2_stats': self._stats_to_dict(o2_stats)
        }

        provenance_str = str(provenance_data)
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def _round_decimal(self, value: float, places: int) -> float:
        """Round to specified decimal places using ROUND_HALF_UP"""
        if value is None:
            return None
        decimal_value = Decimal(str(value))
        quantize_string = '0.' + '0' * places if places > 0 else '1'
        rounded = decimal_value.quantize(Decimal(quantize_string), rounding=ROUND_HALF_UP)
        return float(rounded)
