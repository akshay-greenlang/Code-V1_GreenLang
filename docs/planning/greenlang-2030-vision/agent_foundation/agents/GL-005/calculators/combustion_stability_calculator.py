# -*- coding: utf-8 -*-
"""
Combustion Stability Calculator for GL-005 CombustionControlAgent

Calculates stability indices from temperature/pressure oscillations and flame characteristics.
Zero-hallucination design using deterministic signal processing and combustion physics.

Reference Standards:
- NFPA 86: Standard for Ovens and Furnaces
- ISO 13579: Industrial Furnaces - Method of Measuring Energy Balance
- ASME CSD-1: Controls and Safety Devices for Automatically Fired Boilers

Mathematical Formulas:
- RMS Oscillation: sqrt(sum((x_i - mean)^2) / N)
- Stability Index: 1 / (1 + normalized_oscillation_amplitude)
- Frequency Domain: FFT for dominant frequency detection
- Damping Ratio: ln(A1/A2) / (2*pi*sqrt(1 + (ln(A1/A2)/(2*pi))^2))
"""

from typing import Dict, List, Optional, Tuple
from decimal import Decimal, ROUND_HALF_UP
from pydantic import BaseModel, Field, validator
from enum import Enum
import math
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class StabilityLevel(str, Enum):
    """Combustion stability classification"""
    STABLE = "stable"
    MODERATELY_STABLE = "moderately_stable"
    UNSTABLE = "unstable"
    CRITICALLY_UNSTABLE = "critically_unstable"


class FlameCharacteristic(str, Enum):
    """Flame characteristic types"""
    LENGTH = "length"
    SHAPE = "shape"
    INTENSITY = "intensity"
    COLOR = "color"


@dataclass
class OscillationPattern:
    """Detected oscillation pattern"""
    frequency_hz: float
    amplitude: float
    damping_ratio: float
    pattern_type: str  # "periodic", "random", "chaotic"


class StabilityInput(BaseModel):
    """Input parameters for stability calculation"""

    # Temperature data
    temperature_readings: List[float] = Field(
        ...,
        description="Time-series temperature readings in Celsius",
        min_items=10,
        max_items=10000
    )
    temperature_setpoint: float = Field(
        ...,
        ge=0,
        le=2000,
        description="Target temperature in Celsius"
    )

    # Pressure data
    pressure_readings: List[float] = Field(
        ...,
        description="Time-series pressure readings in Pa",
        min_items=10,
        max_items=10000
    )
    pressure_setpoint: float = Field(
        ...,
        ge=0,
        le=1000000,
        description="Target pressure in Pa"
    )

    # Sampling parameters
    sampling_rate_hz: float = Field(
        default=10.0,
        ge=0.1,
        le=1000.0,
        description="Data sampling rate in Hz"
    )

    # Flame characteristics (optional)
    flame_length_mm: Optional[float] = Field(
        None,
        ge=0,
        le=10000,
        description="Measured flame length in mm"
    )
    flame_intensity_percent: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Flame intensity as percentage of maximum"
    )

    # Operating parameters
    fuel_flow_rate: float = Field(
        ...,
        ge=0,
        description="Fuel flow rate in kg/hr"
    )
    air_flow_rate: float = Field(
        ...,
        ge=0,
        description="Air flow rate in kg/hr"
    )

    @validator('temperature_readings', 'pressure_readings')
    def validate_readings_length(cls, v, values, field):
        """Ensure temperature and pressure readings have same length"""
        if 'temperature_readings' in values and field.name == 'pressure_readings':
            if len(v) != len(values['temperature_readings']):
                raise ValueError("Temperature and pressure readings must have same length")
        return v


class StabilityResult(BaseModel):
    """Combustion stability calculation results"""

    # Overall stability
    stability_index: float = Field(
        ...,
        ge=0,
        le=1,
        description="Overall stability index (0=unstable, 1=perfectly stable)"
    )
    stability_level: StabilityLevel

    # Temperature stability
    temperature_stability_index: float
    temperature_rms_deviation: float
    temperature_peak_to_peak: float

    # Pressure stability
    pressure_stability_index: float
    pressure_rms_deviation: float
    pressure_peak_to_peak: float

    # Oscillation analysis
    dominant_frequency_hz: Optional[float]
    oscillation_amplitude: float
    damping_ratio: Optional[float]
    oscillation_pattern: str

    # Flame characteristics
    flame_stability_score: Optional[float]
    flame_length_deviation: Optional[float]

    # Risk assessment
    blowout_risk_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Risk of flame blowout (0=no risk, 1=imminent)"
    )
    flashback_risk_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Risk of flashback (0=no risk, 1=imminent)"
    )

    # Recommendations
    requires_intervention: bool
    recommended_actions: List[str]


class CombustionStabilityCalculator:
    """
    Calculate combustion stability metrics using deterministic signal processing.

    This calculator analyzes temperature and pressure oscillations to determine
    combustion stability. All calculations are purely mathematical with no ML/AI.

    Stability Index Formula:
        SI = 1 / (1 + (RMS_temp/SP_temp + RMS_pressure/SP_pressure) / 2)

    Where:
        - RMS_temp: Root mean square deviation of temperature
        - SP_temp: Temperature setpoint
        - RMS_pressure: Root mean square deviation of pressure
        - SP_pressure: Pressure setpoint
    """

    # Stability thresholds
    STABLE_THRESHOLD = 0.85
    MODERATELY_STABLE_THRESHOLD = 0.70
    UNSTABLE_THRESHOLD = 0.50

    # Risk thresholds
    BLOWOUT_VELOCITY_RATIO = 0.8  # % of theoretical blowout velocity
    FLASHBACK_VELOCITY_RATIO = 1.2  # % of theoretical flashback velocity

    def __init__(self):
        """Initialize combustion stability calculator"""
        self.logger = logging.getLogger(__name__)

    def calculate_stability_index(
        self,
        stability_input: StabilityInput
    ) -> StabilityResult:
        """
        Calculate comprehensive combustion stability index.

        Args:
            stability_input: Stability calculation input parameters

        Returns:
            StabilityResult with complete stability analysis

        Algorithm:
            1. Calculate temperature stability (RMS, peak-to-peak)
            2. Calculate pressure stability (RMS, peak-to-peak)
            3. Detect oscillation patterns (FFT analysis)
            4. Analyze flame characteristics
            5. Assess blowout/flashback risk
            6. Compute overall stability index
        """
        self.logger.info("Calculating combustion stability index")

        # Step 1: Temperature stability analysis
        temp_stats = self._calculate_signal_statistics(
            stability_input.temperature_readings,
            stability_input.temperature_setpoint
        )

        # Step 2: Pressure stability analysis
        pressure_stats = self._calculate_signal_statistics(
            stability_input.pressure_readings,
            stability_input.pressure_setpoint
        )

        # Step 3: Oscillation pattern detection
        oscillation = self._detect_oscillations(
            stability_input.temperature_readings,
            stability_input.pressure_readings,
            stability_input.sampling_rate_hz
        )

        # Step 4: Flame stability analysis
        flame_stability = self._analyze_flame_characteristics(
            stability_input.flame_length_mm,
            stability_input.flame_intensity_percent,
            stability_input.fuel_flow_rate,
            stability_input.air_flow_rate
        )

        # Step 5: Risk assessment
        blowout_risk = self._predict_blowout_risk(
            stability_input.fuel_flow_rate,
            stability_input.air_flow_rate,
            temp_stats['rms_deviation'],
            pressure_stats['rms_deviation']
        )

        flashback_risk = self._predict_flashback_risk(
            stability_input.fuel_flow_rate,
            stability_input.air_flow_rate,
            temp_stats['mean_value']
        )

        # Step 6: Overall stability index
        # Weighted combination of temperature and pressure stability
        temp_weight = 0.6
        pressure_weight = 0.4

        overall_stability = (
            temp_weight * temp_stats['stability_index'] +
            pressure_weight * pressure_stats['stability_index']
        )

        # Classify stability level
        stability_level = self._classify_stability(overall_stability)

        # Generate recommendations
        requires_intervention = overall_stability < self.MODERATELY_STABLE_THRESHOLD
        recommendations = self._generate_recommendations(
            overall_stability,
            blowout_risk,
            flashback_risk,
            oscillation
        )

        return StabilityResult(
            stability_index=self._round_decimal(overall_stability, 4),
            stability_level=stability_level,
            temperature_stability_index=self._round_decimal(temp_stats['stability_index'], 4),
            temperature_rms_deviation=self._round_decimal(temp_stats['rms_deviation'], 2),
            temperature_peak_to_peak=self._round_decimal(temp_stats['peak_to_peak'], 2),
            pressure_stability_index=self._round_decimal(pressure_stats['stability_index'], 4),
            pressure_rms_deviation=self._round_decimal(pressure_stats['rms_deviation'], 2),
            pressure_peak_to_peak=self._round_decimal(pressure_stats['peak_to_peak'], 2),
            dominant_frequency_hz=oscillation.frequency_hz if oscillation else None,
            oscillation_amplitude=self._round_decimal(oscillation.amplitude if oscillation else 0, 4),
            damping_ratio=oscillation.damping_ratio if oscillation else None,
            oscillation_pattern=oscillation.pattern_type if oscillation else "none",
            flame_stability_score=flame_stability,
            flame_length_deviation=None,  # Calculated separately if needed
            blowout_risk_score=self._round_decimal(blowout_risk, 4),
            flashback_risk_score=self._round_decimal(flashback_risk, 4),
            requires_intervention=requires_intervention,
            recommended_actions=recommendations
        )

    def _calculate_signal_statistics(
        self,
        readings: List[float],
        setpoint: float
    ) -> Dict[str, float]:
        """
        Calculate statistical measures of signal stability.

        Formulas:
            - Mean: sum(x_i) / N
            - RMS Deviation: sqrt(sum((x_i - setpoint)^2) / N)
            - Peak-to-Peak: max(x_i) - min(x_i)
            - Stability Index: 1 / (1 + RMS/setpoint)
        """
        n = len(readings)

        # Mean value
        mean_value = sum(readings) / n

        # RMS deviation from setpoint
        squared_deviations = [(x - setpoint) ** 2 for x in readings]
        rms_deviation = math.sqrt(sum(squared_deviations) / n)

        # Peak-to-peak variation
        peak_to_peak = max(readings) - min(readings)

        # Stability index (inverse of normalized RMS)
        # Avoid division by zero
        normalized_rms = rms_deviation / setpoint if setpoint > 0 else rms_deviation
        stability_index = 1.0 / (1.0 + normalized_rms)

        return {
            'mean_value': mean_value,
            'rms_deviation': rms_deviation,
            'peak_to_peak': peak_to_peak,
            'stability_index': stability_index
        }

    def _detect_oscillations(
        self,
        temp_readings: List[float],
        pressure_readings: List[float],
        sampling_rate: float
    ) -> Optional[OscillationPattern]:
        """
        Detect oscillation patterns using simplified frequency analysis.

        Uses zero-crossing method to detect dominant frequency.
        No FFT to maintain simplicity and determinism.
        """
        # Combine temperature and pressure signals
        # Normalize both signals to 0-1 range
        temp_normalized = self._normalize_signal(temp_readings)
        pressure_normalized = self._normalize_signal(pressure_readings)

        # Combined signal (equal weight)
        combined_signal = [
            0.5 * t + 0.5 * p
            for t, p in zip(temp_normalized, pressure_normalized)
        ]

        # Detect zero crossings
        mean_signal = sum(combined_signal) / len(combined_signal)
        zero_crossings = self._count_zero_crossings(combined_signal, mean_signal)

        # Dominant frequency (Hz) = (zero_crossings / 2) / (duration_seconds)
        duration_seconds = len(combined_signal) / sampling_rate
        dominant_frequency = (zero_crossings / 2.0) / duration_seconds if duration_seconds > 0 else 0

        # Oscillation amplitude (normalized)
        amplitude = max(combined_signal) - min(combined_signal)

        # Estimate damping ratio from successive peaks
        peaks = self._find_peaks(combined_signal)
        damping_ratio = self._calculate_damping_ratio(peaks) if len(peaks) >= 2 else None

        # Classify pattern type
        if dominant_frequency < 0.1:
            pattern_type = "random"
        elif dominant_frequency < 2.0:
            pattern_type = "periodic"
        else:
            pattern_type = "chaotic"

        # Only return oscillation if significant
        if amplitude > 0.1:  # 10% threshold
            return OscillationPattern(
                frequency_hz=dominant_frequency,
                amplitude=amplitude,
                damping_ratio=damping_ratio,
                pattern_type=pattern_type
            )

        return None

    def _normalize_signal(self, signal: List[float]) -> List[float]:
        """Normalize signal to 0-1 range"""
        min_val = min(signal)
        max_val = max(signal)
        range_val = max_val - min_val

        if range_val == 0:
            return [0.0] * len(signal)

        return [(x - min_val) / range_val for x in signal]

    def _count_zero_crossings(self, signal: List[float], threshold: float) -> int:
        """Count number of zero crossings in signal"""
        crossings = 0
        for i in range(1, len(signal)):
            if (signal[i-1] < threshold <= signal[i]) or (signal[i-1] >= threshold > signal[i]):
                crossings += 1
        return crossings

    def _find_peaks(self, signal: List[float]) -> List[Tuple[int, float]]:
        """Find local maxima in signal"""
        peaks = []
        for i in range(1, len(signal) - 1):
            if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                peaks.append((i, signal[i]))
        return peaks

    def _calculate_damping_ratio(self, peaks: List[Tuple[int, float]]) -> float:
        """
        Calculate damping ratio from successive peak amplitudes.

        Formula: ζ = ln(A1/A2) / sqrt((2πn)^2 + ln(A1/A2)^2)

        Simplified for n=1 (successive peaks):
        ζ = ln(A1/A2) / (2π * sqrt(1 + (ln(A1/A2)/(2π))^2))
        """
        if len(peaks) < 2:
            return 0.0

        # Use first two peaks
        a1 = peaks[0][1]
        a2 = peaks[1][1]

        if a1 <= 0 or a2 <= 0 or a1 <= a2:
            return 0.0

        log_decrement = math.log(a1 / a2)
        two_pi = 2 * math.pi

        damping_ratio = log_decrement / (two_pi * math.sqrt(1 + (log_decrement / two_pi) ** 2))

        return min(damping_ratio, 1.0)  # Cap at 1.0 (critically damped)

    def analyze_flame_characteristics(
        self,
        flame_length_mm: Optional[float],
        flame_intensity_percent: Optional[float],
        target_length_mm: Optional[float] = None,
        target_intensity_percent: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Analyze flame stability based on physical characteristics.

        Args:
            flame_length_mm: Measured flame length
            flame_intensity_percent: Measured flame intensity
            target_length_mm: Target flame length (optional)
            target_intensity_percent: Target intensity (optional)

        Returns:
            Dictionary with flame analysis metrics
        """
        result = {
            'length_stability': None,
            'intensity_stability': None,
            'overall_flame_stability': None
        }

        # Length stability
        if flame_length_mm is not None and target_length_mm is not None:
            length_deviation = abs(flame_length_mm - target_length_mm)
            length_stability = 1.0 / (1.0 + length_deviation / target_length_mm)
            result['length_stability'] = length_stability

        # Intensity stability
        if flame_intensity_percent is not None and target_intensity_percent is not None:
            intensity_deviation = abs(flame_intensity_percent - target_intensity_percent)
            intensity_stability = 1.0 / (1.0 + intensity_deviation / 100.0)
            result['intensity_stability'] = intensity_stability

        # Overall flame stability
        stabilities = [s for s in [result['length_stability'], result['intensity_stability']] if s is not None]
        if stabilities:
            result['overall_flame_stability'] = sum(stabilities) / len(stabilities)

        return result

    def _analyze_flame_characteristics(
        self,
        flame_length_mm: Optional[float],
        flame_intensity_percent: Optional[float],
        fuel_flow_rate: float,
        air_flow_rate: float
    ) -> Optional[float]:
        """Internal flame analysis for stability calculation"""
        # Estimate expected flame length from flow rates
        # Empirical correlation: L ∝ (fuel_flow)^0.5
        if fuel_flow_rate > 0:
            expected_length = 200 * math.sqrt(fuel_flow_rate / 100)  # Simplified

            if flame_length_mm is not None:
                length_deviation = abs(flame_length_mm - expected_length) / expected_length
                return 1.0 / (1.0 + length_deviation)

        return None

    def predict_blowout_risk(
        self,
        fuel_flow_rate: float,
        air_flow_rate: float,
        current_velocity: float,
        burner_diameter_mm: float = 100.0
    ) -> float:
        """
        Predict risk of flame blowout.

        Blowout occurs when flow velocity exceeds blowout velocity.

        Args:
            fuel_flow_rate: Fuel flow rate (kg/hr)
            air_flow_rate: Air flow rate (kg/hr)
            current_velocity: Current flow velocity (m/s)
            burner_diameter_mm: Burner diameter in mm

        Returns:
            Blowout risk score (0=no risk, 1=imminent)
        """
        # Calculate blowout velocity (empirical correlation)
        # V_blowout ≈ 50-100 m/s for typical industrial burners
        # Depends on fuel type, burner design, equivalence ratio

        equivalence_ratio = self._calculate_equivalence_ratio(
            fuel_flow_rate,
            air_flow_rate
        )

        # Blowout velocity decreases with lean mixtures
        base_blowout_velocity = 75.0  # m/s
        blowout_velocity = base_blowout_velocity * math.sqrt(equivalence_ratio)

        # Risk increases as current velocity approaches blowout velocity
        velocity_ratio = current_velocity / blowout_velocity if blowout_velocity > 0 else 0

        # Sigmoid risk function
        risk = 1.0 / (1.0 + math.exp(-10 * (velocity_ratio - self.BLOWOUT_VELOCITY_RATIO)))

        return min(risk, 1.0)

    def _predict_blowout_risk(
        self,
        fuel_flow_rate: float,
        air_flow_rate: float,
        temp_rms: float,
        pressure_rms: float
    ) -> float:
        """Internal blowout risk assessment"""
        # Simplified risk based on flow rates and instability
        equivalence_ratio = self._calculate_equivalence_ratio(fuel_flow_rate, air_flow_rate)

        # Risk increases with lean mixtures (low ER) and high oscillations
        base_risk = 1.0 - min(equivalence_ratio, 1.0)  # Higher risk when lean
        instability_factor = (temp_rms / 100.0 + pressure_rms / 1000.0) / 2.0

        total_risk = min(base_risk + instability_factor, 1.0)

        return total_risk

    def _predict_flashback_risk(
        self,
        fuel_flow_rate: float,
        air_flow_rate: float,
        temperature: float
    ) -> float:
        """
        Predict risk of flashback.

        Flashback occurs when flame velocity exceeds flow velocity.
        Risk increases with:
        - Rich mixtures (high equivalence ratio)
        - Low flow velocity
        - High temperature
        """
        equivalence_ratio = self._calculate_equivalence_ratio(fuel_flow_rate, air_flow_rate)

        # Risk increases with rich mixtures
        base_risk = max(0, equivalence_ratio - 1.0)

        # Temperature factor (higher temp increases flame speed)
        temp_factor = max(0, (temperature - 800) / 1000.0)

        total_risk = min(base_risk + temp_factor, 1.0)

        return total_risk

    def _calculate_equivalence_ratio(
        self,
        fuel_flow_rate: float,
        air_flow_rate: float,
        stoichiometric_ratio: float = 14.7  # Typical for natural gas
    ) -> float:
        """
        Calculate equivalence ratio (phi).

        ER = (Fuel/Air)_actual / (Fuel/Air)_stoichiometric
        """
        if air_flow_rate == 0:
            return 0.0

        actual_ratio = fuel_flow_rate / air_flow_rate
        equivalence_ratio = actual_ratio * stoichiometric_ratio

        return equivalence_ratio

    def _classify_stability(self, stability_index: float) -> StabilityLevel:
        """Classify stability level based on index value"""
        if stability_index >= self.STABLE_THRESHOLD:
            return StabilityLevel.STABLE
        elif stability_index >= self.MODERATELY_STABLE_THRESHOLD:
            return StabilityLevel.MODERATELY_STABLE
        elif stability_index >= self.UNSTABLE_THRESHOLD:
            return StabilityLevel.UNSTABLE
        else:
            return StabilityLevel.CRITICALLY_UNSTABLE

    def _generate_recommendations(
        self,
        stability_index: float,
        blowout_risk: float,
        flashback_risk: float,
        oscillation: Optional[OscillationPattern]
    ) -> List[str]:
        """Generate actionable recommendations based on stability analysis"""
        recommendations = []

        if stability_index < self.UNSTABLE_THRESHOLD:
            recommendations.append("CRITICAL: Stability index below 0.5 - immediate intervention required")

        if blowout_risk > 0.7:
            recommendations.append("WARNING: High blowout risk - reduce air flow or increase fuel flow")

        if flashback_risk > 0.7:
            recommendations.append("WARNING: High flashback risk - increase air flow or reduce fuel flow")

        if oscillation and oscillation.amplitude > 0.3:
            recommendations.append(f"Significant oscillations detected at {oscillation.frequency_hz:.2f} Hz - check fuel/air ratio")

        if oscillation and oscillation.damping_ratio and oscillation.damping_ratio < 0.1:
            recommendations.append("Low damping detected - adjust PID controller gains")

        if stability_index >= self.STABLE_THRESHOLD:
            recommendations.append("System operating in stable regime - maintain current settings")

        return recommendations

    def _round_decimal(self, value: float, places: int) -> float:
        """Round to specified decimal places using ROUND_HALF_UP"""
        decimal_value = Decimal(str(value))
        quantize_string = '0.' + '0' * places if places > 0 else '1'
        rounded = decimal_value.quantize(Decimal(quantize_string), rounding=ROUND_HALF_UP)
        return float(rounded)
