# -*- coding: utf-8 -*-
"""
GL-013 PredictMaint Agent - Vibration Analysis Module

This module implements vibration analysis including FFT spectrum analysis
for rotating equipment fault detection. Vibration analysis is the most
widely used predictive maintenance technique.

Key analysis capabilities:
- Overall vibration levels (velocity, acceleration, displacement)
- ISO 10816 zone classification
- FFT spectrum analysis
- Bearing defect frequencies (BPFO, BPFI, BSF, FTF)
- Imbalance, misalignment, and looseness detection
- Trend analysis

All calculations are DETERMINISTIC with provenance tracking.
No ML/LLM in calculation path - ZERO HALLUCINATION.

Example:
    >>> from greenlang.agents.process_heat.gl_013_predictive_maintenance.vibration import (
    ...     VibrationAnalyzer
    ... )
    >>> analyzer = VibrationAnalyzer(config)
    >>> result = analyzer.analyze(reading)
    >>> print(f"ISO Zone: {result.iso_zone}")
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging
import math

from greenlang.agents.process_heat.gl_013_predictive_maintenance.config import (
    AlertSeverity,
    VibrationThresholds,
    PredictiveMaintenanceConfig,
)
from greenlang.agents.process_heat.gl_013_predictive_maintenance.schemas import (
    TrendDirection,
    VibrationAnalysisResult,
    VibrationReading,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Bearing defect frequency multipliers (relative to shaft speed)
# These are approximate and depend on bearing geometry
BEARING_FREQ_MULTIPLIERS = {
    "BPFO": 0.4,  # Ball Pass Frequency Outer Race
    "BPFI": 0.6,  # Ball Pass Frequency Inner Race
    "BSF": 0.4,   # Ball Spin Frequency
    "FTF": 0.4,   # Fundamental Train Frequency (cage)
}

# Common fault frequency relationships
FAULT_FREQUENCIES = {
    "imbalance": [1.0],  # 1x RPM
    "misalignment_angular": [1.0, 2.0],  # 1x, 2x RPM axial
    "misalignment_parallel": [2.0],  # 2x RPM radial
    "mechanical_looseness": [0.5, 1.0, 2.0, 3.0],  # Sub and super harmonics
    "gear_mesh": None,  # Teeth * RPM
    "blade_pass": None,  # Blades * RPM
    "belt_freq": None,  # Belt length dependent
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class BearingGeometry:
    """Bearing geometry for defect frequency calculation."""
    bpfo: float  # Ball Pass Frequency Outer (per revolution)
    bpfi: float  # Ball Pass Frequency Inner (per revolution)
    bsf: float   # Ball Spin Frequency (per revolution)
    ftf: float   # Fundamental Train Frequency (per revolution)


@dataclass
class SpectralPeak:
    """Detected spectral peak."""
    frequency_hz: float
    amplitude: float
    order: float  # Multiple of running speed
    probable_source: str
    severity: str


# =============================================================================
# VIBRATION ANALYZER CLASS
# =============================================================================

class VibrationAnalyzer:
    """
    Vibration Analyzer for rotating equipment fault detection.

    This class performs comprehensive vibration analysis including:
    - ISO 10816 zone classification
    - FFT spectrum analysis
    - Bearing defect detection
    - Mechanical fault diagnosis

    Analysis follows industry standards:
    - ISO 10816 for machine vibration severity
    - ISO 13373 for condition monitoring
    - ISO 15242 for rolling element bearings

    All calculations are DETERMINISTIC - ZERO HALLUCINATION.

    Attributes:
        config: Equipment configuration
        thresholds: Vibration alarm thresholds
        bearing_geometry: Bearing defect frequencies

    Example:
        >>> config = PredictiveMaintenanceConfig(equipment_id="PUMP-001")
        >>> analyzer = VibrationAnalyzer(config)
        >>> result = analyzer.analyze(vibration_reading)
        >>> print(f"ISO Zone: {result.iso_zone}")
    """

    def __init__(
        self,
        config: Optional[PredictiveMaintenanceConfig] = None,
        thresholds: Optional[VibrationThresholds] = None,
    ) -> None:
        """
        Initialize vibration analyzer.

        Args:
            config: Equipment configuration
            thresholds: Vibration alarm thresholds
        """
        self.config = config
        self.thresholds = thresholds or VibrationThresholds()

        # Set bearing geometry from config if available
        self.bearing_geometry: Optional[BearingGeometry] = None
        if config:
            if all([config.bearing_bpfo, config.bearing_bpfi,
                    config.bearing_bsf, config.bearing_ftf]):
                self.bearing_geometry = BearingGeometry(
                    bpfo=config.bearing_bpfo,
                    bpfi=config.bearing_bpfi,
                    bsf=config.bearing_bsf,
                    ftf=config.bearing_ftf,
                )

        logger.info("VibrationAnalyzer initialized")

    def analyze(
        self,
        reading: VibrationReading,
        history: Optional[List[VibrationReading]] = None,
    ) -> VibrationAnalysisResult:
        """
        Perform comprehensive vibration analysis.

        Args:
            reading: Current vibration reading
            history: Historical readings for trend analysis

        Returns:
            VibrationAnalysisResult with fault diagnostics
        """
        start_time = datetime.now(timezone.utc)
        logger.info(f"Analyzing vibration: sensor={reading.sensor_id}")

        # Calculate running speed in Hz
        running_speed_hz = reading.operating_speed_rpm / 60.0

        # ISO 10816 zone classification
        iso_zone = self._classify_iso_10816(reading.velocity_rms_mm_s)

        # Initialize fault flags
        bearing_detected = False
        bearing_type: Optional[str] = None
        imbalance_detected = False
        imbalance_severity: Optional[str] = None
        misalignment_detected = False
        misalignment_type: Optional[str] = None
        looseness_detected = False

        # Spectrum analysis if available
        harmonics: List[Dict[str, float]] = []
        dominant_frequency = running_speed_hz
        dominant_amplitude = reading.velocity_rms_mm_s

        if reading.spectrum and reading.frequency_resolution_hz:
            # Analyze spectrum
            peaks = self._find_spectral_peaks(
                reading.spectrum,
                reading.frequency_resolution_hz,
                running_speed_hz,
            )

            # Detect faults from peaks
            bearing_detected, bearing_type = self._detect_bearing_defects(
                peaks, running_speed_hz
            )

            imbalance_detected, imbalance_severity = self._detect_imbalance(
                peaks, running_speed_hz
            )

            misalignment_detected, misalignment_type = self._detect_misalignment(
                peaks, running_speed_hz, reading.orientation
            )

            looseness_detected = self._detect_looseness(peaks, running_speed_hz)

            # Convert peaks to harmonics list
            harmonics = [
                {
                    "frequency_hz": p.frequency_hz,
                    "amplitude": p.amplitude,
                    "order": p.order,
                    "source": p.probable_source,
                }
                for p in peaks[:10]  # Top 10 peaks
            ]

            # Find dominant frequency
            if peaks:
                dominant = max(peaks, key=lambda p: p.amplitude)
                dominant_frequency = dominant.frequency_hz
                dominant_amplitude = dominant.amplitude

        # Trend analysis
        trend_direction = TrendDirection.STABLE
        trend_rate: Optional[float] = None

        if history and len(history) >= 3:
            trend_direction, trend_rate = self._analyze_trend(reading, history)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            iso_zone,
            bearing_detected,
            bearing_type,
            imbalance_detected,
            imbalance_severity,
            misalignment_detected,
            misalignment_type,
            looseness_detected,
            trend_direction,
        )

        # Calculate provenance
        provenance_hash = self._calculate_provenance(reading, iso_zone)

        logger.info(
            f"Vibration analysis complete: ISO zone={iso_zone.value}, "
            f"velocity={reading.velocity_rms_mm_s:.2f} mm/s"
        )

        return VibrationAnalysisResult(
            sensor_id=reading.sensor_id,
            timestamp=reading.timestamp,
            overall_velocity_mm_s=reading.velocity_rms_mm_s,
            overall_acceleration_g=reading.acceleration_rms_g,
            overall_displacement_um=reading.displacement_um,
            iso_zone=iso_zone,
            dominant_frequency_hz=dominant_frequency,
            dominant_amplitude=dominant_amplitude,
            harmonics_detected=harmonics,
            bearing_defect_detected=bearing_detected,
            bearing_defect_type=bearing_type,
            imbalance_detected=imbalance_detected,
            imbalance_severity=imbalance_severity,
            misalignment_detected=misalignment_detected,
            misalignment_type=misalignment_type,
            looseness_detected=looseness_detected,
            trend_direction=trend_direction,
            trend_rate_pct_per_day=trend_rate,
            recommendations=recommendations,
        )

    def _classify_iso_10816(self, velocity_mm_s: float) -> AlertSeverity:
        """
        Classify vibration per ISO 10816-3 for industrial machines.

        ISO 10816-3 applies to machines mounted on rigid foundations
        with power >15kW and speeds 120-15000 RPM.

        Args:
            velocity_mm_s: RMS velocity in mm/s

        Returns:
            AlertSeverity (ISO zone A/B/C/D)
        """
        if velocity_mm_s <= self.thresholds.velocity_good_mm_s:
            return AlertSeverity.GOOD  # Zone A
        elif velocity_mm_s <= self.thresholds.velocity_acceptable_mm_s:
            return AlertSeverity.ACCEPTABLE  # Zone B
        elif velocity_mm_s <= self.thresholds.velocity_unsatisfactory_mm_s:
            return AlertSeverity.UNSATISFACTORY  # Zone C
        else:
            return AlertSeverity.UNACCEPTABLE  # Zone D

    def _find_spectral_peaks(
        self,
        spectrum: List[float],
        freq_resolution: float,
        running_speed_hz: float,
    ) -> List[SpectralPeak]:
        """
        Find significant peaks in FFT spectrum.

        Args:
            spectrum: FFT amplitude spectrum
            freq_resolution: Frequency resolution (Hz)
            running_speed_hz: Running speed in Hz

        Returns:
            List of detected peaks
        """
        peaks = []

        if not spectrum or len(spectrum) < 10:
            return peaks

        # Find local maxima
        threshold = sum(spectrum) / len(spectrum) * 3  # 3x average

        for i in range(1, len(spectrum) - 1):
            # Local maximum detection
            if (spectrum[i] > spectrum[i-1] and
                spectrum[i] > spectrum[i+1] and
                spectrum[i] > threshold):

                freq = i * freq_resolution
                amplitude = spectrum[i]

                # Calculate order (multiple of running speed)
                order = freq / running_speed_hz if running_speed_hz > 0 else 0

                # Identify probable source
                source = self._identify_peak_source(order, freq, running_speed_hz)

                # Determine severity based on amplitude
                severity = "low"
                if amplitude > threshold * 3:
                    severity = "high"
                elif amplitude > threshold * 2:
                    severity = "medium"

                peaks.append(SpectralPeak(
                    frequency_hz=freq,
                    amplitude=amplitude,
                    order=round(order, 2),
                    probable_source=source,
                    severity=severity,
                ))

        # Sort by amplitude (descending)
        peaks.sort(key=lambda p: p.amplitude, reverse=True)

        return peaks[:20]  # Return top 20 peaks

    def _identify_peak_source(
        self,
        order: float,
        frequency: float,
        running_speed_hz: float,
    ) -> str:
        """
        Identify probable source of spectral peak.

        Args:
            order: Frequency order (multiple of RPM)
            frequency: Peak frequency in Hz
            running_speed_hz: Running speed in Hz

        Returns:
            Probable source description
        """
        # Check common orders
        if abs(order - 1.0) < 0.1:
            return "1x RPM (imbalance/misalignment)"
        elif abs(order - 2.0) < 0.1:
            return "2x RPM (misalignment/looseness)"
        elif abs(order - 0.5) < 0.1:
            return "0.5x RPM (oil whirl/looseness)"
        elif abs(order - 3.0) < 0.1:
            return "3x RPM (misalignment)"
        elif 0.35 < order < 0.48:
            return "Possible cage frequency (FTF)"

        # Check bearing frequencies if geometry available
        if self.bearing_geometry:
            if abs(order - self.bearing_geometry.bpfo) < 0.1:
                return "BPFO (outer race defect)"
            elif abs(order - self.bearing_geometry.bpfi) < 0.1:
                return "BPFI (inner race defect)"
            elif abs(order - self.bearing_geometry.bsf) < 0.1:
                return "BSF (ball/roller defect)"
            elif abs(order - self.bearing_geometry.ftf) < 0.1:
                return "FTF (cage defect)"

        # Check for sub-synchronous
        if order < 1.0:
            return f"{order:.2f}x RPM (sub-synchronous)"

        # Check for harmonics
        if abs(order - round(order)) < 0.1 and order > 3:
            return f"{int(round(order))}x RPM harmonic"

        return f"{order:.2f}x RPM (unknown)"

    def _detect_bearing_defects(
        self,
        peaks: List[SpectralPeak],
        running_speed_hz: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Detect bearing defects from spectral peaks.

        Args:
            peaks: Detected spectral peaks
            running_speed_hz: Running speed in Hz

        Returns:
            Tuple of (defect_detected, defect_type)
        """
        if not self.bearing_geometry:
            # Use approximate bearing frequencies
            bpfo_order = 0.4 * 8  # Approximate for 8-ball bearing
            bpfi_order = 0.6 * 8
            bsf_order = 0.4 * 8
            ftf_order = 0.4
        else:
            bpfo_order = self.bearing_geometry.bpfo
            bpfi_order = self.bearing_geometry.bpfi
            bsf_order = self.bearing_geometry.bsf
            ftf_order = self.bearing_geometry.ftf

        # Check for bearing frequencies with harmonics
        defect_types = []

        for peak in peaks:
            order = peak.order

            # BPFO and harmonics
            for mult in [1, 2, 3]:
                if abs(order - bpfo_order * mult) < 0.2:
                    defect_types.append("BPFO")
                    break

            # BPFI and harmonics
            for mult in [1, 2, 3]:
                if abs(order - bpfi_order * mult) < 0.2:
                    defect_types.append("BPFI")
                    break

            # BSF
            if abs(order - bsf_order) < 0.2:
                defect_types.append("BSF")

            # FTF (usually low amplitude)
            if abs(order - ftf_order) < 0.1:
                defect_types.append("FTF")

        if defect_types:
            # Return most common defect type
            primary_defect = max(set(defect_types), key=defect_types.count)
            return True, primary_defect

        return False, None

    def _detect_imbalance(
        self,
        peaks: List[SpectralPeak],
        running_speed_hz: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Detect imbalance from spectral signature.

        Imbalance shows as dominant 1x RPM in radial direction.

        Args:
            peaks: Detected spectral peaks
            running_speed_hz: Running speed in Hz

        Returns:
            Tuple of (detected, severity)
        """
        # Find 1x peak
        one_x_peaks = [p for p in peaks if abs(p.order - 1.0) < 0.1]

        if not one_x_peaks:
            return False, None

        one_x = one_x_peaks[0]

        # Check if 1x is dominant (should be largest peak for imbalance)
        if peaks and one_x.amplitude >= peaks[0].amplitude * 0.8:
            # Check for minimal harmonics (imbalance is mostly 1x)
            two_x_peaks = [p for p in peaks if abs(p.order - 2.0) < 0.1]

            if not two_x_peaks or two_x_peaks[0].amplitude < one_x.amplitude * 0.5:
                # Determine severity
                if one_x.severity == "high":
                    return True, "severe"
                elif one_x.severity == "medium":
                    return True, "moderate"
                else:
                    return True, "slight"

        return False, None

    def _detect_misalignment(
        self,
        peaks: List[SpectralPeak],
        running_speed_hz: float,
        orientation: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Detect misalignment from spectral signature.

        Angular misalignment: High 1x and 2x axial
        Parallel misalignment: High 2x radial

        Args:
            peaks: Detected spectral peaks
            running_speed_hz: Running speed in Hz
            orientation: Measurement orientation

        Returns:
            Tuple of (detected, type)
        """
        one_x_peaks = [p for p in peaks if abs(p.order - 1.0) < 0.1]
        two_x_peaks = [p for p in peaks if abs(p.order - 2.0) < 0.1]

        one_x_amp = one_x_peaks[0].amplitude if one_x_peaks else 0
        two_x_amp = two_x_peaks[0].amplitude if two_x_peaks else 0

        # Check for significant 2x
        if two_x_amp > 0:
            # Parallel misalignment: dominant 2x in radial
            if orientation in ["radial", "horizontal", "vertical"]:
                if two_x_amp > one_x_amp * 0.5:
                    return True, "parallel"

            # Angular misalignment: high 1x and 2x in axial
            if orientation == "axial":
                if one_x_amp > 0 and two_x_amp > one_x_amp * 0.3:
                    return True, "angular"

        # Combined misalignment
        if one_x_amp > 0 and two_x_amp > one_x_amp * 0.3:
            return True, "combined"

        return False, None

    def _detect_looseness(
        self,
        peaks: List[SpectralPeak],
        running_speed_hz: float,
    ) -> bool:
        """
        Detect mechanical looseness from spectral signature.

        Looseness shows as multiple harmonics and sub-harmonics.

        Args:
            peaks: Detected spectral peaks
            running_speed_hz: Running speed in Hz

        Returns:
            True if looseness detected
        """
        # Count harmonics
        harmonics = [
            p for p in peaks
            if abs(p.order - round(p.order)) < 0.1 and p.order <= 10
        ]

        # Count sub-harmonics (0.5x, etc.)
        sub_harmonics = [
            p for p in peaks
            if 0.4 < p.order < 0.6  # 0.5x region
        ]

        # Looseness typically shows multiple harmonics and 0.5x
        if len(harmonics) >= 4 and len(sub_harmonics) > 0:
            return True

        # Or many harmonics without clear primary fault
        if len(harmonics) >= 6:
            return True

        return False

    def _analyze_trend(
        self,
        current: VibrationReading,
        history: List[VibrationReading],
    ) -> Tuple[TrendDirection, Optional[float]]:
        """
        Analyze vibration trend from historical data.

        Args:
            current: Current reading
            history: Historical readings

        Returns:
            Tuple of (trend_direction, rate_pct_per_day)
        """
        # Sort history by timestamp
        sorted_history = sorted(history, key=lambda x: x.timestamp)

        # Get recent values
        recent = sorted_history[-10:] if len(sorted_history) >= 10 else sorted_history
        velocities = [r.velocity_rms_mm_s for r in recent]
        velocities.append(current.velocity_rms_mm_s)

        if len(velocities) < 3:
            return TrendDirection.STABLE, None

        # Calculate trend
        n = len(velocities)
        mean_y = sum(velocities) / n
        mean_x = (n - 1) / 2

        # Linear regression slope
        numerator = sum(
            (i - mean_x) * (v - mean_y)
            for i, v in enumerate(velocities)
        )
        denominator = sum((i - mean_x) ** 2 for i in range(n))

        if abs(denominator) < 1e-10:
            return TrendDirection.STABLE, None

        slope = numerator / denominator

        # Calculate rate per day (assuming 1 reading per day average)
        first_time = sorted_history[0].timestamp
        last_time = current.timestamp
        days = max(1, (last_time - first_time).total_seconds() / 86400)
        daily_rate = slope / (n / days) if days > 0 else 0

        # Rate as percentage of current value
        rate_pct = (daily_rate / current.velocity_rms_mm_s * 100
                    if current.velocity_rms_mm_s > 0 else 0)

        # Determine direction
        threshold_pct = self.thresholds.trend_increase_pct / 30  # Per day

        if rate_pct > threshold_pct:
            return TrendDirection.INCREASING, round(rate_pct, 2)
        elif rate_pct < -threshold_pct:
            return TrendDirection.DECREASING, round(rate_pct, 2)
        else:
            return TrendDirection.STABLE, round(rate_pct, 2)

    def _generate_recommendations(
        self,
        iso_zone: AlertSeverity,
        bearing_detected: bool,
        bearing_type: Optional[str],
        imbalance_detected: bool,
        imbalance_severity: Optional[str],
        misalignment_detected: bool,
        misalignment_type: Optional[str],
        looseness_detected: bool,
        trend_direction: TrendDirection,
    ) -> List[str]:
        """
        Generate maintenance recommendations.

        Args:
            Multiple analysis flags

        Returns:
            List of recommendations
        """
        recommendations = []

        # ISO zone recommendations
        if iso_zone == AlertSeverity.UNACCEPTABLE:
            recommendations.append(
                "IMMEDIATE ACTION REQUIRED - Vibration in Zone D. "
                "Risk of imminent failure. Plan emergency shutdown."
            )
        elif iso_zone == AlertSeverity.UNSATISFACTORY:
            recommendations.append(
                "Vibration in Zone C - Schedule maintenance soon. "
                "Continued operation may cause damage."
            )

        # Bearing recommendations
        if bearing_detected:
            defect_messages = {
                "BPFO": "Outer race defect detected - Plan bearing replacement",
                "BPFI": "Inner race defect detected - Plan bearing replacement (urgent)",
                "BSF": "Ball/roller defect detected - Monitor closely, plan replacement",
                "FTF": "Cage defect detected - Early stage, increase monitoring",
            }
            msg = defect_messages.get(
                bearing_type,
                f"Bearing defect ({bearing_type}) detected - Investigate"
            )
            recommendations.append(msg)

        # Imbalance recommendations
        if imbalance_detected:
            severity_actions = {
                "severe": "Field balance required urgently",
                "moderate": "Schedule balancing at next opportunity",
                "slight": "Monitor imbalance, balance at next overhaul",
            }
            recommendations.append(
                f"Imbalance detected ({imbalance_severity}). "
                f"{severity_actions.get(imbalance_severity, 'Investigate')}"
            )

        # Misalignment recommendations
        if misalignment_detected:
            type_actions = {
                "angular": "Check coupling condition and angular alignment",
                "parallel": "Check offset/parallel alignment",
                "combined": "Full alignment check required",
            }
            recommendations.append(
                f"Misalignment ({misalignment_type}) detected. "
                f"{type_actions.get(misalignment_type, 'Perform alignment check')}"
            )

        # Looseness recommendations
        if looseness_detected:
            recommendations.append(
                "Mechanical looseness detected. "
                "Check foundation bolts, coupling, and bearing fits."
            )

        # Trend recommendations
        if trend_direction == TrendDirection.INCREASING:
            recommendations.append(
                "Vibration trending upward - Increase monitoring frequency "
                "and investigate root cause."
            )

        # If no specific issues
        if not recommendations:
            if iso_zone == AlertSeverity.GOOD:
                recommendations.append(
                    "Vibration levels excellent (Zone A). "
                    "Continue routine monitoring."
                )
            else:
                recommendations.append(
                    "Vibration levels acceptable (Zone B). "
                    "Continue routine monitoring."
                )

        return recommendations

    def _calculate_provenance(
        self,
        reading: VibrationReading,
        iso_zone: AlertSeverity,
    ) -> str:
        """
        Calculate SHA-256 provenance hash.

        Args:
            reading: Vibration reading
            iso_zone: ISO classification

        Returns:
            SHA-256 hash string
        """
        provenance_str = (
            f"vibration|{reading.sensor_id}|{reading.timestamp.isoformat()}|"
            f"{reading.velocity_rms_mm_s:.6f}|{reading.acceleration_rms_g:.6f}|"
            f"{reading.operating_speed_rpm:.2f}|{iso_zone.value}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def calculate_bearing_frequencies(
        self,
        n_balls: int,
        ball_diameter_mm: float,
        pitch_diameter_mm: float,
        contact_angle_deg: float,
        shaft_speed_rpm: float,
    ) -> Dict[str, float]:
        """
        Calculate bearing defect frequencies from geometry.

        Args:
            n_balls: Number of balls/rollers
            ball_diameter_mm: Ball/roller diameter
            pitch_diameter_mm: Pitch diameter
            contact_angle_deg: Contact angle in degrees
            shaft_speed_rpm: Shaft speed in RPM

        Returns:
            Dictionary of defect frequencies in Hz
        """
        # Convert contact angle to radians
        contact_angle = math.radians(contact_angle_deg)

        # Calculate frequency factors (per shaft revolution)
        bd_pd = ball_diameter_mm / pitch_diameter_mm

        ftf_factor = 0.5 * (1 - bd_pd * math.cos(contact_angle))
        bpfo_factor = n_balls * ftf_factor
        bpfi_factor = n_balls * 0.5 * (1 + bd_pd * math.cos(contact_angle))
        bsf_factor = (pitch_diameter_mm / (2 * ball_diameter_mm) *
                      (1 - (bd_pd * math.cos(contact_angle)) ** 2))

        # Convert to Hz
        shaft_hz = shaft_speed_rpm / 60

        return {
            "FTF": ftf_factor * shaft_hz,
            "BPFO": bpfo_factor * shaft_hz,
            "BPFI": bpfi_factor * shaft_hz,
            "BSF": bsf_factor * shaft_hz,
        }
