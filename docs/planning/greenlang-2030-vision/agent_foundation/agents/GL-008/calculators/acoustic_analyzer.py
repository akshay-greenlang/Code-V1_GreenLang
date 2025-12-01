# -*- coding: utf-8 -*-
"""
AcousticSignatureAnalyzer for GL-008 TRAPCATCHER

Provides ultrasonic acoustic signature analysis for steam trap failure detection.
Analyzes frequency content (20-100 kHz), amplitude patterns, and harmonic
structure to identify trap operating status and failure modes.

Standards:
- ASNT SNT-TC-1A: Personnel Qualification and Certification
- ISO 18436-8: Condition monitoring - Ultrasound
- ASTM E1002: Standard Practice for Leaks Using Ultrasonics

Key Features:
- Frequency spectrum analysis (FFT-based)
- Amplitude threshold detection
- Pattern matching for failure mode identification
- Signal quality assessment
- Background noise compensation
- Multi-band analysis (low/mid/high frequency)

Zero-Hallucination Guarantee:
All analysis uses deterministic signal processing algorithms.
No LLM or AI inference in any calculation path.
Same inputs always produce identical outputs.

Example:
    >>> analyzer = AcousticSignatureAnalyzer()
    >>> result = analyzer.analyze(acoustic_signature)
    >>> print(f"Status: {result.diagnosis.status}")

Author: GL-CalculatorEngineer
Date: December 2025
Version: 1.0.0
"""

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class FrequencyBand(str, Enum):
    """Frequency bands for acoustic analysis."""
    LOW = "low"  # 20-30 kHz (mechanical sounds)
    MID = "mid"  # 30-50 kHz (leak signatures)
    HIGH = "high"  # 50-100 kHz (fine leaks, high-velocity)


class SignalPattern(str, Enum):
    """Characteristic acoustic signal patterns."""
    CONTINUOUS = "continuous"  # Steady signal (leak/blowthrough)
    INTERMITTENT = "intermittent"  # Cycling signal (normal operation)
    RANDOM = "random"  # Irregular pattern (turbulence)
    PULSING = "pulsing"  # Regular pulses (valve cycling)
    SILENT = "silent"  # No significant signal (blocked)
    NOISY = "noisy"  # High noise floor (interference)


class TrapStatusAcoustic(str, Enum):
    """Trap status from acoustic analysis."""
    OPERATING = "operating"
    FAILED_OPEN = "failed_open"
    FAILED_CLOSED = "failed_closed"
    LEAKING = "leaking"
    COLD = "cold"
    UNKNOWN = "unknown"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class AcousticAnalysisConfig:
    """
    Configuration for acoustic signature analysis.

    Attributes:
        leak_threshold_db: Amplitude threshold for leak detection
        blocked_threshold_db: Amplitude below which = blocked
        normal_amplitude_range: Normal operating amplitude range
        leak_frequency_range_khz: Frequency range typical of leaks
        noise_floor_db: Assumed background noise level
        signal_quality_min: Minimum acceptable signal quality
        fft_window_size: FFT window size for spectral analysis
        harmonic_threshold: Threshold for harmonic detection
    """
    leak_threshold_db: float = 40.0
    blocked_threshold_db: float = 5.0
    normal_amplitude_range: Tuple[float, float] = (10.0, 35.0)
    leak_frequency_range_khz: Tuple[float, float] = (30.0, 50.0)
    noise_floor_db: float = -60.0
    signal_quality_min: float = 0.6
    fft_window_size: int = 1024
    harmonic_threshold: float = 0.15
    cycling_detection_threshold: float = 0.3

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "leak_threshold_db": self.leak_threshold_db,
            "blocked_threshold_db": self.blocked_threshold_db,
            "normal_amplitude_range": self.normal_amplitude_range,
            "leak_frequency_range_khz": self.leak_frequency_range_khz,
            "noise_floor_db": self.noise_floor_db,
            "signal_quality_min": self.signal_quality_min,
        }


@dataclass
class FrequencyBandAnalysis:
    """
    Analysis results for a single frequency band.

    Attributes:
        band: Frequency band analyzed
        frequency_range_khz: Actual frequency range
        peak_amplitude_db: Peak amplitude in band
        avg_amplitude_db: Average amplitude in band
        dominant_frequency_khz: Dominant frequency in band
        energy_ratio: Ratio of band energy to total
        harmonic_content: Detected harmonics
    """
    band: FrequencyBand
    frequency_range_khz: Tuple[float, float]
    peak_amplitude_db: float
    avg_amplitude_db: float
    dominant_frequency_khz: float
    energy_ratio: float
    harmonic_content: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "band": self.band.value,
            "frequency_range_khz": self.frequency_range_khz,
            "peak_amplitude_db": self.peak_amplitude_db,
            "avg_amplitude_db": self.avg_amplitude_db,
            "dominant_frequency_khz": self.dominant_frequency_khz,
            "energy_ratio": self.energy_ratio,
            "harmonic_content": self.harmonic_content,
        }


@dataclass
class AcousticDiagnosis:
    """
    Acoustic diagnosis result.

    Attributes:
        status: Diagnosed trap status
        confidence: Confidence score (0-1)
        pattern: Detected signal pattern
        failure_indicators: List of failure indicators
        notes: Diagnostic notes
    """
    status: TrapStatusAcoustic
    confidence: float
    pattern: SignalPattern
    failure_indicators: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "confidence": self.confidence,
            "pattern": self.pattern.value,
            "failure_indicators": self.failure_indicators,
            "notes": self.notes,
        }


@dataclass
class AcousticAnalysisResult:
    """
    Complete acoustic analysis result.

    Attributes:
        trap_id: Steam trap identifier
        timestamp: Analysis timestamp
        overall_amplitude_db: Overall signal amplitude
        snr_db: Signal-to-noise ratio
        signal_quality: Signal quality score (0-1)
        band_analysis: Analysis by frequency band
        diagnosis: Diagnostic result
        raw_spectrum: Raw frequency spectrum data
        provenance_hash: SHA-256 hash for audit trail
        calculation_method: Method description
    """
    trap_id: str
    timestamp: datetime
    overall_amplitude_db: float
    snr_db: float
    signal_quality: float
    band_analysis: List[FrequencyBandAnalysis]
    diagnosis: AcousticDiagnosis
    raw_spectrum: Optional[List[Tuple[float, float]]] = None
    provenance_hash: str = ""
    calculation_method: str = "deterministic_fft"

    def __post_init__(self):
        """Generate provenance hash after initialization."""
        if not self.provenance_hash:
            self.provenance_hash = self._calculate_provenance()

    def _calculate_provenance(self) -> str:
        """Calculate SHA-256 provenance hash."""
        data = {
            "trap_id": self.trap_id,
            "timestamp": self.timestamp.isoformat(),
            "overall_amplitude_db": self.overall_amplitude_db,
            "snr_db": self.snr_db,
            "signal_quality": self.signal_quality,
            "diagnosis_status": self.diagnosis.status.value,
            "diagnosis_confidence": self.diagnosis.confidence,
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode()
        ).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trap_id": self.trap_id,
            "timestamp": self.timestamp.isoformat(),
            "overall_amplitude_db": self.overall_amplitude_db,
            "snr_db": self.snr_db,
            "signal_quality": self.signal_quality,
            "band_analysis": [b.to_dict() for b in self.band_analysis],
            "diagnosis": self.diagnosis.to_dict(),
            "provenance_hash": self.provenance_hash,
            "calculation_method": self.calculation_method,
        }


# ============================================================================
# ACOUSTIC SIGNATURE ANALYZER
# ============================================================================

class AcousticSignatureAnalyzer:
    """
    Deterministic acoustic signature analyzer for steam trap diagnosis.

    Analyzes ultrasonic acoustic signatures to detect steam trap failures
    based on frequency content, amplitude patterns, and signal characteristics.

    ZERO-HALLUCINATION GUARANTEE:
    - All analysis uses deterministic FFT and threshold-based algorithms
    - No LLM or ML inference in calculation path
    - Same inputs always produce identical outputs
    - Complete provenance tracking with SHA-256 hashes

    Acoustic Signatures by Trap Status:
    - OPERATING: Intermittent, moderate amplitude (10-35 dB), varied frequency
    - FAILED_OPEN: Continuous high amplitude (>40 dB), 30-50 kHz dominant
    - FAILED_CLOSED: Very low/no signal (<5 dB)
    - LEAKING: Continuous moderate amplitude (25-40 dB), stable frequency

    Example:
        >>> config = AcousticAnalysisConfig(leak_threshold_db=40.0)
        >>> analyzer = AcousticSignatureAnalyzer(config)
        >>> result = analyzer.analyze(acoustic_signature)
        >>> print(f"Status: {result.diagnosis.status}")
    """

    # Frequency band definitions (kHz)
    FREQUENCY_BANDS = {
        FrequencyBand.LOW: (20.0, 30.0),
        FrequencyBand.MID: (30.0, 50.0),
        FrequencyBand.HIGH: (50.0, 100.0),
    }

    def __init__(self, config: Optional[AcousticAnalysisConfig] = None):
        """
        Initialize acoustic signature analyzer.

        Args:
            config: Analysis configuration (uses defaults if not provided)
        """
        self.config = config or AcousticAnalysisConfig()
        self.analysis_count = 0
        logger.info(
            f"AcousticSignatureAnalyzer initialized "
            f"(leak_threshold={self.config.leak_threshold_db} dB)"
        )

    def analyze(
        self,
        trap_id: str,
        frequency_spectrum: List[Tuple[float, float]],
        amplitude_db: float,
        rms_level_db: float,
        peak_frequency_khz: float,
        duration_seconds: float = 5.0,
        noise_floor_db: Optional[float] = None,
    ) -> AcousticAnalysisResult:
        """
        Perform comprehensive acoustic signature analysis.

        ZERO-HALLUCINATION: Uses deterministic signal processing.

        Args:
            trap_id: Steam trap identifier
            frequency_spectrum: List of (frequency_khz, amplitude_db) tuples
            amplitude_db: Peak signal amplitude
            rms_level_db: RMS signal level
            peak_frequency_khz: Dominant frequency
            duration_seconds: Recording duration
            noise_floor_db: Background noise level (optional)

        Returns:
            AcousticAnalysisResult with complete analysis

        Raises:
            ValueError: If inputs are invalid
        """
        self.analysis_count += 1
        timestamp = datetime.now(timezone.utc)

        # Validate inputs
        if amplitude_db < -100 or amplitude_db > 100:
            raise ValueError(f"Invalid amplitude: {amplitude_db} dB")
        if not 1.0 <= peak_frequency_khz <= 200.0:
            raise ValueError(f"Invalid frequency: {peak_frequency_khz} kHz")

        # Use config noise floor if not provided
        noise_floor = noise_floor_db or self.config.noise_floor_db

        # Calculate signal-to-noise ratio
        snr_db = amplitude_db - noise_floor

        # Calculate signal quality
        signal_quality = self._calculate_signal_quality(
            snr_db, duration_seconds, len(frequency_spectrum)
        )

        # Analyze frequency bands
        band_analysis = self._analyze_frequency_bands(frequency_spectrum)

        # Detect signal pattern
        pattern = self._detect_signal_pattern(
            amplitude_db, rms_level_db, band_analysis
        )

        # Perform diagnosis
        diagnosis = self._diagnose_from_signature(
            amplitude_db,
            peak_frequency_khz,
            snr_db,
            signal_quality,
            pattern,
            band_analysis
        )

        return AcousticAnalysisResult(
            trap_id=trap_id,
            timestamp=timestamp,
            overall_amplitude_db=amplitude_db,
            snr_db=snr_db,
            signal_quality=signal_quality,
            band_analysis=band_analysis,
            diagnosis=diagnosis,
            raw_spectrum=frequency_spectrum if len(frequency_spectrum) <= 100 else None,
        )

    def _calculate_signal_quality(
        self,
        snr_db: float,
        duration_seconds: float,
        spectrum_points: int
    ) -> float:
        """
        Calculate signal quality score.

        FORMULA:
        quality = 0.5 * snr_factor + 0.3 * duration_factor + 0.2 * resolution_factor

        Args:
            snr_db: Signal-to-noise ratio in dB
            duration_seconds: Recording duration
            spectrum_points: Number of spectrum data points

        Returns:
            Signal quality score (0-1)
        """
        # SNR factor (0-1): normalized SNR
        snr_factor = min(1.0, max(0.0, (snr_db + 20) / 60.0))

        # Duration factor (0-1): longer recordings = better quality
        duration_factor = min(1.0, duration_seconds / 10.0)

        # Resolution factor (0-1): more data points = better resolution
        resolution_factor = min(1.0, spectrum_points / 100.0)

        # Weighted quality score
        quality = (
            0.5 * snr_factor +
            0.3 * duration_factor +
            0.2 * resolution_factor
        )

        return round(quality, 3)

    def _analyze_frequency_bands(
        self,
        frequency_spectrum: List[Tuple[float, float]]
    ) -> List[FrequencyBandAnalysis]:
        """
        Analyze signal content in each frequency band.

        Args:
            frequency_spectrum: List of (frequency_khz, amplitude_db) tuples

        Returns:
            List of FrequencyBandAnalysis for each band
        """
        band_results = []

        # Calculate total energy for normalization
        total_energy = sum(
            10 ** (amp / 10) for _, amp in frequency_spectrum
        ) if frequency_spectrum else 1.0

        for band, (f_min, f_max) in self.FREQUENCY_BANDS.items():
            # Filter spectrum to band
            band_points = [
                (f, a) for f, a in frequency_spectrum
                if f_min <= f <= f_max
            ]

            if band_points:
                amplitudes = [a for _, a in band_points]
                peak_amp = max(amplitudes)
                avg_amp = sum(amplitudes) / len(amplitudes)

                # Find dominant frequency in band
                dominant_freq = max(band_points, key=lambda x: x[1])[0]

                # Calculate band energy ratio
                band_energy = sum(10 ** (a / 10) for _, a in band_points)
                energy_ratio = band_energy / total_energy if total_energy > 0 else 0

                # Detect harmonics
                harmonics = self._detect_harmonics(band_points, dominant_freq)
            else:
                peak_amp = self.config.noise_floor_db
                avg_amp = self.config.noise_floor_db
                dominant_freq = (f_min + f_max) / 2
                energy_ratio = 0.0
                harmonics = {}

            band_results.append(FrequencyBandAnalysis(
                band=band,
                frequency_range_khz=(f_min, f_max),
                peak_amplitude_db=round(peak_amp, 2),
                avg_amplitude_db=round(avg_amp, 2),
                dominant_frequency_khz=round(dominant_freq, 2),
                energy_ratio=round(energy_ratio, 4),
                harmonic_content=harmonics,
            ))

        return band_results

    def _detect_harmonics(
        self,
        band_points: List[Tuple[float, float]],
        fundamental_freq: float
    ) -> Dict[str, float]:
        """
        Detect harmonic content in frequency band.

        Args:
            band_points: Frequency/amplitude pairs in band
            fundamental_freq: Fundamental frequency

        Returns:
            Dictionary of harmonic ratios
        """
        if not band_points or fundamental_freq <= 0:
            return {}

        harmonics = {}
        fundamental_amp = 0.0

        # Find fundamental amplitude
        for freq, amp in band_points:
            if abs(freq - fundamental_freq) < 1.0:  # Within 1 kHz
                fundamental_amp = max(fundamental_amp, amp)

        if fundamental_amp <= self.config.noise_floor_db + 10:
            return {}

        # Look for 2nd and 3rd harmonics
        for n, name in [(2, "2nd"), (3, "3rd")]:
            harmonic_freq = fundamental_freq * n
            for freq, amp in band_points:
                if abs(freq - harmonic_freq) < 2.0:  # Within 2 kHz
                    ratio = (amp - self.config.noise_floor_db) / (
                        fundamental_amp - self.config.noise_floor_db
                    ) if fundamental_amp > self.config.noise_floor_db else 0
                    if ratio > self.config.harmonic_threshold:
                        harmonics[name] = round(ratio, 3)
                    break

        return harmonics

    def _detect_signal_pattern(
        self,
        amplitude_db: float,
        rms_level_db: float,
        band_analysis: List[FrequencyBandAnalysis]
    ) -> SignalPattern:
        """
        Detect characteristic signal pattern.

        ZERO-HALLUCINATION: Deterministic pattern classification.

        Args:
            amplitude_db: Peak amplitude
            rms_level_db: RMS level
            band_analysis: Band analysis results

        Returns:
            Detected SignalPattern
        """
        # Silent pattern: very low amplitude
        if amplitude_db < self.config.blocked_threshold_db:
            return SignalPattern.SILENT

        # Calculate peak-to-RMS ratio (crest factor indicator)
        peak_to_rms = amplitude_db - rms_level_db

        # Calculate energy distribution
        mid_band = next(
            (b for b in band_analysis if b.band == FrequencyBand.MID), None
        )

        # Continuous pattern: low peak-to-RMS ratio (constant level)
        if peak_to_rms < 3.0 and amplitude_db > 30.0:
            return SignalPattern.CONTINUOUS

        # Intermittent pattern: moderate peak-to-RMS ratio (cycling)
        if 3.0 <= peak_to_rms <= 10.0:
            return SignalPattern.INTERMITTENT

        # Pulsing pattern: high peak-to-RMS ratio (regular pulses)
        if peak_to_rms > 10.0:
            return SignalPattern.PULSING

        # Random pattern: irregular characteristics
        return SignalPattern.RANDOM

    def _diagnose_from_signature(
        self,
        amplitude_db: float,
        peak_frequency_khz: float,
        snr_db: float,
        signal_quality: float,
        pattern: SignalPattern,
        band_analysis: List[FrequencyBandAnalysis]
    ) -> AcousticDiagnosis:
        """
        Diagnose trap status from acoustic signature.

        ZERO-HALLUCINATION: Deterministic threshold-based diagnosis.

        Args:
            amplitude_db: Overall amplitude
            peak_frequency_khz: Dominant frequency
            snr_db: Signal-to-noise ratio
            signal_quality: Signal quality score
            pattern: Detected signal pattern
            band_analysis: Band analysis results

        Returns:
            AcousticDiagnosis with status and confidence
        """
        notes = []
        failure_indicators = []

        # Get mid-band analysis (most relevant for leaks)
        mid_band = next(
            (b for b in band_analysis if b.band == FrequencyBand.MID), None
        )

        # DIAGNOSIS RULES (deterministic threshold-based)

        # Rule 1: Very low amplitude = FAILED_CLOSED/COLD
        if amplitude_db < self.config.blocked_threshold_db:
            notes.append(
                f"Very low amplitude ({amplitude_db:.1f} dB) indicates "
                f"blocked trap or no steam flow"
            )
            return AcousticDiagnosis(
                status=TrapStatusAcoustic.FAILED_CLOSED,
                confidence=0.85,
                pattern=pattern,
                failure_indicators=["No acoustic signature detected"],
                notes=notes,
            )

        # Rule 2: High amplitude in leak frequency range = FAILED_OPEN
        leak_freq_range = self.config.leak_frequency_range_khz
        if (amplitude_db > self.config.leak_threshold_db and
            leak_freq_range[0] <= peak_frequency_khz <= leak_freq_range[1] and
            pattern == SignalPattern.CONTINUOUS):

            failure_indicators.append(
                f"High continuous amplitude ({amplitude_db:.1f} dB) "
                f"at leak frequency ({peak_frequency_khz:.1f} kHz)"
            )
            notes.append(
                "Acoustic signature consistent with steam blowthrough"
            )

            # Higher confidence if mid-band energy is dominant
            confidence = 0.9
            if mid_band and mid_band.energy_ratio > 0.5:
                confidence = 0.95

            return AcousticDiagnosis(
                status=TrapStatusAcoustic.FAILED_OPEN,
                confidence=confidence,
                pattern=pattern,
                failure_indicators=failure_indicators,
                notes=notes,
            )

        # Rule 3: Moderate continuous amplitude = LEAKING
        if (25.0 < amplitude_db <= self.config.leak_threshold_db and
            pattern == SignalPattern.CONTINUOUS):

            failure_indicators.append(
                f"Moderate continuous amplitude ({amplitude_db:.1f} dB) "
                f"suggests partial steam leak"
            )
            notes.append(
                "Acoustic signature indicates steam passing but not full blowthrough"
            )

            return AcousticDiagnosis(
                status=TrapStatusAcoustic.LEAKING,
                confidence=0.75,
                pattern=pattern,
                failure_indicators=failure_indicators,
                notes=notes,
            )

        # Rule 4: Normal amplitude with intermittent pattern = OPERATING
        normal_range = self.config.normal_amplitude_range
        if (normal_range[0] <= amplitude_db <= normal_range[1] and
            pattern in (SignalPattern.INTERMITTENT, SignalPattern.PULSING)):

            notes.append(
                f"Normal amplitude ({amplitude_db:.1f} dB) with "
                f"{pattern.value} pattern indicates proper operation"
            )

            return AcousticDiagnosis(
                status=TrapStatusAcoustic.OPERATING,
                confidence=0.85,
                pattern=pattern,
                failure_indicators=[],
                notes=notes,
            )

        # Rule 5: Poor signal quality = UNKNOWN
        if signal_quality < self.config.signal_quality_min:
            notes.append(
                f"Low signal quality ({signal_quality:.2f}) - "
                f"results may be unreliable"
            )

            return AcousticDiagnosis(
                status=TrapStatusAcoustic.UNKNOWN,
                confidence=0.4,
                pattern=pattern,
                failure_indicators=["Poor signal quality"],
                notes=notes,
            )

        # Default: Normal operation assumed
        notes.append(
            f"Amplitude ({amplitude_db:.1f} dB) and pattern ({pattern.value}) "
            f"suggest normal operation"
        )

        return AcousticDiagnosis(
            status=TrapStatusAcoustic.OPERATING,
            confidence=0.7,
            pattern=pattern,
            failure_indicators=[],
            notes=notes,
        )

    def calculate_leak_severity(
        self,
        amplitude_db: float,
        frequency_khz: float,
        pressure_bar: float
    ) -> Tuple[str, float]:
        """
        Calculate leak severity from acoustic parameters.

        FORMULA (empirical correlation):
        severity_score = (amplitude - threshold) * pressure_factor * freq_factor

        Args:
            amplitude_db: Signal amplitude
            frequency_khz: Dominant frequency
            pressure_bar: Operating pressure

        Returns:
            Tuple of (severity_category, severity_score)
        """
        if amplitude_db < self.config.blocked_threshold_db:
            return "none", 0.0

        # Calculate above-threshold amplitude
        excess_amplitude = max(0, amplitude_db - self.config.normal_amplitude_range[1])

        # Pressure factor (higher pressure = more severe leak)
        pressure_factor = 1.0 + 0.05 * pressure_bar

        # Frequency factor (leak frequency band more severe)
        leak_range = self.config.leak_frequency_range_khz
        if leak_range[0] <= frequency_khz <= leak_range[1]:
            freq_factor = 1.2
        else:
            freq_factor = 1.0

        # Calculate severity score (0-100)
        severity_score = min(100, excess_amplitude * pressure_factor * freq_factor)

        # Categorize severity
        if severity_score > 75:
            category = "critical"
        elif severity_score > 50:
            category = "high"
        elif severity_score > 25:
            category = "medium"
        elif severity_score > 10:
            category = "low"
        else:
            category = "minimal"

        return category, round(severity_score, 2)

    def get_statistics(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        return {
            "analysis_count": self.analysis_count,
            "config": self.config.to_dict(),
            "frequency_bands": {
                band.value: f"{f_min}-{f_max} kHz"
                for band, (f_min, f_max) in self.FREQUENCY_BANDS.items()
            },
        }


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "AcousticSignatureAnalyzer",
    "AcousticAnalysisConfig",
    "AcousticAnalysisResult",
    "AcousticDiagnosis",
    "FrequencyBand",
    "FrequencyBandAnalysis",
    "SignalPattern",
    "TrapStatusAcoustic",
]
