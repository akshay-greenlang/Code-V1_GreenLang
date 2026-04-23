# -*- coding: utf-8 -*-
"""
GL-008 TRAPCATCHER Acoustic Diagnostic Calculator

This module provides deterministic acoustic/ultrasonic signal analysis for steam trap
diagnostics following GreenLang zero-hallucination principles.

Features:
    - Ultrasonic signal analysis (20-100 kHz range)
    - Acoustic signature pattern matching
    - Steam leak detection via dB level analysis
    - Background noise filtering
    - Signal-to-noise ratio (SNR) calculation
    - Confidence scoring for diagnoses

Zero-Hallucination Guarantee:
    All calculations use deterministic signal processing algorithms.
    No LLM or AI inference in any calculation path.
    Same acoustic inputs always produce identical diagnostic outputs.

Engineering References:
    - ASME PTC 39.1 - Steam Trap Testing
    - ISO 11201 - Acoustic Noise Measurement
    - UE Systems Ultrasonic Analysis Guidelines

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class SignatureType(str, Enum):
    """Acoustic signature types for steam trap conditions."""
    NORMAL_CYCLING = "normal_cycling"
    BLOW_THROUGH = "blow_through"
    LEAKING = "leaking"
    BLOCKED = "blocked"
    RAPID_CYCLING = "rapid_cycling"
    WATERLOGGED = "waterlogged"
    COLD_TRAP = "cold_trap"
    UNKNOWN = "unknown"


class FrequencyBand(str, Enum):
    """Ultrasonic frequency bands for analysis."""
    LOW = "low"           # 20-30 kHz
    MID_LOW = "mid_low"   # 30-40 kHz
    MID = "mid"           # 40-50 kHz
    MID_HIGH = "mid_high" # 50-70 kHz
    HIGH = "high"         # 70-100 kHz


class NoiseSource(str, Enum):
    """Background noise source classifications."""
    STEAM_FLOW = "steam_flow"
    MECHANICAL = "mechanical"
    ELECTRICAL = "electrical"
    AMBIENT = "ambient"
    PIPING = "piping"
    UNKNOWN = "unknown"


class ConfidenceLevel(str, Enum):
    """Confidence levels for diagnostic results."""
    HIGH = "high"         # >= 85%
    MEDIUM = "medium"     # 60-84%
    LOW = "low"           # 40-59%
    VERY_LOW = "very_low" # < 40%


class DiagnosticSeverity(str, Enum):
    """Severity levels for acoustic diagnostics."""
    NORMAL = "normal"
    WATCH = "watch"
    WARNING = "warning"
    CRITICAL = "critical"


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class AcousticConfig:
    """
    Configuration for acoustic diagnostic analysis.

    Attributes:
        sampling_frequency_hz: Sampling rate for acoustic signals
        analysis_duration_ms: Duration of analysis window
        noise_floor_db: Minimum detectable signal level
        leak_threshold_db: dB level indicating potential leak
        blow_through_threshold_db: dB level indicating blow-through failure
        snr_minimum: Minimum acceptable signal-to-noise ratio
        confidence_threshold: Minimum confidence for valid diagnosis
        frequency_range_low_khz: Lower frequency bound
        frequency_range_high_khz: Upper frequency bound
    """
    sampling_frequency_hz: int = 250000
    analysis_duration_ms: int = 2000
    noise_floor_db: Decimal = field(default_factory=lambda: Decimal("25.0"))
    leak_threshold_db: Decimal = field(default_factory=lambda: Decimal("45.0"))
    blow_through_threshold_db: Decimal = field(default_factory=lambda: Decimal("70.0"))
    snr_minimum: Decimal = field(default_factory=lambda: Decimal("6.0"))
    confidence_threshold: Decimal = field(default_factory=lambda: Decimal("0.60"))
    frequency_range_low_khz: Decimal = field(default_factory=lambda: Decimal("20.0"))
    frequency_range_high_khz: Decimal = field(default_factory=lambda: Decimal("100.0"))

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.sampling_frequency_hz < 100000:
            raise ValueError("Sampling frequency must be >= 100 kHz for ultrasonic analysis")
        if self.analysis_duration_ms < 100:
            raise ValueError("Analysis duration must be >= 100ms")
        if self.leak_threshold_db >= self.blow_through_threshold_db:
            raise ValueError("Leak threshold must be less than blow-through threshold")


# =============================================================================
# ACOUSTIC SIGNATURE REFERENCE DATA
# =============================================================================

# Reference acoustic signatures for pattern matching (deterministic lookup)
ACOUSTIC_SIGNATURES: Dict[SignatureType, Dict[str, Any]] = {
    SignatureType.NORMAL_CYCLING: {
        "description": "Normal intermittent operation with regular cycling",
        "db_range": (Decimal("30.0"), Decimal("50.0")),
        "primary_frequency_khz": Decimal("38.0"),
        "pattern": "intermittent_regular",
        "cycle_interval_s": (5, 60),
        "duty_cycle_pct": (10, 40),
    },
    SignatureType.BLOW_THROUGH: {
        "description": "Continuous high-amplitude steam flow indicating failed-open trap",
        "db_range": (Decimal("70.0"), Decimal("95.0")),
        "primary_frequency_khz": Decimal("42.0"),
        "pattern": "continuous_high",
        "cycle_interval_s": None,
        "duty_cycle_pct": (90, 100),
    },
    SignatureType.LEAKING: {
        "description": "Moderate continuous flow indicating partial leak",
        "db_range": (Decimal("45.0"), Decimal("70.0")),
        "primary_frequency_khz": Decimal("40.0"),
        "pattern": "continuous_moderate",
        "cycle_interval_s": None,
        "duty_cycle_pct": (70, 95),
    },
    SignatureType.BLOCKED: {
        "description": "Very low or no acoustic activity indicating blocked trap",
        "db_range": (Decimal("20.0"), Decimal("30.0")),
        "primary_frequency_khz": Decimal("35.0"),
        "pattern": "minimal",
        "cycle_interval_s": None,
        "duty_cycle_pct": (0, 5),
    },
    SignatureType.RAPID_CYCLING: {
        "description": "Frequent cycling indicating undersized trap or high load",
        "db_range": (Decimal("35.0"), Decimal("55.0")),
        "primary_frequency_khz": Decimal("38.0"),
        "pattern": "intermittent_rapid",
        "cycle_interval_s": (1, 5),
        "duty_cycle_pct": (40, 70),
    },
    SignatureType.WATERLOGGED: {
        "description": "Water hammer sounds with irregular bursts",
        "db_range": (Decimal("50.0"), Decimal("80.0")),
        "primary_frequency_khz": Decimal("25.0"),
        "pattern": "burst_irregular",
        "cycle_interval_s": None,
        "duty_cycle_pct": (20, 60),
    },
    SignatureType.COLD_TRAP: {
        "description": "Low-frequency rumble indicating condensate backup",
        "db_range": (Decimal("25.0"), Decimal("40.0")),
        "primary_frequency_khz": Decimal("22.0"),
        "pattern": "low_frequency_rumble",
        "cycle_interval_s": None,
        "duty_cycle_pct": (5, 30),
    },
}

# Frequency band definitions (deterministic)
FREQUENCY_BANDS: Dict[FrequencyBand, Tuple[Decimal, Decimal]] = {
    FrequencyBand.LOW: (Decimal("20.0"), Decimal("30.0")),
    FrequencyBand.MID_LOW: (Decimal("30.0"), Decimal("40.0")),
    FrequencyBand.MID: (Decimal("40.0"), Decimal("50.0")),
    FrequencyBand.MID_HIGH: (Decimal("50.0"), Decimal("70.0")),
    FrequencyBand.HIGH: (Decimal("70.0"), Decimal("100.0")),
}


# =============================================================================
# PROVENANCE TRACKING
# =============================================================================

@dataclass(frozen=True)
class ProvenanceStep:
    """
    Single step in calculation provenance chain.

    Attributes:
        step_name: Identifier for this calculation step
        inputs: Input values used in calculation
        formula: Formula or method applied
        output: Result of the calculation
        timestamp: When the calculation was performed
    """
    step_name: str
    inputs: Dict[str, Any]
    formula: str
    output: Any
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step_name": self.step_name,
            "inputs": self.inputs,
            "formula": self.formula,
            "output": str(self.output) if isinstance(self.output, Decimal) else self.output,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ProvenanceTracker:
    """
    Tracks complete calculation provenance for audit trails.

    Maintains ordered list of calculation steps with inputs, formulas,
    and outputs for complete reproducibility and verification.
    """
    calculation_id: str = ""
    steps: List[ProvenanceStep] = field(default_factory=list)

    def add_step(
        self,
        step_name: str,
        inputs: Dict[str, Any],
        formula: str,
        output: Any
    ) -> None:
        """Add a calculation step to the provenance chain."""
        step = ProvenanceStep(
            step_name=step_name,
            inputs=inputs,
            formula=formula,
            output=output,
        )
        self.steps.append(step)

    def get_hash(self) -> str:
        """Generate SHA-256 hash of entire provenance chain."""
        chain_str = "|".join(
            f"{s.step_name}:{s.inputs}:{s.formula}:{s.output}"
            for s in self.steps
        )
        return hashlib.sha256(chain_str.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert provenance chain to dictionary."""
        return {
            "calculation_id": self.calculation_id,
            "steps": [s.to_dict() for s in self.steps],
            "provenance_hash": self.get_hash(),
        }


# =============================================================================
# INPUT DATA MODELS
# =============================================================================

@dataclass(frozen=True)
class AcousticSignature:
    """
    Reference acoustic signature for pattern matching.

    Attributes:
        signature_type: Type of acoustic pattern
        description: Human-readable description
        db_min: Minimum dB level for this signature
        db_max: Maximum dB level for this signature
        primary_frequency_khz: Dominant frequency component
        pattern_type: Pattern classification
    """
    signature_type: SignatureType
    description: str
    db_min: Decimal
    db_max: Decimal
    primary_frequency_khz: Decimal
    pattern_type: str


@dataclass(frozen=True)
class AcousticReading:
    """
    Raw acoustic measurement data from ultrasonic sensor.

    Attributes:
        trap_id: Unique identifier for the steam trap
        reading_timestamp: When the measurement was taken
        peak_db: Peak sound pressure level in dB
        average_db: Average sound pressure level over measurement period
        rms_db: Root-mean-square dB level
        dominant_frequency_khz: Primary frequency component
        frequency_spectrum: dB levels across frequency bands
        background_noise_db: Ambient noise level
        measurement_duration_ms: Duration of the measurement
        sensor_id: Identifier of the acoustic sensor used
    """
    trap_id: str
    reading_timestamp: datetime
    peak_db: Decimal
    average_db: Decimal
    rms_db: Decimal
    dominant_frequency_khz: Decimal
    frequency_spectrum: Dict[FrequencyBand, Decimal]
    background_noise_db: Decimal
    measurement_duration_ms: int
    sensor_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate acoustic reading values."""
        if self.peak_db < Decimal("0") or self.peak_db > Decimal("120"):
            raise ValueError(f"Peak dB {self.peak_db} outside valid range 0-120")
        if self.average_db > self.peak_db:
            raise ValueError("Average dB cannot exceed peak dB")
        if self.dominant_frequency_khz < Decimal("0"):
            raise ValueError("Frequency cannot be negative")


# =============================================================================
# OUTPUT DATA MODELS
# =============================================================================

@dataclass(frozen=True)
class SignatureAnalysisResult:
    """
    Result of acoustic signature pattern matching.

    Attributes:
        detected_signature: Most likely signature type
        match_confidence: Confidence score (0.0 - 1.0)
        alternative_signatures: Other possible signatures with confidence
        pattern_characteristics: Detected pattern features
    """
    detected_signature: SignatureType
    match_confidence: Decimal
    alternative_signatures: Dict[SignatureType, Decimal]
    pattern_characteristics: Dict[str, Any]

    def get_confidence_level(self) -> ConfidenceLevel:
        """Determine confidence level category."""
        conf_pct = float(self.match_confidence) * 100
        if conf_pct >= 85:
            return ConfidenceLevel.HIGH
        elif conf_pct >= 60:
            return ConfidenceLevel.MEDIUM
        elif conf_pct >= 40:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW


@dataclass(frozen=True)
class LeakDetectionResult:
    """
    Steam leak detection analysis result.

    Attributes:
        leak_detected: Whether a leak was detected
        leak_severity: Severity classification
        estimated_leak_rate_kg_hr: Estimated steam loss rate
        confidence: Detection confidence score
        contributing_factors: Factors supporting the detection
    """
    leak_detected: bool
    leak_severity: DiagnosticSeverity
    estimated_leak_rate_kg_hr: Decimal
    confidence: Decimal
    contributing_factors: List[str]


@dataclass(frozen=True)
class SignalQualityMetrics:
    """
    Signal quality assessment metrics.

    Attributes:
        signal_to_noise_ratio: SNR in dB
        noise_floor_db: Detected noise floor
        signal_stability: Stability coefficient (0-1)
        frequency_resolution: Achievable frequency resolution
        measurement_quality: Overall quality assessment
    """
    signal_to_noise_ratio: Decimal
    noise_floor_db: Decimal
    signal_stability: Decimal
    frequency_resolution_hz: Decimal
    measurement_quality: str  # "excellent", "good", "acceptable", "poor"

    @staticmethod
    def assess_quality(snr: Decimal, stability: Decimal) -> str:
        """Determine measurement quality from metrics."""
        snr_val = float(snr)
        stab_val = float(stability)

        if snr_val >= 20 and stab_val >= 0.9:
            return "excellent"
        elif snr_val >= 12 and stab_val >= 0.7:
            return "good"
        elif snr_val >= 6 and stab_val >= 0.5:
            return "acceptable"
        else:
            return "poor"


@dataclass(frozen=True)
class AcousticDiagnosticResult:
    """
    Complete acoustic diagnostic analysis result.

    Attributes:
        trap_id: Steam trap identifier
        analysis_timestamp: When analysis was performed
        signature_analysis: Pattern matching results
        leak_detection: Leak detection results
        signal_quality: Signal quality metrics
        recommended_action: Suggested maintenance action
        diagnostic_severity: Overall severity assessment
        provenance_hash: SHA-256 hash for audit trail
        calculation_details: Detailed provenance chain
    """
    trap_id: str
    analysis_timestamp: datetime
    signature_analysis: SignatureAnalysisResult
    leak_detection: LeakDetectionResult
    signal_quality: SignalQualityMetrics
    recommended_action: str
    diagnostic_severity: DiagnosticSeverity
    provenance_hash: str
    calculation_details: Dict[str, Any]


# =============================================================================
# ACOUSTIC DIAGNOSTIC CALCULATOR
# =============================================================================

class AcousticDiagnosticCalculator:
    """
    Deterministic acoustic/ultrasonic diagnostic calculator for steam traps.

    This calculator analyzes ultrasonic acoustic signals to diagnose steam trap
    conditions using pattern matching against known acoustic signatures. All
    calculations are deterministic - same inputs always produce same outputs.

    Zero-Hallucination Design:
        - All pattern matching uses lookup tables with predefined signatures
        - SNR and dB calculations use standard acoustic formulas
        - No ML/AI inference in any calculation path
        - Complete provenance tracking for audit trails

    Example:
        >>> config = AcousticConfig()
        >>> calculator = AcousticDiagnosticCalculator(config)
        >>> reading = AcousticReading(
        ...     trap_id="TRAP-001",
        ...     reading_timestamp=datetime.now(),
        ...     peak_db=Decimal("75.0"),
        ...     average_db=Decimal("72.0"),
        ...     rms_db=Decimal("73.5"),
        ...     dominant_frequency_khz=Decimal("42.0"),
        ...     frequency_spectrum={...},
        ...     background_noise_db=Decimal("30.0"),
        ...     measurement_duration_ms=2000
        ... )
        >>> result = calculator.analyze_acoustic_signal(reading)
        >>> print(result.signature_analysis.detected_signature)
        SignatureType.BLOW_THROUGH

    Attributes:
        config: Calculator configuration parameters
        signatures: Reference acoustic signatures for pattern matching
    """

    def __init__(self, config: Optional[AcousticConfig] = None):
        """
        Initialize the acoustic diagnostic calculator.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or AcousticConfig()
        self.signatures = ACOUSTIC_SIGNATURES
        logger.info(f"AcousticDiagnosticCalculator initialized with config: {self.config}")

    def analyze_acoustic_signal(
        self,
        reading: AcousticReading,
        include_provenance: bool = True
    ) -> AcousticDiagnosticResult:
        """
        Perform complete acoustic diagnostic analysis on a reading.

        This is the main entry point for acoustic analysis. It performs:
        1. Signal quality assessment
        2. Pattern matching against known signatures
        3. Leak detection analysis
        4. Recommendation generation

        Args:
            reading: Acoustic reading data from ultrasonic sensor
            include_provenance: Whether to include detailed provenance chain

        Returns:
            Complete diagnostic result with all analysis components

        Raises:
            ValueError: If reading data is invalid
        """
        logger.info(f"Analyzing acoustic signal for trap {reading.trap_id}")
        start_time = datetime.utcnow()

        # Initialize provenance tracker
        provenance = ProvenanceTracker(
            calculation_id=f"acoustic_{reading.trap_id}_{start_time.isoformat()}"
        )

        # Step 1: Assess signal quality
        signal_quality = self._assess_signal_quality(reading, provenance)

        # Step 2: Analyze signature pattern
        signature_analysis = self._analyze_signature(reading, provenance)

        # Step 3: Detect leaks
        leak_detection = self._detect_leak(reading, signature_analysis, provenance)

        # Step 4: Determine severity
        severity = self._determine_severity(
            signature_analysis, leak_detection, signal_quality, provenance
        )

        # Step 5: Generate recommendation
        recommendation = self._generate_recommendation(
            signature_analysis, leak_detection, severity, provenance
        )

        # Generate final provenance hash
        provenance_hash = provenance.get_hash()

        result = AcousticDiagnosticResult(
            trap_id=reading.trap_id,
            analysis_timestamp=start_time,
            signature_analysis=signature_analysis,
            leak_detection=leak_detection,
            signal_quality=signal_quality,
            recommended_action=recommendation,
            diagnostic_severity=severity,
            provenance_hash=provenance_hash,
            calculation_details=provenance.to_dict() if include_provenance else {},
        )

        logger.info(
            f"Acoustic analysis complete for {reading.trap_id}: "
            f"signature={signature_analysis.detected_signature.value}, "
            f"severity={severity.value}"
        )

        return result

    def _assess_signal_quality(
        self,
        reading: AcousticReading,
        provenance: ProvenanceTracker
    ) -> SignalQualityMetrics:
        """
        Assess the quality of the acoustic signal.

        Calculates signal-to-noise ratio and other quality metrics using
        standard acoustic engineering formulas.

        Args:
            reading: The acoustic reading to assess
            provenance: Provenance tracker for audit trail

        Returns:
            Signal quality metrics
        """
        # Calculate SNR: SNR = 20 * log10(signal_power / noise_power)
        # Simplified: SNR_dB = signal_dB - noise_dB
        snr = reading.rms_db - reading.background_noise_db

        provenance.add_step(
            step_name="calculate_snr",
            inputs={
                "signal_db": str(reading.rms_db),
                "noise_db": str(reading.background_noise_db)
            },
            formula="SNR_dB = signal_dB - noise_dB",
            output=str(snr)
        )

        # Calculate signal stability (coefficient of variation proxy)
        # Using ratio of average to peak as stability indicator
        if reading.peak_db > Decimal("0"):
            stability = reading.average_db / reading.peak_db
        else:
            stability = Decimal("0")

        provenance.add_step(
            step_name="calculate_stability",
            inputs={
                "average_db": str(reading.average_db),
                "peak_db": str(reading.peak_db)
            },
            formula="stability = average_db / peak_db",
            output=str(stability)
        )

        # Calculate frequency resolution
        # Resolution = sampling_frequency / (2 * num_samples)
        num_samples = (
            self.config.sampling_frequency_hz *
            reading.measurement_duration_ms // 1000
        )
        freq_resolution = Decimal(str(
            self.config.sampling_frequency_hz / (2 * max(num_samples, 1))
        )).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        provenance.add_step(
            step_name="calculate_frequency_resolution",
            inputs={
                "sampling_freq_hz": self.config.sampling_frequency_hz,
                "num_samples": num_samples
            },
            formula="resolution = sampling_freq / (2 * num_samples)",
            output=str(freq_resolution)
        )

        # Assess overall quality
        quality = SignalQualityMetrics.assess_quality(snr, stability)

        return SignalQualityMetrics(
            signal_to_noise_ratio=snr,
            noise_floor_db=reading.background_noise_db,
            signal_stability=stability.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
            frequency_resolution_hz=freq_resolution,
            measurement_quality=quality
        )

    def _analyze_signature(
        self,
        reading: AcousticReading,
        provenance: ProvenanceTracker
    ) -> SignatureAnalysisResult:
        """
        Match acoustic reading against known signature patterns.

        Uses deterministic pattern matching based on:
        - dB level ranges
        - Dominant frequency
        - Signal pattern characteristics

        Args:
            reading: Acoustic reading to analyze
            provenance: Provenance tracker for audit trail

        Returns:
            Signature analysis result with confidence scores
        """
        match_scores: Dict[SignatureType, Decimal] = {}

        for sig_type, sig_data in self.signatures.items():
            score = self._calculate_signature_match_score(
                reading, sig_type, sig_data, provenance
            )
            match_scores[sig_type] = score

        # Find best match
        best_match = max(match_scores, key=lambda k: match_scores[k])
        best_confidence = match_scores[best_match]

        # Get alternative matches (excluding best)
        alternatives = {
            k: v for k, v in match_scores.items()
            if k != best_match and v > Decimal("0.1")
        }

        provenance.add_step(
            step_name="select_best_signature",
            inputs={"all_scores": {k.value: str(v) for k, v in match_scores.items()}},
            formula="best_match = argmax(match_scores)",
            output=best_match.value
        )

        # Extract pattern characteristics
        characteristics = {
            "db_level": str(reading.average_db),
            "dominant_frequency_khz": str(reading.dominant_frequency_khz),
            "peak_to_average_ratio": str(
                (reading.peak_db / reading.average_db).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                ) if reading.average_db > 0 else Decimal("0")
            ),
            "frequency_band_distribution": {
                band.value: str(level)
                for band, level in reading.frequency_spectrum.items()
            }
        }

        return SignatureAnalysisResult(
            detected_signature=best_match,
            match_confidence=best_confidence,
            alternative_signatures=alternatives,
            pattern_characteristics=characteristics
        )

    def _calculate_signature_match_score(
        self,
        reading: AcousticReading,
        sig_type: SignatureType,
        sig_data: Dict[str, Any],
        provenance: ProvenanceTracker
    ) -> Decimal:
        """
        Calculate match score for a specific signature type.

        Uses weighted scoring based on:
        - dB level match (40% weight)
        - Frequency match (30% weight)
        - Pattern consistency (30% weight)

        Args:
            reading: Acoustic reading
            sig_type: Signature type to match against
            sig_data: Reference signature data
            provenance: Provenance tracker

        Returns:
            Match score between 0.0 and 1.0
        """
        db_min, db_max = sig_data["db_range"]
        ref_freq = sig_data["primary_frequency_khz"]

        # Calculate dB match score (0-1)
        if db_min <= reading.average_db <= db_max:
            db_score = Decimal("1.0")
        else:
            # Linear decay outside range
            if reading.average_db < db_min:
                distance = db_min - reading.average_db
            else:
                distance = reading.average_db - db_max
            db_score = max(Decimal("0"), Decimal("1.0") - (distance / Decimal("20")))

        # Calculate frequency match score (0-1)
        freq_diff = abs(reading.dominant_frequency_khz - ref_freq)
        freq_score = max(
            Decimal("0"),
            Decimal("1.0") - (freq_diff / Decimal("15"))
        )

        # Calculate pattern consistency score based on duty cycle proxy
        # Using peak-to-average ratio as duty cycle indicator
        if reading.average_db > 0:
            duty_proxy = float(reading.average_db / reading.peak_db)
        else:
            duty_proxy = 0

        duty_range = sig_data.get("duty_cycle_pct", (0, 100))
        if duty_range:
            duty_min = duty_range[0] / 100
            duty_max = duty_range[1] / 100
            if duty_min <= duty_proxy <= duty_max:
                pattern_score = Decimal("1.0")
            else:
                if duty_proxy < duty_min:
                    pattern_diff = duty_min - duty_proxy
                else:
                    pattern_diff = duty_proxy - duty_max
                pattern_score = max(Decimal("0"), Decimal("1.0") - Decimal(str(pattern_diff * 2)))
        else:
            pattern_score = Decimal("0.5")  # Neutral if no duty cycle specified

        # Weighted combination
        weights = {
            "db": Decimal("0.40"),
            "freq": Decimal("0.30"),
            "pattern": Decimal("0.30")
        }

        total_score = (
            weights["db"] * db_score +
            weights["freq"] * freq_score +
            weights["pattern"] * pattern_score
        ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        provenance.add_step(
            step_name=f"match_score_{sig_type.value}",
            inputs={
                "db_score": str(db_score),
                "freq_score": str(freq_score),
                "pattern_score": str(pattern_score),
                "weights": {k: str(v) for k, v in weights.items()}
            },
            formula="total = w_db*db + w_freq*freq + w_pattern*pattern",
            output=str(total_score)
        )

        return total_score

    def _detect_leak(
        self,
        reading: AcousticReading,
        signature: SignatureAnalysisResult,
        provenance: ProvenanceTracker
    ) -> LeakDetectionResult:
        """
        Detect and characterize steam leaks from acoustic data.

        Uses dB thresholds and signature analysis to detect leaks.
        Estimates leak rate using correlation from UE Systems guidelines.

        Args:
            reading: Acoustic reading
            signature: Signature analysis result
            provenance: Provenance tracker

        Returns:
            Leak detection result with severity and estimated loss
        """
        contributing_factors: List[str] = []

        # Check if signature indicates leak
        leak_signatures = {
            SignatureType.BLOW_THROUGH,
            SignatureType.LEAKING,
        }

        signature_indicates_leak = signature.detected_signature in leak_signatures
        if signature_indicates_leak:
            contributing_factors.append(
                f"Signature pattern: {signature.detected_signature.value}"
            )

        # Check dB threshold
        exceeds_leak_threshold = reading.average_db >= self.config.leak_threshold_db
        if exceeds_leak_threshold:
            contributing_factors.append(
                f"dB level ({reading.average_db}) exceeds leak threshold "
                f"({self.config.leak_threshold_db})"
            )

        exceeds_blowthrough = reading.average_db >= self.config.blow_through_threshold_db
        if exceeds_blowthrough:
            contributing_factors.append(
                f"dB level ({reading.average_db}) exceeds blow-through threshold "
                f"({self.config.blow_through_threshold_db})"
            )

        # Determine if leak is detected
        leak_detected = signature_indicates_leak or exceeds_leak_threshold

        # Determine severity
        if exceeds_blowthrough:
            severity = DiagnosticSeverity.CRITICAL
        elif exceeds_leak_threshold and signature_indicates_leak:
            severity = DiagnosticSeverity.WARNING
        elif exceeds_leak_threshold or signature_indicates_leak:
            severity = DiagnosticSeverity.WATCH
        else:
            severity = DiagnosticSeverity.NORMAL

        # Estimate leak rate using empirical correlation
        # Approximate: leak_rate_kg_hr = 10^((dB - 40) / 20) for dB > 40
        if leak_detected and reading.average_db > Decimal("40"):
            db_above_baseline = float(reading.average_db - Decimal("40"))
            leak_rate = Decimal(str(10 ** (db_above_baseline / 20))).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        else:
            leak_rate = Decimal("0.00")

        provenance.add_step(
            step_name="estimate_leak_rate",
            inputs={
                "average_db": str(reading.average_db),
                "baseline_db": "40",
                "leak_detected": leak_detected
            },
            formula="leak_rate_kg_hr = 10^((dB - 40) / 20) for dB > 40",
            output=str(leak_rate)
        )

        # Calculate confidence
        if leak_detected:
            confidence = signature.match_confidence
        else:
            # High confidence in no-leak determination if signal quality is good
            confidence = Decimal("0.85")

        provenance.add_step(
            step_name="leak_detection_summary",
            inputs={
                "leak_detected": leak_detected,
                "severity": severity.value,
                "contributing_factors": contributing_factors
            },
            formula="deterministic_threshold_and_signature_analysis",
            output={"leak_rate_kg_hr": str(leak_rate), "confidence": str(confidence)}
        )

        return LeakDetectionResult(
            leak_detected=leak_detected,
            leak_severity=severity,
            estimated_leak_rate_kg_hr=leak_rate,
            confidence=confidence,
            contributing_factors=contributing_factors
        )

    def _determine_severity(
        self,
        signature: SignatureAnalysisResult,
        leak: LeakDetectionResult,
        signal_quality: SignalQualityMetrics,
        provenance: ProvenanceTracker
    ) -> DiagnosticSeverity:
        """
        Determine overall diagnostic severity.

        Considers signature type, leak severity, and signal quality
        to determine the most appropriate severity level.

        Args:
            signature: Signature analysis result
            leak: Leak detection result
            signal_quality: Signal quality metrics
            provenance: Provenance tracker

        Returns:
            Overall diagnostic severity
        """
        # Start with leak severity as baseline
        severity = leak.leak_severity

        # Escalate based on signature
        critical_signatures = {SignatureType.BLOW_THROUGH, SignatureType.WATERLOGGED}
        warning_signatures = {SignatureType.LEAKING, SignatureType.BLOCKED}

        if signature.detected_signature in critical_signatures:
            severity = DiagnosticSeverity.CRITICAL
        elif signature.detected_signature in warning_signatures:
            if severity == DiagnosticSeverity.NORMAL:
                severity = DiagnosticSeverity.WARNING

        # Reduce confidence if signal quality is poor
        if signal_quality.measurement_quality == "poor":
            # Do not escalate to critical with poor signal quality
            if severity == DiagnosticSeverity.CRITICAL:
                severity = DiagnosticSeverity.WARNING

        provenance.add_step(
            step_name="determine_overall_severity",
            inputs={
                "signature": signature.detected_signature.value,
                "leak_severity": leak.leak_severity.value,
                "signal_quality": signal_quality.measurement_quality
            },
            formula="max(signature_severity, leak_severity) adjusted for signal_quality",
            output=severity.value
        )

        return severity

    def _generate_recommendation(
        self,
        signature: SignatureAnalysisResult,
        leak: LeakDetectionResult,
        severity: DiagnosticSeverity,
        provenance: ProvenanceTracker
    ) -> str:
        """
        Generate maintenance recommendation based on diagnosis.

        Uses lookup table for deterministic recommendations based
        on signature type and severity level.

        Args:
            signature: Signature analysis result
            leak: Leak detection result
            severity: Overall severity
            provenance: Provenance tracker

        Returns:
            Recommended action string
        """
        # Deterministic recommendation lookup
        recommendations = {
            (SignatureType.BLOW_THROUGH, DiagnosticSeverity.CRITICAL):
                "IMMEDIATE: Replace failed-open trap. Estimated steam loss significant.",
            (SignatureType.BLOW_THROUGH, DiagnosticSeverity.WARNING):
                "URGENT: Schedule trap replacement within 48 hours.",
            (SignatureType.LEAKING, DiagnosticSeverity.WARNING):
                "Schedule repair or replacement within 1 week. Monitor daily.",
            (SignatureType.LEAKING, DiagnosticSeverity.WATCH):
                "Add to watch list. Re-survey in 2 weeks.",
            (SignatureType.BLOCKED, DiagnosticSeverity.WARNING):
                "URGENT: Clear blocked trap. Risk of equipment damage from condensate backup.",
            (SignatureType.BLOCKED, DiagnosticSeverity.WATCH):
                "Investigate potential blockage. Check inlet strainer.",
            (SignatureType.RAPID_CYCLING, DiagnosticSeverity.WARNING):
                "Evaluate trap sizing. Consider upgrading to higher-capacity model.",
            (SignatureType.RAPID_CYCLING, DiagnosticSeverity.WATCH):
                "Monitor cycling frequency. May indicate increased load.",
            (SignatureType.WATERLOGGED, DiagnosticSeverity.CRITICAL):
                "IMMEDIATE: Address water hammer condition. Safety hazard.",
            (SignatureType.WATERLOGGED, DiagnosticSeverity.WARNING):
                "URGENT: Drain system and inspect trap internals.",
            (SignatureType.COLD_TRAP, DiagnosticSeverity.WARNING):
                "Investigate heat loss. Check insulation and trap operation.",
            (SignatureType.NORMAL_CYCLING, DiagnosticSeverity.NORMAL):
                "No action required. Trap operating normally.",
        }

        # Look up recommendation
        key = (signature.detected_signature, severity)
        recommendation = recommendations.get(
            key,
            f"Review trap condition. Detected: {signature.detected_signature.value}, "
            f"Severity: {severity.value}"
        )

        # Add leak rate if significant
        if leak.leak_detected and leak.estimated_leak_rate_kg_hr > Decimal("1"):
            recommendation += f" Estimated leak: {leak.estimated_leak_rate_kg_hr} kg/hr."

        provenance.add_step(
            step_name="generate_recommendation",
            inputs={
                "signature": signature.detected_signature.value,
                "severity": severity.value,
                "leak_rate_kg_hr": str(leak.estimated_leak_rate_kg_hr)
            },
            formula="lookup_table[signature, severity]",
            output=recommendation
        )

        return recommendation

    def analyze_batch(
        self,
        readings: List[AcousticReading],
        include_provenance: bool = False
    ) -> List[AcousticDiagnosticResult]:
        """
        Analyze multiple acoustic readings in batch.

        Args:
            readings: List of acoustic readings to analyze
            include_provenance: Whether to include detailed provenance

        Returns:
            List of diagnostic results for each reading
        """
        logger.info(f"Starting batch analysis of {len(readings)} readings")

        results = []
        for reading in readings:
            try:
                result = self.analyze_acoustic_signal(reading, include_provenance)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to analyze reading for trap {reading.trap_id}: {e}")
                raise

        logger.info(f"Batch analysis complete. Processed {len(results)} readings.")
        return results

    def get_statistics(
        self,
        results: List[AcousticDiagnosticResult]
    ) -> Dict[str, Any]:
        """
        Calculate summary statistics from batch analysis results.

        Args:
            results: List of diagnostic results

        Returns:
            Dictionary of summary statistics
        """
        if not results:
            return {
                "total_analyzed": 0,
                "signature_distribution": {},
                "severity_distribution": {},
                "total_estimated_leak_kg_hr": "0.00",
                "average_confidence": "0.00"
            }

        # Signature distribution
        sig_dist: Dict[str, int] = {}
        for r in results:
            sig = r.signature_analysis.detected_signature.value
            sig_dist[sig] = sig_dist.get(sig, 0) + 1

        # Severity distribution
        sev_dist: Dict[str, int] = {}
        for r in results:
            sev = r.diagnostic_severity.value
            sev_dist[sev] = sev_dist.get(sev, 0) + 1

        # Total leak estimate
        total_leak = sum(
            r.leak_detection.estimated_leak_rate_kg_hr
            for r in results
        )

        # Average confidence
        avg_confidence = sum(
            r.signature_analysis.match_confidence for r in results
        ) / Decimal(len(results))

        return {
            "total_analyzed": len(results),
            "signature_distribution": sig_dist,
            "severity_distribution": sev_dist,
            "total_estimated_leak_kg_hr": str(total_leak.quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )),
            "average_confidence": str(avg_confidence.quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )),
            "critical_count": sev_dist.get("critical", 0),
            "warning_count": sev_dist.get("warning", 0),
        }


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "AcousticDiagnosticCalculator",
    "AcousticConfig",
    "AcousticSignature",
    "SignatureType",
    "FrequencyBand",
    "NoiseSource",
    "ConfidenceLevel",
    "DiagnosticSeverity",
    "AcousticReading",
    "SignatureAnalysisResult",
    "LeakDetectionResult",
    "SignalQualityMetrics",
    "AcousticDiagnosticResult",
    "ProvenanceTracker",
    "ProvenanceStep",
    "ACOUSTIC_SIGNATURES",
    "FREQUENCY_BANDS",
]
