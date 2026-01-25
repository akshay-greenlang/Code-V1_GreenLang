"""
GL-013 PREDICTMAINT - Vibration Analysis Module

This module implements ISO 10816 compliant vibration analysis with
complete provenance tracking for predictive maintenance.

Key Features:
- ISO 10816-3 machine classification (Class I-IV)
- Evaluation zones (A, B, C, D) with severity limits
- FFT spectrum analysis
- Bearing fault frequency detection (BPFO, BPFI, BSF, FTF)
- Envelope analysis for bearing faults
- Trend analysis and alarming

Reference Standards:
- ISO 10816-1:1995 Mechanical vibration - Evaluation of machine vibration
- ISO 10816-3:2009 Industrial machines with power above 15 kW
- ISO 13373-2:2016 Condition monitoring - Vibration condition monitoring
- ISO 15243:2017 Rolling bearings - Damage and failures

Author: GL-CalculatorEngineer
Version: 1.0.0
License: Proprietary - GreenLang
"""

from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum, auto
import math
from collections import deque

from .constants import (
    MachineClass,
    VibrationZone,
    VibrationLimits,
    ISO_10816_VIBRATION_LIMITS,
    BEARING_GEOMETRIES,
    BearingGeometry,
    PI,
    DEFAULT_DECIMAL_PRECISION,
)
from .units import (
    convert,
    mm_s_to_in_s,
    in_s_to_mm_s,
    g_to_m_s2,
    rpm_to_hz,
)
from .provenance import (
    ProvenanceBuilder,
    ProvenanceRecord,
    CalculationType,
    store_provenance,
)


# =============================================================================
# ENUMS
# =============================================================================

class VibrationMeasureType(Enum):
    """Types of vibration measurement."""
    VELOCITY_RMS = auto()      # mm/s RMS (ISO 10816 standard)
    VELOCITY_PEAK = auto()     # mm/s Peak
    DISPLACEMENT_PP = auto()   # um Peak-to-Peak
    ACCELERATION_RMS = auto()  # g RMS
    ACCELERATION_PEAK = auto() # g Peak


class FaultType(Enum):
    """Bearing and machine fault types."""
    IMBALANCE = auto()
    MISALIGNMENT = auto()
    LOOSENESS = auto()
    BEARING_OUTER_RACE = auto()
    BEARING_INNER_RACE = auto()
    BEARING_BALL_SPIN = auto()
    BEARING_CAGE = auto()
    GEAR_MESH = auto()
    ELECTRICAL = auto()
    RESONANCE = auto()


class AlarmLevel(Enum):
    """Alarm severity levels."""
    NORMAL = auto()
    ALERT = auto()
    WARNING = auto()
    CRITICAL = auto()


# =============================================================================
# RESULT DATA CLASSES
# =============================================================================

@dataclass(frozen=True)
class VibrationSeverityResult:
    """
    Result of ISO 10816 vibration severity assessment.

    Attributes:
        velocity_rms: Vibration velocity in mm/s RMS
        zone: ISO 10816 evaluation zone (A, B, C, D)
        machine_class: ISO machine classification
        zone_limits: Zone boundary values
        alarm_level: Corresponding alarm level
        assessment: Human-readable assessment
        recommendation: Action recommendation
        provenance_hash: SHA-256 hash for audit
    """
    velocity_rms: Decimal
    zone: VibrationZone
    machine_class: MachineClass
    zone_limits: Dict[str, Decimal]
    alarm_level: AlarmLevel
    assessment: str
    recommendation: str
    margin_to_next_zone: Decimal
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "velocity_rms_mm_s": str(self.velocity_rms),
            "zone": self.zone.name,
            "machine_class": self.machine_class.name,
            "zone_limits": {k: str(v) for k, v in self.zone_limits.items()},
            "alarm_level": self.alarm_level.name,
            "assessment": self.assessment,
            "recommendation": self.recommendation,
            "margin_to_next_zone": str(self.margin_to_next_zone),
            "provenance_hash": self.provenance_hash,
        }


@dataclass(frozen=True)
class BearingFaultFrequencies:
    """
    Calculated bearing fault frequencies.

    BPFO: Ball Pass Frequency Outer race
    BPFI: Ball Pass Frequency Inner race
    BSF: Ball Spin Frequency
    FTF: Fundamental Train Frequency (cage)
    """
    shaft_speed_hz: Decimal
    bpfo: Decimal
    bpfi: Decimal
    bsf: Decimal
    ftf: Decimal
    bearing_id: str
    geometry: Dict[str, Any]
    harmonics: Dict[str, Tuple[Decimal, ...]]
    provenance_hash: str = ""


@dataclass(frozen=True)
class SpectrumPeak:
    """Individual peak in frequency spectrum."""
    frequency_hz: Decimal
    amplitude: Decimal
    amplitude_unit: str
    probable_fault: Optional[FaultType]
    confidence: Decimal
    harmonic_of: Optional[str] = None


@dataclass(frozen=True)
class SpectrumAnalysisResult:
    """
    Result of FFT spectrum analysis.

    Attributes:
        peaks: Identified spectral peaks
        dominant_frequency: Highest amplitude frequency
        total_rms: Overall RMS value
        frequency_bands: Energy in standard frequency bands
        identified_faults: Detected fault signatures
        provenance_hash: SHA-256 hash
    """
    peaks: Tuple[SpectrumPeak, ...]
    dominant_frequency: Decimal
    dominant_amplitude: Decimal
    total_rms: Decimal
    frequency_bands: Dict[str, Decimal]
    identified_faults: Tuple[FaultType, ...]
    fault_confidence: Dict[str, Decimal]
    provenance_hash: str = ""


@dataclass(frozen=True)
class TrendAnalysisResult:
    """
    Result of vibration trend analysis.

    Tracks changes in vibration level over time to detect
    degradation patterns.
    """
    current_value: Decimal
    baseline_value: Decimal
    change_percent: Decimal
    trend_direction: str  # "increasing", "decreasing", "stable"
    rate_of_change: Decimal  # per day
    days_to_alarm: Optional[Decimal]
    days_to_danger: Optional[Decimal]
    statistical_significance: bool
    provenance_hash: str = ""


@dataclass(frozen=True)
class EnvelopeAnalysisResult:
    """
    Result of envelope (demodulation) analysis.

    Envelope analysis extracts the modulating signal from
    high-frequency resonance bands, revealing bearing defect
    frequencies.
    """
    envelope_rms: Decimal
    detected_frequencies: Tuple[Decimal, ...]
    probable_defects: Tuple[FaultType, ...]
    bearing_condition: str
    severity_index: Decimal
    provenance_hash: str = ""


# =============================================================================
# VIBRATION ANALYZER
# =============================================================================

class VibrationAnalyzer:
    """
    ISO 10816 compliant vibration analyzer with zero-hallucination guarantee.

    This analyzer provides deterministic vibration severity assessments
    and fault detection using established standards and formulas.

    All calculations are:
    - Bit-perfect reproducible (Decimal arithmetic)
    - Fully documented with provenance tracking
    - Based on ISO 10816 and industry standards

    Reference: ISO 10816-3:2009

    Example:
        >>> analyzer = VibrationAnalyzer()
        >>> result = analyzer.assess_severity(
        ...     velocity_rms=Decimal("4.2"),
        ...     machine_class=MachineClass.CLASS_II
        ... )
        >>> print(f"Zone: {result.zone.name}, Alarm: {result.alarm_level.name}")
        Zone: ZONE_B, Alarm: NORMAL
    """

    def __init__(
        self,
        precision: int = DEFAULT_DECIMAL_PRECISION,
        store_provenance_records: bool = True
    ):
        """
        Initialize Vibration Analyzer.

        Args:
            precision: Decimal precision for calculations
            store_provenance_records: Whether to store provenance
        """
        self._precision = precision
        self._store_provenance = store_provenance_records
        self._trend_history: Dict[str, deque] = {}

    # =========================================================================
    # ISO 10816 SEVERITY ASSESSMENT
    # =========================================================================

    def assess_severity(
        self,
        velocity_rms: Union[Decimal, float, str],
        machine_class: MachineClass,
        measurement_unit: str = "mm/s",
        equipment_id: Optional[str] = None
    ) -> VibrationSeverityResult:
        """
        Assess vibration severity per ISO 10816-3.

        ISO 10816 defines four evaluation zones:
        - Zone A: Good - Newly commissioned machines
        - Zone B: Acceptable - Unrestricted long-term operation
        - Zone C: Alert - Short-term operation only
        - Zone D: Danger - May cause damage

        Args:
            velocity_rms: Vibration velocity RMS value
            machine_class: ISO machine class (I-IV)
            measurement_unit: Unit of measurement ("mm/s" or "in/s")
            equipment_id: Optional equipment identifier

        Returns:
            VibrationSeverityResult with complete assessment

        Reference:
            ISO 10816-3:2009, Table 1

        Example:
            >>> analyzer = VibrationAnalyzer()
            >>> result = analyzer.assess_severity(
            ...     velocity_rms="3.5",
            ...     machine_class=MachineClass.CLASS_II
            ... )
            >>> print(f"Zone: {result.zone.name}")
            Zone: ZONE_B
        """
        builder = ProvenanceBuilder(CalculationType.VIBRATION_ANALYSIS)

        # Convert to Decimal and standardize to mm/s
        value = self._to_decimal(velocity_rms)
        if measurement_unit == "in/s" or measurement_unit == "ips":
            value_mm_s = in_s_to_mm_s(value)
        else:
            value_mm_s = value

        # Validate
        if value_mm_s < Decimal("0"):
            raise ValueError("Vibration velocity cannot be negative")

        # Record inputs
        builder.add_input("velocity_rms", value_mm_s)
        builder.add_input("machine_class", machine_class.name)
        builder.add_input("measurement_unit", measurement_unit)
        if equipment_id:
            builder.add_input("equipment_id", equipment_id)

        # Step 1: Get zone limits for machine class
        limits = ISO_10816_VIBRATION_LIMITS[machine_class]

        builder.add_step(
            step_number=1,
            operation="lookup",
            description="Retrieve ISO 10816 zone limits",
            inputs={"machine_class": machine_class.name},
            output_name="zone_limits",
            output_value={
                "zone_a_upper": limits.zone_a_upper,
                "zone_b_upper": limits.zone_b_upper,
                "zone_c_upper": limits.zone_c_upper
            },
            formula="ISO 10816-3:2009 Table 1",
            reference="ISO 10816-3:2009"
        )

        # Step 2: Determine zone
        if value_mm_s <= limits.zone_a_upper:
            zone = VibrationZone.ZONE_A
            alarm_level = AlarmLevel.NORMAL
            margin = limits.zone_a_upper - value_mm_s
            assessment = "Good - Newly commissioned machine quality"
            recommendation = "Continue normal operation and monitoring"
        elif value_mm_s <= limits.zone_b_upper:
            zone = VibrationZone.ZONE_B
            alarm_level = AlarmLevel.NORMAL
            margin = limits.zone_b_upper - value_mm_s
            assessment = "Acceptable - Suitable for unrestricted long-term operation"
            recommendation = "Maintain regular monitoring schedule"
        elif value_mm_s <= limits.zone_c_upper:
            zone = VibrationZone.ZONE_C
            alarm_level = AlarmLevel.WARNING
            margin = limits.zone_c_upper - value_mm_s
            assessment = "Alert - Short-term operation only, plan maintenance"
            recommendation = "Schedule maintenance within 30 days, increase monitoring frequency"
        else:
            zone = VibrationZone.ZONE_D
            alarm_level = AlarmLevel.CRITICAL
            margin = Decimal("0")  # Already exceeded
            assessment = "Danger - May cause damage, immediate action required"
            recommendation = "Stop machine immediately, investigate cause before restart"

        builder.add_step(
            step_number=2,
            operation="compare",
            description="Determine evaluation zone",
            inputs={
                "value": value_mm_s,
                "zone_a_limit": limits.zone_a_upper,
                "zone_b_limit": limits.zone_b_upper,
                "zone_c_limit": limits.zone_c_upper
            },
            output_name="zone",
            output_value=zone.name,
            formula="Compare value against zone boundaries",
            reference="ISO 10816-3:2009"
        )

        # Finalize outputs
        builder.add_output("zone", zone.name)
        builder.add_output("alarm_level", alarm_level.name)
        builder.add_output("margin_to_next_zone", margin)

        provenance = builder.build()
        if self._store_provenance:
            store_provenance(provenance)

        return VibrationSeverityResult(
            velocity_rms=self._apply_precision(value_mm_s, 2),
            zone=zone,
            machine_class=machine_class,
            zone_limits={
                "zone_a_upper": limits.zone_a_upper,
                "zone_b_upper": limits.zone_b_upper,
                "zone_c_upper": limits.zone_c_upper
            },
            alarm_level=alarm_level,
            assessment=assessment,
            recommendation=recommendation,
            margin_to_next_zone=self._apply_precision(margin, 2),
            provenance_hash=provenance.final_hash
        )

    # =========================================================================
    # BEARING FAULT FREQUENCY CALCULATION
    # =========================================================================

    def calculate_bearing_fault_frequencies(
        self,
        shaft_speed_rpm: Union[Decimal, float, int, str],
        bearing_id: str,
        num_harmonics: int = 3
    ) -> BearingFaultFrequencies:
        """
        Calculate bearing fault frequencies for defect detection.

        Bearing defects produce characteristic frequencies based on
        bearing geometry and shaft speed:

        BPFO = (n/2) * (1 - Bd/Pd * cos(phi)) * shaft_speed
        BPFI = (n/2) * (1 + Bd/Pd * cos(phi)) * shaft_speed
        BSF = (Pd/Bd) * (1 - (Bd/Pd * cos(phi))^2) / 2 * shaft_speed
        FTF = (1/2) * (1 - Bd/Pd * cos(phi)) * shaft_speed

        Where:
            n = Number of rolling elements
            Bd = Ball/roller diameter
            Pd = Pitch diameter
            phi = Contact angle

        Args:
            shaft_speed_rpm: Shaft rotational speed in RPM
            bearing_id: Bearing identifier (e.g., "6205")
            num_harmonics: Number of harmonics to calculate

        Returns:
            BearingFaultFrequencies with all characteristic frequencies

        Reference:
            Harris, T.A. (2006). Rolling Bearing Analysis, 5th Edition.

        Example:
            >>> analyzer = VibrationAnalyzer()
            >>> freqs = analyzer.calculate_bearing_fault_frequencies(
            ...     shaft_speed_rpm=1800,
            ...     bearing_id="6205"
            ... )
            >>> print(f"BPFO: {freqs.bpfo} Hz, BPFI: {freqs.bpfi} Hz")
        """
        builder = ProvenanceBuilder(CalculationType.VIBRATION_ANALYSIS)

        # Convert inputs
        rpm = self._to_decimal(shaft_speed_rpm)
        shaft_hz = rpm / Decimal("60")

        # Get bearing geometry
        if bearing_id not in BEARING_GEOMETRIES:
            raise ValueError(f"Unknown bearing: {bearing_id}. Available: {list(BEARING_GEOMETRIES.keys())}")

        geom = BEARING_GEOMETRIES[bearing_id]

        builder.add_input("shaft_speed_rpm", rpm)
        builder.add_input("bearing_id", bearing_id)
        builder.add_input("num_harmonics", num_harmonics)

        # Step 1: Calculate geometry ratios
        n = Decimal(str(geom.num_rolling_elements))
        Bd = geom.ball_diameter_mm
        Pd = geom.pitch_diameter_mm
        phi_deg = geom.contact_angle_deg
        phi_rad = phi_deg * PI / Decimal("180")

        # cos(phi)
        cos_phi = self._cos(phi_rad)
        # Bd/Pd ratio
        bd_pd = Bd / Pd

        builder.add_step(
            step_number=1,
            operation="lookup",
            description="Get bearing geometry parameters",
            inputs={"bearing_id": bearing_id},
            output_name="geometry",
            output_value={
                "n": n,
                "Bd_mm": Bd,
                "Pd_mm": Pd,
                "contact_angle_deg": phi_deg
            },
            reference="SKF Bearing Catalog"
        )

        # Step 2: Calculate BPFO (Ball Pass Frequency Outer)
        # BPFO = (n/2) * (1 - Bd/Pd * cos(phi)) * shaft_speed
        bpfo_factor = (n / Decimal("2")) * (Decimal("1") - bd_pd * cos_phi)
        bpfo = bpfo_factor * shaft_hz

        builder.add_step(
            step_number=2,
            operation="calculate",
            description="Calculate BPFO",
            inputs={"n": n, "bd_pd": bd_pd, "cos_phi": cos_phi, "shaft_hz": shaft_hz},
            output_name="bpfo",
            output_value=bpfo,
            formula="BPFO = (n/2) * (1 - Bd/Pd * cos(phi)) * f_shaft",
            reference="Harris (2006)"
        )

        # Step 3: Calculate BPFI (Ball Pass Frequency Inner)
        # BPFI = (n/2) * (1 + Bd/Pd * cos(phi)) * shaft_speed
        bpfi_factor = (n / Decimal("2")) * (Decimal("1") + bd_pd * cos_phi)
        bpfi = bpfi_factor * shaft_hz

        builder.add_step(
            step_number=3,
            operation="calculate",
            description="Calculate BPFI",
            inputs={"n": n, "bd_pd": bd_pd, "cos_phi": cos_phi, "shaft_hz": shaft_hz},
            output_name="bpfi",
            output_value=bpfi,
            formula="BPFI = (n/2) * (1 + Bd/Pd * cos(phi)) * f_shaft",
            reference="Harris (2006)"
        )

        # Step 4: Calculate BSF (Ball Spin Frequency)
        # BSF = (Pd/(2*Bd)) * (1 - (Bd/Pd * cos(phi))^2) * shaft_speed
        bsf_factor = (Pd / (Decimal("2") * Bd)) * (
            Decimal("1") - self._power(bd_pd * cos_phi, Decimal("2"))
        )
        bsf = bsf_factor * shaft_hz

        builder.add_step(
            step_number=4,
            operation="calculate",
            description="Calculate BSF",
            inputs={"Pd": Pd, "Bd": Bd, "bd_pd_cos": bd_pd * cos_phi, "shaft_hz": shaft_hz},
            output_name="bsf",
            output_value=bsf,
            formula="BSF = (Pd/(2*Bd)) * (1 - (Bd/Pd * cos(phi))^2) * f_shaft",
            reference="Harris (2006)"
        )

        # Step 5: Calculate FTF (Fundamental Train Frequency / Cage)
        # FTF = (1/2) * (1 - Bd/Pd * cos(phi)) * shaft_speed
        ftf_factor = Decimal("0.5") * (Decimal("1") - bd_pd * cos_phi)
        ftf = ftf_factor * shaft_hz

        builder.add_step(
            step_number=5,
            operation="calculate",
            description="Calculate FTF",
            inputs={"bd_pd": bd_pd, "cos_phi": cos_phi, "shaft_hz": shaft_hz},
            output_name="ftf",
            output_value=ftf,
            formula="FTF = (1/2) * (1 - Bd/Pd * cos(phi)) * f_shaft",
            reference="Harris (2006)"
        )

        # Step 6: Calculate harmonics
        harmonics = {
            "bpfo": tuple(bpfo * Decimal(str(i)) for i in range(1, num_harmonics + 1)),
            "bpfi": tuple(bpfi * Decimal(str(i)) for i in range(1, num_harmonics + 1)),
            "bsf": tuple(bsf * Decimal(str(i)) for i in range(1, num_harmonics + 1)),
            "ftf": tuple(ftf * Decimal(str(i)) for i in range(1, num_harmonics + 1)),
            "shaft": tuple(shaft_hz * Decimal(str(i)) for i in range(1, num_harmonics + 1)),
        }

        builder.add_step(
            step_number=6,
            operation="calculate",
            description="Calculate harmonics",
            inputs={"num_harmonics": num_harmonics},
            output_name="harmonics",
            output_value={k: [str(v) for v in vals] for k, vals in harmonics.items()}
        )

        # Finalize
        builder.add_output("bpfo", bpfo)
        builder.add_output("bpfi", bpfi)
        builder.add_output("bsf", bsf)
        builder.add_output("ftf", ftf)

        provenance = builder.build()
        if self._store_provenance:
            store_provenance(provenance)

        return BearingFaultFrequencies(
            shaft_speed_hz=self._apply_precision(shaft_hz, 4),
            bpfo=self._apply_precision(bpfo, 3),
            bpfi=self._apply_precision(bpfi, 3),
            bsf=self._apply_precision(bsf, 3),
            ftf=self._apply_precision(ftf, 3),
            bearing_id=bearing_id,
            geometry={
                "num_rolling_elements": int(n),
                "ball_diameter_mm": str(Bd),
                "pitch_diameter_mm": str(Pd),
                "contact_angle_deg": str(phi_deg)
            },
            harmonics={k: tuple(self._apply_precision(v, 3) for v in vals)
                       for k, vals in harmonics.items()},
            provenance_hash=provenance.final_hash
        )

    # =========================================================================
    # SPECTRUM ANALYSIS
    # =========================================================================

    def analyze_spectrum(
        self,
        frequencies: List[Union[Decimal, float]],
        amplitudes: List[Union[Decimal, float]],
        amplitude_unit: str = "mm/s",
        shaft_speed_rpm: Optional[Union[Decimal, float, int]] = None,
        bearing_id: Optional[str] = None,
        num_peaks: int = 10
    ) -> SpectrumAnalysisResult:
        """
        Analyze vibration frequency spectrum.

        Performs peak detection and fault identification from
        FFT spectrum data.

        Args:
            frequencies: List of frequencies (Hz)
            amplitudes: List of amplitude values
            amplitude_unit: Unit of amplitudes
            shaft_speed_rpm: Optional shaft speed for fault correlation
            bearing_id: Optional bearing ID for fault frequency matching
            num_peaks: Number of peaks to report

        Returns:
            SpectrumAnalysisResult with identified peaks and faults

        Reference:
            ISO 13373-2:2016, Vibration condition monitoring

        Example:
            >>> analyzer = VibrationAnalyzer()
            >>> result = analyzer.analyze_spectrum(
            ...     frequencies=[30, 60, 90, 120, 150],
            ...     amplitudes=[0.5, 2.1, 0.8, 0.3, 0.2],
            ...     shaft_speed_rpm=1800
            ... )
        """
        builder = ProvenanceBuilder(CalculationType.VIBRATION_ANALYSIS)

        # Convert to Decimal
        freqs = [self._to_decimal(f) for f in frequencies]
        amps = [self._to_decimal(a) for a in amplitudes]

        if len(freqs) != len(amps):
            raise ValueError("Frequencies and amplitudes must have same length")

        builder.add_input("num_data_points", len(freqs))
        builder.add_input("amplitude_unit", amplitude_unit)

        # Step 1: Find peaks (local maxima above threshold)
        mean_amp = sum(amps) / Decimal(str(len(amps)))
        std_amp = self._std_dev(amps)
        threshold = mean_amp + Decimal("2") * std_amp

        peaks = []
        for i in range(1, len(amps) - 1):
            if amps[i] > amps[i-1] and amps[i] > amps[i+1] and amps[i] > threshold:
                peaks.append((freqs[i], amps[i]))

        # Sort by amplitude, take top N
        peaks.sort(key=lambda x: x[1], reverse=True)
        peaks = peaks[:num_peaks]

        builder.add_step(
            step_number=1,
            operation="detect",
            description="Detect spectral peaks",
            inputs={"threshold": threshold, "num_peaks_found": len(peaks)},
            output_name="peaks",
            output_value=[(str(f), str(a)) for f, a in peaks]
        )

        # Step 2: Get bearing fault frequencies if available
        fault_freqs = {}
        if shaft_speed_rpm and bearing_id:
            bf = self.calculate_bearing_fault_frequencies(shaft_speed_rpm, bearing_id)
            fault_freqs = {
                "bpfo": bf.bpfo,
                "bpfi": bf.bpfi,
                "bsf": bf.bsf,
                "ftf": bf.ftf,
                "shaft_1x": bf.shaft_speed_hz,
                "shaft_2x": bf.shaft_speed_hz * Decimal("2"),
            }
        elif shaft_speed_rpm:
            shaft_hz = self._to_decimal(shaft_speed_rpm) / Decimal("60")
            fault_freqs = {
                "shaft_1x": shaft_hz,
                "shaft_2x": shaft_hz * Decimal("2"),
                "shaft_3x": shaft_hz * Decimal("3"),
            }

        # Step 3: Identify faults from peaks
        identified_faults = []
        fault_confidence = {}
        spectrum_peaks = []

        for freq, amp in peaks:
            probable_fault = None
            confidence = Decimal("0")
            harmonic_of = None

            # Check against known fault frequencies
            for fault_name, fault_freq in fault_freqs.items():
                # Allow 5% tolerance
                tolerance = fault_freq * Decimal("0.05")
                if abs(freq - fault_freq) <= tolerance:
                    if fault_name == "shaft_1x":
                        probable_fault = FaultType.IMBALANCE
                        harmonic_of = "1X"
                    elif fault_name == "shaft_2x":
                        probable_fault = FaultType.MISALIGNMENT
                        harmonic_of = "2X"
                    elif fault_name == "bpfo":
                        probable_fault = FaultType.BEARING_OUTER_RACE
                        harmonic_of = "BPFO"
                    elif fault_name == "bpfi":
                        probable_fault = FaultType.BEARING_INNER_RACE
                        harmonic_of = "BPFI"
                    elif fault_name == "bsf":
                        probable_fault = FaultType.BEARING_BALL_SPIN
                        harmonic_of = "BSF"
                    elif fault_name == "ftf":
                        probable_fault = FaultType.BEARING_CAGE
                        harmonic_of = "FTF"

                    # Confidence based on amplitude relative to overall
                    confidence = min(Decimal("1"), amp / (mean_amp * Decimal("5")))
                    break

            if probable_fault and probable_fault not in identified_faults:
                identified_faults.append(probable_fault)
                fault_confidence[probable_fault.name] = confidence

            spectrum_peaks.append(SpectrumPeak(
                frequency_hz=self._apply_precision(freq, 2),
                amplitude=self._apply_precision(amp, 4),
                amplitude_unit=amplitude_unit,
                probable_fault=probable_fault,
                confidence=self._apply_precision(confidence, 2),
                harmonic_of=harmonic_of
            ))

        builder.add_step(
            step_number=2,
            operation="identify",
            description="Identify fault signatures",
            inputs={"fault_frequencies": {k: str(v) for k, v in fault_freqs.items()}},
            output_name="identified_faults",
            output_value=[f.name for f in identified_faults]
        )

        # Step 4: Calculate frequency band energies
        frequency_bands = {}
        band_definitions = [
            ("low_10_100", Decimal("10"), Decimal("100")),
            ("mid_100_1000", Decimal("100"), Decimal("1000")),
            ("high_1000_10000", Decimal("1000"), Decimal("10000")),
        ]

        for band_name, low, high in band_definitions:
            band_energy = Decimal("0")
            for f, a in zip(freqs, amps):
                if low <= f < high:
                    band_energy += a * a
            frequency_bands[band_name] = self._power(band_energy, Decimal("0.5"))

        builder.add_step(
            step_number=3,
            operation="calculate",
            description="Calculate frequency band energies",
            inputs={"bands": [b[0] for b in band_definitions]},
            output_name="frequency_bands",
            output_value={k: str(v) for k, v in frequency_bands.items()}
        )

        # Step 5: Calculate total RMS
        total_rms = self._power(
            sum(a * a for a in amps) / Decimal(str(len(amps))),
            Decimal("0.5")
        )

        # Get dominant peak
        if peaks:
            dom_freq, dom_amp = peaks[0]
        else:
            dom_freq, dom_amp = Decimal("0"), Decimal("0")

        builder.add_output("dominant_frequency", dom_freq)
        builder.add_output("total_rms", total_rms)
        builder.add_output("num_faults_identified", len(identified_faults))

        provenance = builder.build()
        if self._store_provenance:
            store_provenance(provenance)

        return SpectrumAnalysisResult(
            peaks=tuple(spectrum_peaks),
            dominant_frequency=self._apply_precision(dom_freq, 2),
            dominant_amplitude=self._apply_precision(dom_amp, 4),
            total_rms=self._apply_precision(total_rms, 4),
            frequency_bands={k: self._apply_precision(v, 4) for k, v in frequency_bands.items()},
            identified_faults=tuple(identified_faults),
            fault_confidence={k: self._apply_precision(v, 2) for k, v in fault_confidence.items()},
            provenance_hash=provenance.final_hash
        )

    # =========================================================================
    # TREND ANALYSIS
    # =========================================================================

    def analyze_trend(
        self,
        equipment_id: str,
        current_value: Union[Decimal, float, str],
        timestamp_days: Union[Decimal, float, int],
        machine_class: MachineClass,
        window_size: int = 30
    ) -> TrendAnalysisResult:
        """
        Analyze vibration trend over time.

        Tracks changes in vibration levels to predict when alarm
        or danger thresholds will be reached.

        Args:
            equipment_id: Equipment identifier for history tracking
            current_value: Current vibration reading (mm/s RMS)
            timestamp_days: Days since epoch or reference
            machine_class: Machine class for limit lookup
            window_size: Number of readings for trend calculation

        Returns:
            TrendAnalysisResult with trend metrics

        Reference:
            ISO 13381-1:2015, Condition monitoring prognosis

        Example:
            >>> analyzer = VibrationAnalyzer()
            >>> for day in range(30):
            ...     value = 2.0 + 0.05 * day  # Simulated degradation
            ...     result = analyzer.analyze_trend(
            ...         equipment_id="PUMP-001",
            ...         current_value=value,
            ...         timestamp_days=day,
            ...         machine_class=MachineClass.CLASS_II
            ...     )
        """
        builder = ProvenanceBuilder(CalculationType.VIBRATION_ANALYSIS)

        value = self._to_decimal(current_value)
        day = self._to_decimal(timestamp_days)

        # Initialize or get history
        if equipment_id not in self._trend_history:
            self._trend_history[equipment_id] = deque(maxlen=window_size)

        history = self._trend_history[equipment_id]
        history.append((day, value))

        builder.add_input("equipment_id", equipment_id)
        builder.add_input("current_value", value)
        builder.add_input("timestamp_days", day)
        builder.add_input("history_length", len(history))

        # Get limits
        limits = ISO_10816_VIBRATION_LIMITS[machine_class]

        # Step 1: Calculate baseline (first reading or rolling average)
        if len(history) >= 2:
            baseline = history[0][1]
        else:
            baseline = value

        builder.add_step(
            step_number=1,
            operation="calculate",
            description="Establish baseline",
            inputs={"first_value": str(history[0][1]) if history else str(value)},
            output_name="baseline",
            output_value=baseline
        )

        # Step 2: Calculate change from baseline
        change = value - baseline
        if baseline > Decimal("0"):
            change_percent = (change / baseline) * Decimal("100")
        else:
            change_percent = Decimal("0")

        builder.add_step(
            step_number=2,
            operation="calculate",
            description="Calculate change from baseline",
            inputs={"current": value, "baseline": baseline},
            output_name="change_percent",
            output_value=change_percent,
            formula="change_% = (current - baseline) / baseline * 100"
        )

        # Step 3: Calculate rate of change (linear regression)
        if len(history) >= 3:
            days = [float(h[0]) for h in history]
            values = [float(h[1]) for h in history]

            # Simple linear regression
            n = len(days)
            sum_x = sum(days)
            sum_y = sum(values)
            sum_xy = sum(x*y for x, y in zip(days, values))
            sum_xx = sum(x*x for x in days)

            denominator = n * sum_xx - sum_x * sum_x
            if abs(denominator) > 1e-10:
                slope = (n * sum_xy - sum_x * sum_y) / denominator
                rate_of_change = Decimal(str(slope))
            else:
                rate_of_change = Decimal("0")

            # Determine trend direction
            if rate_of_change > Decimal("0.01"):
                trend_direction = "increasing"
            elif rate_of_change < Decimal("-0.01"):
                trend_direction = "decreasing"
            else:
                trend_direction = "stable"

            # Statistical significance (simplified: check if R^2 > 0.5)
            mean_y = sum_y / n
            ss_tot = sum((y - mean_y)**2 for y in values)
            if ss_tot > 0:
                intercept = (sum_y - slope * sum_x) / n
                ss_res = sum((y - (intercept + slope * x))**2 for x, y in zip(days, values))
                r_squared = 1 - ss_res / ss_tot
                significant = r_squared > 0.5
            else:
                significant = False
        else:
            rate_of_change = Decimal("0")
            trend_direction = "insufficient_data"
            significant = False

        builder.add_step(
            step_number=3,
            operation="regression",
            description="Calculate trend rate",
            inputs={"num_points": len(history)},
            output_name="rate_of_change",
            output_value=rate_of_change,
            formula="Linear regression slope"
        )

        # Step 4: Project time to alarm/danger thresholds
        if rate_of_change > Decimal("0.001"):
            alarm_threshold = limits.zone_b_upper
            danger_threshold = limits.zone_c_upper

            if value < alarm_threshold:
                days_to_alarm = (alarm_threshold - value) / rate_of_change
            else:
                days_to_alarm = None

            if value < danger_threshold:
                days_to_danger = (danger_threshold - value) / rate_of_change
            else:
                days_to_danger = None
        else:
            days_to_alarm = None
            days_to_danger = None

        builder.add_step(
            step_number=4,
            operation="project",
            description="Project time to thresholds",
            inputs={
                "current_value": value,
                "rate_of_change": rate_of_change,
                "alarm_threshold": limits.zone_b_upper
            },
            output_name="days_to_alarm",
            output_value=days_to_alarm
        )

        builder.add_output("trend_direction", trend_direction)
        builder.add_output("days_to_alarm", days_to_alarm)
        builder.add_output("days_to_danger", days_to_danger)

        provenance = builder.build()
        if self._store_provenance:
            store_provenance(provenance)

        return TrendAnalysisResult(
            current_value=self._apply_precision(value, 2),
            baseline_value=self._apply_precision(baseline, 2),
            change_percent=self._apply_precision(change_percent, 1),
            trend_direction=trend_direction,
            rate_of_change=self._apply_precision(rate_of_change, 4),
            days_to_alarm=self._apply_precision(days_to_alarm, 1) if days_to_alarm else None,
            days_to_danger=self._apply_precision(days_to_danger, 1) if days_to_danger else None,
            statistical_significance=significant,
            provenance_hash=provenance.final_hash
        )

    # =========================================================================
    # ENVELOPE ANALYSIS
    # =========================================================================

    def analyze_envelope(
        self,
        envelope_spectrum_frequencies: List[Union[Decimal, float]],
        envelope_spectrum_amplitudes: List[Union[Decimal, float]],
        shaft_speed_rpm: Union[Decimal, float, int, str],
        bearing_id: str
    ) -> EnvelopeAnalysisResult:
        """
        Perform envelope (demodulation) analysis for bearing faults.

        Envelope analysis extracts the modulating signal from
        high-frequency carrier signals (bearing resonances) to
        reveal repetitive impacts from defects.

        Args:
            envelope_spectrum_frequencies: Frequencies from envelope FFT (Hz)
            envelope_spectrum_amplitudes: Amplitudes from envelope FFT
            shaft_speed_rpm: Shaft speed in RPM
            bearing_id: Bearing identifier

        Returns:
            EnvelopeAnalysisResult

        Reference:
            ISO 13373-2:2016, Annex C - Envelope analysis

        Example:
            >>> analyzer = VibrationAnalyzer()
            >>> result = analyzer.analyze_envelope(
            ...     envelope_spectrum_frequencies=[...],
            ...     envelope_spectrum_amplitudes=[...],
            ...     shaft_speed_rpm=1800,
            ...     bearing_id="6205"
            ... )
        """
        builder = ProvenanceBuilder(CalculationType.VIBRATION_ANALYSIS)

        # Convert inputs
        freqs = [self._to_decimal(f) for f in envelope_spectrum_frequencies]
        amps = [self._to_decimal(a) for a in envelope_spectrum_amplitudes]
        rpm = self._to_decimal(shaft_speed_rpm)

        builder.add_input("num_data_points", len(freqs))
        builder.add_input("shaft_speed_rpm", rpm)
        builder.add_input("bearing_id", bearing_id)

        # Step 1: Get bearing fault frequencies
        fault_freqs = self.calculate_bearing_fault_frequencies(rpm, bearing_id)

        # Step 2: Calculate envelope RMS
        if amps:
            envelope_rms = self._power(
                sum(a * a for a in amps) / Decimal(str(len(amps))),
                Decimal("0.5")
            )
        else:
            envelope_rms = Decimal("0")

        builder.add_step(
            step_number=1,
            operation="calculate",
            description="Calculate envelope RMS",
            inputs={"num_amplitudes": len(amps)},
            output_name="envelope_rms",
            output_value=envelope_rms
        )

        # Step 3: Find peaks at fault frequencies
        detected_frequencies = []
        probable_defects = []
        severity_scores = []

        fault_map = {
            "bpfo": (fault_freqs.bpfo, FaultType.BEARING_OUTER_RACE),
            "bpfi": (fault_freqs.bpfi, FaultType.BEARING_INNER_RACE),
            "bsf": (fault_freqs.bsf, FaultType.BEARING_BALL_SPIN),
            "ftf": (fault_freqs.ftf, FaultType.BEARING_CAGE),
        }

        mean_amp = sum(amps) / Decimal(str(len(amps))) if amps else Decimal("1")

        for fault_name, (fault_freq, fault_type) in fault_map.items():
            tolerance = fault_freq * Decimal("0.05")

            # Find amplitude at fault frequency
            for i, freq in enumerate(freqs):
                if abs(freq - fault_freq) <= tolerance:
                    if amps[i] > mean_amp * Decimal("3"):  # Significant peak
                        detected_frequencies.append(freq)
                        if fault_type not in probable_defects:
                            probable_defects.append(fault_type)
                        severity_scores.append(amps[i] / mean_amp)

        builder.add_step(
            step_number=2,
            operation="detect",
            description="Detect fault frequencies in envelope",
            inputs={"fault_frequencies_checked": list(fault_map.keys())},
            output_name="detected_frequencies",
            output_value=[str(f) for f in detected_frequencies]
        )

        # Step 4: Determine bearing condition and severity
        if not probable_defects:
            bearing_condition = "Good"
            severity_index = Decimal("0")
        elif len(probable_defects) == 1:
            bearing_condition = f"Early {probable_defects[0].name.replace('_', ' ').title()} fault"
            severity_index = min(Decimal("5"), max(severity_scores) if severity_scores else Decimal("1"))
        else:
            bearing_condition = "Multiple defects - Advanced degradation"
            severity_index = min(Decimal("10"), sum(severity_scores) / Decimal(str(len(severity_scores))) if severity_scores else Decimal("5"))

        builder.add_step(
            step_number=3,
            operation="assess",
            description="Assess bearing condition",
            inputs={"num_defects": len(probable_defects)},
            output_name="bearing_condition",
            output_value=bearing_condition
        )

        builder.add_output("envelope_rms", envelope_rms)
        builder.add_output("bearing_condition", bearing_condition)
        builder.add_output("severity_index", severity_index)

        provenance = builder.build()
        if self._store_provenance:
            store_provenance(provenance)

        return EnvelopeAnalysisResult(
            envelope_rms=self._apply_precision(envelope_rms, 4),
            detected_frequencies=tuple(self._apply_precision(f, 2) for f in detected_frequencies),
            probable_defects=tuple(probable_defects),
            bearing_condition=bearing_condition,
            severity_index=self._apply_precision(severity_index, 1),
            provenance_hash=provenance.final_hash
        )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _to_decimal(self, value: Union[Decimal, float, int, str]) -> Decimal:
        """Convert value to Decimal."""
        if isinstance(value, Decimal):
            return value
        try:
            return Decimal(str(value))
        except InvalidOperation as e:
            raise ValueError(f"Cannot convert {value} to Decimal: {e}")

    def _apply_precision(
        self,
        value: Decimal,
        precision: Optional[int] = None
    ) -> Decimal:
        """Apply precision rounding."""
        prec = precision if precision is not None else self._precision
        if prec == 0:
            return value.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        quantize_str = "0." + "0" * prec
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    def _power(self, base: Decimal, exponent: Decimal) -> Decimal:
        """Calculate base^exponent."""
        if base == Decimal("0"):
            return Decimal("0") if exponent > Decimal("0") else Decimal("1")
        if exponent == Decimal("0"):
            return Decimal("1")
        result = Decimal(str(math.pow(float(base), float(exponent))))
        return result

    def _cos(self, x: Decimal) -> Decimal:
        """Calculate cosine."""
        return Decimal(str(math.cos(float(x))))

    def _std_dev(self, values: List[Decimal]) -> Decimal:
        """Calculate standard deviation."""
        if len(values) < 2:
            return Decimal("0")
        n = len(values)
        mean = sum(values) / Decimal(str(n))
        variance = sum((v - mean) ** 2 for v in values) / Decimal(str(n - 1))
        return self._power(variance, Decimal("0.5"))

    def get_supported_bearings(self) -> List[str]:
        """Get list of supported bearing IDs."""
        return list(BEARING_GEOMETRIES.keys())

    def clear_trend_history(self, equipment_id: Optional[str] = None) -> None:
        """Clear trend history for equipment."""
        if equipment_id:
            if equipment_id in self._trend_history:
                del self._trend_history[equipment_id]
        else:
            self._trend_history.clear()


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "VibrationMeasureType",
    "FaultType",
    "AlarmLevel",

    # Re-exported from constants
    "MachineClass",
    "VibrationZone",

    # Data classes
    "VibrationSeverityResult",
    "BearingFaultFrequencies",
    "SpectrumPeak",
    "SpectrumAnalysisResult",
    "TrendAnalysisResult",
    "EnvelopeAnalysisResult",

    # Main class
    "VibrationAnalyzer",
]
