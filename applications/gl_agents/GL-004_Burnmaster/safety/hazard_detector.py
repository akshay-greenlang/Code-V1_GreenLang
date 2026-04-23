"""
CombustionHazardDetector - Detects combustion-specific hazards.

This module detects flameout, delayed ignition, flashback, and oscillation
hazards specific to combustion systems. Early detection enables preventive
action before hazardous conditions develop.

CRITICAL: Hazard detection is advisory only. The optimizer does not control
safety systems - it uses hazard detection to constrain optimization actions.

Example:
    >>> detector = CombustionHazardDetector(unit_id="BLR-001")
    >>> hazard = detector.detect_flameout_risk(flame_signal, threshold)
    >>> if hazard == HazardLevel.HIGH:
    ...     # Switch to observe-only mode
    ...     pass
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum
import hashlib
import logging
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class HazardLevel(str, Enum):
    """Level of detected hazard."""
    NONE = "none"  # No hazard detected
    LOW = "low"  # Minor concern, monitor
    MODERATE = "moderate"  # Elevated concern, limit optimization
    HIGH = "high"  # Significant concern, restrict operations
    CRITICAL = "critical"  # Immediate concern, observe-only mode


class HazardType(str, Enum):
    """Type of combustion hazard."""
    FLAMEOUT = "flameout"
    DELAYED_IGNITION = "delayed_ignition"
    FLASHBACK = "flashback"
    COMBUSTION_OSCILLATION = "combustion_oscillation"
    FUEL_RICH = "fuel_rich"
    LEAN_BLOWOFF = "lean_blowoff"
    THERMAL_SHOCK = "thermal_shock"


class Hazard(BaseModel):
    """Individual hazard detection result."""
    hazard_type: HazardType = Field(..., description="Type of hazard")
    level: HazardLevel = Field(..., description="Hazard severity level")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence")
    trigger_value: Optional[float] = Field(None, description="Value triggering detection")
    threshold: Optional[float] = Field(None, description="Detection threshold")
    description: str = Field(..., description="Human-readable description")
    recommended_action: str = Field(..., description="Recommended response")
    detection_timestamp: datetime = Field(default_factory=datetime.utcnow)
    provenance_hash: str = Field(..., description="SHA-256 hash for audit")


class HazardAssessment(BaseModel):
    """Comprehensive hazard assessment."""
    unit_id: str = Field(..., description="Unit identifier")
    overall_level: HazardLevel = Field(..., description="Overall hazard level")
    hazards: List[Hazard] = Field(default_factory=list)
    safe_to_optimize: bool = Field(..., description="Whether optimization is safe")
    observe_only_recommended: bool = Field(default=False)
    restrictions: List[str] = Field(default_factory=list)
    assessment_timestamp: datetime = Field(default_factory=datetime.utcnow)
    provenance_hash: str = Field(..., description="SHA-256 hash for audit")


class CombustionHazardDetector:
    """
    CombustionHazardDetector identifies combustion-specific hazards.

    CRITICAL SAFETY INVARIANT:
    - Hazard detection is advisory - does not control safety systems
    - All detections are logged for audit
    - High hazard levels trigger automatic restrictions on optimization

    Attributes:
        unit_id: Identifier for the combustion unit
        detection_history: Historical hazard detections
        oscillation_buffer: Buffer for oscillation analysis

    Example:
        >>> detector = CombustionHazardDetector(unit_id="BLR-001")
        >>> flameout_risk = detector.detect_flameout_risk(25.0, 20.0)
        >>> if flameout_risk == HazardLevel.HIGH:
        ...     # Restrict optimization, alert operator
        ...     pass
    """

    # Hazard detection thresholds
    OSCILLATION_FREQ_MIN = 0.5  # Hz - minimum frequency of concern
    OSCILLATION_FREQ_MAX = 50.0  # Hz - maximum frequency of concern
    OSCILLATION_AMPLITUDE_THRESHOLD = 0.05  # 5% of signal amplitude

    def __init__(self, unit_id: str):
        """Initialize CombustionHazardDetector."""
        self.unit_id = unit_id
        self.detection_history: List[Hazard] = []
        self._oscillation_buffer: Dict[str, List[float]] = {}
        self._creation_time = datetime.utcnow()
        logger.info(f"CombustionHazardDetector initialized for unit {unit_id}")

    def detect_flameout_risk(
        self,
        flame_signal: float,
        threshold: float
    ) -> HazardLevel:
        """
        Detect risk of flameout condition.

        Flameout occurs when flame is extinguished unintentionally.
        Early detection allows preventive action.

        Args:
            flame_signal: Current flame scanner signal
            threshold: Minimum acceptable flame signal

        Returns:
            HazardLevel indicating flameout risk
        """
        if flame_signal >= threshold * 1.5:
            level = HazardLevel.NONE
            description = "Flame signal strong and stable"
            action = "Normal operation"
        elif flame_signal >= threshold * 1.2:
            level = HazardLevel.LOW
            description = "Flame signal adequate but below optimal"
            action = "Monitor flame stability"
        elif flame_signal >= threshold:
            level = HazardLevel.MODERATE
            description = "Flame signal approaching minimum threshold"
            action = "Increase monitoring, consider load reduction"
        elif flame_signal >= threshold * 0.7:
            level = HazardLevel.HIGH
            description = "Flame signal below minimum - flameout risk"
            action = "Reduce load, alert operator, prepare for shutdown"
        else:
            level = HazardLevel.CRITICAL
            description = "CRITICAL: Flame signal very low - imminent flameout"
            action = "ALERT: Verify flame immediately, standby for trip"

        confidence = self._calculate_confidence(flame_signal, threshold)

        hazard = self._create_hazard(
            hazard_type=HazardType.FLAMEOUT,
            level=level,
            confidence=confidence,
            trigger_value=flame_signal,
            threshold=threshold,
            description=description,
            action=action
        )

        self.detection_history.append(hazard)

        if level in [HazardLevel.HIGH, HazardLevel.CRITICAL]:
            logger.warning(f"FLAMEOUT RISK: {level.value} - signal={flame_signal:.1f}, threshold={threshold:.1f}")

        return level

    def detect_delayed_ignition_risk(
        self,
        conditions: Dict[str, Any]
    ) -> HazardLevel:
        """
        Detect risk of delayed ignition.

        Delayed ignition occurs when fuel accumulates before ignition,
        causing a potentially dangerous pressure surge.

        Args:
            conditions: Dictionary with ignition-related parameters:
                - pilot_proven: bool
                - main_fuel_valve_opening: bool
                - time_since_pilot: float (seconds)
                - fuel_pressure: float
                - combustion_chamber_temp: float

        Returns:
            HazardLevel indicating delayed ignition risk
        """
        pilot_proven = conditions.get('pilot_proven', False)
        main_valve_opening = conditions.get('main_fuel_valve_opening', False)
        time_since_pilot = conditions.get('time_since_pilot', 0)
        chamber_temp = conditions.get('combustion_chamber_temp', 0)

        # Risk increases if fuel valve opens without proven pilot
        if main_valve_opening and not pilot_proven:
            level = HazardLevel.CRITICAL
            description = "CRITICAL: Main fuel valve opening without proven pilot"
            action = "BLOCK fuel valve opening - pilot not proven"
            confidence = 0.95
        # Risk if pilot just lit and main is opening too quickly
        elif main_valve_opening and time_since_pilot < 5:
            level = HazardLevel.HIGH
            description = "Main fuel opening too quickly after pilot ignition"
            action = "Delay main fuel opening, verify pilot stability"
            confidence = 0.85
        # Cold chamber increases delayed ignition risk
        elif main_valve_opening and chamber_temp < 200:
            level = HazardLevel.MODERATE
            description = "Cold chamber may delay main flame ignition"
            action = "Proceed slowly, monitor for ignition"
            confidence = 0.7
        elif main_valve_opening and not conditions.get('igniter_proven', True):
            level = HazardLevel.HIGH
            description = "Ignition source not proven for main flame"
            action = "Verify ignition source before fuel admission"
            confidence = 0.9
        else:
            level = HazardLevel.NONE
            description = "No delayed ignition risk detected"
            action = "Normal operation"
            confidence = 0.8

        hazard = self._create_hazard(
            hazard_type=HazardType.DELAYED_IGNITION,
            level=level,
            confidence=confidence,
            trigger_value=time_since_pilot,
            threshold=5.0,
            description=description,
            action=action
        )

        self.detection_history.append(hazard)

        if level in [HazardLevel.HIGH, HazardLevel.CRITICAL]:
            logger.warning(f"DELAYED IGNITION RISK: {level.value}")

        return level

    def detect_flashback_risk(
        self,
        velocity: float,
        flame_speed: float
    ) -> HazardLevel:
        """
        Detect risk of flame flashback.

        Flashback occurs when flame propagates back into the fuel supply,
        potentially damaging equipment or causing explosion.

        Args:
            velocity: Fuel/air mixture velocity at burner (m/s or ft/s)
            flame_speed: Flame propagation speed (same units as velocity)

        Returns:
            HazardLevel indicating flashback risk
        """
        if velocity <= 0 or flame_speed <= 0:
            level = HazardLevel.CRITICAL
            description = "Invalid velocity or flame speed values"
            action = "Verify measurement accuracy"
            confidence = 0.5
        else:
            velocity_ratio = velocity / flame_speed

            if velocity_ratio >= 3.0:
                level = HazardLevel.NONE
                description = "Safe velocity margin above flame speed"
                action = "Normal operation"
                confidence = 0.9
            elif velocity_ratio >= 2.0:
                level = HazardLevel.LOW
                description = "Adequate velocity margin"
                action = "Monitor velocity"
                confidence = 0.85
            elif velocity_ratio >= 1.5:
                level = HazardLevel.MODERATE
                description = "Velocity margin below recommended minimum"
                action = "Increase fuel velocity or reduce load"
                confidence = 0.8
            elif velocity_ratio >= 1.0:
                level = HazardLevel.HIGH
                description = "Velocity approaching flame speed - flashback possible"
                action = "Increase velocity immediately, reduce load"
                confidence = 0.9
            else:
                level = HazardLevel.CRITICAL
                description = "CRITICAL: Velocity below flame speed - flashback imminent"
                action = "EMERGENCY: Increase velocity or shutdown"
                confidence = 0.95

        hazard = self._create_hazard(
            hazard_type=HazardType.FLASHBACK,
            level=level,
            confidence=confidence,
            trigger_value=velocity,
            threshold=flame_speed,
            description=description,
            action=action
        )

        self.detection_history.append(hazard)

        if level in [HazardLevel.HIGH, HazardLevel.CRITICAL]:
            logger.warning(f"FLASHBACK RISK: {level.value} - velocity={velocity:.2f}, flame_speed={flame_speed:.2f}")

        return level

    def detect_oscillation_hazard(
        self,
        pressure_signal: np.ndarray
    ) -> HazardLevel:
        """
        Detect combustion oscillation/instability hazard.

        Combustion oscillation can cause equipment damage, noise, and
        potentially dangerous pressure fluctuations.

        Args:
            pressure_signal: Array of pressure measurements (uniform sampling)

        Returns:
            HazardLevel indicating oscillation hazard
        """
        if len(pressure_signal) < 50:
            # Insufficient data
            hazard = self._create_hazard(
                hazard_type=HazardType.COMBUSTION_OSCILLATION,
                level=HazardLevel.NONE,
                confidence=0.3,
                description="Insufficient data for oscillation analysis",
                action="Continue monitoring"
            )
            self.detection_history.append(hazard)
            return HazardLevel.NONE

        try:
            # Calculate signal statistics
            mean_pressure = np.mean(pressure_signal)
            std_pressure = np.std(pressure_signal)
            amplitude_ratio = std_pressure / max(abs(mean_pressure), 0.001)

            # Perform FFT to detect oscillation frequencies
            fft_result = np.fft.fft(pressure_signal)
            frequencies = np.fft.fftfreq(len(pressure_signal))
            magnitudes = np.abs(fft_result)

            # Find dominant frequency (excluding DC component)
            magnitudes[0] = 0  # Remove DC
            peak_idx = np.argmax(magnitudes[:len(magnitudes)//2])
            dominant_freq = abs(frequencies[peak_idx])
            peak_magnitude = magnitudes[peak_idx]
            normalized_peak = peak_magnitude / (len(pressure_signal) * max(abs(mean_pressure), 0.001))

            # Determine hazard level
            if amplitude_ratio < 0.02 and normalized_peak < 0.1:
                level = HazardLevel.NONE
                description = "Stable combustion, no significant oscillation"
                action = "Normal operation"
                confidence = 0.9
            elif amplitude_ratio < 0.05 or normalized_peak < 0.2:
                level = HazardLevel.LOW
                description = "Minor pressure fluctuations detected"
                action = "Monitor for increasing amplitude"
                confidence = 0.8
            elif amplitude_ratio < 0.1 or normalized_peak < 0.4:
                level = HazardLevel.MODERATE
                description = f"Combustion oscillation detected at ~{dominant_freq:.1f} Hz"
                action = "Adjust air/fuel ratio, consider load reduction"
                confidence = 0.85
            elif amplitude_ratio < 0.2 or normalized_peak < 0.6:
                level = HazardLevel.HIGH
                description = f"Significant oscillation at {dominant_freq:.1f} Hz"
                action = "Reduce load, adjust combustion parameters"
                confidence = 0.9
            else:
                level = HazardLevel.CRITICAL
                description = f"CRITICAL: Severe combustion oscillation at {dominant_freq:.1f} Hz"
                action = "REDUCE LOAD IMMEDIATELY, prepare for shutdown"
                confidence = 0.95

        except Exception as e:
            logger.error(f"Oscillation analysis failed: {e}")
            level = HazardLevel.MODERATE
            description = f"Analysis error: {str(e)}"
            action = "Manual inspection recommended"
            confidence = 0.4

        hazard = self._create_hazard(
            hazard_type=HazardType.COMBUSTION_OSCILLATION,
            level=level,
            confidence=confidence,
            trigger_value=amplitude_ratio if 'amplitude_ratio' in dir() else None,
            description=description,
            action=action
        )

        self.detection_history.append(hazard)

        if level in [HazardLevel.HIGH, HazardLevel.CRITICAL]:
            logger.warning(f"OSCILLATION HAZARD: {level.value}")

        return level

    def compute_overall_hazard_level(
        self,
        hazards: List[Hazard]
    ) -> HazardLevel:
        """
        Compute overall hazard level from multiple hazard detections.

        The overall level is the maximum of individual hazards, with
        additional consideration for multiple simultaneous hazards.

        Args:
            hazards: List of individual hazard detections

        Returns:
            Overall HazardLevel
        """
        if not hazards:
            return HazardLevel.NONE

        # Get maximum individual level
        level_order = {
            HazardLevel.NONE: 0,
            HazardLevel.LOW: 1,
            HazardLevel.MODERATE: 2,
            HazardLevel.HIGH: 3,
            HazardLevel.CRITICAL: 4
        }

        max_level = max(hazards, key=lambda h: level_order[h.level]).level

        # Escalate if multiple moderate+ hazards
        elevated_count = sum(
            1 for h in hazards
            if level_order[h.level] >= level_order[HazardLevel.MODERATE]
        )

        if elevated_count >= 3 and max_level == HazardLevel.MODERATE:
            max_level = HazardLevel.HIGH
        elif elevated_count >= 2 and max_level == HazardLevel.HIGH:
            max_level = HazardLevel.CRITICAL

        logger.info(
            f"Overall hazard level: {max_level.value} "
            f"(from {len(hazards)} hazards, {elevated_count} elevated)"
        )

        return max_level

    def assess_all_hazards(
        self,
        flame_signal: float,
        flame_threshold: float,
        ignition_conditions: Optional[Dict[str, Any]] = None,
        velocity: Optional[float] = None,
        flame_speed: Optional[float] = None,
        pressure_signal: Optional[np.ndarray] = None
    ) -> HazardAssessment:
        """
        Perform comprehensive hazard assessment.

        Args:
            flame_signal: Current flame scanner signal
            flame_threshold: Minimum acceptable flame signal
            ignition_conditions: Optional ignition-related conditions
            velocity: Optional fuel velocity for flashback analysis
            flame_speed: Optional flame speed for flashback analysis
            pressure_signal: Optional pressure data for oscillation analysis

        Returns:
            HazardAssessment with all detected hazards
        """
        hazards: List[Hazard] = []

        # Always check flameout
        self.detect_flameout_risk(flame_signal, flame_threshold)
        hazards.append(self.detection_history[-1])

        # Check delayed ignition if conditions provided
        if ignition_conditions:
            self.detect_delayed_ignition_risk(ignition_conditions)
            hazards.append(self.detection_history[-1])

        # Check flashback if velocity data provided
        if velocity is not None and flame_speed is not None:
            self.detect_flashback_risk(velocity, flame_speed)
            hazards.append(self.detection_history[-1])

        # Check oscillation if pressure data provided
        if pressure_signal is not None:
            self.detect_oscillation_hazard(pressure_signal)
            hazards.append(self.detection_history[-1])

        # Compute overall level
        overall_level = self.compute_overall_hazard_level(hazards)

        # Determine if safe to optimize
        level_order = {
            HazardLevel.NONE: 0,
            HazardLevel.LOW: 1,
            HazardLevel.MODERATE: 2,
            HazardLevel.HIGH: 3,
            HazardLevel.CRITICAL: 4
        }

        safe_to_optimize = level_order[overall_level] < level_order[HazardLevel.HIGH]
        observe_only = level_order[overall_level] >= level_order[HazardLevel.CRITICAL]

        # Generate restrictions
        restrictions = []
        if overall_level == HazardLevel.MODERATE:
            restrictions.append("Limit optimization to conservative moves only")
            restrictions.append("Do not push toward efficiency limits")
        elif overall_level == HazardLevel.HIGH:
            restrictions.append("Optimization restricted to safety-improving moves only")
            restrictions.append("No efficiency optimization allowed")
        elif overall_level == HazardLevel.CRITICAL:
            restrictions.append("ALL optimization suspended")
            restrictions.append("OBSERVE-ONLY mode enforced")

        provenance_hash = hashlib.sha256(
            f"{self.unit_id}_{overall_level}_{len(hazards)}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()

        assessment = HazardAssessment(
            unit_id=self.unit_id,
            overall_level=overall_level,
            hazards=hazards,
            safe_to_optimize=safe_to_optimize,
            observe_only_recommended=observe_only,
            restrictions=restrictions,
            provenance_hash=provenance_hash
        )

        if observe_only:
            logger.critical(
                f"HAZARD ASSESSMENT: {overall_level.value} - OBSERVE-ONLY recommended"
            )
        elif not safe_to_optimize:
            logger.warning(
                f"HAZARD ASSESSMENT: {overall_level.value} - optimization restricted"
            )
        else:
            logger.info(f"HAZARD ASSESSMENT: {overall_level.value} - safe to optimize")

        return assessment

    def _create_hazard(
        self,
        hazard_type: HazardType,
        level: HazardLevel,
        confidence: float,
        description: str,
        action: str,
        trigger_value: Optional[float] = None,
        threshold: Optional[float] = None
    ) -> Hazard:
        """Create a Hazard object with provenance."""
        provenance_hash = hashlib.sha256(
            f"{hazard_type}_{level}_{confidence}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()

        return Hazard(
            hazard_type=hazard_type,
            level=level,
            confidence=confidence,
            trigger_value=trigger_value,
            threshold=threshold,
            description=description,
            recommended_action=action,
            provenance_hash=provenance_hash
        )

    def _calculate_confidence(
        self,
        value: float,
        threshold: float
    ) -> float:
        """Calculate detection confidence based on value vs threshold."""
        if threshold <= 0:
            return 0.5

        ratio = value / threshold
        if ratio >= 1.5:
            return 0.95  # Very confident flame is good
        elif ratio >= 1.0:
            return 0.85  # Confident but near threshold
        elif ratio >= 0.7:
            return 0.90  # Confident there is risk
        else:
            return 0.95  # Very confident there is risk
