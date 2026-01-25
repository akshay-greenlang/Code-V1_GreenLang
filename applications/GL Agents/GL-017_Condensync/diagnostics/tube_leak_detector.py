# -*- coding: utf-8 -*-
"""
Tube Leak Detector for GL-017 CONDENSYNC

Detects condenser tube leaks through hotwell conductivity monitoring,
condensate quality analysis, and pattern recognition for tube leak signatures.

Zero-Hallucination Guarantee:
- All detection uses threshold-based deterministic logic
- Pattern matching with documented algorithms
- No ML inference for critical detections

Key Features:
- Hotwell conductivity monitoring
- Condensate quality analysis (cation conductivity, sodium, silica)
- Pattern recognition for tube leak signatures
- Alert generation with confidence levels
- Severity classification and impact assessment

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

import hashlib
import json
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class LeakSeverity(Enum):
    """Tube leak severity classification."""
    NONE = "none"           # No detectable leak
    TRACE = "trace"         # Barely detectable
    MINOR = "minor"         # Small leak, limited impact
    MODERATE = "moderate"   # Noticeable leak, chemistry impact
    MAJOR = "major"         # Significant leak, turbine risk
    SEVERE = "severe"       # Large leak, immediate action


class LeakType(Enum):
    """Type of tube leak."""
    PINHOLE = "pinhole"             # Small hole, gradual leak
    CRACK = "crack"                 # Stress crack, variable leak
    EROSION = "erosion"             # Inlet erosion damage
    CORROSION = "corrosion"         # Corrosion penetration
    VIBRATION_FATIGUE = "vibration" # Vibration-induced fatigue
    JOINT_FAILURE = "joint_failure" # Tube-to-tubesheet joint
    UNKNOWN = "unknown"


class CoolingWaterType(Enum):
    """Cooling water type."""
    SEAWATER = "seawater"
    BRACKISH = "brackish"
    RIVER = "river"
    LAKE = "lake"
    TOWER_FRESH = "tower_fresh"
    TOWER_TREATED = "tower_treated"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class DetectionConfidence(Enum):
    """Detection confidence level."""
    HIGH = "high"           # > 90% confidence
    MEDIUM = "medium"       # 70-90% confidence
    LOW = "low"             # 50-70% confidence
    UNCERTAIN = "uncertain"  # < 50% confidence


class ChemistryTrend(Enum):
    """Chemistry parameter trend."""
    STABLE = "stable"
    RISING = "rising"
    FALLING = "falling"
    SPIKE = "spike"
    ERRATIC = "erratic"


# ============================================================================
# THRESHOLD CONSTANTS
# ============================================================================

# Conductivity thresholds (uS/cm) by cooling water type
CONDUCTIVITY_THRESHOLDS = {
    CoolingWaterType.SEAWATER: {
        LeakSeverity.TRACE: 0.2,
        LeakSeverity.MINOR: 0.5,
        LeakSeverity.MODERATE: 1.0,
        LeakSeverity.MAJOR: 2.0,
        LeakSeverity.SEVERE: 5.0,
    },
    CoolingWaterType.BRACKISH: {
        LeakSeverity.TRACE: 0.15,
        LeakSeverity.MINOR: 0.4,
        LeakSeverity.MODERATE: 0.8,
        LeakSeverity.MAJOR: 1.5,
        LeakSeverity.SEVERE: 3.0,
    },
    CoolingWaterType.RIVER: {
        LeakSeverity.TRACE: 0.1,
        LeakSeverity.MINOR: 0.2,
        LeakSeverity.MODERATE: 0.5,
        LeakSeverity.MAJOR: 1.0,
        LeakSeverity.SEVERE: 2.0,
    },
    CoolingWaterType.LAKE: {
        LeakSeverity.TRACE: 0.08,
        LeakSeverity.MINOR: 0.15,
        LeakSeverity.MODERATE: 0.4,
        LeakSeverity.MAJOR: 0.8,
        LeakSeverity.SEVERE: 1.5,
    },
    CoolingWaterType.TOWER_FRESH: {
        LeakSeverity.TRACE: 0.1,
        LeakSeverity.MINOR: 0.25,
        LeakSeverity.MODERATE: 0.6,
        LeakSeverity.MAJOR: 1.2,
        LeakSeverity.SEVERE: 2.5,
    },
    CoolingWaterType.TOWER_TREATED: {
        LeakSeverity.TRACE: 0.12,
        LeakSeverity.MINOR: 0.3,
        LeakSeverity.MODERATE: 0.7,
        LeakSeverity.MAJOR: 1.5,
        LeakSeverity.SEVERE: 3.0,
    },
}

# Sodium thresholds (ppb)
SODIUM_THRESHOLDS = {
    LeakSeverity.TRACE: 2.0,
    LeakSeverity.MINOR: 5.0,
    LeakSeverity.MODERATE: 10.0,
    LeakSeverity.MAJOR: 20.0,
    LeakSeverity.SEVERE: 50.0,
}

# Silica thresholds (ppb)
SILICA_THRESHOLDS = {
    LeakSeverity.TRACE: 5.0,
    LeakSeverity.MINOR: 10.0,
    LeakSeverity.MODERATE: 20.0,
    LeakSeverity.MAJOR: 50.0,
    LeakSeverity.SEVERE: 100.0,
}

# Cation conductivity thresholds (uS/cm)
CATION_CONDUCTIVITY_THRESHOLDS = {
    LeakSeverity.TRACE: 0.15,
    LeakSeverity.MINOR: 0.3,
    LeakSeverity.MODERATE: 0.5,
    LeakSeverity.MAJOR: 1.0,
    LeakSeverity.SEVERE: 2.0,
}

# Chloride thresholds (ppb) - critical for seawater
CHLORIDE_THRESHOLDS = {
    LeakSeverity.TRACE: 3.0,
    LeakSeverity.MINOR: 10.0,
    LeakSeverity.MODERATE: 25.0,
    LeakSeverity.MAJOR: 50.0,
    LeakSeverity.SEVERE: 100.0,
}

# Normal baseline values
NORMAL_BASELINES = {
    "conductivity_us_cm": 0.055,
    "cation_conductivity_us_cm": 0.08,
    "sodium_ppb": 1.0,
    "silica_ppb": 3.0,
    "chloride_ppb": 2.0,
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class TubeLeakDetectorConfig:
    """Configuration for tube leak detector."""

    # Cooling water type
    cooling_water_type: CoolingWaterType = CoolingWaterType.TOWER_TREATED

    # Baseline chemistry values (override defaults)
    baseline_conductivity: Optional[float] = None
    baseline_sodium: Optional[float] = None
    baseline_silica: Optional[float] = None
    baseline_chloride: Optional[float] = None

    # Rolling window for trend analysis
    rolling_window_minutes: int = 30

    # Minimum data points for detection
    min_data_points: int = 5

    # Spike detection threshold (multiples of std dev)
    spike_threshold_sigma: float = 3.0

    # Rate of change threshold for rapid leak detection
    rapid_change_threshold_pct_per_hour: float = 50.0

    # Detection sensitivity
    sensitivity: str = "normal"  # low, normal, high


@dataclass
class ChemistryDataPoint:
    """Single chemistry data point."""
    timestamp: datetime
    specific_conductivity_us_cm: Optional[float] = None
    cation_conductivity_us_cm: Optional[float] = None
    sodium_ppb: Optional[float] = None
    silica_ppb: Optional[float] = None
    chloride_ppb: Optional[float] = None
    dissolved_oxygen_ppb: Optional[float] = None
    ph: Optional[float] = None
    hotwell_level_pct: Optional[float] = None
    condensate_flow_t_hr: Optional[float] = None
    condenser_load_pct: Optional[float] = None
    is_validated: bool = True
    quality_flag: str = "good"


@dataclass
class ChemistryTrendAnalysis:
    """Analysis of chemistry parameter trend."""
    parameter_name: str
    current_value: float
    baseline_value: float
    elevation_from_baseline: float
    elevation_pct: float
    trend: ChemistryTrend
    rate_of_change_per_hour: float
    rolling_mean: float
    rolling_std: float
    is_spiking: bool
    spike_magnitude: Optional[float] = None


@dataclass
class LeakRateEstimate:
    """Estimated tube leak rate."""
    leak_rate_kg_hr: float
    leak_rate_uncertainty_kg_hr: float
    estimation_method: str
    confidence: DetectionConfidence
    supporting_evidence: List[str]


@dataclass
class LeakSignature:
    """Pattern signature for tube leak."""
    primary_indicator: str
    secondary_indicators: List[str]
    pattern_description: str
    leak_type: LeakType
    confidence: float


@dataclass
class TubeLeakAlert:
    """Tube leak alert."""
    alert_id: str
    timestamp: datetime
    severity: AlertSeverity
    alert_type: str
    message: str
    leak_severity: LeakSeverity
    primary_indicator: str
    indicator_value: float
    threshold: float
    recommended_action: Optional[str] = None
    evidence: List[str] = field(default_factory=list)


@dataclass
class TurbineRiskAssessment:
    """Assessment of turbine damage risk from tube leak."""
    risk_level: str  # low, moderate, high, critical
    chloride_risk: bool
    sodium_risk: bool
    estimated_hours_to_limit: Optional[float]
    recommended_action: str
    evidence: List[str]


@dataclass
class TubeLeakDetection:
    """Complete tube leak detection result."""
    condenser_id: str
    detection_timestamp: datetime

    # Overall assessment
    leak_detected: bool
    leak_severity: LeakSeverity
    detection_confidence: DetectionConfidence

    # Leak characteristics
    leak_type: LeakType
    leak_signature: Optional[LeakSignature]

    # Chemistry analysis
    conductivity_analysis: Optional[ChemistryTrendAnalysis]
    cation_conductivity_analysis: Optional[ChemistryTrendAnalysis]
    sodium_analysis: Optional[ChemistryTrendAnalysis]
    silica_analysis: Optional[ChemistryTrendAnalysis]
    chloride_analysis: Optional[ChemistryTrendAnalysis]

    # Leak rate estimate
    leak_rate_estimate: Optional[LeakRateEstimate]

    # Turbine risk
    turbine_risk: TurbineRiskAssessment

    # Impact assessment
    chemistry_hold_required: bool
    estimated_repair_urgency_hours: float

    # Alerts
    active_alerts: List[TubeLeakAlert]

    # Provenance
    methodology: str
    data_points_analyzed: int
    provenance_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "condenser_id": self.condenser_id,
            "detection_timestamp": self.detection_timestamp.isoformat(),
            "leak_detected": self.leak_detected,
            "leak_severity": self.leak_severity.value,
            "detection_confidence": self.detection_confidence.value,
            "leak_type": self.leak_type.value,
            "chemistry": {
                "conductivity": {
                    "current": self.conductivity_analysis.current_value
                    if self.conductivity_analysis else None,
                    "elevation_pct": self.conductivity_analysis.elevation_pct
                    if self.conductivity_analysis else None,
                    "trend": self.conductivity_analysis.trend.value
                    if self.conductivity_analysis else None,
                },
                "sodium": {
                    "current_ppb": self.sodium_analysis.current_value
                    if self.sodium_analysis else None,
                    "elevation_pct": self.sodium_analysis.elevation_pct
                    if self.sodium_analysis else None,
                },
                "chloride": {
                    "current_ppb": self.chloride_analysis.current_value
                    if self.chloride_analysis else None,
                    "elevation_pct": self.chloride_analysis.elevation_pct
                    if self.chloride_analysis else None,
                },
            },
            "leak_rate": {
                "rate_kg_hr": self.leak_rate_estimate.leak_rate_kg_hr
                if self.leak_rate_estimate else None,
                "confidence": self.leak_rate_estimate.confidence.value
                if self.leak_rate_estimate else None,
            },
            "turbine_risk": {
                "level": self.turbine_risk.risk_level,
                "hours_to_limit": self.turbine_risk.estimated_hours_to_limit,
            },
            "chemistry_hold_required": self.chemistry_hold_required,
            "repair_urgency_hours": self.estimated_repair_urgency_hours,
            "active_alerts": [
                {"severity": a.severity.value, "type": a.alert_type, "message": a.message}
                for a in self.active_alerts
            ],
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class CondenserChemistryProfile:
    """Condenser chemistry profile."""
    condenser_id: str
    cooling_water_type: CoolingWaterType = CoolingWaterType.TOWER_TREATED
    tube_material: str = "titanium"
    baseline_conductivity_us_cm: float = 0.055
    baseline_cation_conductivity_us_cm: float = 0.08
    baseline_sodium_ppb: float = 1.0
    baseline_silica_ppb: float = 3.0
    baseline_chloride_ppb: float = 2.0
    cw_conductivity_us_cm: float = 500.0  # Cooling water conductivity


# ============================================================================
# MAIN TUBE LEAK DETECTOR CLASS
# ============================================================================

class TubeLeakDetector:
    """
    Tube leak detection system for power plant condensers.

    Monitors hotwell conductivity, condensate quality parameters,
    and applies pattern recognition for tube leak signatures.

    Zero-Hallucination Guarantee:
    - Detection uses deterministic threshold logic
    - Leak rate estimates from mass balance calculations
    - Pattern matching with documented algorithms

    Example:
        >>> detector = TubeLeakDetector()
        >>> data = [ChemistryDataPoint(timestamp=..., specific_conductivity_us_cm=0.15), ...]
        >>> profile = CondenserChemistryProfile(condenser_id="COND-01")
        >>> detection = detector.detect_tube_leak(data, profile)
        >>> print(f"Leak detected: {detection.leak_detected}")
    """

    VERSION = "1.0.0"
    METHODOLOGY = "Multi-parameter threshold analysis with pattern recognition"

    def __init__(self, config: Optional[TubeLeakDetectorConfig] = None):
        """
        Initialize tube leak detector.

        Args:
            config: Detector configuration (optional)
        """
        self.config = config or TubeLeakDetectorConfig()
        self._sensitivity_factor = self._get_sensitivity_factor()
        logger.info(f"TubeLeakDetector initialized with {self.config.sensitivity} sensitivity")

    def _get_sensitivity_factor(self) -> float:
        """Get sensitivity multiplier for thresholds."""
        sensitivity_map = {
            "low": 1.5,      # Less sensitive, higher thresholds
            "normal": 1.0,
            "high": 0.7,     # More sensitive, lower thresholds
        }
        return sensitivity_map.get(self.config.sensitivity, 1.0)

    # ========================================================================
    # CHEMISTRY TREND ANALYSIS
    # ========================================================================

    def _analyze_parameter_trend(
        self,
        data_points: List[ChemistryDataPoint],
        param_name: str,
        extractor,
        baseline: float
    ) -> Optional[ChemistryTrendAnalysis]:
        """
        Analyze trend for a single chemistry parameter.

        Args:
            data_points: List of chemistry data points
            param_name: Parameter name
            extractor: Function to extract parameter value
            baseline: Baseline value

        Returns:
            ChemistryTrendAnalysis or None if insufficient data
        """
        # Extract valid values
        values_with_time = [
            (dp.timestamp, extractor(dp))
            for dp in data_points
            if extractor(dp) is not None and dp.is_validated
        ]

        if len(values_with_time) < 3:
            return None

        # Sort by timestamp
        values_with_time.sort(key=lambda x: x[0])
        timestamps, values = zip(*values_with_time)

        # Current value (latest)
        current_value = values[-1]

        # Rolling statistics
        window_values = list(values[-min(len(values), 10):])
        rolling_mean = statistics.mean(window_values)
        rolling_std = statistics.stdev(window_values) if len(window_values) > 1 else 0.0

        # Elevation from baseline
        elevation = current_value - baseline
        elevation_pct = (elevation / baseline) * 100 if baseline > 0 else 0

        # Rate of change
        if len(values) >= 2:
            time_diff_hours = (timestamps[-1] - timestamps[0]).total_seconds() / 3600
            if time_diff_hours > 0:
                value_change = values[-1] - values[0]
                rate_of_change = value_change / time_diff_hours
            else:
                rate_of_change = 0.0
        else:
            rate_of_change = 0.0

        # Detect spike
        is_spiking = False
        spike_magnitude = None
        if rolling_std > 0:
            z_score = (current_value - rolling_mean) / rolling_std
            if z_score > self.config.spike_threshold_sigma:
                is_spiking = True
                spike_magnitude = z_score

        # Classify trend
        trend = self._classify_trend(values, rate_of_change, rolling_std, is_spiking)

        return ChemistryTrendAnalysis(
            parameter_name=param_name,
            current_value=current_value,
            baseline_value=baseline,
            elevation_from_baseline=elevation,
            elevation_pct=elevation_pct,
            trend=trend,
            rate_of_change_per_hour=rate_of_change,
            rolling_mean=rolling_mean,
            rolling_std=rolling_std,
            is_spiking=is_spiking,
            spike_magnitude=spike_magnitude,
        )

    def _classify_trend(
        self,
        values: Tuple,
        rate_of_change: float,
        std_dev: float,
        is_spiking: bool
    ) -> ChemistryTrend:
        """Classify the trend pattern."""
        if is_spiking:
            return ChemistryTrend.SPIKE

        # Calculate coefficient of variation
        mean_val = statistics.mean(values) if values else 0
        cv = std_dev / mean_val if mean_val > 0 else 0

        # High variability
        if cv > 0.3:
            return ChemistryTrend.ERRATIC

        # Steady rising
        if rate_of_change > 0.01 * mean_val:  # > 1% per hour
            return ChemistryTrend.RISING

        # Falling
        if rate_of_change < -0.01 * mean_val:
            return ChemistryTrend.FALLING

        return ChemistryTrend.STABLE

    # ========================================================================
    # SEVERITY CLASSIFICATION
    # ========================================================================

    def _classify_severity_by_conductivity(
        self,
        conductivity: float,
        baseline: float,
        cw_type: CoolingWaterType
    ) -> LeakSeverity:
        """Classify severity based on conductivity."""
        elevation = conductivity - baseline
        sf = self._sensitivity_factor

        thresholds = CONDUCTIVITY_THRESHOLDS.get(
            cw_type,
            CONDUCTIVITY_THRESHOLDS[CoolingWaterType.TOWER_TREATED]
        )

        for severity in [LeakSeverity.SEVERE, LeakSeverity.MAJOR, LeakSeverity.MODERATE,
                        LeakSeverity.MINOR, LeakSeverity.TRACE]:
            if elevation >= thresholds[severity] * sf:
                return severity

        return LeakSeverity.NONE

    def _classify_severity_by_sodium(self, sodium_ppb: float, baseline: float) -> LeakSeverity:
        """Classify severity based on sodium."""
        elevation = sodium_ppb - baseline
        sf = self._sensitivity_factor

        for severity in [LeakSeverity.SEVERE, LeakSeverity.MAJOR, LeakSeverity.MODERATE,
                        LeakSeverity.MINOR, LeakSeverity.TRACE]:
            if elevation >= SODIUM_THRESHOLDS[severity] * sf:
                return severity

        return LeakSeverity.NONE

    def _classify_severity_by_chloride(self, chloride_ppb: float, baseline: float) -> LeakSeverity:
        """Classify severity based on chloride."""
        elevation = chloride_ppb - baseline
        sf = self._sensitivity_factor

        for severity in [LeakSeverity.SEVERE, LeakSeverity.MAJOR, LeakSeverity.MODERATE,
                        LeakSeverity.MINOR, LeakSeverity.TRACE]:
            if elevation >= CHLORIDE_THRESHOLDS[severity] * sf:
                return severity

        return LeakSeverity.NONE

    def _classify_overall_severity(
        self,
        conductivity_severity: LeakSeverity,
        sodium_severity: LeakSeverity,
        chloride_severity: LeakSeverity,
        silica_severity: LeakSeverity
    ) -> LeakSeverity:
        """Determine overall severity from multiple indicators."""
        severity_order = [
            LeakSeverity.SEVERE,
            LeakSeverity.MAJOR,
            LeakSeverity.MODERATE,
            LeakSeverity.MINOR,
            LeakSeverity.TRACE,
            LeakSeverity.NONE,
        ]

        severities = [conductivity_severity, sodium_severity, chloride_severity, silica_severity]

        # Return worst severity
        for sev in severity_order:
            if sev in severities:
                return sev

        return LeakSeverity.NONE

    # ========================================================================
    # LEAK TYPE IDENTIFICATION
    # ========================================================================

    def _identify_leak_type(
        self,
        conductivity_analysis: Optional[ChemistryTrendAnalysis],
        sodium_analysis: Optional[ChemistryTrendAnalysis],
        chloride_analysis: Optional[ChemistryTrendAnalysis]
    ) -> Tuple[LeakType, LeakSignature]:
        """
        Identify leak type from chemistry patterns.

        Args:
            conductivity_analysis: Conductivity trend analysis
            sodium_analysis: Sodium trend analysis
            chloride_analysis: Chloride trend analysis

        Returns:
            Tuple of (LeakType, LeakSignature)
        """
        indicators = []

        # Analyze patterns
        if conductivity_analysis:
            if conductivity_analysis.is_spiking:
                indicators.append("conductivity_spike")
            elif conductivity_analysis.trend == ChemistryTrend.RISING:
                indicators.append("conductivity_rising")

        if sodium_analysis:
            if sodium_analysis.is_spiking:
                indicators.append("sodium_spike")
            elif sodium_analysis.trend == ChemistryTrend.RISING:
                indicators.append("sodium_rising")

        if chloride_analysis:
            if chloride_analysis.is_spiking:
                indicators.append("chloride_spike")
            elif chloride_analysis.trend == ChemistryTrend.RISING:
                indicators.append("chloride_rising")

        # Pattern matching for leak types
        if "conductivity_spike" in indicators or "sodium_spike" in indicators:
            leak_type = LeakType.CRACK
            pattern = "Sudden spike suggests stress crack with intermittent flow"
            confidence = 0.75
        elif "conductivity_rising" in indicators and "sodium_rising" in indicators:
            if chloride_analysis and chloride_analysis.elevation_pct > 50:
                leak_type = LeakType.CORROSION
                pattern = "Gradual rise with chloride suggests corrosion penetration"
                confidence = 0.80
            else:
                leak_type = LeakType.PINHOLE
                pattern = "Steady gradual increase suggests small pinhole leak"
                confidence = 0.70
        elif conductivity_analysis and conductivity_analysis.trend == ChemistryTrend.ERRATIC:
            leak_type = LeakType.VIBRATION_FATIGUE
            pattern = "Erratic pattern suggests vibration-induced fatigue crack"
            confidence = 0.65
        else:
            leak_type = LeakType.UNKNOWN
            pattern = "Pattern does not match known leak signatures"
            confidence = 0.40

        primary = "conductivity" if conductivity_analysis else "unknown"
        secondary = [ind for ind in indicators if ind != primary]

        signature = LeakSignature(
            primary_indicator=primary,
            secondary_indicators=secondary,
            pattern_description=pattern,
            leak_type=leak_type,
            confidence=confidence,
        )

        return leak_type, signature

    # ========================================================================
    # LEAK RATE ESTIMATION
    # ========================================================================

    def _estimate_leak_rate(
        self,
        conductivity_elevation: float,
        profile: CondenserChemistryProfile,
        condensate_flow: Optional[float]
    ) -> Optional[LeakRateEstimate]:
        """
        Estimate leak rate using mass balance.

        Args:
            conductivity_elevation: Conductivity elevation (uS/cm)
            profile: Chemistry profile
            condensate_flow: Condensate flow rate (t/hr)

        Returns:
            LeakRateEstimate or None if insufficient data
        """
        if conductivity_elevation <= 0:
            return LeakRateEstimate(
                leak_rate_kg_hr=0.0,
                leak_rate_uncertainty_kg_hr=0.0,
                estimation_method="none",
                confidence=DetectionConfidence.HIGH,
                supporting_evidence=["No conductivity elevation detected"],
            )

        # Default condensate flow if not provided
        if condensate_flow is None:
            condensate_flow = 500.0  # Typical 500 t/hr

        # Mass balance: CW_leak * CW_cond = Condensate_flow * Cond_elevation
        # CW_leak = Condensate_flow * (Cond_elevation / CW_cond)
        cw_conductivity = profile.cw_conductivity_us_cm

        if cw_conductivity > 0:
            # Convert to kg/hr (condensate flow in t/hr = 1000 kg/hr)
            leak_rate = condensate_flow * 1000 * (conductivity_elevation / cw_conductivity)
        else:
            leak_rate = 0.0

        # Uncertainty (typically 20-30% for this method)
        uncertainty = leak_rate * 0.25

        evidence = [
            f"Conductivity elevation: {conductivity_elevation:.3f} uS/cm",
            f"CW conductivity: {cw_conductivity:.0f} uS/cm",
            f"Condensate flow: {condensate_flow:.0f} t/hr",
        ]

        return LeakRateEstimate(
            leak_rate_kg_hr=leak_rate,
            leak_rate_uncertainty_kg_hr=uncertainty,
            estimation_method="mass_balance_conductivity",
            confidence=DetectionConfidence.MEDIUM,
            supporting_evidence=evidence,
        )

    # ========================================================================
    # TURBINE RISK ASSESSMENT
    # ========================================================================

    def _assess_turbine_risk(
        self,
        sodium_analysis: Optional[ChemistryTrendAnalysis],
        chloride_analysis: Optional[ChemistryTrendAnalysis],
        leak_severity: LeakSeverity
    ) -> TurbineRiskAssessment:
        """
        Assess risk of turbine damage from tube leak.

        Args:
            sodium_analysis: Sodium trend analysis
            chloride_analysis: Chloride trend analysis
            leak_severity: Overall leak severity

        Returns:
            TurbineRiskAssessment
        """
        evidence = []

        # Chloride risk (stress corrosion cracking of turbine blades)
        chloride_risk = False
        if chloride_analysis:
            if chloride_analysis.current_value > 10:
                chloride_risk = True
                evidence.append(f"Chloride at {chloride_analysis.current_value:.0f} ppb - SCC risk")

        # Sodium risk (NaOH attack on turbine steel)
        sodium_risk = False
        if sodium_analysis:
            if sodium_analysis.current_value > 10:
                sodium_risk = True
                evidence.append(f"Sodium at {sodium_analysis.current_value:.0f} ppb - caustic attack risk")

        # Determine risk level
        if leak_severity in [LeakSeverity.SEVERE, LeakSeverity.MAJOR]:
            risk_level = "critical"
            hours_to_limit = 2.0
            action = "Immediate unit shutdown recommended"
        elif chloride_risk or sodium_risk:
            risk_level = "high"
            hours_to_limit = 8.0
            action = "Reduce load and prepare for shutdown within 8 hours"
        elif leak_severity == LeakSeverity.MODERATE:
            risk_level = "moderate"
            hours_to_limit = 24.0
            action = "Monitor closely, plan shutdown within 24 hours"
        elif leak_severity in [LeakSeverity.MINOR, LeakSeverity.TRACE]:
            risk_level = "low"
            hours_to_limit = 72.0
            action = "Increased monitoring, plan repair at next opportunity"
        else:
            risk_level = "none"
            hours_to_limit = None
            action = "Continue normal operation"

        return TurbineRiskAssessment(
            risk_level=risk_level,
            chloride_risk=chloride_risk,
            sodium_risk=sodium_risk,
            estimated_hours_to_limit=hours_to_limit,
            recommended_action=action,
            evidence=evidence,
        )

    # ========================================================================
    # ALERT GENERATION
    # ========================================================================

    def _generate_alerts(
        self,
        leak_severity: LeakSeverity,
        conductivity_analysis: Optional[ChemistryTrendAnalysis],
        sodium_analysis: Optional[ChemistryTrendAnalysis],
        chloride_analysis: Optional[ChemistryTrendAnalysis],
        turbine_risk: TurbineRiskAssessment,
        condenser_id: str
    ) -> List[TubeLeakAlert]:
        """Generate alerts based on detection results."""
        alerts = []
        now = datetime.now(timezone.utc)

        # Severity-based alerts
        if leak_severity == LeakSeverity.SEVERE:
            alerts.append(TubeLeakAlert(
                alert_id=f"{condenser_id}_TUBE_SEVERE_{now.strftime('%Y%m%d%H%M')}",
                timestamp=now,
                severity=AlertSeverity.EMERGENCY,
                alert_type="SEVERE_TUBE_LEAK",
                message="Severe tube leak detected - immediate action required",
                leak_severity=leak_severity,
                primary_indicator="conductivity",
                indicator_value=conductivity_analysis.current_value if conductivity_analysis else 0,
                threshold=CONDUCTIVITY_THRESHOLDS[self.config.cooling_water_type][LeakSeverity.SEVERE],
                recommended_action="Initiate emergency shutdown procedure",
                evidence=["Chemistry excursion exceeds safe operating limits"],
            ))
        elif leak_severity == LeakSeverity.MAJOR:
            alerts.append(TubeLeakAlert(
                alert_id=f"{condenser_id}_TUBE_MAJOR_{now.strftime('%Y%m%d%H%M')}",
                timestamp=now,
                severity=AlertSeverity.CRITICAL,
                alert_type="MAJOR_TUBE_LEAK",
                message="Major tube leak detected - prepare for shutdown",
                leak_severity=leak_severity,
                primary_indicator="conductivity",
                indicator_value=conductivity_analysis.current_value if conductivity_analysis else 0,
                threshold=CONDUCTIVITY_THRESHOLDS[self.config.cooling_water_type][LeakSeverity.MAJOR],
                recommended_action="Reduce load and prepare for orderly shutdown",
                evidence=["Significant chemistry deviation detected"],
            ))
        elif leak_severity == LeakSeverity.MODERATE:
            alerts.append(TubeLeakAlert(
                alert_id=f"{condenser_id}_TUBE_MODERATE_{now.strftime('%Y%m%d%H%M')}",
                timestamp=now,
                severity=AlertSeverity.WARNING,
                alert_type="MODERATE_TUBE_LEAK",
                message="Moderate tube leak detected - increased monitoring required",
                leak_severity=leak_severity,
                primary_indicator="conductivity",
                indicator_value=conductivity_analysis.current_value if conductivity_analysis else 0,
                threshold=CONDUCTIVITY_THRESHOLDS[self.config.cooling_water_type][LeakSeverity.MODERATE],
                recommended_action="Increase sampling frequency, prepare for shutdown",
                evidence=["Chemistry deviation above normal limits"],
            ))

        # Chloride-specific alert
        if chloride_analysis and chloride_analysis.current_value > 20:
            alerts.append(TubeLeakAlert(
                alert_id=f"{condenser_id}_CHLORIDE_HIGH_{now.strftime('%Y%m%d%H%M')}",
                timestamp=now,
                severity=AlertSeverity.CRITICAL,
                alert_type="HIGH_CHLORIDE",
                message=f"Chloride at {chloride_analysis.current_value:.0f} ppb - turbine SCC risk",
                leak_severity=leak_severity,
                primary_indicator="chloride",
                indicator_value=chloride_analysis.current_value,
                threshold=20.0,
                recommended_action="Chloride limit exceeded - turbine blade damage risk",
                evidence=[f"Chloride {chloride_analysis.elevation_pct:.0f}% above baseline"],
            ))

        # Sodium-specific alert
        if sodium_analysis and sodium_analysis.current_value > 15:
            alerts.append(TubeLeakAlert(
                alert_id=f"{condenser_id}_SODIUM_HIGH_{now.strftime('%Y%m%d%H%M')}",
                timestamp=now,
                severity=AlertSeverity.WARNING,
                alert_type="HIGH_SODIUM",
                message=f"Sodium at {sodium_analysis.current_value:.0f} ppb - caustic attack risk",
                leak_severity=leak_severity,
                primary_indicator="sodium",
                indicator_value=sodium_analysis.current_value,
                threshold=15.0,
                recommended_action="Monitor sodium closely, risk of caustic attack",
                evidence=[f"Sodium {sodium_analysis.elevation_pct:.0f}% above baseline"],
            ))

        # Spike detection alert
        if conductivity_analysis and conductivity_analysis.is_spiking:
            alerts.append(TubeLeakAlert(
                alert_id=f"{condenser_id}_SPIKE_{now.strftime('%Y%m%d%H%M')}",
                timestamp=now,
                severity=AlertSeverity.WARNING,
                alert_type="CHEMISTRY_SPIKE",
                message="Sudden conductivity spike detected - possible tube crack",
                leak_severity=leak_severity,
                primary_indicator="conductivity_spike",
                indicator_value=conductivity_analysis.current_value,
                threshold=conductivity_analysis.rolling_mean + 3 * conductivity_analysis.rolling_std,
                recommended_action="Investigate sudden chemistry change",
                evidence=[f"Spike magnitude: {conductivity_analysis.spike_magnitude:.1f} sigma"],
            ))

        return alerts

    # ========================================================================
    # PROVENANCE
    # ========================================================================

    def _compute_provenance_hash(
        self,
        condenser_id: str,
        leak_severity: LeakSeverity,
        conductivity: Optional[float],
        data_points_count: int
    ) -> str:
        """Compute SHA-256 hash for detection provenance."""
        data = {
            "version": self.VERSION,
            "condenser_id": condenser_id,
            "leak_severity": leak_severity.value,
            "conductivity": round(conductivity, 6) if conductivity else None,
            "data_points": data_points_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    # ========================================================================
    # MAIN DETECTION METHOD
    # ========================================================================

    def detect_tube_leak(
        self,
        data_points: List[ChemistryDataPoint],
        profile: CondenserChemistryProfile,
        detection_timestamp: Optional[datetime] = None
    ) -> TubeLeakDetection:
        """
        Detect and classify condenser tube leak.

        Args:
            data_points: Historical chemistry data points
            profile: Condenser chemistry profile
            detection_timestamp: Timestamp for detection (default: now)

        Returns:
            TubeLeakDetection with complete analysis

        Raises:
            ValueError: If insufficient data points
        """
        if detection_timestamp is None:
            detection_timestamp = datetime.now(timezone.utc)

        logger.info(f"Performing tube leak detection for {profile.condenser_id}")

        # Validate input
        valid_points = [dp for dp in data_points if dp.is_validated]
        if len(valid_points) < self.config.min_data_points:
            raise ValueError(
                f"Insufficient data points: {len(valid_points)} < {self.config.min_data_points}"
            )

        # Sort by timestamp
        valid_points.sort(key=lambda dp: dp.timestamp)

        # Analyze each parameter
        conductivity_analysis = self._analyze_parameter_trend(
            valid_points, "specific_conductivity",
            lambda dp: dp.specific_conductivity_us_cm,
            profile.baseline_conductivity_us_cm
        )

        cation_cond_analysis = self._analyze_parameter_trend(
            valid_points, "cation_conductivity",
            lambda dp: dp.cation_conductivity_us_cm,
            profile.baseline_cation_conductivity_us_cm
        )

        sodium_analysis = self._analyze_parameter_trend(
            valid_points, "sodium",
            lambda dp: dp.sodium_ppb,
            profile.baseline_sodium_ppb
        )

        silica_analysis = self._analyze_parameter_trend(
            valid_points, "silica",
            lambda dp: dp.silica_ppb,
            profile.baseline_silica_ppb
        )

        chloride_analysis = self._analyze_parameter_trend(
            valid_points, "chloride",
            lambda dp: dp.chloride_ppb,
            profile.baseline_chloride_ppb
        )

        # Classify severity for each parameter
        cond_severity = LeakSeverity.NONE
        if conductivity_analysis:
            cond_severity = self._classify_severity_by_conductivity(
                conductivity_analysis.current_value,
                profile.baseline_conductivity_us_cm,
                profile.cooling_water_type
            )

        sodium_severity = LeakSeverity.NONE
        if sodium_analysis:
            sodium_severity = self._classify_severity_by_sodium(
                sodium_analysis.current_value,
                profile.baseline_sodium_ppb
            )

        chloride_severity = LeakSeverity.NONE
        if chloride_analysis:
            chloride_severity = self._classify_severity_by_chloride(
                chloride_analysis.current_value,
                profile.baseline_chloride_ppb
            )

        silica_severity = LeakSeverity.NONE
        if silica_analysis:
            elevation = silica_analysis.current_value - profile.baseline_silica_ppb
            for sev in [LeakSeverity.SEVERE, LeakSeverity.MAJOR, LeakSeverity.MODERATE,
                       LeakSeverity.MINOR, LeakSeverity.TRACE]:
                if elevation >= SILICA_THRESHOLDS[sev] * self._sensitivity_factor:
                    silica_severity = sev
                    break

        # Overall severity
        leak_severity = self._classify_overall_severity(
            cond_severity, sodium_severity, chloride_severity, silica_severity
        )

        # Detect leak
        leak_detected = leak_severity != LeakSeverity.NONE

        # Determine confidence
        indicators_elevated = sum([
            1 for s in [cond_severity, sodium_severity, chloride_severity, silica_severity]
            if s != LeakSeverity.NONE
        ])

        if indicators_elevated >= 3:
            detection_confidence = DetectionConfidence.HIGH
        elif indicators_elevated >= 2:
            detection_confidence = DetectionConfidence.MEDIUM
        elif indicators_elevated == 1:
            detection_confidence = DetectionConfidence.LOW
        else:
            detection_confidence = DetectionConfidence.HIGH  # High confidence of no leak

        # Identify leak type
        leak_type, leak_signature = self._identify_leak_type(
            conductivity_analysis, sodium_analysis, chloride_analysis
        )

        # Estimate leak rate
        leak_rate_estimate = None
        if conductivity_analysis and conductivity_analysis.elevation_from_baseline > 0:
            condensate_flow = None
            for dp in reversed(valid_points):
                if dp.condensate_flow_t_hr:
                    condensate_flow = dp.condensate_flow_t_hr
                    break
            leak_rate_estimate = self._estimate_leak_rate(
                conductivity_analysis.elevation_from_baseline,
                profile,
                condensate_flow
            )

        # Turbine risk assessment
        turbine_risk = self._assess_turbine_risk(sodium_analysis, chloride_analysis, leak_severity)

        # Chemistry hold required?
        chemistry_hold = leak_severity in [LeakSeverity.MAJOR, LeakSeverity.SEVERE]

        # Repair urgency
        urgency_hours_map = {
            LeakSeverity.SEVERE: 2.0,
            LeakSeverity.MAJOR: 8.0,
            LeakSeverity.MODERATE: 24.0,
            LeakSeverity.MINOR: 72.0,
            LeakSeverity.TRACE: 168.0,  # 1 week
            LeakSeverity.NONE: float('inf'),
        }
        repair_urgency = urgency_hours_map.get(leak_severity, float('inf'))

        # Generate alerts
        alerts = self._generate_alerts(
            leak_severity, conductivity_analysis, sodium_analysis,
            chloride_analysis, turbine_risk, profile.condenser_id
        )

        # Provenance hash
        current_cond = conductivity_analysis.current_value if conductivity_analysis else None
        provenance_hash = self._compute_provenance_hash(
            profile.condenser_id, leak_severity, current_cond, len(valid_points)
        )

        logger.info(
            f"Tube leak detection complete for {profile.condenser_id}: "
            f"detected={leak_detected}, severity={leak_severity.value}, "
            f"type={leak_type.value}"
        )

        return TubeLeakDetection(
            condenser_id=profile.condenser_id,
            detection_timestamp=detection_timestamp,
            leak_detected=leak_detected,
            leak_severity=leak_severity,
            detection_confidence=detection_confidence,
            leak_type=leak_type,
            leak_signature=leak_signature if leak_detected else None,
            conductivity_analysis=conductivity_analysis,
            cation_conductivity_analysis=cation_cond_analysis,
            sodium_analysis=sodium_analysis,
            silica_analysis=silica_analysis,
            chloride_analysis=chloride_analysis,
            leak_rate_estimate=leak_rate_estimate,
            turbine_risk=turbine_risk,
            chemistry_hold_required=chemistry_hold,
            estimated_repair_urgency_hours=repair_urgency,
            active_alerts=alerts,
            methodology=self.METHODOLOGY,
            data_points_analyzed=len(valid_points),
            provenance_hash=provenance_hash,
        )

    def generate_detection_report(
        self,
        detection: TubeLeakDetection
    ) -> str:
        """
        Generate detailed detection report.

        Args:
            detection: TubeLeakDetection result

        Returns:
            Formatted report text
        """
        lines = [
            "=" * 80,
            "          CONDENSER TUBE LEAK DETECTION REPORT",
            "=" * 80,
            "",
            f"Condenser: {detection.condenser_id}",
            f"Detection Time: {detection.detection_timestamp.strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            "DETECTION SUMMARY",
            "-" * 40,
            f"  Leak Detected:     {detection.leak_detected}",
            f"  Severity:          {detection.leak_severity.value.upper()}",
            f"  Confidence:        {detection.detection_confidence.value}",
            f"  Leak Type:         {detection.leak_type.value}",
            "",
            "CHEMISTRY ANALYSIS",
            "-" * 40,
        ]

        if detection.conductivity_analysis:
            ca = detection.conductivity_analysis
            lines.append(
                f"  Conductivity:      {ca.current_value:.3f} uS/cm "
                f"(baseline: {ca.baseline_value:.3f}, +{ca.elevation_pct:.0f}%)"
            )

        if detection.sodium_analysis:
            sa = detection.sodium_analysis
            lines.append(
                f"  Sodium:            {sa.current_value:.1f} ppb "
                f"(baseline: {sa.baseline_value:.1f}, +{sa.elevation_pct:.0f}%)"
            )

        if detection.chloride_analysis:
            cl = detection.chloride_analysis
            lines.append(
                f"  Chloride:          {cl.current_value:.1f} ppb "
                f"(baseline: {cl.baseline_value:.1f}, +{cl.elevation_pct:.0f}%)"
            )

        if detection.silica_analysis:
            si = detection.silica_analysis
            lines.append(
                f"  Silica:            {si.current_value:.1f} ppb "
                f"(baseline: {si.baseline_value:.1f}, +{si.elevation_pct:.0f}%)"
            )

        lines.extend([
            "",
            "TURBINE RISK ASSESSMENT",
            "-" * 40,
            f"  Risk Level:        {detection.turbine_risk.risk_level.upper()}",
            f"  Chloride Risk:     {detection.turbine_risk.chloride_risk}",
            f"  Sodium Risk:       {detection.turbine_risk.sodium_risk}",
            f"  Hours to Limit:    {detection.turbine_risk.estimated_hours_to_limit}",
            f"  Recommendation:    {detection.turbine_risk.recommended_action}",
            "",
            "ACTION REQUIRED",
            "-" * 40,
            f"  Chemistry Hold:    {detection.chemistry_hold_required}",
            f"  Repair Urgency:    {detection.estimated_repair_urgency_hours:.0f} hours",
        ])

        if detection.leak_rate_estimate:
            lines.extend([
                "",
                f"  Est. Leak Rate:    {detection.leak_rate_estimate.leak_rate_kg_hr:.1f} +/- "
                f"{detection.leak_rate_estimate.leak_rate_uncertainty_kg_hr:.1f} kg/hr",
            ])

        if detection.active_alerts:
            lines.extend([
                "",
                "ACTIVE ALERTS",
                "-" * 40,
            ])
            for alert in detection.active_alerts:
                lines.append(f"  [{alert.severity.value.upper():8s}] {alert.message}")

        lines.extend([
            "",
            "=" * 80,
            f"Methodology: {detection.methodology}",
            f"Provenance Hash: {detection.provenance_hash}",
            "=" * 80,
        ])

        return "\n".join(lines)
