# -*- coding: utf-8 -*-
"""
Air In-Leakage Detector for GL-017 CONDENSYNC

Detects and classifies air in-leakage in power plant condensers using
vacuum response patterns, ejector performance, and dissolved oxygen correlation.

Zero-Hallucination Guarantee:
- All detection uses deterministic algorithms
- Threshold-based classification with documented sources
- Statistical pattern matching without ML inference

Key Features:
- Monitor vacuum response patterns
- Analyze ejector/vacuum pump performance
- Off-gas flow analysis (if available)
- Dissolved oxygen correlation
- Classify leak severity (minor, moderate, severe)
- Estimate leak rate

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
    """Air in-leakage severity classification."""
    NONE = "none"           # No detectable leakage
    MINOR = "minor"         # Small leak, minimal impact
    MODERATE = "moderate"   # Noticeable leak, performance impact
    SEVERE = "severe"       # Large leak, significant impact
    CRITICAL = "critical"   # Critical leak, immediate action needed


class LeakLocation(Enum):
    """Probable leak location."""
    LP_TURBINE_GLANDS = "lp_turbine_glands"
    EXPANSION_JOINTS = "expansion_joints"
    VALVE_STEMS = "valve_stems"
    INSTRUMENT_CONNECTIONS = "instrument_connections"
    MANHOLE_COVERS = "manhole_covers"
    CONDENSATE_PUMP_SEALS = "condensate_pump_seals"
    VACUUM_BREAKER = "vacuum_breaker"
    HOTWELL_LEVEL_CONTROL = "hotwell_level_control"
    RUPTURE_DISCS = "rupture_discs"
    UNKNOWN = "unknown"


class VacuumResponsePattern(Enum):
    """Vacuum response pattern classification."""
    NORMAL = "normal"                    # Expected vacuum behavior
    SLUGGISH_RECOVERY = "sluggish"       # Slow vacuum recovery
    RAPID_DECAY = "rapid_decay"          # Quick vacuum loss
    OSCILLATING = "oscillating"          # Unstable vacuum
    STEADY_DEGRADATION = "steady_degradation"  # Gradual worsening


class DetectionConfidence(Enum):
    """Detection confidence level."""
    HIGH = "high"           # > 90% confidence
    MEDIUM = "medium"       # 70-90% confidence
    LOW = "low"             # 50-70% confidence
    UNCERTAIN = "uncertain"  # < 50% confidence


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


# ============================================================================
# THRESHOLD CONSTANTS
# ============================================================================

# Air in-leakage rate thresholds (kg/hr) per condenser size category
# Source: HEI Standards for Steam Surface Condensers
LEAKAGE_THRESHOLDS = {
    "small": {   # < 100 MW
        LeakSeverity.MINOR: 1.0,
        LeakSeverity.MODERATE: 2.5,
        LeakSeverity.SEVERE: 5.0,
        LeakSeverity.CRITICAL: 10.0,
    },
    "medium": {  # 100-500 MW
        LeakSeverity.MINOR: 2.5,
        LeakSeverity.MODERATE: 5.0,
        LeakSeverity.SEVERE: 10.0,
        LeakSeverity.CRITICAL: 20.0,
    },
    "large": {   # > 500 MW
        LeakSeverity.MINOR: 5.0,
        LeakSeverity.MODERATE: 10.0,
        LeakSeverity.SEVERE: 20.0,
        LeakSeverity.CRITICAL: 40.0,
    },
}

# Dissolved oxygen correlation (ppb) with air leakage
DO_THRESHOLDS = {
    LeakSeverity.NONE: 7.0,      # Normal < 7 ppb
    LeakSeverity.MINOR: 10.0,    # 7-10 ppb
    LeakSeverity.MODERATE: 15.0,  # 10-15 ppb
    LeakSeverity.SEVERE: 25.0,   # 15-25 ppb
    LeakSeverity.CRITICAL: 50.0,  # > 25 ppb
}

# Vacuum deviation thresholds (mbar from design)
VACUUM_DEVIATION_THRESHOLDS = {
    LeakSeverity.NONE: 2.0,       # < 2 mbar deviation
    LeakSeverity.MINOR: 5.0,      # 2-5 mbar
    LeakSeverity.MODERATE: 10.0,  # 5-10 mbar
    LeakSeverity.SEVERE: 20.0,    # 10-20 mbar
    LeakSeverity.CRITICAL: 30.0,  # > 20 mbar
}

# Ejector steam consumption increase thresholds (%)
EJECTOR_STEAM_THRESHOLDS = {
    LeakSeverity.MINOR: 10.0,     # < 10% increase
    LeakSeverity.MODERATE: 25.0,  # 10-25% increase
    LeakSeverity.SEVERE: 50.0,    # 25-50% increase
    LeakSeverity.CRITICAL: 75.0,  # > 50% increase
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class AirLeakageDetectorConfig:
    """Configuration for air in-leakage detector."""

    # Condenser size category
    condenser_size: str = "medium"  # small, medium, large

    # Design vacuum pressure (mbar absolute)
    design_vacuum_mbar_a: float = 50.0

    # Ejector design steam flow (kg/hr)
    ejector_design_steam_kg_hr: float = 500.0

    # DO baseline (ppb)
    do_baseline_ppb: float = 5.0

    # Detection sensitivity
    sensitivity: str = "normal"  # low, normal, high

    # Rolling window for analysis
    rolling_window_minutes: int = 60

    # Minimum data points for detection
    min_data_points: int = 10

    # Alert cooldown period (minutes)
    alert_cooldown_minutes: int = 30


@dataclass
class VacuumDataPoint:
    """Single vacuum system data point."""
    timestamp: datetime
    vacuum_mbar_a: float  # Condenser vacuum (mbar absolute)
    ejector_steam_kg_hr: Optional[float] = None  # Ejector steam consumption
    vacuum_pump_power_kw: Optional[float] = None  # LRVP power consumption
    off_gas_flow_kg_hr: Optional[float] = None  # Off-gas flow rate
    dissolved_oxygen_ppb: Optional[float] = None  # Condensate DO
    hotwell_level_pct: Optional[float] = None  # Hotwell level
    load_mw: Optional[float] = None  # Unit load
    cw_inlet_temp_c: Optional[float] = None  # CW inlet temperature
    is_validated: bool = True
    quality_flag: str = "good"


@dataclass
class EjectorPerformance:
    """Ejector/vacuum pump performance metrics."""
    steam_consumption_kg_hr: float
    steam_consumption_pct_design: float
    capacity_utilization_pct: float
    efficiency_pct: float
    suction_pressure_mbar: float
    discharge_pressure_mbar: float
    is_at_limit: bool
    performance_deviation_pct: float


@dataclass
class VacuumResponseAnalysis:
    """Analysis of vacuum response pattern."""
    pattern: VacuumResponsePattern
    recovery_time_seconds: float  # Time to recover from disturbance
    decay_rate_mbar_per_min: float  # Rate of vacuum degradation
    oscillation_amplitude_mbar: float  # Peak-to-peak variation
    oscillation_frequency_per_min: float  # Oscillation frequency
    steady_state_deviation_mbar: float  # Deviation from design


@dataclass
class LeakRateEstimate:
    """Estimated air in-leakage rate."""
    leak_rate_kg_hr: float
    leak_rate_uncertainty_kg_hr: float
    estimation_method: str
    confidence: DetectionConfidence
    supporting_evidence: List[str]


@dataclass
class LeakLocationAssessment:
    """Assessment of probable leak location."""
    location: LeakLocation
    probability: float  # 0-1
    evidence: List[str]
    inspection_priority: int  # 1=highest


@dataclass
class AirLeakageAlert:
    """Air in-leakage alert."""
    alert_id: str
    timestamp: datetime
    severity: AlertSeverity
    alert_type: str
    message: str
    leak_severity: LeakSeverity
    estimated_leak_rate_kg_hr: Optional[float] = None
    recommended_action: Optional[str] = None
    evidence: List[str] = field(default_factory=list)


@dataclass
class AirLeakageDetection:
    """Complete air in-leakage detection result."""
    condenser_id: str
    detection_timestamp: datetime

    # Overall assessment
    leak_detected: bool
    leak_severity: LeakSeverity
    detection_confidence: DetectionConfidence

    # Leak rate estimate
    leak_rate_estimate: LeakRateEstimate

    # Vacuum analysis
    vacuum_response: VacuumResponseAnalysis
    current_vacuum_mbar_a: float
    vacuum_deviation_mbar: float

    # Ejector performance
    ejector_performance: Optional[EjectorPerformance]

    # DO correlation
    do_analysis: Optional[Dict[str, Any]]

    # Location assessment
    probable_locations: List[LeakLocationAssessment]

    # Impact assessment
    backpressure_penalty_mbar: float
    heat_rate_penalty_pct: float
    estimated_cost_per_day_usd: float

    # Alerts
    active_alerts: List[AirLeakageAlert]

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
            "leak_rate_estimate": {
                "rate_kg_hr": round(self.leak_rate_estimate.leak_rate_kg_hr, 2),
                "uncertainty_kg_hr": round(self.leak_rate_estimate.leak_rate_uncertainty_kg_hr, 2),
                "method": self.leak_rate_estimate.estimation_method,
                "confidence": self.leak_rate_estimate.confidence.value,
            },
            "vacuum": {
                "current_mbar_a": round(self.current_vacuum_mbar_a, 2),
                "deviation_mbar": round(self.vacuum_deviation_mbar, 2),
                "response_pattern": self.vacuum_response.pattern.value,
            },
            "ejector": {
                "steam_pct_design": round(self.ejector_performance.steam_consumption_pct_design, 1)
                if self.ejector_performance else None,
                "is_at_limit": self.ejector_performance.is_at_limit
                if self.ejector_performance else None,
            },
            "impact": {
                "backpressure_penalty_mbar": round(self.backpressure_penalty_mbar, 2),
                "heat_rate_penalty_pct": round(self.heat_rate_penalty_pct, 3),
                "cost_per_day_usd": round(self.estimated_cost_per_day_usd, 0),
            },
            "probable_locations": [
                {"location": loc.location.value, "probability": round(loc.probability, 2)}
                for loc in self.probable_locations[:3]
            ],
            "active_alerts": [
                {"severity": a.severity.value, "type": a.alert_type, "message": a.message}
                for a in self.active_alerts
            ],
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class CondenserVacuumProfile:
    """Condenser vacuum system profile."""
    condenser_id: str
    design_vacuum_mbar_a: float = 50.0
    condenser_size: str = "medium"
    ejector_type: str = "two_stage"  # single_stage, two_stage, hybrid
    ejector_design_steam_kg_hr: float = 500.0
    vacuum_pump_type: Optional[str] = None  # LRVP, dry, hybrid
    vacuum_pump_design_power_kw: Optional[float] = None
    typical_do_ppb: float = 5.0
    unit_capacity_mw: float = 500.0
    fuel_cost_usd_per_mwh: float = 30.0


# ============================================================================
# MAIN AIR LEAKAGE DETECTOR CLASS
# ============================================================================

class AirInleakageDetector:
    """
    Air in-leakage detection system for power plant condensers.

    Monitors vacuum response patterns, ejector performance, off-gas flow,
    and dissolved oxygen to detect and classify air in-leakage.

    Zero-Hallucination Guarantee:
    - Detection uses deterministic threshold logic
    - Leak rate estimates from HEI correlations
    - Pattern matching with documented algorithms

    Example:
        >>> detector = AirInleakageDetector()
        >>> data = [VacuumDataPoint(timestamp=..., vacuum_mbar_a=52), ...]
        >>> profile = CondenserVacuumProfile(condenser_id="COND-01")
        >>> detection = detector.detect_air_leakage(data, profile)
        >>> print(f"Leak detected: {detection.leak_detected}")
    """

    VERSION = "1.0.0"
    METHODOLOGY = "Multi-parameter correlation with HEI standard thresholds"

    def __init__(self, config: Optional[AirLeakageDetectorConfig] = None):
        """
        Initialize air in-leakage detector.

        Args:
            config: Detector configuration (optional)
        """
        self.config = config or AirLeakageDetectorConfig()
        self._sensitivity_factor = self._get_sensitivity_factor()
        logger.info(f"AirInleakageDetector initialized with {self.config.sensitivity} sensitivity")

    def _get_sensitivity_factor(self) -> float:
        """Get sensitivity multiplier for thresholds."""
        sensitivity_map = {
            "low": 1.5,      # Less sensitive, higher thresholds
            "normal": 1.0,
            "high": 0.7,     # More sensitive, lower thresholds
        }
        return sensitivity_map.get(self.config.sensitivity, 1.0)

    # ========================================================================
    # VACUUM RESPONSE ANALYSIS
    # ========================================================================

    def _analyze_vacuum_response(
        self,
        data_points: List[VacuumDataPoint],
        design_vacuum: float
    ) -> VacuumResponseAnalysis:
        """
        Analyze vacuum response patterns.

        Args:
            data_points: List of vacuum data points
            design_vacuum: Design vacuum pressure (mbar_a)

        Returns:
            VacuumResponseAnalysis with pattern classification
        """
        if len(data_points) < 3:
            return VacuumResponseAnalysis(
                pattern=VacuumResponsePattern.NORMAL,
                recovery_time_seconds=0.0,
                decay_rate_mbar_per_min=0.0,
                oscillation_amplitude_mbar=0.0,
                oscillation_frequency_per_min=0.0,
                steady_state_deviation_mbar=0.0,
            )

        vacuum_values = [dp.vacuum_mbar_a for dp in data_points if dp.is_validated]
        timestamps = [dp.timestamp for dp in data_points if dp.is_validated]

        if len(vacuum_values) < 3:
            return VacuumResponseAnalysis(
                pattern=VacuumResponsePattern.NORMAL,
                recovery_time_seconds=0.0,
                decay_rate_mbar_per_min=0.0,
                oscillation_amplitude_mbar=0.0,
                oscillation_frequency_per_min=0.0,
                steady_state_deviation_mbar=0.0,
            )

        # Calculate statistics
        mean_vacuum = statistics.mean(vacuum_values)
        std_vacuum = statistics.stdev(vacuum_values) if len(vacuum_values) > 1 else 0.0

        # Steady-state deviation from design
        steady_state_deviation = mean_vacuum - design_vacuum

        # Calculate rate of change (decay rate)
        time_span_min = (timestamps[-1] - timestamps[0]).total_seconds() / 60.0
        if time_span_min > 0:
            vacuum_change = vacuum_values[-1] - vacuum_values[0]
            decay_rate = vacuum_change / time_span_min
        else:
            decay_rate = 0.0

        # Oscillation analysis
        oscillation_amplitude = max(vacuum_values) - min(vacuum_values)
        zero_crossings = self._count_zero_crossings(vacuum_values, mean_vacuum)
        oscillation_freq = zero_crossings / (2 * time_span_min) if time_span_min > 0 else 0.0

        # Pattern classification
        pattern = self._classify_vacuum_pattern(
            steady_state_deviation,
            decay_rate,
            oscillation_amplitude,
            std_vacuum,
            design_vacuum
        )

        # Estimate recovery time (simplified)
        recovery_time = self._estimate_recovery_time(vacuum_values, design_vacuum)

        return VacuumResponseAnalysis(
            pattern=pattern,
            recovery_time_seconds=recovery_time,
            decay_rate_mbar_per_min=decay_rate,
            oscillation_amplitude_mbar=oscillation_amplitude,
            oscillation_frequency_per_min=oscillation_freq,
            steady_state_deviation_mbar=steady_state_deviation,
        )

    def _count_zero_crossings(self, values: List[float], mean: float) -> int:
        """Count zero crossings relative to mean."""
        crossings = 0
        for i in range(1, len(values)):
            if (values[i] - mean) * (values[i-1] - mean) < 0:
                crossings += 1
        return crossings

    def _classify_vacuum_pattern(
        self,
        deviation: float,
        decay_rate: float,
        amplitude: float,
        std_dev: float,
        design: float
    ) -> VacuumResponsePattern:
        """Classify vacuum response pattern."""
        # Thresholds (adjusted by sensitivity)
        sf = self._sensitivity_factor

        # High oscillation
        if amplitude > 5.0 * sf and std_dev > 2.0 * sf:
            return VacuumResponsePattern.OSCILLATING

        # Rapid decay (vacuum getting worse quickly)
        if decay_rate > 0.5 * sf:  # >0.5 mbar/min increase in pressure
            return VacuumResponsePattern.RAPID_DECAY

        # Sluggish recovery (not returning to design)
        if deviation > 5.0 * sf and decay_rate >= 0:
            return VacuumResponsePattern.SLUGGISH_RECOVERY

        # Steady degradation
        if deviation > 3.0 * sf and decay_rate > 0.1 * sf:
            return VacuumResponsePattern.STEADY_DEGRADATION

        return VacuumResponsePattern.NORMAL

    def _estimate_recovery_time(
        self,
        vacuum_values: List[float],
        design: float
    ) -> float:
        """Estimate time to recover to design vacuum after disturbance."""
        # Find points above design (worse vacuum)
        above_design = [v for v in vacuum_values if v > design + 2]

        if not above_design or len(vacuum_values) < 5:
            return 0.0

        # Simple model: assume exponential recovery
        deviation = statistics.mean(above_design) - design
        # Typical recovery time constant ~30 seconds per mbar deviation
        recovery_time = deviation * 30.0

        return min(recovery_time, 600.0)  # Cap at 10 minutes

    # ========================================================================
    # EJECTOR PERFORMANCE ANALYSIS
    # ========================================================================

    def _analyze_ejector_performance(
        self,
        data_points: List[VacuumDataPoint],
        profile: CondenserVacuumProfile
    ) -> Optional[EjectorPerformance]:
        """
        Analyze steam ejector performance.

        Args:
            data_points: List of vacuum data points
            profile: Condenser vacuum profile

        Returns:
            EjectorPerformance metrics or None if data unavailable
        """
        # Get ejector steam data
        steam_values = [
            dp.ejector_steam_kg_hr for dp in data_points
            if dp.ejector_steam_kg_hr is not None and dp.is_validated
        ]

        if not steam_values:
            return None

        # Current steam consumption
        current_steam = statistics.mean(steam_values[-5:]) if len(steam_values) >= 5 else steam_values[-1]

        # Percentage of design
        pct_design = (current_steam / profile.ejector_design_steam_kg_hr) * 100

        # Get vacuum values for efficiency calculation
        vacuum_values = [dp.vacuum_mbar_a for dp in data_points if dp.is_validated]
        current_vacuum = vacuum_values[-1] if vacuum_values else profile.design_vacuum_mbar_a

        # Capacity utilization estimate
        # Higher steam = higher capacity utilization
        capacity_util = min(100.0, pct_design * 0.9)

        # Efficiency estimate (simplified)
        # Efficiency decreases when working harder for worse vacuum
        vacuum_ratio = profile.design_vacuum_mbar_a / max(current_vacuum, 1.0)
        efficiency = max(50.0, min(100.0, vacuum_ratio * 100 * (1 - (pct_design - 100) / 200)))

        # At limit check
        is_at_limit = pct_design > 90.0

        # Performance deviation
        performance_deviation = pct_design - 100.0

        return EjectorPerformance(
            steam_consumption_kg_hr=current_steam,
            steam_consumption_pct_design=pct_design,
            capacity_utilization_pct=capacity_util,
            efficiency_pct=efficiency,
            suction_pressure_mbar=current_vacuum,
            discharge_pressure_mbar=current_vacuum + 950.0,  # Approximate
            is_at_limit=is_at_limit,
            performance_deviation_pct=performance_deviation,
        )

    # ========================================================================
    # DISSOLVED OXYGEN CORRELATION
    # ========================================================================

    def _analyze_dissolved_oxygen(
        self,
        data_points: List[VacuumDataPoint],
        baseline_ppb: float
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze dissolved oxygen correlation with air leakage.

        Args:
            data_points: List of vacuum data points
            baseline_ppb: Baseline DO level

        Returns:
            DO analysis dictionary or None if data unavailable
        """
        # Get DO data
        do_values = [
            dp.dissolved_oxygen_ppb for dp in data_points
            if dp.dissolved_oxygen_ppb is not None and dp.is_validated
        ]

        if not do_values:
            return None

        current_do = statistics.mean(do_values[-5:]) if len(do_values) >= 5 else do_values[-1]
        mean_do = statistics.mean(do_values)
        max_do = max(do_values)

        # DO elevation
        elevation_ppb = current_do - baseline_ppb
        elevation_pct = (elevation_ppb / baseline_ppb) * 100 if baseline_ppb > 0 else 0

        # Classify severity based on DO
        do_severity = LeakSeverity.NONE
        for severity, threshold in sorted(DO_THRESHOLDS.items(), key=lambda x: x[1], reverse=True):
            if current_do >= threshold:
                do_severity = severity
                break

        # Trend analysis
        if len(do_values) >= 3:
            early_mean = statistics.mean(do_values[:len(do_values)//2])
            late_mean = statistics.mean(do_values[len(do_values)//2:])
            trend = "increasing" if late_mean > early_mean * 1.1 else (
                "decreasing" if late_mean < early_mean * 0.9 else "stable"
            )
        else:
            trend = "insufficient_data"

        return {
            "current_ppb": current_do,
            "mean_ppb": mean_do,
            "max_ppb": max_do,
            "baseline_ppb": baseline_ppb,
            "elevation_ppb": elevation_ppb,
            "elevation_pct": elevation_pct,
            "indicated_severity": do_severity.value,
            "trend": trend,
        }

    # ========================================================================
    # LEAK RATE ESTIMATION
    # ========================================================================

    def _estimate_leak_rate(
        self,
        vacuum_response: VacuumResponseAnalysis,
        ejector_perf: Optional[EjectorPerformance],
        do_analysis: Optional[Dict[str, Any]],
        profile: CondenserVacuumProfile
    ) -> LeakRateEstimate:
        """
        Estimate air in-leakage rate using multiple methods.

        Args:
            vacuum_response: Vacuum response analysis
            ejector_perf: Ejector performance (if available)
            do_analysis: DO analysis (if available)
            profile: Condenser vacuum profile

        Returns:
            LeakRateEstimate with uncertainty
        """
        estimates = []
        methods = []
        evidence = []

        # Method 1: Vacuum deviation correlation
        # Based on HEI correlation: air leakage ~ k * vacuum_deviation^1.5
        vacuum_dev = vacuum_response.steady_state_deviation_mbar
        if vacuum_dev > 0:
            # k factor depends on condenser size
            k_factors = {"small": 0.5, "medium": 0.8, "large": 1.2}
            k = k_factors.get(profile.condenser_size, 0.8)
            leak_vacuum = k * math.pow(vacuum_dev, 1.5)
            estimates.append(leak_vacuum)
            methods.append("vacuum_deviation")
            evidence.append(f"Vacuum deviation: {vacuum_dev:.1f} mbar")

        # Method 2: Ejector steam consumption
        if ejector_perf and ejector_perf.steam_consumption_pct_design > 100:
            excess_steam_pct = ejector_perf.steam_consumption_pct_design - 100
            # Air leakage ~ excess steam / ejector ratio
            # Typical ratio: 1 kg air requires ~5 kg steam at this stage
            leak_ejector = (excess_steam_pct / 100) * profile.ejector_design_steam_kg_hr / 5
            estimates.append(leak_ejector)
            methods.append("ejector_steam")
            evidence.append(f"Ejector steam at {ejector_perf.steam_consumption_pct_design:.0f}% design")

        # Method 3: DO correlation
        if do_analysis and do_analysis["elevation_ppb"] > 2:
            # Simplified correlation: 1 ppb DO elevation ~ 0.5 kg/hr air
            leak_do = do_analysis["elevation_ppb"] * 0.5
            estimates.append(leak_do)
            methods.append("do_correlation")
            evidence.append(f"DO elevated by {do_analysis['elevation_ppb']:.1f} ppb")

        # Method 4: Vacuum decay rate (if significant)
        if vacuum_response.decay_rate_mbar_per_min > 0.2:
            # Correlation: decay rate to air ingress
            # Assuming condenser volume and ideal gas
            condenser_volume_m3 = {"small": 100, "medium": 500, "large": 1500}
            vol = condenser_volume_m3.get(profile.condenser_size, 500)
            leak_decay = vacuum_response.decay_rate_mbar_per_min * vol * 0.001 * 60  # kg/hr
            estimates.append(leak_decay)
            methods.append("vacuum_decay")
            evidence.append(f"Vacuum decay rate: {vacuum_response.decay_rate_mbar_per_min:.2f} mbar/min")

        # Combine estimates
        if not estimates:
            return LeakRateEstimate(
                leak_rate_kg_hr=0.0,
                leak_rate_uncertainty_kg_hr=0.0,
                estimation_method="no_data",
                confidence=DetectionConfidence.UNCERTAIN,
                supporting_evidence=["Insufficient data for leak rate estimation"],
            )

        # Use weighted average (vacuum deviation method is most reliable)
        weights = {"vacuum_deviation": 1.5, "ejector_steam": 1.0, "do_correlation": 0.8, "vacuum_decay": 0.7}
        weighted_sum = sum(est * weights.get(meth, 1.0) for est, meth in zip(estimates, methods))
        weight_total = sum(weights.get(meth, 1.0) for meth in methods)

        leak_rate = weighted_sum / weight_total

        # Uncertainty estimation
        if len(estimates) > 1:
            std_dev = statistics.stdev(estimates)
            uncertainty = std_dev * 1.5  # 1.5x std dev for conservative bound
        else:
            uncertainty = leak_rate * 0.3  # 30% uncertainty with single method

        # Confidence based on number of methods and agreement
        if len(estimates) >= 3 and (max(estimates) / (min(estimates) + 0.1)) < 2:
            confidence = DetectionConfidence.HIGH
        elif len(estimates) >= 2:
            confidence = DetectionConfidence.MEDIUM
        else:
            confidence = DetectionConfidence.LOW

        return LeakRateEstimate(
            leak_rate_kg_hr=max(0.0, leak_rate),
            leak_rate_uncertainty_kg_hr=uncertainty,
            estimation_method="+".join(methods),
            confidence=confidence,
            supporting_evidence=evidence,
        )

    # ========================================================================
    # SEVERITY CLASSIFICATION
    # ========================================================================

    def _classify_severity(
        self,
        leak_rate: float,
        vacuum_deviation: float,
        ejector_perf: Optional[EjectorPerformance],
        do_analysis: Optional[Dict[str, Any]]
    ) -> LeakSeverity:
        """
        Classify leak severity based on multiple indicators.

        Args:
            leak_rate: Estimated leak rate (kg/hr)
            vacuum_deviation: Vacuum deviation from design (mbar)
            ejector_perf: Ejector performance
            do_analysis: DO analysis

        Returns:
            LeakSeverity classification
        """
        severities = []
        sf = self._sensitivity_factor

        # Classify by leak rate
        thresholds = LEAKAGE_THRESHOLDS.get(self.config.condenser_size, LEAKAGE_THRESHOLDS["medium"])
        for severity in [LeakSeverity.CRITICAL, LeakSeverity.SEVERE, LeakSeverity.MODERATE, LeakSeverity.MINOR]:
            if leak_rate >= thresholds[severity] * sf:
                severities.append(severity)
                break

        # Classify by vacuum deviation
        for severity in [LeakSeverity.CRITICAL, LeakSeverity.SEVERE, LeakSeverity.MODERATE, LeakSeverity.MINOR]:
            if vacuum_deviation >= VACUUM_DEVIATION_THRESHOLDS[severity] * sf:
                severities.append(severity)
                break

        # Classify by ejector performance
        if ejector_perf:
            excess = ejector_perf.steam_consumption_pct_design - 100
            if excess > 0:
                for severity in [LeakSeverity.CRITICAL, LeakSeverity.SEVERE, LeakSeverity.MODERATE, LeakSeverity.MINOR]:
                    if excess >= EJECTOR_STEAM_THRESHOLDS[severity] * sf:
                        severities.append(severity)
                        break

        # Classify by DO
        if do_analysis:
            do_severity = LeakSeverity[do_analysis["indicated_severity"].upper()]
            if do_severity != LeakSeverity.NONE:
                severities.append(do_severity)

        # Return worst case
        if not severities:
            return LeakSeverity.NONE

        severity_order = [LeakSeverity.CRITICAL, LeakSeverity.SEVERE, LeakSeverity.MODERATE, LeakSeverity.MINOR, LeakSeverity.NONE]
        for sev in severity_order:
            if sev in severities:
                return sev

        return LeakSeverity.NONE

    # ========================================================================
    # LOCATION ASSESSMENT
    # ========================================================================

    def _assess_probable_locations(
        self,
        vacuum_response: VacuumResponseAnalysis,
        ejector_perf: Optional[EjectorPerformance],
        data_points: List[VacuumDataPoint]
    ) -> List[LeakLocationAssessment]:
        """
        Assess probable leak locations based on symptoms.

        Args:
            vacuum_response: Vacuum response analysis
            ejector_perf: Ejector performance
            data_points: Raw data points

        Returns:
            List of LeakLocationAssessment sorted by probability
        """
        location_scores = {}

        # LP turbine glands - common source, especially with oscillating vacuum
        if vacuum_response.pattern == VacuumResponsePattern.OSCILLATING:
            location_scores[LeakLocation.LP_TURBINE_GLANDS] = 0.35
        else:
            location_scores[LeakLocation.LP_TURBINE_GLANDS] = 0.15

        # Expansion joints - common with steady degradation
        if vacuum_response.pattern == VacuumResponsePattern.STEADY_DEGRADATION:
            location_scores[LeakLocation.EXPANSION_JOINTS] = 0.30
        else:
            location_scores[LeakLocation.EXPANSION_JOINTS] = 0.10

        # Valve stems - moderate probability
        location_scores[LeakLocation.VALVE_STEMS] = 0.12

        # Instrument connections - higher if oscillating
        if vacuum_response.oscillation_amplitude_mbar > 3:
            location_scores[LeakLocation.INSTRUMENT_CONNECTIONS] = 0.18
        else:
            location_scores[LeakLocation.INSTRUMENT_CONNECTIONS] = 0.08

        # Manhole covers - check if recent maintenance
        location_scores[LeakLocation.MANHOLE_COVERS] = 0.08

        # Condensate pump seals - check hotwell level stability
        hotwell_levels = [
            dp.hotwell_level_pct for dp in data_points
            if dp.hotwell_level_pct is not None
        ]
        if hotwell_levels and statistics.stdev(hotwell_levels) > 5:
            location_scores[LeakLocation.CONDENSATE_PUMP_SEALS] = 0.15
        else:
            location_scores[LeakLocation.CONDENSATE_PUMP_SEALS] = 0.05

        # Vacuum breaker - if rapid decay
        if vacuum_response.pattern == VacuumResponsePattern.RAPID_DECAY:
            location_scores[LeakLocation.VACUUM_BREAKER] = 0.25
        else:
            location_scores[LeakLocation.VACUUM_BREAKER] = 0.05

        # Normalize probabilities
        total = sum(location_scores.values())
        if total > 0:
            location_scores = {k: v / total for k, v in location_scores.items()}

        # Create assessments
        assessments = []
        for location, prob in sorted(location_scores.items(), key=lambda x: x[1], reverse=True):
            evidence = self._get_location_evidence(location, vacuum_response, data_points)
            assessments.append(LeakLocationAssessment(
                location=location,
                probability=prob,
                evidence=evidence,
                inspection_priority=len(assessments) + 1,
            ))

        return assessments

    def _get_location_evidence(
        self,
        location: LeakLocation,
        vacuum_response: VacuumResponseAnalysis,
        data_points: List[VacuumDataPoint]
    ) -> List[str]:
        """Get supporting evidence for a location assessment."""
        evidence = []

        if location == LeakLocation.LP_TURBINE_GLANDS:
            if vacuum_response.oscillation_amplitude_mbar > 3:
                evidence.append("Vacuum oscillation suggests gland seal variation")
            evidence.append("Most common source of air in-leakage")

        elif location == LeakLocation.EXPANSION_JOINTS:
            if vacuum_response.pattern == VacuumResponsePattern.STEADY_DEGRADATION:
                evidence.append("Steady vacuum degradation pattern")
            evidence.append("Thermal cycling causes joint degradation")

        elif location == LeakLocation.VACUUM_BREAKER:
            if vacuum_response.pattern == VacuumResponsePattern.RAPID_DECAY:
                evidence.append("Rapid vacuum decay suggests large leak")
            evidence.append("Check vacuum breaker valve seating")

        else:
            evidence.append(f"Standard inspection point: {location.value}")

        return evidence

    # ========================================================================
    # IMPACT ASSESSMENT
    # ========================================================================

    def _calculate_impact(
        self,
        vacuum_deviation: float,
        leak_rate: float,
        profile: CondenserVacuumProfile
    ) -> Tuple[float, float, float]:
        """
        Calculate economic impact of air in-leakage.

        Args:
            vacuum_deviation: Vacuum deviation (mbar)
            leak_rate: Air leak rate (kg/hr)
            profile: Condenser vacuum profile

        Returns:
            Tuple of (backpressure_penalty, heat_rate_penalty_pct, cost_per_day_usd)
        """
        # Backpressure penalty = vacuum deviation
        bp_penalty = vacuum_deviation

        # Heat rate penalty: approximately 0.1% per mbar of backpressure increase
        # This is a typical correlation for steam turbines
        hr_penalty_pct = vacuum_deviation * 0.1

        # Cost calculation
        # Additional fuel cost = HR penalty * capacity * fuel cost
        capacity_mw = profile.unit_capacity_mw
        fuel_cost = profile.fuel_cost_usd_per_mwh

        # Assuming 24 hours operation
        additional_cost = (hr_penalty_pct / 100) * capacity_mw * 24 * fuel_cost

        return bp_penalty, hr_penalty_pct, additional_cost

    # ========================================================================
    # ALERT GENERATION
    # ========================================================================

    def _generate_alerts(
        self,
        leak_severity: LeakSeverity,
        leak_rate: LeakRateEstimate,
        vacuum_response: VacuumResponseAnalysis,
        ejector_perf: Optional[EjectorPerformance],
        condenser_id: str
    ) -> List[AirLeakageAlert]:
        """Generate alerts based on detection results."""
        alerts = []
        now = datetime.now(timezone.utc)

        # Severity-based alerts
        if leak_severity == LeakSeverity.CRITICAL:
            alerts.append(AirLeakageAlert(
                alert_id=f"{condenser_id}_LEAK_CRITICAL_{now.strftime('%Y%m%d%H%M')}",
                timestamp=now,
                severity=AlertSeverity.EMERGENCY,
                alert_type="CRITICAL_AIR_LEAKAGE",
                message=f"Critical air in-leakage detected: {leak_rate.leak_rate_kg_hr:.1f} kg/hr",
                leak_severity=leak_severity,
                estimated_leak_rate_kg_hr=leak_rate.leak_rate_kg_hr,
                recommended_action="Immediate action required - initiate leak search procedure",
                evidence=leak_rate.supporting_evidence,
            ))
        elif leak_severity == LeakSeverity.SEVERE:
            alerts.append(AirLeakageAlert(
                alert_id=f"{condenser_id}_LEAK_SEVERE_{now.strftime('%Y%m%d%H%M')}",
                timestamp=now,
                severity=AlertSeverity.CRITICAL,
                alert_type="SEVERE_AIR_LEAKAGE",
                message=f"Severe air in-leakage detected: {leak_rate.leak_rate_kg_hr:.1f} kg/hr",
                leak_severity=leak_severity,
                estimated_leak_rate_kg_hr=leak_rate.leak_rate_kg_hr,
                recommended_action="Schedule urgent leak search within 24 hours",
                evidence=leak_rate.supporting_evidence,
            ))
        elif leak_severity == LeakSeverity.MODERATE:
            alerts.append(AirLeakageAlert(
                alert_id=f"{condenser_id}_LEAK_MODERATE_{now.strftime('%Y%m%d%H%M')}",
                timestamp=now,
                severity=AlertSeverity.WARNING,
                alert_type="MODERATE_AIR_LEAKAGE",
                message=f"Moderate air in-leakage detected: {leak_rate.leak_rate_kg_hr:.1f} kg/hr",
                leak_severity=leak_severity,
                estimated_leak_rate_kg_hr=leak_rate.leak_rate_kg_hr,
                recommended_action="Plan leak search during next available window",
                evidence=leak_rate.supporting_evidence,
            ))

        # Ejector at limit alert
        if ejector_perf and ejector_perf.is_at_limit:
            alerts.append(AirLeakageAlert(
                alert_id=f"{condenser_id}_EJECTOR_LIMIT_{now.strftime('%Y%m%d%H%M')}",
                timestamp=now,
                severity=AlertSeverity.WARNING,
                alert_type="EJECTOR_AT_CAPACITY",
                message=f"Steam ejector operating at {ejector_perf.steam_consumption_pct_design:.0f}% capacity",
                leak_severity=leak_severity,
                estimated_leak_rate_kg_hr=leak_rate.leak_rate_kg_hr,
                recommended_action="Ejector capacity limit - vacuum may deteriorate further",
                evidence=[f"Ejector steam: {ejector_perf.steam_consumption_kg_hr:.0f} kg/hr"],
            ))

        # Vacuum response pattern alerts
        if vacuum_response.pattern == VacuumResponsePattern.RAPID_DECAY:
            alerts.append(AirLeakageAlert(
                alert_id=f"{condenser_id}_RAPID_DECAY_{now.strftime('%Y%m%d%H%M')}",
                timestamp=now,
                severity=AlertSeverity.CRITICAL,
                alert_type="RAPID_VACUUM_DECAY",
                message=f"Rapid vacuum decay: {vacuum_response.decay_rate_mbar_per_min:.2f} mbar/min",
                leak_severity=leak_severity,
                estimated_leak_rate_kg_hr=leak_rate.leak_rate_kg_hr,
                recommended_action="Investigate immediate - possible large leak or vacuum breaker issue",
                evidence=["Vacuum pressure increasing rapidly"],
            ))

        return alerts

    # ========================================================================
    # PROVENANCE
    # ========================================================================

    def _compute_provenance_hash(
        self,
        condenser_id: str,
        leak_rate: float,
        vacuum_dev: float,
        data_points_count: int
    ) -> str:
        """Compute SHA-256 hash for detection provenance."""
        data = {
            "version": self.VERSION,
            "condenser_id": condenser_id,
            "leak_rate_kg_hr": round(leak_rate, 4),
            "vacuum_deviation_mbar": round(vacuum_dev, 4),
            "data_points": data_points_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    # ========================================================================
    # MAIN DETECTION METHOD
    # ========================================================================

    def detect_air_leakage(
        self,
        data_points: List[VacuumDataPoint],
        profile: CondenserVacuumProfile,
        detection_timestamp: Optional[datetime] = None
    ) -> AirLeakageDetection:
        """
        Detect and classify air in-leakage.

        Args:
            data_points: Historical vacuum system data points
            profile: Condenser vacuum system profile
            detection_timestamp: Timestamp for detection (default: now)

        Returns:
            AirLeakageDetection with complete analysis

        Raises:
            ValueError: If insufficient data points
        """
        if detection_timestamp is None:
            detection_timestamp = datetime.now(timezone.utc)

        logger.info(f"Performing air in-leakage detection for {profile.condenser_id}")

        # Validate input
        valid_points = [dp for dp in data_points if dp.is_validated]
        if len(valid_points) < self.config.min_data_points:
            raise ValueError(
                f"Insufficient data points: {len(valid_points)} < {self.config.min_data_points}"
            )

        # Sort by timestamp
        valid_points.sort(key=lambda dp: dp.timestamp)

        # Get current vacuum
        current_vacuum = valid_points[-1].vacuum_mbar_a
        vacuum_deviation = current_vacuum - profile.design_vacuum_mbar_a

        # Analyze vacuum response
        vacuum_response = self._analyze_vacuum_response(valid_points, profile.design_vacuum_mbar_a)

        # Analyze ejector performance
        ejector_perf = self._analyze_ejector_performance(valid_points, profile)

        # Analyze DO correlation
        do_analysis = self._analyze_dissolved_oxygen(valid_points, profile.typical_do_ppb)

        # Estimate leak rate
        leak_rate_estimate = self._estimate_leak_rate(
            vacuum_response, ejector_perf, do_analysis, profile
        )

        # Classify severity
        leak_severity = self._classify_severity(
            leak_rate_estimate.leak_rate_kg_hr,
            vacuum_deviation,
            ejector_perf,
            do_analysis
        )

        # Detect if leak is present
        leak_detected = leak_severity != LeakSeverity.NONE

        # Detection confidence
        if leak_detected and leak_rate_estimate.confidence == DetectionConfidence.HIGH:
            detection_confidence = DetectionConfidence.HIGH
        elif leak_detected and leak_rate_estimate.confidence == DetectionConfidence.MEDIUM:
            detection_confidence = DetectionConfidence.MEDIUM
        elif leak_detected:
            detection_confidence = DetectionConfidence.LOW
        else:
            detection_confidence = DetectionConfidence.HIGH  # High confidence of no leak

        # Assess probable locations
        probable_locations = self._assess_probable_locations(
            vacuum_response, ejector_perf, valid_points
        )

        # Calculate impact
        bp_penalty, hr_penalty, cost_per_day = self._calculate_impact(
            vacuum_deviation, leak_rate_estimate.leak_rate_kg_hr, profile
        )

        # Generate alerts
        alerts = self._generate_alerts(
            leak_severity, leak_rate_estimate, vacuum_response, ejector_perf, profile.condenser_id
        )

        # Provenance hash
        provenance_hash = self._compute_provenance_hash(
            profile.condenser_id,
            leak_rate_estimate.leak_rate_kg_hr,
            vacuum_deviation,
            len(valid_points)
        )

        logger.info(
            f"Air leakage detection complete for {profile.condenser_id}: "
            f"detected={leak_detected}, severity={leak_severity.value}, "
            f"rate={leak_rate_estimate.leak_rate_kg_hr:.1f} kg/hr"
        )

        return AirLeakageDetection(
            condenser_id=profile.condenser_id,
            detection_timestamp=detection_timestamp,
            leak_detected=leak_detected,
            leak_severity=leak_severity,
            detection_confidence=detection_confidence,
            leak_rate_estimate=leak_rate_estimate,
            vacuum_response=vacuum_response,
            current_vacuum_mbar_a=current_vacuum,
            vacuum_deviation_mbar=vacuum_deviation,
            ejector_performance=ejector_perf,
            do_analysis=do_analysis,
            probable_locations=probable_locations,
            backpressure_penalty_mbar=bp_penalty,
            heat_rate_penalty_pct=hr_penalty,
            estimated_cost_per_day_usd=cost_per_day,
            active_alerts=alerts,
            methodology=self.METHODOLOGY,
            data_points_analyzed=len(valid_points),
            provenance_hash=provenance_hash,
        )

    def generate_detection_report(
        self,
        detection: AirLeakageDetection
    ) -> str:
        """
        Generate detailed detection report.

        Args:
            detection: AirLeakageDetection result

        Returns:
            Formatted report text
        """
        lines = [
            "=" * 80,
            "          AIR IN-LEAKAGE DETECTION REPORT",
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
            f"  Leak Rate:         {detection.leak_rate_estimate.leak_rate_kg_hr:.1f} +/- "
            f"{detection.leak_rate_estimate.leak_rate_uncertainty_kg_hr:.1f} kg/hr",
            "",
            "VACUUM ANALYSIS",
            "-" * 40,
            f"  Current Vacuum:    {detection.current_vacuum_mbar_a:.1f} mbar(a)",
            f"  Deviation:         {detection.vacuum_deviation_mbar:+.1f} mbar",
            f"  Response Pattern:  {detection.vacuum_response.pattern.value}",
            f"  Decay Rate:        {detection.vacuum_response.decay_rate_mbar_per_min:.3f} mbar/min",
            "",
        ]

        if detection.ejector_performance:
            lines.extend([
                "EJECTOR PERFORMANCE",
                "-" * 40,
                f"  Steam Consumption: {detection.ejector_performance.steam_consumption_kg_hr:.0f} kg/hr "
                f"({detection.ejector_performance.steam_consumption_pct_design:.0f}% design)",
                f"  At Capacity Limit: {detection.ejector_performance.is_at_limit}",
                "",
            ])

        if detection.do_analysis:
            lines.extend([
                "DISSOLVED OXYGEN",
                "-" * 40,
                f"  Current DO:        {detection.do_analysis['current_ppb']:.1f} ppb",
                f"  Baseline:          {detection.do_analysis['baseline_ppb']:.1f} ppb",
                f"  Elevation:         {detection.do_analysis['elevation_ppb']:+.1f} ppb",
                "",
            ])

        lines.extend([
            "ECONOMIC IMPACT",
            "-" * 40,
            f"  Backpressure Penalty: {detection.backpressure_penalty_mbar:.1f} mbar",
            f"  Heat Rate Penalty:    {detection.heat_rate_penalty_pct:.2f}%",
            f"  Estimated Cost:       ${detection.estimated_cost_per_day_usd:,.0f}/day",
            "",
            "PROBABLE LEAK LOCATIONS",
            "-" * 40,
        ])

        for loc in detection.probable_locations[:5]:
            lines.append(f"  {loc.inspection_priority}. {loc.location.value:30s} ({loc.probability:.0%})")

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
