# -*- coding: utf-8 -*-
"""
Combustion Quality Index (CQI) Calculator for GL-005 COMBUSENSE

Implements the CQI scoring system as specified in the GL-005 Playbook.
CQI is a composite score (0-100) combining physics-based computations,
statistical health metrics, and real-time diagnostics.

CQI Structure:
    CQI_total = w1*CQI_efficiency + w2*CQI_emissions + w3*CQI_stability + w4*CQI_safety + w5*CQI_data

Default Weights (per playbook):
    - Efficiency/air-fuel optimality: 0.30
    - Emissions (CO/NOx): 0.30
    - Stability (flame + variability): 0.20
    - Safety boundary status: 0.15
    - Data integrity/confidence: 0.05

Reference Standards:
    - ASME PTC 4.1: Boiler efficiency calculations
    - IEC 61511: SIL 2 safety boundary compliance
    - NFPA 85: Combustion systems hazards code

Author: GreenLang GL-005 Team
Version: 1.0.0
Performance Target: <5ms per CQI calculation
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - CQI PARAMETERS
# =============================================================================

# Default CQI weights (per playbook Section 8.1)
DEFAULT_CQI_WEIGHTS = {
    "efficiency": 0.30,
    "emissions": 0.30,
    "stability": 0.20,
    "safety": 0.15,
    "data": 0.05,
}

# O2 band parameters (typical natural gas combustion)
DEFAULT_O2_TARGET = 3.0  # %
DEFAULT_O2_LOW = 2.0     # %
DEFAULT_O2_HIGH = 4.5    # %

# CO/NOx targets (ppm)
DEFAULT_CO_TARGET = 50.0   # ppm
DEFAULT_NOX_TARGET = 30.0  # ppm

# Safety cap when critical condition active
CQI_SAFETY_CAP = 30.0

# Logistic function steepness for emissions scoring
EMISSIONS_LOGISTIC_K = 0.1


# =============================================================================
# ENUMERATIONS
# =============================================================================

class CQIGrade(str, Enum):
    """CQI grade classification"""
    EXCELLENT = "excellent"  # 90-100
    GOOD = "good"            # 75-89
    ACCEPTABLE = "acceptable"  # 60-74
    POOR = "poor"            # 40-59
    CRITICAL = "critical"    # 0-39


class OperatingMode(str, Enum):
    """Combustion operating modes"""
    PURGE = "purge"
    IGNITE = "ignite"
    RUN = "run"
    SHUTDOWN = "shutdown"
    TRIP = "trip"
    STANDBY = "standby"


class SafetyStatus(str, Enum):
    """Safety boundary status"""
    NORMAL = "normal"
    WARNING = "warning"
    BYPASS_ACTIVE = "bypass_active"
    INTERLOCK_TRIPPED = "interlock_tripped"
    ENVELOPE_EXCEEDED = "envelope_exceeded"


class DataQualityStatus(str, Enum):
    """Data quality status for signals"""
    GOOD = "good"
    UNCERTAIN = "uncertain"
    BAD = "bad"
    CALIBRATING = "calibrating"
    INVALID = "invalid"


# =============================================================================
# FROZEN DATACLASSES
# =============================================================================

@dataclass(frozen=True)
class CQISubScores:
    """Immutable CQI sub-scores breakdown"""
    efficiency: float  # 0-100
    emissions: float   # 0-100
    stability: float   # 0-100
    safety: float      # 0-100
    data: float        # 0-100


@dataclass(frozen=True)
class CQIResult:
    """Immutable CQI calculation result"""
    cqi_total: float
    grade: CQIGrade
    sub_scores: CQISubScores
    weights: Dict[str, float]
    confidence: float
    is_capped: bool
    cap_reason: Optional[str]
    active_incidents: List[str]
    timestamp: datetime
    asset_id: str
    operating_mode: OperatingMode
    provenance_hash: str
    calculation_time_ms: float


@dataclass(frozen=True)
class EfficiencyMetrics:
    """Metrics for efficiency sub-score calculation"""
    o2_actual: float
    o2_target: float
    o2_deviation: float
    excess_air_percent: float
    afr_efficiency: float
    thermal_efficiency: float


@dataclass(frozen=True)
class EmissionsMetrics:
    """Metrics for emissions sub-score calculation"""
    co_ppm: float
    co_target: float
    co_score: float
    nox_ppm: float
    nox_target: float
    nox_score: float
    o2_corrected_co: float
    o2_corrected_nox: float


@dataclass(frozen=True)
class StabilityMetrics:
    """Metrics for stability sub-score calculation"""
    flame_intensity_mean: float
    flame_intensity_std: float
    flame_variability_index: float
    pressure_oscillation_hz: float
    temperature_variance: float
    stability_index: float


@dataclass(frozen=True)
class SafetyMetrics:
    """Metrics for safety sub-score calculation"""
    interlock_health: float
    bypass_active: bool
    bypass_duration_seconds: float
    envelope_status: SafetyStatus
    trip_count_24h: int
    safety_margin_percent: float


@dataclass(frozen=True)
class DataQualityMetrics:
    """Metrics for data quality sub-score calculation"""
    signal_availability: float
    sensor_health_scores: Dict[str, float]
    calibration_status: Dict[str, bool]
    timestamp_quality: float
    overall_quality: float


# =============================================================================
# INPUT MODELS
# =============================================================================

class SignalQuality(BaseModel):
    """Quality information for a signal"""
    status: DataQualityStatus = DataQualityStatus.GOOD
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    is_calibrating: bool = False
    last_valid_timestamp: Optional[datetime] = None


class CombustionSignals(BaseModel):
    """Input signals for CQI calculation"""

    # Core signals (MVSS - Minimum Viable Signal Set)
    fuel_flow_kg_s: float = Field(..., ge=0, description="Fuel mass flow rate (kg/s)")
    air_flow_kg_s: float = Field(..., ge=0, description="Combustion air flow (kg/s)")
    o2_percent: float = Field(..., ge=0, le=21, description="Flue gas O2 (vol% dry)")
    co_ppm: float = Field(..., ge=0, description="Flue gas CO (ppm dry)")
    nox_ppm: float = Field(..., ge=0, description="Flue gas NOx (ppm dry)")

    # Flame signals
    flame_intensity: float = Field(default=80.0, ge=0, le=100, description="Flame intensity (0-100%)")
    flame_intensity_history: List[float] = Field(default_factory=list, description="Recent flame intensity readings")

    # Temperature/Pressure
    combustion_air_temp_c: float = Field(default=25.0, description="Combustion air temperature (C)")
    furnace_temp_c: float = Field(default=800.0, ge=0, le=2000)
    furnace_pressure_pa: float = Field(default=101325.0, ge=0)

    # Safety interlocks
    flame_present: bool = True
    emergency_stop_active: bool = False
    interlocks_healthy: bool = True
    bypass_active: bool = False
    bypass_duration_s: float = 0.0

    # Operating mode
    operating_mode: OperatingMode = OperatingMode.RUN
    burners_in_service: int = Field(default=1, ge=0)
    load_percent: float = Field(default=75.0, ge=0, le=100)

    # Signal quality
    o2_quality: SignalQuality = Field(default_factory=SignalQuality)
    co_quality: SignalQuality = Field(default_factory=SignalQuality)
    nox_quality: SignalQuality = Field(default_factory=SignalQuality)
    flow_quality: SignalQuality = Field(default_factory=SignalQuality)

    # Historical data for trends
    recent_trip_count: int = Field(default=0, ge=0, description="Trip count in last 24h")

    @model_validator(mode='after')
    def validate_signals(self) -> 'CombustionSignals':
        """Validate signal consistency"""
        # Air-fuel ratio sanity check
        if self.fuel_flow_kg_s > 0:
            afr = self.air_flow_kg_s / self.fuel_flow_kg_s
            if afr < 5 or afr > 50:
                logger.warning(f"Unusual AFR detected: {afr:.2f}")
        return self


class CQIConfiguration(BaseModel):
    """Configuration for CQI calculation"""

    # Weights
    efficiency_weight: float = Field(default=0.30, ge=0, le=1)
    emissions_weight: float = Field(default=0.30, ge=0, le=1)
    stability_weight: float = Field(default=0.20, ge=0, le=1)
    safety_weight: float = Field(default=0.15, ge=0, le=1)
    data_weight: float = Field(default=0.05, ge=0, le=1)

    # O2 targets
    o2_target: float = Field(default=3.0, ge=0, le=21)
    o2_low: float = Field(default=2.0, ge=0, le=21)
    o2_high: float = Field(default=4.5, ge=0, le=21)

    # Emissions targets
    co_target_ppm: float = Field(default=50.0, ge=0)
    nox_target_ppm: float = Field(default=30.0, ge=0)

    # Stability thresholds
    flame_variability_baseline: float = Field(default=5.0, ge=0)
    stability_alpha: float = Field(default=0.5, ge=0, le=2)

    # Safety
    safety_cap_cqi: float = Field(default=30.0, ge=0, le=100)
    bypass_penalty_per_minute: float = Field(default=2.0, ge=0)

    # Asset info
    asset_id: str = Field(default="default_asset")
    fuel_type: str = Field(default="natural_gas")

    @model_validator(mode='after')
    def validate_weights(self) -> 'CQIConfiguration':
        """Ensure weights sum to 1.0"""
        total = (self.efficiency_weight + self.emissions_weight +
                 self.stability_weight + self.safety_weight + self.data_weight)
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"CQI weights must sum to 1.0, got {total:.4f}")
        return self


# =============================================================================
# CQI CALCULATOR
# =============================================================================

class CQICalculator:
    """
    Combustion Quality Index Calculator

    Implements the CQI scoring system per GL-005 Playbook Section 8.
    All calculations are deterministic and produce auditable results.
    """

    def __init__(self, config: Optional[CQIConfiguration] = None):
        """
        Initialize CQI Calculator

        Args:
            config: CQI configuration parameters
        """
        self.config = config or CQIConfiguration()
        self._active_incidents: List[str] = []
        logger.info(f"CQI Calculator initialized for asset: {self.config.asset_id}")

    def calculate(self, signals: CombustionSignals) -> CQIResult:
        """
        Calculate CQI from combustion signals

        This is the main entry point for CQI calculation.

        Args:
            signals: Current combustion signals

        Returns:
            CQIResult with total score, sub-scores, and metadata
        """
        start_time = time.perf_counter()
        timestamp = datetime.now(timezone.utc)

        # Check operating mode - only score in RUN mode
        if signals.operating_mode != OperatingMode.RUN:
            return self._create_non_run_result(signals, timestamp, start_time)

        # Calculate sub-scores
        efficiency_score, efficiency_metrics = self._calculate_efficiency_score(signals)
        emissions_score, emissions_metrics = self._calculate_emissions_score(signals)
        stability_score, stability_metrics = self._calculate_stability_score(signals)
        safety_score, safety_metrics = self._calculate_safety_score(signals)
        data_score, data_metrics = self._calculate_data_quality_score(signals)

        # Calculate weighted total
        cqi_total = (
            self.config.efficiency_weight * efficiency_score +
            self.config.emissions_weight * emissions_score +
            self.config.stability_weight * stability_score +
            self.config.safety_weight * safety_score +
            self.config.data_weight * data_score
        )

        # Apply safety caps (Section 8.3)
        is_capped, cap_reason = self._check_safety_caps(signals, safety_metrics)
        if is_capped:
            cqi_total = min(cqi_total, self.config.safety_cap_cqi)

        # Calculate confidence based on data quality
        confidence = self._calculate_confidence(signals, data_metrics)

        # Determine grade
        grade = self._determine_grade(cqi_total)

        # Create sub-scores object
        sub_scores = CQISubScores(
            efficiency=round(efficiency_score, 2),
            emissions=round(emissions_score, 2),
            stability=round(stability_score, 2),
            safety=round(safety_score, 2),
            data=round(data_score, 2)
        )

        # Build weights dict
        weights = {
            "efficiency": self.config.efficiency_weight,
            "emissions": self.config.emissions_weight,
            "stability": self.config.stability_weight,
            "safety": self.config.safety_weight,
            "data": self.config.data_weight,
        }

        # Calculate provenance hash
        calc_time_ms = (time.perf_counter() - start_time) * 1000
        provenance_hash = self._calculate_provenance_hash(
            cqi_total, sub_scores, timestamp, signals
        )

        return CQIResult(
            cqi_total=round(cqi_total, 2),
            grade=grade,
            sub_scores=sub_scores,
            weights=weights,
            confidence=round(confidence, 3),
            is_capped=is_capped,
            cap_reason=cap_reason,
            active_incidents=list(self._active_incidents),
            timestamp=timestamp,
            asset_id=self.config.asset_id,
            operating_mode=signals.operating_mode,
            provenance_hash=provenance_hash,
            calculation_time_ms=round(calc_time_ms, 3)
        )

    def _calculate_efficiency_score(
        self, signals: CombustionSignals
    ) -> Tuple[float, EfficiencyMetrics]:
        """
        Calculate efficiency sub-score (Section 8.2)

        Uses piecewise-linear penalty for O2 deviation from target band.
        Score = 100 * (1 - clip(penalty, 0, 1))
        """
        o2_actual = signals.o2_percent
        o2_target = self.config.o2_target
        o2_low = self.config.o2_low
        o2_high = self.config.o2_high

        # Calculate O2 deviation penalty
        if o2_low <= o2_actual <= o2_high:
            # Within acceptable band - no penalty
            penalty = 0.0
        elif o2_actual < o2_low:
            # Below band - lean/rich risk
            penalty = (o2_low - o2_actual) / o2_low
        else:
            # Above band - excess air
            penalty = (o2_actual - o2_high) / (21 - o2_high)

        # Clip penalty to [0, 1]
        penalty = max(0.0, min(1.0, penalty))

        # Calculate score
        o2_score = 100.0 * (1.0 - penalty)

        # Calculate excess air from O2
        # Approximation: excess_air % ~ O2 / (21 - O2) * 100
        if o2_actual < 21:
            excess_air = (o2_actual / (21.0 - o2_actual)) * 100.0
        else:
            excess_air = 100.0

        # AFR efficiency component
        if signals.fuel_flow_kg_s > 0:
            afr = signals.air_flow_kg_s / signals.fuel_flow_kg_s
            # Typical stoichiometric AFR for natural gas is ~17.2
            stoich_afr = 17.2
            afr_ratio = afr / stoich_afr
            # Ideal AFR ratio is ~1.1-1.2 for safe lean operation
            if 1.05 <= afr_ratio <= 1.25:
                afr_efficiency = 100.0
            else:
                afr_deviation = min(abs(afr_ratio - 1.15), 0.5)
                afr_efficiency = 100.0 * (1.0 - afr_deviation)
        else:
            afr_efficiency = 0.0

        # Combined efficiency score (weighted average)
        efficiency_score = 0.7 * o2_score + 0.3 * afr_efficiency

        # Thermal efficiency placeholder (would use heat balance calculation)
        thermal_efficiency = max(0, 100 - excess_air * 0.5)

        metrics = EfficiencyMetrics(
            o2_actual=o2_actual,
            o2_target=o2_target,
            o2_deviation=o2_actual - o2_target,
            excess_air_percent=excess_air,
            afr_efficiency=afr_efficiency,
            thermal_efficiency=thermal_efficiency
        )

        return efficiency_score, metrics

    def _calculate_emissions_score(
        self, signals: CombustionSignals
    ) -> Tuple[float, EmissionsMetrics]:
        """
        Calculate emissions sub-score (Section 8.2)

        Uses logistic penalty for CO and NOx excursions.
        CQI_CO = 100 / (1 + exp(k*(CO - CO_target)))
        """
        co_ppm = signals.co_ppm
        nox_ppm = signals.nox_ppm
        co_target = self.config.co_target_ppm
        nox_target = self.config.nox_target_ppm

        # Logistic scoring for CO
        k_co = EMISSIONS_LOGISTIC_K
        if co_ppm <= co_target * 0.5:
            co_score = 100.0
        else:
            co_score = 100.0 / (1.0 + math.exp(k_co * (co_ppm - co_target)))

        # Logistic scoring for NOx
        k_nox = EMISSIONS_LOGISTIC_K * 1.5  # NOx is more strictly penalized
        if nox_ppm <= nox_target * 0.5:
            nox_score = 100.0
        else:
            nox_score = 100.0 / (1.0 + math.exp(k_nox * (nox_ppm - nox_target)))

        # O2-corrected emissions (normalize to reference O2, e.g., 3%)
        o2_ref = 3.0
        o2_measured = signals.o2_percent
        if o2_measured < 20.5:  # Avoid division by zero
            correction_factor = (21.0 - o2_ref) / (21.0 - o2_measured)
            o2_corrected_co = co_ppm * correction_factor
            o2_corrected_nox = nox_ppm * correction_factor
        else:
            o2_corrected_co = co_ppm
            o2_corrected_nox = nox_ppm

        # Combined emissions score (equal weighting CO and NOx)
        emissions_score = 0.5 * co_score + 0.5 * nox_score

        metrics = EmissionsMetrics(
            co_ppm=co_ppm,
            co_target=co_target,
            co_score=co_score,
            nox_ppm=nox_ppm,
            nox_target=nox_target,
            nox_score=nox_score,
            o2_corrected_co=o2_corrected_co,
            o2_corrected_nox=o2_corrected_nox
        )

        return emissions_score, metrics

    def _calculate_stability_score(
        self, signals: CombustionSignals
    ) -> Tuple[float, StabilityMetrics]:
        """
        Calculate stability sub-score (Section 8.2)

        Uses flame signal variability and exponential penalty.
        CQI_stability = 100 * exp(-alpha * max(0, (sigma/sigma_0 - 1)))
        """
        # Get flame intensity history or use current value
        if signals.flame_intensity_history and len(signals.flame_intensity_history) >= 5:
            flame_data = signals.flame_intensity_history[-60:]  # Last 60 readings
            flame_mean = sum(flame_data) / len(flame_data)
            flame_std = math.sqrt(
                sum((x - flame_mean) ** 2 for x in flame_data) / len(flame_data)
            )
        else:
            flame_mean = signals.flame_intensity
            flame_std = 0.0  # No history available

        # Calculate variability index
        baseline_std = self.config.flame_variability_baseline
        if baseline_std > 0:
            variability_ratio = flame_std / baseline_std
        else:
            variability_ratio = 0.0

        # Exponential penalty for high variability
        alpha = self.config.stability_alpha
        penalty_factor = max(0.0, variability_ratio - 1.0)
        stability_index = math.exp(-alpha * penalty_factor)
        stability_score = 100.0 * stability_index

        # Penalty for low flame intensity
        if signals.flame_intensity < 50.0:
            intensity_penalty = (50.0 - signals.flame_intensity) / 50.0
            stability_score *= (1.0 - intensity_penalty * 0.3)

        # Penalty if flame not present
        if not signals.flame_present:
            stability_score = 0.0

        metrics = StabilityMetrics(
            flame_intensity_mean=flame_mean,
            flame_intensity_std=flame_std,
            flame_variability_index=variability_ratio,
            pressure_oscillation_hz=0.0,  # Placeholder
            temperature_variance=0.0,  # Placeholder
            stability_index=stability_index
        )

        return max(0.0, min(100.0, stability_score)), metrics

    def _calculate_safety_score(
        self, signals: CombustionSignals
    ) -> Tuple[float, SafetyMetrics]:
        """
        Calculate safety sub-score (Section 8.3, 11)

        Monitors safety boundary conditions and interlock health.
        """
        score = 100.0

        # Interlock health
        interlock_health = 100.0 if signals.interlocks_healthy else 50.0

        # Emergency stop penalty
        if signals.emergency_stop_active:
            score = 0.0
            envelope_status = SafetyStatus.INTERLOCK_TRIPPED
        elif signals.bypass_active:
            # Prolonged bypass penalty
            bypass_penalty = min(
                signals.bypass_duration_s / 60.0 * self.config.bypass_penalty_per_minute,
                50.0
            )
            score -= bypass_penalty
            envelope_status = SafetyStatus.BYPASS_ACTIVE
        elif not signals.interlocks_healthy:
            score -= 30.0
            envelope_status = SafetyStatus.WARNING
        else:
            envelope_status = SafetyStatus.NORMAL

        # Trip history penalty
        if signals.recent_trip_count > 0:
            trip_penalty = min(signals.recent_trip_count * 10.0, 30.0)
            score -= trip_penalty

        # Flame presence is critical
        if not signals.flame_present and signals.operating_mode == OperatingMode.RUN:
            score = 0.0

        # Calculate safety margin
        safety_margin = max(0.0, score - self.config.safety_cap_cqi)

        metrics = SafetyMetrics(
            interlock_health=interlock_health,
            bypass_active=signals.bypass_active,
            bypass_duration_seconds=signals.bypass_duration_s,
            envelope_status=envelope_status,
            trip_count_24h=signals.recent_trip_count,
            safety_margin_percent=safety_margin
        )

        return max(0.0, min(100.0, score)), metrics

    def _calculate_data_quality_score(
        self, signals: CombustionSignals
    ) -> Tuple[float, DataQualityMetrics]:
        """
        Calculate data quality sub-score (Section 6.4)

        Assesses signal availability, sensor health, and calibration status.
        """
        # Collect quality statuses
        quality_scores = {
            "o2": self._quality_to_score(signals.o2_quality),
            "co": self._quality_to_score(signals.co_quality),
            "nox": self._quality_to_score(signals.nox_quality),
            "flow": self._quality_to_score(signals.flow_quality),
        }

        # Calculate average
        avg_quality = sum(quality_scores.values()) / len(quality_scores)

        # Calibration status
        calibration_status = {
            "o2": not signals.o2_quality.is_calibrating,
            "co": not signals.co_quality.is_calibrating,
            "nox": not signals.nox_quality.is_calibrating,
        }

        # Penalize if any critical sensor is calibrating
        calibrating_penalty = sum(
            1 for v in calibration_status.values() if not v
        ) * 10.0

        # Signal availability (all present = 100%)
        signal_availability = 100.0  # Assume all present if we got here

        # Timestamp quality (assume good for now)
        timestamp_quality = 100.0

        # Overall data quality score
        data_score = avg_quality - calibrating_penalty

        metrics = DataQualityMetrics(
            signal_availability=signal_availability,
            sensor_health_scores=quality_scores,
            calibration_status=calibration_status,
            timestamp_quality=timestamp_quality,
            overall_quality=avg_quality
        )

        return max(0.0, min(100.0, data_score)), metrics

    def _quality_to_score(self, quality: SignalQuality) -> float:
        """Convert signal quality to numeric score"""
        status_scores = {
            DataQualityStatus.GOOD: 100.0,
            DataQualityStatus.UNCERTAIN: 70.0,
            DataQualityStatus.BAD: 30.0,
            DataQualityStatus.CALIBRATING: 50.0,
            DataQualityStatus.INVALID: 0.0,
        }
        base_score = status_scores.get(quality.status, 50.0)
        return base_score * quality.confidence

    def _check_safety_caps(
        self, signals: CombustionSignals, safety_metrics: SafetyMetrics
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if CQI should be capped due to safety conditions (Section 8.3)
        """
        if signals.emergency_stop_active:
            return True, "Emergency stop active"

        if signals.bypass_active and signals.bypass_duration_s > 300:  # 5 minutes
            return True, f"Prolonged bypass active ({signals.bypass_duration_s:.0f}s)"

        if not signals.flame_present and signals.operating_mode == OperatingMode.RUN:
            return True, "Flame not detected during RUN mode"

        if safety_metrics.trip_count_24h >= 3:
            return True, f"Multiple trips in 24h ({safety_metrics.trip_count_24h})"

        return False, None

    def _calculate_confidence(
        self, signals: CombustionSignals, data_metrics: DataQualityMetrics
    ) -> float:
        """Calculate overall confidence in CQI value"""
        # Base confidence on data quality
        confidence = data_metrics.overall_quality / 100.0

        # Reduce confidence if any analyzer is calibrating
        if (signals.o2_quality.is_calibrating or
            signals.co_quality.is_calibrating or
            signals.nox_quality.is_calibrating):
            confidence *= 0.8

        # Reduce confidence in non-steady operation
        if signals.load_percent < 30 or signals.load_percent > 95:
            confidence *= 0.9

        return max(0.0, min(1.0, confidence))

    def _determine_grade(self, cqi: float) -> CQIGrade:
        """Determine CQI grade from score"""
        if cqi >= 90:
            return CQIGrade.EXCELLENT
        elif cqi >= 75:
            return CQIGrade.GOOD
        elif cqi >= 60:
            return CQIGrade.ACCEPTABLE
        elif cqi >= 40:
            return CQIGrade.POOR
        else:
            return CQIGrade.CRITICAL

    def _create_non_run_result(
        self, signals: CombustionSignals, timestamp: datetime, start_time: float
    ) -> CQIResult:
        """Create CQI result for non-RUN operating modes"""
        calc_time_ms = (time.perf_counter() - start_time) * 1000

        # Provide minimal scores for non-run modes
        sub_scores = CQISubScores(
            efficiency=0.0,
            emissions=0.0,
            stability=0.0,
            safety=0.0 if signals.emergency_stop_active else 50.0,
            data=50.0
        )

        provenance_hash = self._calculate_provenance_hash(
            0.0, sub_scores, timestamp, signals
        )

        return CQIResult(
            cqi_total=0.0,
            grade=CQIGrade.CRITICAL,
            sub_scores=sub_scores,
            weights={
                "efficiency": self.config.efficiency_weight,
                "emissions": self.config.emissions_weight,
                "stability": self.config.stability_weight,
                "safety": self.config.safety_weight,
                "data": self.config.data_weight,
            },
            confidence=0.0,
            is_capped=True,
            cap_reason=f"Operating mode is {signals.operating_mode.value}, not RUN",
            active_incidents=[],
            timestamp=timestamp,
            asset_id=self.config.asset_id,
            operating_mode=signals.operating_mode,
            provenance_hash=provenance_hash,
            calculation_time_ms=round(calc_time_ms, 3)
        )

    def _calculate_provenance_hash(
        self, cqi: float, sub_scores: CQISubScores,
        timestamp: datetime, signals: CombustionSignals
    ) -> str:
        """Calculate deterministic provenance hash for audit trail"""
        hashable_data = {
            "cqi_total": round(cqi, 4),
            "efficiency": round(sub_scores.efficiency, 4),
            "emissions": round(sub_scores.emissions, 4),
            "stability": round(sub_scores.stability, 4),
            "safety": round(sub_scores.safety, 4),
            "data": round(sub_scores.data, 4),
            "timestamp": timestamp.isoformat(),
            "o2": round(signals.o2_percent, 4),
            "co": round(signals.co_ppm, 4),
            "nox": round(signals.nox_ppm, 4),
            "asset_id": self.config.asset_id,
        }
        hash_input = json.dumps(hashable_data, sort_keys=True)
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def add_incident(self, incident_id: str) -> None:
        """Add an active incident to be included in CQI results"""
        if incident_id not in self._active_incidents:
            self._active_incidents.append(incident_id)

    def remove_incident(self, incident_id: str) -> None:
        """Remove a resolved incident"""
        if incident_id in self._active_incidents:
            self._active_incidents.remove(incident_id)

    def to_sse_event(self, result: CQIResult) -> Dict[str, Any]:
        """
        Convert CQI result to SSE event format (Section 12.3)

        Returns dict matching the playbook SSE event contract.
        """
        return {
            "schema_version": "1.0",
            "asset_id": result.asset_id,
            "ts_utc": result.timestamp.isoformat(),
            "cqi_total": result.cqi_total,
            "cqi_components": {
                "efficiency": result.sub_scores.efficiency,
                "emissions": result.sub_scores.emissions,
                "stability": result.sub_scores.stability,
                "safety": result.sub_scores.safety,
                "data": result.sub_scores.data,
            },
            "confidence": result.confidence,
            "active_incidents": result.active_incidents,
            "grade": result.grade.value,
            "is_capped": result.is_capped,
            "cap_reason": result.cap_reason,
            "provenance_hash": result.provenance_hash,
        }


# =============================================================================
# MODULE-LEVEL FUNCTIONS
# =============================================================================

def create_default_calculator(asset_id: str = "default") -> CQICalculator:
    """Create a CQI calculator with default configuration"""
    config = CQIConfiguration(asset_id=asset_id)
    return CQICalculator(config)


def calculate_cqi_quick(
    o2_percent: float,
    co_ppm: float,
    nox_ppm: float,
    fuel_flow: float = 1.0,
    air_flow: float = 17.2,
    flame_intensity: float = 80.0,
    asset_id: str = "default"
) -> CQIResult:
    """
    Quick CQI calculation with minimal inputs

    Useful for testing and simple integrations.
    """
    signals = CombustionSignals(
        fuel_flow_kg_s=fuel_flow,
        air_flow_kg_s=air_flow,
        o2_percent=o2_percent,
        co_ppm=co_ppm,
        nox_ppm=nox_ppm,
        flame_intensity=flame_intensity,
    )
    calculator = create_default_calculator(asset_id)
    return calculator.calculate(signals)
