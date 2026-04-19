"""
GL-018 UnifiedCombustionOptimizer - Burner Control Module

Burner tuning, flame stability monitoring, and BMS coordination per NFPA 85.
Provides Flame Stability Index (FSI) calculation and burner optimization.

Features:
    - Flame Stability Index (FSI) calculation
    - Burner tuning recommendations
    - BMS sequence management per NFPA 85
    - Multi-burner coordination
    - Flame detection supervision

Standards:
    - NFPA 85 (Boiler and Combustion Systems Hazards Code)
    - NFPA 86 (Ovens and Furnaces)
    - IEC 61511 (Functional Safety)

Example:
    >>> from greenlang.agents.process_heat.gl_018_unified_combustion import BurnerController
    >>> controller = BurnerController(config)
    >>> fsi = controller.calculate_flame_stability_index(burner_data)
    >>> print(f"Flame Stability Index: {fsi:.2f}")
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import logging
import math
import statistics

from pydantic import BaseModel, Field

from .config import BMSSequence, FlameStabilityConfig, BMSConfig, BurnerConfig
from .schemas import (
    BurnerStatus,
    FlameStabilityAnalysis,
    BurnerTuningResult,
    BMSStatus,
    Alert,
    AlertSeverity,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================


class FlameDetectorType(Enum):
    """Flame detector types."""
    UV = "uv"
    IR = "ir"
    UV_IR = "uv_ir"
    SCANNER = "scanner"
    THERMOCOUPLE = "thermocouple"


class BurnerMode(Enum):
    """Burner operating modes."""
    OFF = auto()
    PILOT = auto()
    LOW_FIRE = auto()
    MODULATING = auto()
    HIGH_FIRE = auto()
    FAULT = auto()


@dataclass(frozen=True)
class FSIWeights:
    """Weights for Flame Stability Index calculation."""
    signal_strength: float = 0.30
    signal_variance: float = 0.25
    flicker_frequency: float = 0.20
    combustion_noise: float = 0.15
    o2_stability: float = 0.10


DEFAULT_FSI_WEIGHTS = FSIWeights()


# Optimal air register positions by fuel type and load
OPTIMAL_AIR_REGISTER: Dict[str, Dict[str, float]] = {
    "natural_gas": {
        "low_fire": 30.0,
        "mid_fire": 50.0,
        "high_fire": 75.0,
    },
    "no2_fuel_oil": {
        "low_fire": 35.0,
        "mid_fire": 55.0,
        "high_fire": 80.0,
    },
}


# =============================================================================
# FLAME STABILITY ANALYZER
# =============================================================================


class FlameStabilityAnalyzer:
    """
    Flame stability analysis with Flame Stability Index (FSI).

    The FSI is a composite metric (0-1) that indicates flame quality:
    - 1.0: Perfect, stable flame
    - 0.85-1.0: Optimal operating range
    - 0.70-0.85: Normal, acceptable
    - 0.50-0.70: Warning, tuning needed
    - <0.50: Alarm, intervention required

    Components of FSI:
    1. Signal Strength: Average flame detector signal
    2. Signal Variance: Stability of flame signal
    3. Flicker Frequency: Combustion oscillation frequency
    4. Combustion Noise: Acoustic signature quality
    5. O2 Stability: Consistency of excess air

    Example:
        >>> analyzer = FlameStabilityAnalyzer()
        >>> fsi = analyzer.calculate_fsi(
        ...     flame_signals=[85.0, 82.0, 88.0, 84.0, 86.0],
        ...     o2_readings=[3.0, 3.1, 2.9, 3.0, 3.1]
        ... )
    """

    def __init__(
        self,
        config: Optional[FlameStabilityConfig] = None,
        weights: FSIWeights = DEFAULT_FSI_WEIGHTS,
    ) -> None:
        """
        Initialize flame stability analyzer.

        Args:
            config: Flame stability configuration
            weights: FSI component weights
        """
        self.config = config or FlameStabilityConfig()
        self.weights = weights
        self._calculation_count = 0
        logger.info("FlameStabilityAnalyzer initialized")

    def calculate_fsi(
        self,
        flame_signals: List[float],
        o2_readings: Optional[List[float]] = None,
        flicker_frequency_hz: Optional[float] = None,
        noise_level_db: Optional[float] = None,
    ) -> Tuple[float, str]:
        """
        Calculate Flame Stability Index.

        Args:
            flame_signals: List of flame detector signals (0-100%)
            o2_readings: Optional list of O2 readings
            flicker_frequency_hz: Measured flicker frequency
            noise_level_db: Combustion noise level

        Returns:
            Tuple of (FSI value 0-1, status string)
        """
        self._calculation_count += 1

        if not flame_signals or len(flame_signals) < 3:
            return 0.0, "INSUFFICIENT_DATA"

        # Component 1: Signal Strength Score
        avg_signal = statistics.mean(flame_signals)
        strength_score = self._calculate_strength_score(avg_signal)

        # Component 2: Signal Variance Score
        signal_std = statistics.stdev(flame_signals) if len(flame_signals) > 1 else 0
        variance_score = self._calculate_variance_score(signal_std)

        # Component 3: Flicker Frequency Score
        if flicker_frequency_hz is not None:
            flicker_score = self._calculate_flicker_score(flicker_frequency_hz)
        else:
            flicker_score = 0.8  # Neutral if not measured

        # Component 4: Combustion Noise Score
        if noise_level_db is not None:
            noise_score = self._calculate_noise_score(noise_level_db)
        else:
            noise_score = 0.8  # Neutral if not measured

        # Component 5: O2 Stability Score
        if o2_readings and len(o2_readings) > 1:
            o2_std = statistics.stdev(o2_readings)
            o2_score = self._calculate_o2_stability_score(o2_std)
        else:
            o2_score = 0.8  # Neutral if not measured

        # Calculate weighted FSI
        fsi = (
            self.weights.signal_strength * strength_score +
            self.weights.signal_variance * variance_score +
            self.weights.flicker_frequency * flicker_score +
            self.weights.combustion_noise * noise_score +
            self.weights.o2_stability * o2_score
        )

        # Determine status
        if fsi >= self.config.fsi_optimal_min:
            status = "optimal"
        elif fsi >= self.config.fsi_warning_threshold:
            status = "normal"
        elif fsi >= self.config.fsi_alarm_threshold:
            status = "warning"
        else:
            status = "alarm"

        return round(fsi, 3), status

    def analyze_multi_burner(
        self,
        burner_readings: Dict[str, List[float]],
        o2_readings: Optional[List[float]] = None,
    ) -> FlameStabilityAnalysis:
        """
        Analyze flame stability for multiple burners.

        Args:
            burner_readings: Dict of burner_id -> flame signal list
            o2_readings: Optional common O2 readings

        Returns:
            FlameStabilityAnalysis with per-burner and aggregate results
        """
        per_burner_fsi = {}
        per_burner_status = {}
        all_signals = []

        for burner_id, signals in burner_readings.items():
            if signals:
                fsi, status = self.calculate_fsi(signals, o2_readings)
                per_burner_fsi[burner_id] = fsi
                per_burner_status[burner_id] = status == "optimal" or status == "normal"
                all_signals.extend(signals)

        # Calculate aggregate FSI
        if per_burner_fsi:
            aggregate_fsi = statistics.mean(per_burner_fsi.values())
        else:
            aggregate_fsi = 0.0

        # Determine aggregate status
        if aggregate_fsi >= self.config.fsi_optimal_min:
            fsi_status = "optimal"
        elif aggregate_fsi >= self.config.fsi_warning_threshold:
            fsi_status = "normal"
        elif aggregate_fsi >= self.config.fsi_alarm_threshold:
            fsi_status = "warning"
        else:
            fsi_status = "alarm"

        # Calculate intensity stats
        intensity_avg = statistics.mean(all_signals) if all_signals else 0.0
        intensity_var = statistics.variance(all_signals) if len(all_signals) > 1 else 0.0

        # Determine if tuning needed
        tuning_required = fsi_status in ["warning", "alarm"]
        tuning_recommendations = []

        if tuning_required:
            # Identify which burners need attention
            for burner_id, fsi in per_burner_fsi.items():
                if fsi < self.config.fsi_warning_threshold:
                    tuning_recommendations.append(
                        f"Burner {burner_id}: FSI={fsi:.2f}, tune air register and check fuel pressure"
                    )

        return FlameStabilityAnalysis(
            flame_stability_index=round(aggregate_fsi, 3),
            fsi_status=fsi_status,
            flame_intensity_avg=round(intensity_avg, 1),
            flame_intensity_variance=round(intensity_var, 2),
            burner_flame_status=per_burner_status,
            burner_fsi=per_burner_fsi,
            tuning_required=tuning_required,
            tuning_recommendations=tuning_recommendations,
        )

    def _calculate_strength_score(self, avg_signal: float) -> float:
        """Calculate signal strength component score."""
        # Optimal range: 70-90%
        # Score decreases outside this range
        if avg_signal >= 70 and avg_signal <= 90:
            return 1.0
        elif avg_signal >= 50 and avg_signal < 70:
            return 0.5 + (avg_signal - 50) / 40
        elif avg_signal > 90:
            return 1.0 - (avg_signal - 90) / 20
        elif avg_signal >= self.config.flame_signal_min_pct:
            return avg_signal / self.config.flame_signal_min_pct * 0.5
        else:
            return 0.0

    def _calculate_variance_score(self, signal_std: float) -> float:
        """Calculate signal variance component score."""
        # Lower variance = better stability
        # Optimal: std < 3%
        # Acceptable: std < 8%
        # Poor: std > 15%
        if signal_std <= 3:
            return 1.0
        elif signal_std <= 8:
            return 1.0 - (signal_std - 3) / 10
        elif signal_std <= 15:
            return 0.5 - (signal_std - 8) / 14
        else:
            return 0.0

    def _calculate_flicker_score(self, frequency_hz: float) -> float:
        """Calculate flicker frequency component score."""
        # Optimal flicker: 2-5 Hz (healthy combustion)
        # Too low: potential instability
        # Too high: potential pulsation
        target = self.config.flame_flicker_frequency_hz

        if abs(frequency_hz - target) <= 1.0:
            return 1.0
        elif abs(frequency_hz - target) <= 2.0:
            return 0.8
        elif abs(frequency_hz - target) <= 4.0:
            return 0.5
        else:
            return 0.2

    def _calculate_noise_score(self, noise_db: float) -> float:
        """Calculate combustion noise component score."""
        # Optimal: 80-90 dB
        # Too quiet: weak flame
        # Too loud: combustion instability
        if noise_db >= 80 and noise_db <= 90:
            return 1.0
        elif noise_db >= 70 and noise_db < 80:
            return 0.5 + (noise_db - 70) / 20
        elif noise_db > 90 and noise_db <= 100:
            return 1.0 - (noise_db - 90) / 20
        elif noise_db > 100:
            return 0.5 - (noise_db - 100) / 20
        else:
            return 0.3

    def _calculate_o2_stability_score(self, o2_std: float) -> float:
        """Calculate O2 stability component score."""
        # Optimal: std < 0.3%
        # Acceptable: std < 0.5%
        # Poor: std > 1.0%
        if o2_std <= 0.3:
            return 1.0
        elif o2_std <= 0.5:
            return 0.8
        elif o2_std <= 1.0:
            return 0.5
        else:
            return max(0.0, 0.5 - (o2_std - 1.0) / 2)


# =============================================================================
# BURNER TUNING CONTROLLER
# =============================================================================


class BurnerTuningController:
    """
    Burner tuning and optimization controller.

    Provides recommendations for:
    - Air register adjustment
    - Fuel pressure optimization
    - Load distribution for multi-burner systems
    - Startup/shutdown sequence tuning

    Example:
        >>> controller = BurnerTuningController()
        >>> recommendation = controller.calculate_tuning(
        ...     burner_status=burner,
        ...     flue_gas_o2=3.5,
        ...     target_o2=3.0,
        ...     co_ppm=25.0
        ... )
    """

    def __init__(self, burner_config: Optional[BurnerConfig] = None) -> None:
        """
        Initialize burner tuning controller.

        Args:
            burner_config: Burner configuration
        """
        self.config = burner_config
        logger.info("BurnerTuningController initialized")

    def calculate_tuning(
        self,
        burner_status: BurnerStatus,
        flue_gas_o2_pct: float,
        target_o2_pct: float,
        co_ppm: float,
        nox_ppm: Optional[float] = None,
        load_pct: float = 75.0,
        fuel_type: str = "natural_gas",
    ) -> BurnerTuningResult:
        """
        Calculate burner tuning recommendations.

        Args:
            burner_status: Current burner status
            flue_gas_o2_pct: Current O2 reading
            target_o2_pct: Target O2 setpoint
            co_ppm: Current CO reading
            nox_ppm: Current NOx reading (optional)
            load_pct: Current load percentage
            fuel_type: Fuel type

        Returns:
            BurnerTuningResult with recommendations
        """
        current_air_register = burner_status.air_register_position_pct
        current_fuel_pressure = None  # Would come from actual data

        # Calculate O2 deviation
        o2_deviation = flue_gas_o2_pct - target_o2_pct

        # Determine optimal air register based on load
        load_key = self._get_load_key(load_pct)
        fuel_key = fuel_type.lower().replace(" ", "_")
        optimal_registers = OPTIMAL_AIR_REGISTER.get(
            fuel_key, OPTIMAL_AIR_REGISTER["natural_gas"]
        )
        base_optimal = optimal_registers.get(load_key, 50.0)

        # Adjust for O2 deviation
        # Each 1% O2 deviation = ~3% air register change
        air_register_adjustment = -o2_deviation * 3.0

        # Safety: Limit adjustments if CO is high
        if co_ppm > 100:
            # Don't reduce air if CO is elevated
            air_register_adjustment = max(0, air_register_adjustment)

        # Calculate recommended position
        recommended_air_register = current_air_register + air_register_adjustment
        recommended_air_register = max(10.0, min(95.0, recommended_air_register))

        # Expected changes
        expected_o2_change = -air_register_adjustment / 3.0
        expected_co_change = 0.0

        if air_register_adjustment > 0:
            # Opening air should reduce CO
            expected_co_change = -air_register_adjustment * 2
        elif air_register_adjustment < 0 and co_ppm < 50:
            # Only predict CO increase if currently low
            expected_co_change = -air_register_adjustment * 1.5

        # NOx impact
        expected_nox_change = 0.0
        if nox_ppm is not None:
            if air_register_adjustment > 0:
                # More air = cooler flame = lower NOx
                expected_nox_change = -air_register_adjustment * 0.5
            else:
                expected_nox_change = -air_register_adjustment * 0.3

        # Determine priority
        if abs(o2_deviation) > 2.0 or co_ppm > 100:
            priority = "high"
        elif abs(o2_deviation) > 1.0:
            priority = "medium"
        else:
            priority = "low"

        # Confidence based on data quality
        confidence = 85.0
        if co_ppm > 200:
            confidence = 70.0  # Less certain when combustion is poor

        return BurnerTuningResult(
            burner_id=burner_status.burner_id,
            current_air_register_pct=round(current_air_register, 1),
            current_fuel_pressure_psig=current_fuel_pressure,
            recommended_air_register_pct=round(recommended_air_register, 1),
            recommended_fuel_pressure_psig=current_fuel_pressure,
            expected_o2_change_pct=round(expected_o2_change, 2),
            expected_co_change_ppm=round(expected_co_change, 1),
            expected_nox_change_ppm=round(expected_nox_change, 1),
            priority=priority,
            confidence_pct=confidence,
        )

    def optimize_load_distribution(
        self,
        burner_statuses: List[BurnerStatus],
        total_demand_pct: float,
        burner_fsi: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Optimize load distribution across multiple burners.

        Prioritizes burners with better FSI (flame stability).

        Args:
            burner_statuses: List of burner statuses
            total_demand_pct: Total load demand
            burner_fsi: FSI by burner ID

        Returns:
            Dict of burner_id -> recommended firing rate
        """
        if not burner_statuses:
            return {}

        n_burners = len(burner_statuses)
        base_load = total_demand_pct / n_burners

        distribution = {}

        # Calculate FSI-weighted distribution
        total_fsi = sum(burner_fsi.values()) if burner_fsi else n_burners
        if total_fsi == 0:
            total_fsi = n_burners

        for burner in burner_statuses:
            fsi = burner_fsi.get(burner.burner_id, 0.8)

            # Weight by FSI
            weight = fsi / (total_fsi / n_burners)
            burner_load = base_load * weight

            # Enforce min/max
            burner_load = max(25.0, min(burner_load, 100.0))
            distribution[burner.burner_id] = round(burner_load, 1)

        # Normalize to match total demand
        current_total = sum(distribution.values())
        if current_total > 0:
            scale = (total_demand_pct * n_burners / 100) * 100 / current_total
            for burner_id in distribution:
                distribution[burner_id] = round(
                    distribution[burner_id] * scale, 1
                )

        return distribution

    def _get_load_key(self, load_pct: float) -> str:
        """Get load category key."""
        if load_pct < 40:
            return "low_fire"
        elif load_pct < 70:
            return "mid_fire"
        else:
            return "high_fire"


# =============================================================================
# BMS SEQUENCE CONTROLLER
# =============================================================================


class BMSSequenceController:
    """
    Burner Management System sequence controller per NFPA 85.

    Manages startup, shutdown, and operating sequences with
    proper timing and interlock verification.

    Sequences:
    1. Pre-purge: Verify air flow, purge furnace
    2. Pilot trial: Light and prove pilot
    3. Main flame trial: Light and prove main burner
    4. Running: Normal modulating operation
    5. Post-purge: Purge after shutdown

    Example:
        >>> controller = BMSSequenceController(bms_config)
        >>> status = controller.get_sequence_status()
        >>> can_fire = controller.verify_permissives()
    """

    def __init__(self, config: BMSConfig) -> None:
        """
        Initialize BMS sequence controller.

        Args:
            config: BMS configuration
        """
        self.config = config
        self._current_sequence = BMSSequence.IDLE
        self._sequence_start_time: Optional[datetime] = None
        self._interlocks_satisfied = False
        self._purge_complete = False
        logger.info("BMSSequenceController initialized")

    def get_status(
        self,
        flame_signals: Dict[str, float],
        air_flow_verified: bool,
        interlocks: Dict[str, bool],
    ) -> BMSStatus:
        """
        Get current BMS status.

        Args:
            flame_signals: Flame detector signals by detector ID
            air_flow_verified: Air flow interlock status
            interlocks: Dict of interlock_name -> status

        Returns:
            BMSStatus with current state
        """
        # Check interlocks
        tripped = [name for name, status in interlocks.items() if not status]
        all_satisfied = len(tripped) == 0

        # Flame detection
        pilot_proven = any(
            v > 30 for k, v in flame_signals.items() if "pilot" in k.lower()
        )
        main_proven = any(
            v > 30 for k, v in flame_signals.items() if "main" in k.lower()
        )
        if not pilot_proven and not main_proven:
            # Use any signal if not differentiated
            main_proven = any(v > 30 for v in flame_signals.values())

        # Detector status
        detector_status = {
            det_id: "normal" if sig > 30 else "no_flame"
            for det_id, sig in flame_signals.items()
        }

        # Calculate remaining time if in timed sequence
        time_remaining = None
        if self._sequence_start_time:
            elapsed = (datetime.now(timezone.utc) - self._sequence_start_time).total_seconds()
            if self._current_sequence == BMSSequence.PRE_PURGE:
                time_remaining = max(0, self.config.pre_purge_time_s - elapsed)
            elif self._current_sequence == BMSSequence.PILOT_TRIAL:
                time_remaining = max(0, self.config.pilot_trial_time_s - elapsed)
            elif self._current_sequence == BMSSequence.MAIN_FLAME_TRIAL:
                time_remaining = max(0, self.config.main_flame_trial_time_s - elapsed)
            elif self._current_sequence == BMSSequence.POST_PURGE:
                time_remaining = max(0, self.config.post_purge_time_s - elapsed)

        # Ready to fire
        ready = (
            all_satisfied and
            self._purge_complete and
            self._current_sequence != BMSSequence.LOCKOUT
        )

        return BMSStatus(
            current_sequence=self._current_sequence,
            sequence_time_remaining_s=time_remaining,
            all_interlocks_satisfied=all_satisfied,
            active_interlocks=list(interlocks.keys()),
            tripped_interlocks=tripped,
            purge_complete=self._purge_complete,
            purge_air_flow_verified=air_flow_verified,
            pilot_flame_proven=pilot_proven,
            main_flame_proven=main_proven,
            flame_detector_status=detector_status,
            ready_to_fire=ready,
            permissive_satisfied=all_satisfied,
        )

    def verify_startup_permissives(
        self,
        interlocks: Dict[str, bool],
        air_flow_pct: float,
    ) -> Tuple[bool, List[str]]:
        """
        Verify startup permissives per NFPA 85.

        Args:
            interlocks: Dict of interlock_name -> status
            air_flow_pct: Current air flow as % of capacity

        Returns:
            Tuple of (all_satisfied, list of failed permissives)
        """
        failed = []

        # Check required interlocks
        required_interlocks = [
            "fuel_supply_pressure",
            "combustion_air_pressure",
            "low_water_cutoff",
            "high_pressure_limit",
            "flame_failure_relay",
        ]

        for interlock in required_interlocks:
            if interlock in interlocks and not interlocks[interlock]:
                failed.append(f"Interlock not satisfied: {interlock}")

        # Check air flow for purge
        if air_flow_pct < self.config.purge_air_flow_pct:
            failed.append(
                f"Insufficient air flow for purge: {air_flow_pct:.1f}% "
                f"< {self.config.purge_air_flow_pct}% required"
            )

        # Check low fire interlock
        if self.config.low_fire_interlock:
            if "low_fire_position" in interlocks:
                if not interlocks["low_fire_position"]:
                    failed.append("Not in low fire position for lightoff")

        return len(failed) == 0, failed

    def calculate_purge_time(
        self,
        furnace_volume_ft3: float,
        air_flow_cfm: float,
    ) -> float:
        """
        Calculate required purge time per NFPA 85.

        NFPA 85 requires minimum 4 volume changes during purge.

        Args:
            furnace_volume_ft3: Furnace volume in cubic feet
            air_flow_cfm: Air flow in CFM

        Returns:
            Required purge time in seconds
        """
        if air_flow_cfm <= 0:
            return self.config.pre_purge_time_s  # Use default

        # Calculate time for required volume changes
        minutes_per_change = furnace_volume_ft3 / air_flow_cfm
        required_time_min = minutes_per_change * self.config.purge_volume_changes
        required_time_s = required_time_min * 60

        # At least the configured minimum
        return max(required_time_s, self.config.pre_purge_time_s)

    def get_flame_detector_voting_result(
        self,
        detector_signals: Dict[str, float],
        min_signal_pct: float = 30.0,
    ) -> Tuple[bool, str]:
        """
        Evaluate flame detector voting logic.

        Supports: 1oo1, 1oo2, 2oo2, 2oo3

        Args:
            detector_signals: Dict of detector_id -> signal percent
            min_signal_pct: Minimum signal for flame proven

        Returns:
            Tuple of (flame_proven, voting_result)
        """
        n_detectors = len(detector_signals)
        n_detecting = sum(
            1 for v in detector_signals.values() if v >= min_signal_pct
        )

        logic = self.config.flame_detector_redundancy

        if logic == "1oo1":
            # Any single detector proves flame
            proven = n_detecting >= 1
            result = f"{n_detecting}/{n_detectors} detecting"

        elif logic == "1oo2":
            # Either of 2 detectors proves flame
            proven = n_detecting >= 1 and n_detectors >= 2
            result = f"{n_detecting}/2 detecting"

        elif logic == "2oo2":
            # Both detectors must detect
            proven = n_detecting >= 2
            result = f"{n_detecting}/2 detecting (both required)"

        elif logic == "2oo3":
            # 2 of 3 detectors must detect
            proven = n_detecting >= 2 and n_detectors >= 3
            result = f"{n_detecting}/3 detecting (2 required)"

        else:
            # Default to 1oo1
            proven = n_detecting >= 1
            result = f"{n_detecting}/{n_detectors} detecting (default 1oo1)"

        return proven, result

    def set_sequence(self, sequence: BMSSequence) -> None:
        """Set current BMS sequence."""
        self._current_sequence = sequence
        self._sequence_start_time = datetime.now(timezone.utc)
        logger.info(f"BMS sequence changed to: {sequence}")

    def complete_purge(self) -> None:
        """Mark purge as complete."""
        self._purge_complete = True
        logger.info("Purge cycle complete")

    def trigger_lockout(self, reason: str) -> None:
        """Trigger BMS lockout."""
        self._current_sequence = BMSSequence.LOCKOUT
        self._sequence_start_time = None
        logger.critical(f"BMS LOCKOUT triggered: {reason}")

    def reset_lockout(self) -> bool:
        """
        Reset from lockout state.

        Returns:
            True if reset successful
        """
        if self._current_sequence == BMSSequence.LOCKOUT:
            self._current_sequence = BMSSequence.IDLE
            self._purge_complete = False
            self._sequence_start_time = None
            logger.info("BMS lockout reset")
            return True
        return False
