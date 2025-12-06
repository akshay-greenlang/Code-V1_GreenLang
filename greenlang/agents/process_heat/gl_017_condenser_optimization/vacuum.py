"""
GL-017 CONDENSYNC Agent - Vacuum System Monitoring Module

This module implements vacuum system monitoring for steam surface condensers,
including steam jet ejector performance, air removal capacity tracking,
and vacuum decay analysis.

All calculations are deterministic with zero hallucination.

Example:
    >>> monitor = VacuumSystemMonitor(config)
    >>> result = monitor.analyze_vacuum_system(
    ...     condenser_vacuum=1.5,
    ...     motive_steam_pressure=150.0,
    ...     air_removal_scfm=40.0,
    ... )
    >>> print(f"Vacuum normal: {result.vacuum_normal}")
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import hashlib
import logging
import math

from pydantic import BaseModel, Field

from greenlang.agents.process_heat.gl_017_condenser_optimization.config import (
    VacuumSystemConfig,
    VacuumEquipmentType,
    PerformanceConfig,
)
from greenlang.agents.process_heat.gl_017_condenser_optimization.schemas import (
    VacuumSystemResult,
    VacuumSystemInput,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - Vacuum System Engineering Data
# =============================================================================

class VacuumConstants:
    """Vacuum system engineering constants."""

    # Steam jet ejector design parameters
    # Specific steam consumption (lb steam / lb air) typical values
    SPECIFIC_STEAM_SINGLE_STAGE = 150.0
    SPECIFIC_STEAM_TWO_STAGE = 80.0
    SPECIFIC_STEAM_THREE_STAGE = 40.0

    # Minimum motive steam pressure ratios
    MIN_MOTIVE_PRESSURE_RATIO = 1.5  # Motive pressure / discharge pressure

    # Air equivalent weights
    # Weight of non-condensables equivalent to 1 lb dry air at 70F
    AIR_EQUIVALENT = {
        "air": 1.00,
        "nitrogen": 0.97,
        "oxygen": 1.11,
        "co2": 1.53,
        "water_vapor": 0.62,
    }

    # Standard conditions
    STANDARD_TEMP_F = 70.0
    STANDARD_PRESSURE_PSIA = 14.696

    # Vacuum decay limits
    ACCEPTABLE_DECAY_RATE_INHG_MIN = 0.1
    WARNING_DECAY_RATE_INHG_MIN = 0.2
    ALARM_DECAY_RATE_INHG_MIN = 0.5

    # Temperature effect on air ingress
    # Warmer condensate = more air release
    TEMP_AIR_RELEASE_FACTOR = 0.02  # Per degree F above 100F


@dataclass
class VacuumReading:
    """Historical vacuum reading."""
    timestamp: datetime
    vacuum_inhga: float
    load_pct: float
    air_removal_scfm: float
    motive_steam_pressure_psig: float


class SteamJetEjectorModel:
    """
    Steam jet ejector performance model.

    Models ejector performance including capacity curves,
    efficiency, and operating limits.
    """

    def __init__(
        self,
        stages: int = 2,
        design_capacity_scfm: float = 50.0,
        design_motive_pressure_psig: float = 150.0,
        design_suction_inhga: float = 1.5,
    ) -> None:
        """
        Initialize ejector model.

        Args:
            stages: Number of ejector stages
            design_capacity_scfm: Design air removal capacity (SCFM)
            design_motive_pressure_psig: Design motive steam pressure (psig)
            design_suction_inhga: Design suction pressure (inHgA)
        """
        self.stages = stages
        self.design_capacity_scfm = design_capacity_scfm
        self.design_motive_pressure_psig = design_motive_pressure_psig
        self.design_suction_inhga = design_suction_inhga

        # Calculate design specific steam consumption
        if stages == 1:
            self.design_specific_steam = VacuumConstants.SPECIFIC_STEAM_SINGLE_STAGE
        elif stages == 2:
            self.design_specific_steam = VacuumConstants.SPECIFIC_STEAM_TWO_STAGE
        else:
            self.design_specific_steam = VacuumConstants.SPECIFIC_STEAM_THREE_STAGE

    def calculate_capacity(
        self,
        motive_steam_pressure_psig: float,
        suction_pressure_inhga: float,
    ) -> float:
        """
        Calculate ejector capacity at given conditions.

        Uses characteristic curve for steam jet ejectors.

        Args:
            motive_steam_pressure_psig: Motive steam pressure (psig)
            suction_pressure_inhga: Suction pressure (inHgA)

        Returns:
            Air removal capacity (SCFM)
        """
        # Pressure ratio effects
        motive_ratio = (
            motive_steam_pressure_psig / self.design_motive_pressure_psig
        )

        # Capacity varies with sqrt of motive pressure ratio
        motive_factor = math.sqrt(max(0.1, motive_ratio))

        # Suction pressure effect
        # Lower suction = lower capacity (characteristic curve)
        suction_ratio = suction_pressure_inhga / self.design_suction_inhga

        if suction_ratio > 1.5:
            # Overloaded - capacity increases but efficiency drops
            suction_factor = 1.0 + 0.2 * (suction_ratio - 1.5)
        elif suction_ratio < 0.5:
            # Operating at lower vacuum than design - reduced capacity
            suction_factor = suction_ratio * 1.5
        else:
            suction_factor = 1.0

        capacity = self.design_capacity_scfm * motive_factor * suction_factor

        return max(0.0, capacity)

    def calculate_steam_consumption(
        self,
        air_removal_scfm: float,
        motive_steam_pressure_psig: float,
        suction_pressure_inhga: float,
    ) -> float:
        """
        Calculate motive steam consumption.

        Args:
            air_removal_scfm: Air removal rate (SCFM)
            motive_steam_pressure_psig: Motive steam pressure (psig)
            suction_pressure_inhga: Suction pressure (inHgA)

        Returns:
            Steam consumption (lb/hr)
        """
        # Convert SCFM to lb/hr of air (at standard conditions)
        # Air density at standard conditions: ~0.075 lb/ft3
        air_lb_hr = air_removal_scfm * 60 * 0.075

        # Specific steam consumption varies with conditions
        specific_steam = self.design_specific_steam

        # Pressure effects
        motive_ratio = (
            motive_steam_pressure_psig / self.design_motive_pressure_psig
        )
        if motive_ratio < 0.8:
            # Low motive pressure = higher specific steam
            specific_steam *= (1.0 / motive_ratio) ** 0.5

        # Calculate steam consumption
        steam_lb_hr = air_lb_hr * specific_steam

        return steam_lb_hr

    def calculate_efficiency(
        self,
        actual_capacity_scfm: float,
        actual_steam_lb_hr: float,
        motive_steam_pressure_psig: float,
    ) -> float:
        """
        Calculate ejector efficiency.

        Efficiency = (Ideal steam / Actual steam) * 100

        Args:
            actual_capacity_scfm: Measured air removal (SCFM)
            actual_steam_lb_hr: Measured steam consumption (lb/hr)
            motive_steam_pressure_psig: Motive steam pressure (psig)

        Returns:
            Efficiency (%)
        """
        if actual_steam_lb_hr <= 0:
            return 0.0

        # Calculate ideal steam consumption
        ideal_steam = self.calculate_steam_consumption(
            actual_capacity_scfm,
            self.design_motive_pressure_psig,
            self.design_suction_inhga,
        )

        efficiency = (ideal_steam / actual_steam_lb_hr) * 100

        return min(100.0, max(0.0, efficiency))


class VacuumSystemMonitor:
    """
    Vacuum system monitoring and analysis.

    This class monitors condenser vacuum system performance including
    steam jet ejectors, vacuum pumps, and air removal capacity.

    Features:
        - Vacuum trend analysis
        - Air removal capacity tracking
        - Ejector efficiency monitoring
        - Vacuum decay testing
        - Maintenance recommendations

    Attributes:
        config: Vacuum system configuration
        performance_config: Performance configuration
        ejector_model: Steam jet ejector performance model

    Example:
        >>> config = VacuumSystemConfig()
        >>> monitor = VacuumSystemMonitor(config)
        >>> result = monitor.analyze_vacuum_system(input_data)
    """

    def __init__(
        self,
        vacuum_config: VacuumSystemConfig,
        performance_config: PerformanceConfig,
    ) -> None:
        """
        Initialize the vacuum system monitor.

        Args:
            vacuum_config: Vacuum system configuration
            performance_config: Performance configuration
        """
        self.config = vacuum_config
        self.performance_config = performance_config
        self._history: List[VacuumReading] = []
        self._calculation_count = 0

        # Initialize ejector model if using ejectors
        if self.config.primary_equipment == VacuumEquipmentType.STEAM_JET_EJECTOR:
            self.ejector_model = SteamJetEjectorModel(
                stages=self.config.ejector_stages,
                design_capacity_scfm=self.config.air_removal_capacity_scfm,
                design_motive_pressure_psig=self.config.motive_steam_pressure_psig,
                design_suction_inhga=self.config.design_vacuum_inhga,
            )
        else:
            self.ejector_model = None

        logger.info(
            f"VacuumSystemMonitor initialized: "
            f"Equipment={self.config.primary_equipment.value}"
        )

    def analyze_vacuum_system(
        self,
        condenser_vacuum_inhga: float,
        motive_steam_pressure_psig: Optional[float] = None,
        motive_steam_flow_lb_hr: Optional[float] = None,
        air_removal_scfm: Optional[float] = None,
        load_pct: float = 100.0,
        saturation_temp_f: Optional[float] = None,
        cw_inlet_temp_f: Optional[float] = None,
    ) -> VacuumSystemResult:
        """
        Analyze vacuum system performance.

        Args:
            condenser_vacuum_inhga: Current vacuum (inHgA)
            motive_steam_pressure_psig: Motive steam pressure (psig)
            motive_steam_flow_lb_hr: Motive steam flow (lb/hr)
            air_removal_scfm: Measured air removal (SCFM)
            load_pct: Unit load (%)
            saturation_temp_f: Saturation temperature (F)
            cw_inlet_temp_f: Cooling water inlet temperature (F)

        Returns:
            VacuumSystemResult with analysis
        """
        logger.debug(
            f"Analyzing vacuum system: vacuum={condenser_vacuum_inhga:.2f} inHgA"
        )
        self._calculation_count += 1

        # Use config defaults if not provided
        motive_pressure = (
            motive_steam_pressure_psig or
            self.config.motive_steam_pressure_psig
        )

        # Calculate expected vacuum
        expected_vacuum = self._calculate_expected_vacuum(
            load_pct, cw_inlet_temp_f
        )

        # Calculate vacuum deviation
        vacuum_deviation = condenser_vacuum_inhga - expected_vacuum

        # Determine if vacuum is normal
        vacuum_normal = self._is_vacuum_normal(
            condenser_vacuum_inhga, expected_vacuum
        )

        # Estimate air removal capacity
        capacity_pct = self._calculate_capacity_utilization(
            air_removal_scfm, condenser_vacuum_inhga, motive_pressure
        )

        # Estimate air ingress
        estimated_air_ingress = self._estimate_air_ingress(
            condenser_vacuum_inhga, expected_vacuum, air_removal_scfm
        )

        # Check for excessive air ingress
        air_ingress_excessive = (
            estimated_air_ingress > self.config.air_removal_capacity_scfm * 0.8
        )

        # Calculate ejector efficiency if applicable
        ejector_efficiency = None
        motive_steam_consumption = None
        motive_steam_specific = None

        if self.ejector_model and motive_steam_flow_lb_hr and air_removal_scfm:
            ejector_efficiency = self.ejector_model.calculate_efficiency(
                air_removal_scfm, motive_steam_flow_lb_hr, motive_pressure
            )
            motive_steam_consumption = motive_steam_flow_lb_hr
            if air_removal_scfm > 0:
                motive_steam_specific = motive_steam_flow_lb_hr / air_removal_scfm

        # Determine maintenance needs
        maintenance_required = self._check_maintenance_required(
            condenser_vacuum_inhga, expected_vacuum,
            ejector_efficiency, capacity_pct
        )
        recommended_action = self._recommend_action(
            vacuum_normal, air_ingress_excessive,
            ejector_efficiency, capacity_pct
        )

        # Record history
        self._record_reading(
            condenser_vacuum_inhga, load_pct,
            air_removal_scfm or 0.0, motive_pressure
        )

        result = VacuumSystemResult(
            vacuum_normal=vacuum_normal,
            current_vacuum_inhga=round(condenser_vacuum_inhga, 3),
            expected_vacuum_inhga=round(expected_vacuum, 3),
            vacuum_deviation_inhg=round(vacuum_deviation, 3),
            air_removal_capacity_pct=round(capacity_pct, 1),
            estimated_air_ingress_scfm=round(estimated_air_ingress, 2),
            air_ingress_excessive=air_ingress_excessive,
            ejector_efficiency_pct=(
                round(ejector_efficiency, 1) if ejector_efficiency else None
            ),
            motive_steam_consumption_lb_hr=(
                round(motive_steam_consumption, 0)
                if motive_steam_consumption else None
            ),
            motive_steam_specific_lb_scfm=(
                round(motive_steam_specific, 1)
                if motive_steam_specific else None
            ),
            maintenance_required=maintenance_required,
            recommended_action=recommended_action,
        )

        logger.info(
            f"Vacuum analysis complete: normal={vacuum_normal}, "
            f"deviation={vacuum_deviation:.3f} inHg"
        )

        return result

    def perform_vacuum_decay_test(
        self,
        initial_vacuum_inhga: float,
        final_vacuum_inhga: float,
        duration_minutes: float,
    ) -> Dict[str, any]:
        """
        Analyze vacuum decay test results.

        A vacuum decay test is performed by isolating the condenser
        from the vacuum equipment and measuring the rate of vacuum loss.

        Args:
            initial_vacuum_inhga: Initial vacuum (inHgA)
            final_vacuum_inhga: Final vacuum after isolation (inHgA)
            duration_minutes: Test duration (minutes)

        Returns:
            Dictionary with test results and analysis
        """
        logger.info(
            f"Analyzing vacuum decay test: "
            f"{initial_vacuum_inhga:.2f} -> {final_vacuum_inhga:.2f} inHgA "
            f"over {duration_minutes:.1f} minutes"
        )

        # Calculate decay rate
        vacuum_change = final_vacuum_inhga - initial_vacuum_inhga
        decay_rate = vacuum_change / duration_minutes  # inHg/min

        # Determine severity
        acceptable = self.config.acceptable_decay_rate_inhg_min
        if decay_rate <= acceptable:
            status = "acceptable"
            severity = "none"
        elif decay_rate <= VacuumConstants.WARNING_DECAY_RATE_INHG_MIN:
            status = "marginal"
            severity = "minor"
        elif decay_rate <= VacuumConstants.ALARM_DECAY_RATE_INHG_MIN:
            status = "excessive"
            severity = "moderate"
        else:
            status = "severe"
            severity = "severe"

        # Estimate air ingress from decay rate
        # Approximate: 1 inHg/min decay = ~100 SCFM air ingress
        estimated_ingress_scfm = decay_rate * 100

        # Determine recommended action
        if status == "acceptable":
            action = "No action required"
        elif status == "marginal":
            action = "Schedule leak survey"
        elif status == "excessive":
            action = "Perform tracer gas leak detection"
        else:
            action = "Urgent: Major air in-leakage, immediate leak repair required"

        result = {
            "test_passed": status == "acceptable",
            "decay_rate_inhg_min": round(decay_rate, 4),
            "acceptable_rate_inhg_min": acceptable,
            "status": status,
            "severity": severity,
            "estimated_air_ingress_scfm": round(estimated_ingress_scfm, 1),
            "recommended_action": action,
            "test_duration_min": duration_minutes,
            "vacuum_change_inhg": round(vacuum_change, 3),
        }

        return result

    def _calculate_expected_vacuum(
        self,
        load_pct: float,
        cw_inlet_temp_f: Optional[float] = None,
    ) -> float:
        """
        Calculate expected vacuum based on operating conditions.

        Args:
            load_pct: Unit load (%)
            cw_inlet_temp_f: Cooling water inlet temperature (F)

        Returns:
            Expected vacuum (inHgA)
        """
        design_vacuum = self.config.design_vacuum_inhga

        # Load correction
        # Lower load = better vacuum (less heat rejection)
        load_factor = load_pct / 100.0
        load_correction = (load_factor - 1.0) * 0.2

        # Temperature correction (if provided)
        temp_correction = 0.0
        if cw_inlet_temp_f is not None:
            design_inlet = self.performance_config.design_inlet_temp_f
            temp_diff = cw_inlet_temp_f - design_inlet
            # Higher inlet temp = higher (worse) vacuum
            temp_correction = temp_diff * 0.02

        expected = design_vacuum + load_correction + temp_correction
        return max(0.5, min(5.0, expected))

    def _is_vacuum_normal(
        self,
        actual_vacuum: float,
        expected_vacuum: float,
    ) -> bool:
        """
        Determine if vacuum is within normal range.

        Args:
            actual_vacuum: Actual vacuum (inHgA)
            expected_vacuum: Expected vacuum (inHgA)

        Returns:
            True if vacuum is normal
        """
        deviation = abs(actual_vacuum - expected_vacuum)
        max_deviation = 0.3  # inHg

        # Check absolute limits
        if actual_vacuum > self.config.min_vacuum_inhga:
            return False
        if actual_vacuum < self.config.max_vacuum_inhga:
            return False

        return deviation <= max_deviation

    def _calculate_capacity_utilization(
        self,
        air_removal_scfm: Optional[float],
        condenser_vacuum: float,
        motive_pressure: float,
    ) -> float:
        """
        Calculate air removal capacity utilization.

        Args:
            air_removal_scfm: Measured air removal (SCFM)
            condenser_vacuum: Current vacuum (inHgA)
            motive_pressure: Motive steam pressure (psig)

        Returns:
            Capacity utilization (%)
        """
        if air_removal_scfm is None:
            return 100.0  # Assume at capacity if not measured

        # Calculate available capacity
        if self.ejector_model:
            available = self.ejector_model.calculate_capacity(
                motive_pressure, condenser_vacuum
            )
        else:
            available = self.config.air_removal_capacity_scfm

        if available <= 0:
            return 100.0

        utilization = (air_removal_scfm / available) * 100
        return min(150.0, utilization)

    def _estimate_air_ingress(
        self,
        actual_vacuum: float,
        expected_vacuum: float,
        air_removal_scfm: Optional[float],
    ) -> float:
        """
        Estimate air ingress rate.

        Args:
            actual_vacuum: Actual vacuum (inHgA)
            expected_vacuum: Expected vacuum (inHgA)
            air_removal_scfm: Measured air removal (SCFM)

        Returns:
            Estimated air ingress (SCFM)
        """
        if air_removal_scfm is not None:
            # Air removal = air ingress at steady state
            return air_removal_scfm

        # Estimate from vacuum deviation
        vacuum_deviation = actual_vacuum - expected_vacuum
        if vacuum_deviation <= 0:
            return 0.0

        # Rule of thumb: 0.1 inHg deviation per 5 SCFM additional air
        estimated_ingress = vacuum_deviation * 50

        return estimated_ingress

    def _check_maintenance_required(
        self,
        actual_vacuum: float,
        expected_vacuum: float,
        ejector_efficiency: Optional[float],
        capacity_utilization: float,
    ) -> bool:
        """
        Check if maintenance is required.

        Args:
            actual_vacuum: Actual vacuum (inHgA)
            expected_vacuum: Expected vacuum (inHgA)
            ejector_efficiency: Ejector efficiency (%)
            capacity_utilization: Capacity utilization (%)

        Returns:
            True if maintenance required
        """
        # Vacuum deviation check
        deviation = actual_vacuum - expected_vacuum
        if deviation > 0.5:
            return True

        # Efficiency check
        if ejector_efficiency is not None and ejector_efficiency < 70:
            return True

        # Capacity check
        if capacity_utilization > 90:
            return True

        return False

    def _recommend_action(
        self,
        vacuum_normal: bool,
        air_ingress_excessive: bool,
        ejector_efficiency: Optional[float],
        capacity_utilization: float,
    ) -> Optional[str]:
        """
        Recommend corrective action.

        Args:
            vacuum_normal: Is vacuum normal
            air_ingress_excessive: Is air ingress excessive
            ejector_efficiency: Ejector efficiency (%)
            capacity_utilization: Capacity utilization (%)

        Returns:
            Recommended action or None
        """
        if vacuum_normal and not air_ingress_excessive:
            return None

        if air_ingress_excessive:
            return "Perform air in-leakage survey and repair identified leaks"

        if ejector_efficiency is not None and ejector_efficiency < 60:
            return "Inspect and clean steam jet ejectors; check for nozzle wear"

        if capacity_utilization > 95:
            return "Air removal capacity at limit; investigate air sources"

        if not vacuum_normal:
            return "Investigate cause of vacuum deviation"

        return None

    def _record_reading(
        self,
        vacuum: float,
        load: float,
        air_removal: float,
        motive_pressure: float,
    ) -> None:
        """Record a vacuum reading."""
        reading = VacuumReading(
            timestamp=datetime.now(timezone.utc),
            vacuum_inhga=vacuum,
            load_pct=load,
            air_removal_scfm=air_removal,
            motive_steam_pressure_psig=motive_pressure,
        )
        self._history.append(reading)

        # Trim old history
        cutoff = datetime.now(timezone.utc).timestamp() - (7 * 24 * 3600)
        self._history = [
            r for r in self._history
            if r.timestamp.timestamp() > cutoff
        ]

    def get_vacuum_trend(
        self,
        hours: int = 24,
    ) -> List[Tuple[datetime, float]]:
        """
        Get vacuum trend data.

        Args:
            hours: Hours of history

        Returns:
            List of (timestamp, vacuum) tuples
        """
        cutoff = datetime.now(timezone.utc).timestamp() - (hours * 3600)

        trend = [
            (r.timestamp, r.vacuum_inhga)
            for r in self._history
            if r.timestamp.timestamp() > cutoff
        ]

        return sorted(trend, key=lambda x: x[0])

    @property
    def calculation_count(self) -> int:
        """Get total calculation count."""
        return self._calculation_count
