"""
GL-017 CONDENSYNC Agent - Cooling Tower Optimization Module

This module implements cooling tower optimization including cycles of
concentration calculation, blowdown optimization, water balance, and
chemistry compliance monitoring.

All calculations are deterministic with zero hallucination.
Based on Cooling Technology Institute (CTI) standards.

Example:
    >>> optimizer = CoolingTowerOptimizer(config)
    >>> result = optimizer.analyze_cooling_tower(
    ...     hot_water_temp=105.0,
    ...     cold_water_temp=85.0,
    ...     wet_bulb_temp=78.0,
    ...     circulation_flow=100000.0,
    ... )
    >>> print(f"Cycles: {result.cycles_of_concentration:.1f}")
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import hashlib
import logging
import math

from pydantic import BaseModel, Field

from greenlang.agents.process_heat.gl_017_condenser_optimization.config import (
    CoolingTowerConfig,
)
from greenlang.agents.process_heat.gl_017_condenser_optimization.schemas import (
    CoolingTowerResult,
    CoolingTowerInput,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - Cooling Tower Engineering Data
# =============================================================================

class CoolingTowerConstants:
    """Cooling tower engineering constants."""

    # Water properties
    WATER_DENSITY_LB_GAL = 8.34
    LATENT_HEAT_BTU_LB = 1000.0  # Approximate

    # Evaporation loss factor (% per degree F range)
    EVAPORATION_FACTOR = 0.0008  # 0.08% per degree F

    # Drift loss (% of circulation)
    TYPICAL_DRIFT_LOSS_PCT = 0.005  # Modern towers
    OLD_DRIFT_LOSS_PCT = 0.02  # Older towers without drift eliminators

    # Minimum cycles of concentration
    MIN_CYCLES = 1.5  # Below this, excessive water waste

    # Water chemistry scaling indices
    LANGELIER_SCALING = {
        (-2.0, -0.5): "severe_corrosion",
        (-0.5, 0.0): "mild_corrosion",
        (0.0, 0.5): "balanced",
        (0.5, 1.0): "mild_scaling",
        (1.0, 2.0): "severe_scaling",
    }

    # Typical water costs
    WATER_COST_PER_1000_GAL = 3.0
    SEWER_COST_PER_1000_GAL = 4.0
    CHEMICAL_COST_PER_1000_GAL = 0.5


@dataclass
class CoolingTowerReading:
    """Historical cooling tower reading."""
    timestamp: datetime
    hot_water_temp_f: float
    cold_water_temp_f: float
    wet_bulb_temp_f: float
    cycles: float
    blowdown_gpm: float


class CoolingTowerOptimizer:
    """
    Cooling tower optimization and analysis.

    This class optimizes cooling tower operation including:
    - Cycles of concentration management
    - Blowdown optimization
    - Water balance calculations
    - Chemistry compliance monitoring
    - Performance analysis

    Attributes:
        config: Cooling tower configuration

    Example:
        >>> config = CoolingTowerConfig()
        >>> optimizer = CoolingTowerOptimizer(config)
        >>> result = optimizer.analyze_cooling_tower(input_data)
    """

    def __init__(
        self,
        config: CoolingTowerConfig,
    ) -> None:
        """
        Initialize the cooling tower optimizer.

        Args:
            config: Cooling tower configuration
        """
        self.config = config
        self._history: List[CoolingTowerReading] = []
        self._calculation_count = 0

        logger.info(
            f"CoolingTowerOptimizer initialized: "
            f"design_range={config.design_range_f}F, "
            f"design_approach={config.design_approach_f}F"
        )

    def analyze_cooling_tower(
        self,
        hot_water_temp_f: float,
        cold_water_temp_f: float,
        wet_bulb_temp_f: float,
        circulation_flow_gpm: float,
        makeup_flow_gpm: Optional[float] = None,
        blowdown_flow_gpm: Optional[float] = None,
        makeup_conductivity_umhos: Optional[float] = None,
        tower_conductivity_umhos: Optional[float] = None,
        ph: Optional[float] = None,
        calcium_ppm: Optional[float] = None,
        silica_ppm: Optional[float] = None,
        chlorides_ppm: Optional[float] = None,
        dry_bulb_temp_f: Optional[float] = None,
        fans_operating: int = 1,
    ) -> CoolingTowerResult:
        """
        Analyze cooling tower performance.

        Calculates thermal efficiency, water balance, and chemistry
        compliance.

        Args:
            hot_water_temp_f: Hot water temperature (F)
            cold_water_temp_f: Cold water temperature (F)
            wet_bulb_temp_f: Wet bulb temperature (F)
            circulation_flow_gpm: Circulation flow (GPM)
            makeup_flow_gpm: Makeup water flow (GPM)
            blowdown_flow_gpm: Blowdown flow (GPM)
            makeup_conductivity_umhos: Makeup conductivity (umhos/cm)
            tower_conductivity_umhos: Tower conductivity (umhos/cm)
            ph: Tower water pH
            calcium_ppm: Calcium hardness (ppm as CaCO3)
            silica_ppm: Silica (ppm as SiO2)
            chlorides_ppm: Chlorides (ppm)
            dry_bulb_temp_f: Dry bulb temperature (F)
            fans_operating: Number of fans operating

        Returns:
            CoolingTowerResult with analysis
        """
        logger.debug(
            f"Analyzing cooling tower: range={hot_water_temp_f-cold_water_temp_f:.1f}F, "
            f"approach={cold_water_temp_f-wet_bulb_temp_f:.1f}F"
        )
        self._calculation_count += 1

        # Calculate thermal performance
        range_f = hot_water_temp_f - cold_water_temp_f
        approach_f = cold_water_temp_f - wet_bulb_temp_f

        # Calculate thermal efficiency
        thermal_efficiency = self._calculate_thermal_efficiency(
            range_f, approach_f, wet_bulb_temp_f, dry_bulb_temp_f
        )

        # Calculate L/G ratio
        l_g_ratio = self._calculate_lg_ratio(
            circulation_flow_gpm, range_f, wet_bulb_temp_f
        )

        # Calculate water balance
        evaporation_gpm = self._calculate_evaporation(
            circulation_flow_gpm, range_f
        )
        drift_gpm = self._calculate_drift(circulation_flow_gpm)

        # Calculate cycles of concentration
        cycles = self._calculate_cycles(
            makeup_flow_gpm, evaporation_gpm, drift_gpm,
            makeup_conductivity_umhos, tower_conductivity_umhos
        )

        # Calculate blowdown
        actual_blowdown = blowdown_flow_gpm or 0.0
        required_blowdown = self._calculate_required_blowdown(
            evaporation_gpm, drift_gpm, cycles
        )
        makeup_required = self._calculate_required_makeup(
            evaporation_gpm, required_blowdown, drift_gpm
        )

        # Check chemistry compliance
        chemistry_compliant, deviations = self._check_chemistry_compliance(
            ph, calcium_ppm, silica_ppm, chlorides_ppm,
            tower_conductivity_umhos
        )

        # Assess scaling/corrosion potential
        scaling_potential = self._assess_scaling_potential(
            ph, calcium_ppm, hot_water_temp_f
        )
        corrosion_potential = self._assess_corrosion_potential(
            ph, chlorides_ppm
        )

        # Calculate optimal cycles and blowdown
        optimal_cycles = self._calculate_optimal_cycles(
            makeup_conductivity_umhos, calcium_ppm, silica_ppm
        )
        optimal_blowdown = self._calculate_required_blowdown(
            evaporation_gpm, drift_gpm, optimal_cycles
        )

        # Calculate savings potential
        water_savings = self._calculate_water_savings(
            actual_blowdown, optimal_blowdown
        )
        chemical_savings = self._calculate_chemical_savings(
            cycles, optimal_cycles
        )

        # Record reading
        self._record_reading(
            hot_water_temp_f, cold_water_temp_f, wet_bulb_temp_f,
            cycles, actual_blowdown
        )

        result = CoolingTowerResult(
            thermal_efficiency_pct=round(thermal_efficiency, 1),
            approach_f=round(approach_f, 2),
            range_f=round(range_f, 2),
            liquid_to_gas_ratio=round(l_g_ratio, 3),
            cycles_of_concentration=round(cycles, 2),
            evaporation_rate_gpm=round(evaporation_gpm, 1),
            drift_loss_gpm=round(drift_gpm, 2),
            blowdown_rate_gpm=round(actual_blowdown, 1),
            makeup_required_gpm=round(makeup_required, 1),
            chemistry_compliant=chemistry_compliant,
            chemistry_deviations=deviations,
            scaling_potential=scaling_potential,
            corrosion_potential=corrosion_potential,
            optimal_cycles=round(optimal_cycles, 2),
            optimal_blowdown_gpm=round(optimal_blowdown, 1),
            water_savings_potential_gpm=round(water_savings, 1),
            chemical_cost_savings_pct=round(chemical_savings, 1),
        )

        logger.info(
            f"Cooling tower analysis complete: "
            f"efficiency={thermal_efficiency:.1f}%, cycles={cycles:.1f}"
        )

        return result

    def optimize_blowdown(
        self,
        current_cycles: float,
        evaporation_gpm: float,
        drift_gpm: float,
        makeup_conductivity_umhos: float,
        target_cycles: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Optimize blowdown rate.

        Calculates optimal blowdown to achieve target cycles while
        maintaining chemistry limits.

        Args:
            current_cycles: Current cycles of concentration
            evaporation_gpm: Evaporation rate (GPM)
            drift_gpm: Drift loss (GPM)
            makeup_conductivity_umhos: Makeup water conductivity
            target_cycles: Target cycles (optional, uses optimal)

        Returns:
            Dictionary with optimization results
        """
        if target_cycles is None:
            target_cycles = self._calculate_optimal_cycles(
                makeup_conductivity_umhos, None, None
            )

        # Current blowdown
        current_blowdown = self._calculate_required_blowdown(
            evaporation_gpm, drift_gpm, current_cycles
        )

        # Optimal blowdown
        optimal_blowdown = self._calculate_required_blowdown(
            evaporation_gpm, drift_gpm, target_cycles
        )

        # Calculate savings
        blowdown_reduction = current_blowdown - optimal_blowdown
        makeup_reduction = blowdown_reduction  # 1:1 relationship

        # Cost savings
        water_savings_day = (
            makeup_reduction * 60 * 24 / 1000 *
            CoolingTowerConstants.WATER_COST_PER_1000_GAL
        )
        sewer_savings_day = (
            blowdown_reduction * 60 * 24 / 1000 *
            CoolingTowerConstants.SEWER_COST_PER_1000_GAL
        )
        chemical_savings_day = (
            makeup_reduction * 60 * 24 / 1000 *
            CoolingTowerConstants.CHEMICAL_COST_PER_1000_GAL *
            0.5  # 50% chemical reduction
        )

        return {
            "current_cycles": current_cycles,
            "target_cycles": target_cycles,
            "current_blowdown_gpm": round(current_blowdown, 2),
            "optimal_blowdown_gpm": round(optimal_blowdown, 2),
            "blowdown_reduction_gpm": round(blowdown_reduction, 2),
            "makeup_reduction_gpm": round(makeup_reduction, 2),
            "daily_water_savings_usd": round(water_savings_day, 2),
            "daily_sewer_savings_usd": round(sewer_savings_day, 2),
            "daily_chemical_savings_usd": round(chemical_savings_day, 2),
            "total_daily_savings_usd": round(
                water_savings_day + sewer_savings_day + chemical_savings_day, 2
            ),
            "annual_savings_usd": round(
                (water_savings_day + sewer_savings_day + chemical_savings_day) * 365, 0
            ),
        }

    def _calculate_thermal_efficiency(
        self,
        range_f: float,
        approach_f: float,
        wet_bulb_f: float,
        dry_bulb_f: Optional[float] = None,
    ) -> float:
        """
        Calculate cooling tower thermal efficiency.

        Efficiency = Range / (Range + Approach) * 100
        Also known as cooling effectiveness.

        Args:
            range_f: Temperature range (F)
            approach_f: Approach temperature (F)
            wet_bulb_f: Wet bulb temperature (F)
            dry_bulb_f: Dry bulb temperature (F)

        Returns:
            Thermal efficiency (%)
        """
        if range_f + approach_f <= 0:
            return 0.0

        # Basic effectiveness
        effectiveness = range_f / (range_f + approach_f) * 100

        # Compare to design
        design_range = self.config.design_range_f
        design_approach = self.config.design_approach_f

        if design_range > 0:
            range_ratio = range_f / design_range
            approach_ratio = approach_f / design_approach if design_approach > 0 else 1.0

            # Adjust for operating conditions
            # Better approach at same wet bulb = better performance
            adjusted_efficiency = effectiveness * min(1.2, 1.0 / approach_ratio)
        else:
            adjusted_efficiency = effectiveness

        return min(150.0, max(0.0, adjusted_efficiency))

    def _calculate_lg_ratio(
        self,
        circulation_gpm: float,
        range_f: float,
        wet_bulb_f: float,
    ) -> float:
        """
        Calculate liquid-to-gas (L/G) ratio.

        L/G = (h2 - h1) / (t1 - t2)

        Where h is enthalpy of air and t is water temperature.
        Simplified calculation using typical values.

        Args:
            circulation_gpm: Circulation flow (GPM)
            range_f: Temperature range (F)
            wet_bulb_f: Wet bulb temperature (F)

        Returns:
            L/G ratio
        """
        # Typical L/G ratio for mechanical draft towers: 0.8 - 1.5
        # Higher range needs more air (lower L/G)

        # Simplified correlation
        # L/G increases with approach, decreases with range
        design_lg = 1.0  # Typical design point

        # Adjust for operating conditions
        design_range = self.config.design_range_f
        if design_range > 0:
            lg_ratio = design_lg * (design_range / range_f) ** 0.5
        else:
            lg_ratio = design_lg

        return max(0.5, min(2.0, lg_ratio))

    def _calculate_evaporation(
        self,
        circulation_gpm: float,
        range_f: float,
    ) -> float:
        """
        Calculate evaporation rate.

        Evaporation = Circulation * Range * Factor

        Factor is typically 0.08% per degree F range.

        Args:
            circulation_gpm: Circulation flow (GPM)
            range_f: Temperature range (F)

        Returns:
            Evaporation rate (GPM)
        """
        evaporation = (
            circulation_gpm *
            range_f *
            CoolingTowerConstants.EVAPORATION_FACTOR
        )
        return evaporation

    def _calculate_drift(
        self,
        circulation_gpm: float,
    ) -> float:
        """
        Calculate drift loss.

        Args:
            circulation_gpm: Circulation flow (GPM)

        Returns:
            Drift loss (GPM)
        """
        drift = (
            circulation_gpm *
            CoolingTowerConstants.TYPICAL_DRIFT_LOSS_PCT / 100
        )
        return drift

    def _calculate_cycles(
        self,
        makeup_gpm: Optional[float],
        evaporation_gpm: float,
        drift_gpm: float,
        makeup_conductivity: Optional[float],
        tower_conductivity: Optional[float],
    ) -> float:
        """
        Calculate cycles of concentration.

        Cycles = Tower conductivity / Makeup conductivity
        Or: Cycles = Makeup / Blowdown (at steady state)

        Args:
            makeup_gpm: Makeup flow (GPM)
            evaporation_gpm: Evaporation rate (GPM)
            drift_gpm: Drift loss (GPM)
            makeup_conductivity: Makeup conductivity (umhos/cm)
            tower_conductivity: Tower conductivity (umhos/cm)

        Returns:
            Cycles of concentration
        """
        # Prefer conductivity method if available
        if makeup_conductivity and tower_conductivity:
            if makeup_conductivity > 0:
                cycles = tower_conductivity / makeup_conductivity
                return max(1.0, min(20.0, cycles))

        # Fall back to mass balance method
        if makeup_gpm and evaporation_gpm > 0:
            blowdown_gpm = makeup_gpm - evaporation_gpm - drift_gpm
            if blowdown_gpm > 0:
                cycles = makeup_gpm / blowdown_gpm
                return max(1.0, min(20.0, cycles))

        # Default to target
        return self.config.target_cycles_concentration

    def _calculate_required_blowdown(
        self,
        evaporation_gpm: float,
        drift_gpm: float,
        cycles: float,
    ) -> float:
        """
        Calculate required blowdown for target cycles.

        Blowdown = Evaporation / (Cycles - 1) - Drift

        Args:
            evaporation_gpm: Evaporation rate (GPM)
            drift_gpm: Drift loss (GPM)
            cycles: Target cycles

        Returns:
            Required blowdown (GPM)
        """
        if cycles <= 1:
            cycles = 1.5  # Minimum practical cycles

        blowdown = evaporation_gpm / (cycles - 1) - drift_gpm
        return max(0.0, blowdown)

    def _calculate_required_makeup(
        self,
        evaporation_gpm: float,
        blowdown_gpm: float,
        drift_gpm: float,
    ) -> float:
        """
        Calculate required makeup water.

        Makeup = Evaporation + Blowdown + Drift

        Args:
            evaporation_gpm: Evaporation rate (GPM)
            blowdown_gpm: Blowdown rate (GPM)
            drift_gpm: Drift loss (GPM)

        Returns:
            Required makeup (GPM)
        """
        return evaporation_gpm + blowdown_gpm + drift_gpm

    def _check_chemistry_compliance(
        self,
        ph: Optional[float],
        calcium_ppm: Optional[float],
        silica_ppm: Optional[float],
        chlorides_ppm: Optional[float],
        conductivity: Optional[float],
    ) -> Tuple[bool, List[str]]:
        """
        Check chemistry compliance.

        Args:
            ph: pH value
            calcium_ppm: Calcium hardness (ppm)
            silica_ppm: Silica (ppm)
            chlorides_ppm: Chlorides (ppm)
            conductivity: Conductivity (umhos/cm)

        Returns:
            Tuple of (compliant, list of deviations)
        """
        deviations = []

        if ph is not None:
            if ph < 6.5 or ph > 9.5:
                deviations.append(f"pH {ph:.1f} outside range 6.5-9.5")

        if calcium_ppm is not None:
            if calcium_ppm > self.config.max_calcium_ppm:
                deviations.append(
                    f"Calcium {calcium_ppm:.0f} > limit {self.config.max_calcium_ppm}"
                )

        if silica_ppm is not None:
            if silica_ppm > self.config.max_silica_ppm:
                deviations.append(
                    f"Silica {silica_ppm:.0f} > limit {self.config.max_silica_ppm}"
                )

        if chlorides_ppm is not None:
            if chlorides_ppm > self.config.max_chlorides_ppm:
                deviations.append(
                    f"Chlorides {chlorides_ppm:.0f} > limit {self.config.max_chlorides_ppm}"
                )

        if conductivity is not None:
            if conductivity > self.config.max_conductivity_umhos:
                deviations.append(
                    f"Conductivity {conductivity:.0f} > limit {self.config.max_conductivity_umhos}"
                )

        return len(deviations) == 0, deviations

    def _assess_scaling_potential(
        self,
        ph: Optional[float],
        calcium_ppm: Optional[float],
        temperature_f: float,
    ) -> str:
        """
        Assess scaling potential.

        Uses simplified Langelier Saturation Index approach.

        Args:
            ph: pH value
            calcium_ppm: Calcium hardness (ppm)
            temperature_f: Water temperature (F)

        Returns:
            Scaling potential (low, moderate, high)
        """
        if ph is None or calcium_ppm is None:
            return "unknown"

        # Simplified LSI calculation
        # LSI = pH - pHs where pHs is saturation pH

        # Temperature correction
        temp_factor = 0.02 * (temperature_f - 77)

        # Calcium correction
        ca_factor = 0.4 if calcium_ppm > 400 else 0.2 if calcium_ppm > 200 else 0.0

        # pH effect
        if ph > 8.5:
            ph_factor = 0.3
        elif ph > 7.5:
            ph_factor = 0.1
        else:
            ph_factor = -0.1

        scaling_score = temp_factor + ca_factor + ph_factor

        if scaling_score > 0.5:
            return "high"
        elif scaling_score > 0.2:
            return "moderate"
        else:
            return "low"

    def _assess_corrosion_potential(
        self,
        ph: Optional[float],
        chlorides_ppm: Optional[float],
    ) -> str:
        """
        Assess corrosion potential.

        Args:
            ph: pH value
            chlorides_ppm: Chlorides (ppm)

        Returns:
            Corrosion potential (low, moderate, high)
        """
        if ph is None:
            return "unknown"

        corrosion_score = 0.0

        # Low pH increases corrosion
        if ph < 7.0:
            corrosion_score += 0.4
        elif ph < 7.5:
            corrosion_score += 0.2

        # Chlorides increase corrosion
        if chlorides_ppm is not None:
            if chlorides_ppm > 300:
                corrosion_score += 0.4
            elif chlorides_ppm > 150:
                corrosion_score += 0.2

        if corrosion_score > 0.5:
            return "high"
        elif corrosion_score > 0.2:
            return "moderate"
        else:
            return "low"

    def _calculate_optimal_cycles(
        self,
        makeup_conductivity: Optional[float],
        calcium_ppm: Optional[float],
        silica_ppm: Optional[float],
    ) -> float:
        """
        Calculate optimal cycles of concentration.

        Balances water savings against chemistry limits.

        Args:
            makeup_conductivity: Makeup conductivity (umhos/cm)
            calcium_ppm: Makeup calcium (ppm)
            silica_ppm: Makeup silica (ppm)

        Returns:
            Optimal cycles
        """
        max_cycles = self.config.max_cycles_concentration

        # Limit by conductivity
        if makeup_conductivity:
            max_cond = self.config.max_conductivity_umhos
            cycles_by_cond = max_cond / makeup_conductivity
            max_cycles = min(max_cycles, cycles_by_cond)

        # Limit by calcium
        if calcium_ppm:
            max_ca = self.config.max_calcium_ppm
            cycles_by_ca = max_ca / calcium_ppm
            max_cycles = min(max_cycles, cycles_by_ca)

        # Limit by silica
        if silica_ppm:
            max_si = self.config.max_silica_ppm
            cycles_by_si = max_si / silica_ppm
            max_cycles = min(max_cycles, cycles_by_si)

        # Apply safety margin
        optimal = max_cycles * 0.9

        # Ensure minimum
        return max(self.config.min_cycles_concentration, optimal)

    def _calculate_water_savings(
        self,
        actual_blowdown: float,
        optimal_blowdown: float,
    ) -> float:
        """
        Calculate potential water savings.

        Args:
            actual_blowdown: Actual blowdown (GPM)
            optimal_blowdown: Optimal blowdown (GPM)

        Returns:
            Water savings (GPM)
        """
        savings = actual_blowdown - optimal_blowdown
        return max(0.0, savings)

    def _calculate_chemical_savings(
        self,
        current_cycles: float,
        optimal_cycles: float,
    ) -> float:
        """
        Calculate potential chemical cost savings.

        Higher cycles = less makeup = less chemical treatment.

        Args:
            current_cycles: Current cycles
            optimal_cycles: Optimal cycles

        Returns:
            Chemical savings (%)
        """
        if current_cycles <= 0 or optimal_cycles <= 0:
            return 0.0

        if optimal_cycles > current_cycles:
            # Higher cycles = proportionally less chemical use
            savings_pct = (1 - current_cycles / optimal_cycles) * 100
            return max(0.0, savings_pct)

        return 0.0

    def _record_reading(
        self,
        hot_temp: float,
        cold_temp: float,
        wet_bulb: float,
        cycles: float,
        blowdown: float,
    ) -> None:
        """Record a cooling tower reading."""
        reading = CoolingTowerReading(
            timestamp=datetime.now(timezone.utc),
            hot_water_temp_f=hot_temp,
            cold_water_temp_f=cold_temp,
            wet_bulb_temp_f=wet_bulb,
            cycles=cycles,
            blowdown_gpm=blowdown,
        )
        self._history.append(reading)

        # Trim old history
        cutoff = datetime.now(timezone.utc).timestamp() - (7 * 24 * 3600)
        self._history = [
            r for r in self._history
            if r.timestamp.timestamp() > cutoff
        ]

    def get_performance_trend(
        self,
        hours: int = 24,
    ) -> Dict[str, List[Tuple[datetime, float]]]:
        """
        Get performance trend data.

        Args:
            hours: Hours of history

        Returns:
            Dictionary of performance trends
        """
        cutoff = datetime.now(timezone.utc).timestamp() - (hours * 3600)

        relevant = [
            r for r in self._history
            if r.timestamp.timestamp() > cutoff
        ]

        trends = {
            "range": [],
            "approach": [],
            "cycles": [],
            "blowdown": [],
        }

        for r in sorted(relevant, key=lambda x: x.timestamp):
            trends["range"].append(
                (r.timestamp, r.hot_water_temp_f - r.cold_water_temp_f)
            )
            trends["approach"].append(
                (r.timestamp, r.cold_water_temp_f - r.wet_bulb_temp_f)
            )
            trends["cycles"].append((r.timestamp, r.cycles))
            trends["blowdown"].append((r.timestamp, r.blowdown_gpm))

        return trends

    @property
    def calculation_count(self) -> int:
        """Get total calculation count."""
        return self._calculation_count
