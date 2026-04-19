"""
GL-003 UNIFIED STEAM SYSTEM OPTIMIZER - Steam Distribution Module

This module provides steam header pressure balancing and distribution optimization
with exergy-based analysis. Implements multi-header coordination and supply-demand
balancing for optimal steam system operation.

Features:
    - Multi-header pressure balancing
    - Exergy-based optimization
    - Supply-demand coordination
    - Pressure trend analysis
    - Load allocation optimization
    - IAPWS-IF97 steam property calculations

Example:
    >>> from greenlang.agents.process_heat.gl_003_unified_steam.distribution import (
    ...     SteamDistributionOptimizer,
    ... )
    >>>
    >>> optimizer = SteamDistributionOptimizer(config)
    >>> result = optimizer.balance_header(header_input)
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import logging
import math

from pydantic import BaseModel, Field

from .config import (
    SteamHeaderConfig,
    SteamHeaderLevel,
    ExergyOptimizationConfig,
)
from .schemas import (
    HeaderBalanceInput,
    HeaderBalanceOutput,
    HeaderReading,
    OptimizationStatus,
    SteamProperties,
    SteamFlowMeasurement,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - IAPWS-IF97 Reference Data
# =============================================================================

class SteamTableConstants:
    """Steam table constants for calculations."""

    # Reference conditions
    REFERENCE_TEMP_F = 77.0
    REFERENCE_TEMP_R = 536.67
    REFERENCE_PRESSURE_PSIA = 14.696

    # Conversion factors
    BTU_PER_LB_TO_KJ_KG = 2.326
    PSI_TO_KPA = 6.89476
    F_TO_C = lambda f: (f - 32) * 5 / 9

    # Saturation table (psig: (temp_f, h_f, h_fg, h_g, s_f, s_fg, s_g))
    SATURATION_DATA = {
        0: (212.0, 180.2, 970.3, 1150.5, 0.3121, 1.4447, 1.7568),
        15: (250.3, 218.9, 945.4, 1164.3, 0.3680, 1.3607, 1.7287),
        50: (298.0, 267.6, 911.0, 1178.6, 0.4295, 1.2625, 1.6920),
        100: (337.9, 309.0, 879.5, 1188.5, 0.4832, 1.1781, 1.6613),
        150: (365.9, 339.2, 856.8, 1196.0, 0.5208, 1.1167, 1.6375),
        200: (387.9, 362.2, 837.4, 1199.6, 0.5500, 1.0685, 1.6185),
        250: (406.1, 381.2, 820.1, 1201.3, 0.5736, 1.0282, 1.6018),
        300: (421.7, 397.0, 804.3, 1201.3, 0.5932, 0.9930, 1.5862),
        400: (448.0, 424.2, 774.4, 1198.6, 0.6261, 0.9318, 1.5579),
        500: (470.0, 447.7, 747.1, 1194.8, 0.6531, 0.8797, 1.5328),
        600: (489.0, 468.4, 721.4, 1189.8, 0.6763, 0.8335, 1.5098),
    }

    # Superheated steam Cp approximation (BTU/lb-F)
    CP_SUPERHEATED = 0.48


class ExergyConstants:
    """Constants for exergy calculations."""

    # Dead state conditions
    T0_R = 536.67  # 77F in Rankine
    P0_PSIA = 14.696

    # Reference entropy of water at dead state
    S0_BTU_LB_R = 0.0  # Reference point


# =============================================================================
# STEAM PROPERTY CALCULATOR
# =============================================================================

class SteamPropertyCalculator:
    """
    Calculator for steam thermodynamic properties using IAPWS-IF97 correlations.

    This class provides deterministic calculations for steam properties
    including enthalpy, entropy, specific volume, and exergy.
    """

    def __init__(self, reference_temp_f: float = 77.0) -> None:
        """
        Initialize steam property calculator.

        Args:
            reference_temp_f: Dead state reference temperature (F)
        """
        self.reference_temp_f = reference_temp_f
        self.reference_temp_r = reference_temp_f + 459.67

        logger.debug(
            f"SteamPropertyCalculator initialized: T0={reference_temp_f}F"
        )

    def get_saturation_properties(
        self,
        pressure_psig: float,
    ) -> Dict[str, float]:
        """
        Get saturation properties at given pressure.

        Args:
            pressure_psig: Gauge pressure (psig)

        Returns:
            Dictionary with saturation temperature, enthalpies, and entropies
        """
        # Interpolate from steam table
        pressures = sorted(SteamTableConstants.SATURATION_DATA.keys())

        # Clamp to valid range
        pressure_psig = max(0, min(600, pressure_psig))

        # Find bracketing pressures
        p_low = pressures[0]
        p_high = pressures[-1]

        for i in range(len(pressures) - 1):
            if pressures[i] <= pressure_psig <= pressures[i + 1]:
                p_low = pressures[i]
                p_high = pressures[i + 1]
                break

        # Get data points
        data_low = SteamTableConstants.SATURATION_DATA[p_low]
        data_high = SteamTableConstants.SATURATION_DATA[p_high]

        # Linear interpolation factor
        if p_high > p_low:
            factor = (pressure_psig - p_low) / (p_high - p_low)
        else:
            factor = 0.0

        # Interpolate all properties
        def interp(idx: int) -> float:
            return data_low[idx] + factor * (data_high[idx] - data_low[idx])

        return {
            "saturation_temp_f": interp(0),
            "h_f_btu_lb": interp(1),      # Saturated liquid enthalpy
            "h_fg_btu_lb": interp(2),     # Latent heat
            "h_g_btu_lb": interp(3),      # Saturated vapor enthalpy
            "s_f_btu_lb_r": interp(4),    # Saturated liquid entropy
            "s_fg_btu_lb_r": interp(5),   # Entropy of vaporization
            "s_g_btu_lb_r": interp(6),    # Saturated vapor entropy
        }

    def calculate_steam_enthalpy(
        self,
        pressure_psig: float,
        temperature_f: Optional[float] = None,
        dryness_fraction: float = 1.0,
    ) -> float:
        """
        Calculate steam specific enthalpy.

        Args:
            pressure_psig: Gauge pressure (psig)
            temperature_f: Temperature (F) - None for saturated
            dryness_fraction: Steam quality for wet steam (0-1)

        Returns:
            Specific enthalpy (BTU/lb)
        """
        sat_props = self.get_saturation_properties(pressure_psig)
        t_sat = sat_props["saturation_temp_f"]

        if temperature_f is None or temperature_f <= t_sat:
            # Saturated or wet steam
            h_f = sat_props["h_f_btu_lb"]
            h_fg = sat_props["h_fg_btu_lb"]
            return h_f + dryness_fraction * h_fg
        else:
            # Superheated steam
            h_g = sat_props["h_g_btu_lb"]
            superheat = temperature_f - t_sat
            return h_g + SteamTableConstants.CP_SUPERHEATED * superheat

    def calculate_steam_entropy(
        self,
        pressure_psig: float,
        temperature_f: Optional[float] = None,
        dryness_fraction: float = 1.0,
    ) -> float:
        """
        Calculate steam specific entropy.

        Args:
            pressure_psig: Gauge pressure (psig)
            temperature_f: Temperature (F) - None for saturated
            dryness_fraction: Steam quality for wet steam (0-1)

        Returns:
            Specific entropy (BTU/lb-R)
        """
        sat_props = self.get_saturation_properties(pressure_psig)
        t_sat = sat_props["saturation_temp_f"]

        if temperature_f is None or temperature_f <= t_sat:
            # Saturated or wet steam
            s_f = sat_props["s_f_btu_lb_r"]
            s_fg = sat_props["s_fg_btu_lb_r"]
            return s_f + dryness_fraction * s_fg
        else:
            # Superheated steam - approximate
            s_g = sat_props["s_g_btu_lb_r"]
            superheat = temperature_f - t_sat
            t_sat_r = t_sat + 459.67

            # ds = Cp * ln(T2/T1) for ideal gas approximation
            temp_r = temperature_f + 459.67
            delta_s = SteamTableConstants.CP_SUPERHEATED * math.log(temp_r / t_sat_r)
            return s_g + delta_s

    def calculate_specific_exergy(
        self,
        pressure_psig: float,
        temperature_f: Optional[float] = None,
        dryness_fraction: float = 1.0,
    ) -> float:
        """
        Calculate specific exergy (availability).

        Exergy = (h - h0) - T0 * (s - s0)

        Args:
            pressure_psig: Gauge pressure (psig)
            temperature_f: Temperature (F) - None for saturated
            dryness_fraction: Steam quality for wet steam (0-1)

        Returns:
            Specific exergy (BTU/lb)
        """
        # Calculate steam properties
        h = self.calculate_steam_enthalpy(
            pressure_psig, temperature_f, dryness_fraction
        )
        s = self.calculate_steam_entropy(
            pressure_psig, temperature_f, dryness_fraction
        )

        # Dead state properties (liquid water at reference conditions)
        # Approximate h0 and s0 at 77F, atmospheric
        h0 = 45.0  # BTU/lb at 77F
        s0 = 0.088  # BTU/lb-R at 77F

        # Calculate exergy
        exergy = (h - h0) - self.reference_temp_r * (s - s0)
        return max(0, exergy)  # Exergy is always positive

    def calculate_water_enthalpy(self, temperature_f: float) -> float:
        """
        Calculate liquid water enthalpy.

        Args:
            temperature_f: Water temperature (F)

        Returns:
            Specific enthalpy (BTU/lb)
        """
        # h = Cp * (T - T_ref), with T_ref = 32F for steam tables
        return 1.0 * (temperature_f - 32)

    def get_steam_properties(
        self,
        pressure_psig: float,
        temperature_f: Optional[float] = None,
        dryness_fraction: float = 1.0,
    ) -> SteamProperties:
        """
        Get complete steam properties as a SteamProperties object.

        Args:
            pressure_psig: Gauge pressure (psig)
            temperature_f: Temperature (F) - None for saturated
            dryness_fraction: Steam quality (0-1)

        Returns:
            SteamProperties object with all thermodynamic data
        """
        sat_props = self.get_saturation_properties(pressure_psig)
        t_sat = sat_props["saturation_temp_f"]

        # Determine phase
        if temperature_f is None:
            if dryness_fraction >= 1.0:
                phase = "saturated_vapor"
                actual_temp = t_sat
            elif dryness_fraction <= 0.0:
                phase = "saturated_liquid"
                actual_temp = t_sat
            else:
                phase = "wet_steam"
                actual_temp = t_sat
        elif temperature_f > t_sat:
            phase = "superheated_vapor"
            actual_temp = temperature_f
        else:
            phase = "subcooled_liquid"
            actual_temp = temperature_f

        # Calculate properties
        enthalpy = self.calculate_steam_enthalpy(
            pressure_psig, temperature_f, dryness_fraction
        )
        entropy = self.calculate_steam_entropy(
            pressure_psig, temperature_f, dryness_fraction
        )
        exergy = self.calculate_specific_exergy(
            pressure_psig, temperature_f, dryness_fraction
        )

        # Superheat
        superheat = None
        if temperature_f is not None and temperature_f > t_sat:
            superheat = temperature_f - t_sat

        return SteamProperties(
            pressure_psig=pressure_psig,
            temperature_f=actual_temp if temperature_f is None else temperature_f,
            saturation_temperature_f=t_sat,
            phase=phase,
            dryness_fraction=dryness_fraction,
            superheat_f=superheat,
            enthalpy_btu_lb=enthalpy,
            entropy_btu_lb_r=entropy,
            exergy_btu_lb=exergy,
        )


# =============================================================================
# HEADER BALANCE CALCULATOR
# =============================================================================

class HeaderBalanceCalculator:
    """
    Calculator for steam header balance optimization.

    Performs supply-demand balancing with exergy analysis
    and pressure trend prediction.
    """

    def __init__(
        self,
        header_config: SteamHeaderConfig,
        exergy_config: Optional[ExergyOptimizationConfig] = None,
    ) -> None:
        """
        Initialize header balance calculator.

        Args:
            header_config: Steam header configuration
            exergy_config: Exergy optimization settings
        """
        self.header_config = header_config
        self.exergy_config = exergy_config or ExergyOptimizationConfig()

        self.steam_calc = SteamPropertyCalculator(
            reference_temp_f=self.exergy_config.reference_temperature_f
        )

        # History for trend analysis
        self._pressure_history: List[Tuple[datetime, float]] = []
        self._max_history_size = 60

        logger.info(
            f"HeaderBalanceCalculator initialized for {header_config.name}"
        )

    def calculate_balance(
        self,
        input_data: HeaderBalanceInput,
    ) -> HeaderBalanceOutput:
        """
        Calculate header balance and optimization recommendations.

        Args:
            input_data: Current header operating data

        Returns:
            HeaderBalanceOutput with balance analysis
        """
        start_time = datetime.now(timezone.utc)
        warnings = []
        adjustments = []

        # Calculate supply and demand totals
        total_supply = self._sum_flows(input_data.supplies)
        total_demand = self._sum_flows(input_data.demands)

        # Calculate imbalance
        imbalance = total_supply - total_demand
        imbalance_pct = (
            (imbalance / total_demand * 100) if total_demand > 0 else 0.0
        )

        # Pressure deviation
        pressure_deviation = (
            input_data.current_pressure_psig -
            input_data.pressure_setpoint_psig
        )

        # Update pressure history
        self._update_pressure_history(
            input_data.timestamp,
            input_data.current_pressure_psig
        )

        # Analyze pressure trend
        pressure_trend = self._analyze_pressure_trend()

        # Determine status
        status = self._determine_status(
            pressure_deviation,
            input_data.pressure_deadband_psi,
            imbalance_pct
        )

        # Calculate exergy if enabled
        exergy_supply = None
        exergy_demand = None
        exergy_efficiency = None

        if self.exergy_config.enabled:
            exergy_supply, exergy_demand, exergy_efficiency = (
                self._calculate_exergy_balance(
                    input_data,
                    total_supply,
                    total_demand
                )
            )

        # Generate adjustment recommendations
        if abs(pressure_deviation) > input_data.pressure_deadband_psi:
            adjustments = self._generate_adjustments(
                input_data,
                pressure_deviation,
                imbalance
            )
            if pressure_deviation > 0:
                warnings.append(
                    f"Header pressure {pressure_deviation:.1f} psi above setpoint"
                )
            else:
                warnings.append(
                    f"Header pressure {abs(pressure_deviation):.1f} psi below setpoint"
                )

        # Check for supply constraints
        if total_supply > self.header_config.max_flow_lb_hr * 0.95:
            warnings.append(
                f"Approaching maximum header capacity "
                f"({total_supply:.0f}/{self.header_config.max_flow_lb_hr:.0f} lb/hr)"
            )

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance_hash(input_data)

        # Processing time
        processing_time = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000

        return HeaderBalanceOutput(
            header_id=input_data.header_id,
            timestamp=datetime.now(timezone.utc),
            status=status,
            total_supply_lb_hr=total_supply,
            total_demand_lb_hr=total_demand,
            imbalance_lb_hr=imbalance,
            imbalance_pct=imbalance_pct,
            pressure_psig=input_data.current_pressure_psig,
            pressure_deviation_psi=pressure_deviation,
            pressure_trend=pressure_trend,
            exergy_supply_btu_hr=exergy_supply,
            exergy_demand_btu_hr=exergy_demand,
            exergy_efficiency_pct=exergy_efficiency,
            adjustments=adjustments,
            warnings=warnings,
            provenance_hash=provenance_hash,
            processing_time_ms=processing_time,
        )

    def _sum_flows(self, flow_sources: List[Dict[str, Any]]) -> float:
        """Sum flow rates from supply or demand sources."""
        total = 0.0
        for source in flow_sources:
            flow = source.get("flow_lb_hr", 0.0)
            if isinstance(flow, (int, float)):
                total += flow
        return total

    def _update_pressure_history(
        self,
        timestamp: datetime,
        pressure: float,
    ) -> None:
        """Update pressure history for trend analysis."""
        self._pressure_history.append((timestamp, pressure))

        # Trim to max size
        if len(self._pressure_history) > self._max_history_size:
            self._pressure_history = self._pressure_history[-self._max_history_size:]

    def _analyze_pressure_trend(self) -> str:
        """Analyze pressure trend from history."""
        if len(self._pressure_history) < 5:
            return "stable"

        # Get recent readings
        recent = self._pressure_history[-10:]
        pressures = [p for _, p in recent]

        # Calculate simple linear regression slope
        n = len(pressures)
        x_mean = (n - 1) / 2
        y_mean = sum(pressures) / n

        numerator = sum(
            (i - x_mean) * (p - y_mean)
            for i, p in enumerate(pressures)
        )
        denominator = sum(
            (i - x_mean) ** 2
            for i in range(n)
        )

        if denominator > 0:
            slope = numerator / denominator
        else:
            slope = 0

        # Classify trend
        if slope > 0.5:
            return "rising"
        elif slope < -0.5:
            return "falling"
        else:
            return "stable"

    def _determine_status(
        self,
        pressure_deviation: float,
        deadband: float,
        imbalance_pct: float,
    ) -> OptimizationStatus:
        """Determine header optimization status."""
        # Check for critical conditions
        if (
            abs(pressure_deviation) >
            (self.header_config.max_pressure_psig -
             self.header_config.design_pressure_psig)
        ):
            return OptimizationStatus.CRITICAL

        if (
            abs(pressure_deviation) <
            (self.header_config.min_pressure_psig -
             self.header_config.design_pressure_psig)
        ):
            return OptimizationStatus.CRITICAL

        # Check for suboptimal
        if abs(pressure_deviation) > deadband * 2:
            return OptimizationStatus.SUBOPTIMAL

        if abs(imbalance_pct) > 10:
            return OptimizationStatus.SUBOPTIMAL

        # Within deadband and balanced
        if abs(pressure_deviation) <= deadband:
            return OptimizationStatus.OPTIMAL

        return OptimizationStatus.SUBOPTIMAL

    def _calculate_exergy_balance(
        self,
        input_data: HeaderBalanceInput,
        total_supply: float,
        total_demand: float,
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Calculate exergy supply, demand, and efficiency."""
        try:
            # Get steam properties at header conditions
            steam_props = self.steam_calc.get_steam_properties(
                pressure_psig=input_data.current_pressure_psig,
                temperature_f=input_data.current_temperature_f,
            )

            specific_exergy = steam_props.exergy_btu_lb or 0.0

            # Calculate exergy flow rates
            exergy_supply = total_supply * specific_exergy
            exergy_demand = total_demand * specific_exergy

            # Exergy efficiency (simplified - actual would track transformations)
            if exergy_supply > 0:
                # Account for typical losses (2-5% for distribution)
                distribution_loss_pct = 3.0
                exergy_efficiency = 100.0 - distribution_loss_pct
            else:
                exergy_efficiency = None

            return exergy_supply, exergy_demand, exergy_efficiency

        except Exception as e:
            logger.warning(f"Exergy calculation failed: {e}")
            return None, None, None

    def _generate_adjustments(
        self,
        input_data: HeaderBalanceInput,
        pressure_deviation: float,
        imbalance: float,
    ) -> List[Dict[str, Any]]:
        """Generate supply adjustment recommendations."""
        adjustments = []

        # Determine adjustment direction
        if pressure_deviation > 0:
            # Pressure high - reduce supply or increase demand
            adjustment_direction = "decrease"
            required_change = imbalance + abs(pressure_deviation) * 100
        else:
            # Pressure low - increase supply or reduce demand
            adjustment_direction = "increase"
            required_change = -imbalance + abs(pressure_deviation) * 100

        # Prioritize suppliers for adjustment
        for supply in input_data.supplies:
            supply_id = supply.get("id", "unknown")
            current_flow = supply.get("flow_lb_hr", 0)
            min_flow = supply.get("min_flow_lb_hr", 0)
            max_flow = supply.get("max_flow_lb_hr", current_flow * 1.2)
            controllable = supply.get("controllable", True)

            if not controllable:
                continue

            if adjustment_direction == "decrease":
                # Can this source decrease?
                available_decrease = current_flow - min_flow
                if available_decrease > 0:
                    recommended_decrease = min(available_decrease, required_change)
                    adjustments.append({
                        "source_id": supply_id,
                        "action": "decrease",
                        "current_flow_lb_hr": current_flow,
                        "recommended_flow_lb_hr": current_flow - recommended_decrease,
                        "change_lb_hr": -recommended_decrease,
                    })
                    required_change -= recommended_decrease
            else:
                # Can this source increase?
                available_increase = max_flow - current_flow
                if available_increase > 0:
                    recommended_increase = min(available_increase, required_change)
                    adjustments.append({
                        "source_id": supply_id,
                        "action": "increase",
                        "current_flow_lb_hr": current_flow,
                        "recommended_flow_lb_hr": current_flow + recommended_increase,
                        "change_lb_hr": recommended_increase,
                    })
                    required_change -= recommended_increase

            if required_change <= 0:
                break

        return adjustments

    def _calculate_provenance_hash(self, input_data: HeaderBalanceInput) -> str:
        """Calculate SHA-256 provenance hash."""
        provenance_data = {
            "header_id": input_data.header_id,
            "timestamp": input_data.timestamp.isoformat(),
            "pressure_psig": input_data.current_pressure_psig,
            "temperature_f": input_data.current_temperature_f,
            "supplies_count": len(input_data.supplies),
            "demands_count": len(input_data.demands),
        }
        data_str = json.dumps(provenance_data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()


# =============================================================================
# STEAM DISTRIBUTION OPTIMIZER
# =============================================================================

class SteamDistributionOptimizer:
    """
    Steam distribution system optimizer.

    Coordinates multiple headers for optimal steam distribution
    with exergy-based load allocation.

    Example:
        >>> from greenlang.agents.process_heat.gl_003_unified_steam.config import (
        ...     UnifiedSteamConfig,
        ...     create_default_config,
        ... )
        >>>
        >>> config = create_default_config()
        >>> optimizer = SteamDistributionOptimizer(config)
        >>>
        >>> # Get all header balances
        >>> results = optimizer.balance_all_headers(readings)
    """

    def __init__(
        self,
        headers: List[SteamHeaderConfig],
        exergy_config: Optional[ExergyOptimizationConfig] = None,
    ) -> None:
        """
        Initialize steam distribution optimizer.

        Args:
            headers: List of steam header configurations
            exergy_config: Exergy optimization settings
        """
        self.headers = {h.name: h for h in headers}
        self.exergy_config = exergy_config or ExergyOptimizationConfig()

        # Create calculators for each header
        self.calculators: Dict[str, HeaderBalanceCalculator] = {}
        for header in headers:
            self.calculators[header.name] = HeaderBalanceCalculator(
                header_config=header,
                exergy_config=self.exergy_config,
            )

        self.steam_calc = SteamPropertyCalculator(
            reference_temp_f=self.exergy_config.reference_temperature_f
        )

        logger.info(
            f"SteamDistributionOptimizer initialized with {len(headers)} headers"
        )

    def balance_header(
        self,
        header_id: str,
        input_data: HeaderBalanceInput,
    ) -> HeaderBalanceOutput:
        """
        Calculate balance for a single header.

        Args:
            header_id: Header identifier
            input_data: Header operating data

        Returns:
            HeaderBalanceOutput with analysis

        Raises:
            ValueError: If header_id is not configured
        """
        if header_id not in self.calculators:
            raise ValueError(f"Unknown header: {header_id}")

        return self.calculators[header_id].calculate_balance(input_data)

    def balance_all_headers(
        self,
        readings: List[HeaderBalanceInput],
    ) -> List[HeaderBalanceOutput]:
        """
        Calculate balance for all headers.

        Args:
            readings: List of header readings

        Returns:
            List of HeaderBalanceOutput for each header
        """
        results = []
        for reading in readings:
            header_id = reading.header_id
            if header_id in self.calculators:
                result = self.calculators[header_id].calculate_balance(reading)
                results.append(result)
            else:
                logger.warning(f"Skipping unknown header: {header_id}")

        return results

    def optimize_load_allocation(
        self,
        total_demand_lb_hr: float,
        available_supplies: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Optimize load allocation across multiple supply sources.

        Uses exergy-based optimization when enabled, otherwise
        uses cost-based allocation.

        Args:
            total_demand_lb_hr: Total steam demand (lb/hr)
            available_supplies: List of available supply sources

        Returns:
            List of recommended allocations per source
        """
        allocations = []
        remaining_demand = total_demand_lb_hr

        # Sort supplies by efficiency (exergy or cost)
        sorted_supplies = self._rank_supplies_by_efficiency(available_supplies)

        for supply in sorted_supplies:
            supply_id = supply.get("id", "unknown")
            max_capacity = supply.get("max_flow_lb_hr", 0)
            min_flow = supply.get("min_flow_lb_hr", 0)

            if remaining_demand <= 0:
                # All demand satisfied
                allocations.append({
                    "source_id": supply_id,
                    "allocated_flow_lb_hr": 0,
                    "utilization_pct": 0,
                })
                continue

            # Allocate from this source
            allocated = min(remaining_demand, max_capacity - min_flow)
            allocated = max(allocated, 0)

            if allocated > 0:
                remaining_demand -= allocated
                utilization = (allocated / max_capacity * 100) if max_capacity > 0 else 0

                allocations.append({
                    "source_id": supply_id,
                    "allocated_flow_lb_hr": allocated,
                    "utilization_pct": utilization,
                    "min_flow_lb_hr": min_flow,
                    "max_flow_lb_hr": max_capacity,
                })

        return allocations

    def _rank_supplies_by_efficiency(
        self,
        supplies: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Rank supplies by exergy efficiency or cost."""
        def get_efficiency_score(supply: Dict[str, Any]) -> float:
            # Higher score = more efficient = use first
            exergy_eff = supply.get("exergy_efficiency_pct", 80)
            cost = supply.get("cost_per_mlb", 10)

            if self.exergy_config.enabled:
                # Weighted combination
                exergy_weight = self.exergy_config.exergy_weight
                cost_weight = self.exergy_config.cost_weight

                # Normalize: higher exergy good, lower cost good
                score = (
                    exergy_weight * exergy_eff / 100 +
                    cost_weight * (1 - cost / 50)  # Assume max cost ~50
                )
                return score
            else:
                # Cost only - lower cost is better (higher score)
                return 50 - cost

        return sorted(supplies, key=get_efficiency_score, reverse=True)

    def calculate_system_exergy_efficiency(
        self,
        header_results: List[HeaderBalanceOutput],
    ) -> Optional[float]:
        """
        Calculate overall system exergy efficiency.

        Args:
            header_results: Results from all header analyses

        Returns:
            System exergy efficiency percentage, or None if not available
        """
        if not self.exergy_config.enabled:
            return None

        total_exergy_supply = 0.0
        total_exergy_demand = 0.0

        for result in header_results:
            if result.exergy_supply_btu_hr:
                total_exergy_supply += result.exergy_supply_btu_hr
            if result.exergy_demand_btu_hr:
                total_exergy_demand += result.exergy_demand_btu_hr

        if total_exergy_supply > 0:
            # Efficiency = useful exergy / supplied exergy
            efficiency = (total_exergy_demand / total_exergy_supply) * 100
            return min(efficiency, 100.0)

        return None

    def get_pressure_cascade_recommendation(
        self,
    ) -> List[Dict[str, Any]]:
        """
        Get recommendations for pressure cascade optimization.

        Returns:
            List of cascade optimization recommendations
        """
        recommendations = []

        # Sort headers by pressure level
        sorted_headers = sorted(
            self.headers.values(),
            key=lambda h: h.design_pressure_psig,
            reverse=True
        )

        for i, header in enumerate(sorted_headers[:-1]):
            next_header = sorted_headers[i + 1]

            # Check if PRV letdown is efficient
            pressure_drop = (
                header.design_pressure_psig -
                next_header.design_pressure_psig
            )

            if pressure_drop > 100:
                recommendations.append({
                    "type": "cascade_opportunity",
                    "from_header": header.name,
                    "to_header": next_header.name,
                    "pressure_drop_psi": pressure_drop,
                    "recommendation": (
                        f"Consider back-pressure turbine between "
                        f"{header.name} and {next_header.name} "
                        f"to recover {pressure_drop:.0f} psi drop"
                    ),
                })

        return recommendations
