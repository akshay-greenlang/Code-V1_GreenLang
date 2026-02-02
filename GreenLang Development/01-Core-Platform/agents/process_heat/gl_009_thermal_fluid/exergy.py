"""
GL-009 THERMALIQ Agent - Exergy (2nd Law) Analysis

This module provides exergy analysis calculations for thermal fluid systems,
enabling 2nd Law thermodynamic efficiency evaluation. Unlike 1st Law analysis
(energy balance), exergy analysis reveals the quality of energy and identifies
irreversibilities in the system.

Key concepts:
    - Exergy: Maximum useful work obtainable as system reaches equilibrium
    - Dead state: Reference environment (typically 77F, 14.696 psia)
    - Exergy destruction: Irreversibilities due to heat transfer, mixing, friction
    - 2nd Law efficiency: Ratio of exergy output to exergy input

All calculations are deterministic - ZERO HALLUCINATION guaranteed.

Reference:
    - Bejan, A. "Advanced Engineering Thermodynamics"
    - Kotas, T.J. "The Exergy Method of Thermal Plant Analysis"

Example:
    >>> from greenlang.agents.process_heat.gl_009_thermal_fluid.exergy import (
    ...     ExergyAnalyzer,
    ...     ExergyConfig,
    ... )
    >>> analyzer = ExergyAnalyzer()
    >>> result = analyzer.analyze_system(
    ...     hot_temp_f=600.0,
    ...     cold_temp_f=400.0,
    ...     heat_duty_btu_hr=5_000_000,
    ... )
    >>> print(f"Exergy efficiency: {result.exergy_efficiency_pct:.1f}%")
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import logging
import math

from pydantic import BaseModel, Field

from .schemas import ExergyAnalysis, ThermalFluidType
from .fluid_properties import ThermalFluidPropertyDatabase
from .config import ExergyConfig

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Reference dead state conditions
DEFAULT_REFERENCE_TEMP_F = 77.0  # 25C
DEFAULT_REFERENCE_TEMP_R = 536.67  # Rankine
DEFAULT_REFERENCE_PRESSURE_PSIA = 14.696

# Conversion factors
R_RANKINE_OFFSET = 459.67
BTU_PER_KWH = 3412.14


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class ExergyStream:
    """Represents an exergy stream in the system."""

    stream_id: str
    temperature_f: float
    mass_flow_lb_hr: float
    specific_heat_btu_lb_f: float
    exergy_rate_btu_hr: float
    entropy_rate_btu_hr_r: float


class ExergyDestructionBreakdown(BaseModel):
    """Breakdown of exergy destruction by source."""

    heater_destruction_btu_hr: float = Field(
        default=0.0,
        ge=0,
        description="Exergy destroyed in heater (BTU/hr)"
    )
    heater_destruction_pct: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Heater destruction as % of input"
    )
    heat_exchanger_destruction_btu_hr: float = Field(
        default=0.0,
        ge=0,
        description="Exergy destroyed in HX (BTU/hr)"
    )
    heat_exchanger_destruction_pct: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="HX destruction as % of input"
    )
    piping_destruction_btu_hr: float = Field(
        default=0.0,
        ge=0,
        description="Exergy destroyed in piping (BTU/hr)"
    )
    piping_destruction_pct: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Piping destruction as % of input"
    )
    mixing_destruction_btu_hr: float = Field(
        default=0.0,
        ge=0,
        description="Exergy destroyed by mixing (BTU/hr)"
    )
    mixing_destruction_pct: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Mixing destruction as % of input"
    )
    pump_destruction_btu_hr: float = Field(
        default=0.0,
        ge=0,
        description="Exergy destroyed in pump (BTU/hr)"
    )
    pump_destruction_pct: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Pump destruction as % of input"
    )
    total_destruction_btu_hr: float = Field(
        default=0.0,
        ge=0,
        description="Total exergy destruction (BTU/hr)"
    )


# =============================================================================
# EXERGY ANALYZER
# =============================================================================

class ExergyAnalyzer:
    """
    Exergy (2nd Law) analyzer for thermal fluid systems.

    This class calculates exergy efficiency and identifies sources of
    irreversibility in thermal fluid heating systems. Unlike 1st Law
    analysis (which only considers energy quantity), exergy analysis
    considers energy quality and reveals optimization opportunities.

    Key calculations:
        - Carnot efficiency (theoretical maximum)
        - Exergy input from fuel/heat source
        - Exergy transferred to process
        - Exergy destruction by component
        - 2nd Law efficiency

    Example:
        >>> analyzer = ExergyAnalyzer(reference_temp_f=77.0)
        >>> result = analyzer.analyze_heat_transfer(
        ...     hot_temp_f=600.0,
        ...     cold_temp_f=400.0,
        ...     heat_duty_btu_hr=5_000_000,
        ... )
    """

    def __init__(
        self,
        reference_temp_f: float = DEFAULT_REFERENCE_TEMP_F,
        reference_pressure_psia: float = DEFAULT_REFERENCE_PRESSURE_PSIA,
        fluid_type: Optional[ThermalFluidType] = None,
    ) -> None:
        """
        Initialize the exergy analyzer.

        Args:
            reference_temp_f: Dead state reference temperature (F)
            reference_pressure_psia: Dead state reference pressure (psia)
            fluid_type: Optional thermal fluid type for property lookups
        """
        self.reference_temp_f = reference_temp_f
        self.reference_temp_r = reference_temp_f + R_RANKINE_OFFSET
        self.reference_pressure_psia = reference_pressure_psia

        self.fluid_type = fluid_type
        self._property_db = ThermalFluidPropertyDatabase() if fluid_type else None

        self._calculation_count = 0

        logger.info(
            f"ExergyAnalyzer initialized with T0={reference_temp_f}F"
        )

    def analyze_system(
        self,
        hot_temp_f: float,
        cold_temp_f: float,
        heat_duty_btu_hr: float,
        heater_efficiency_pct: float = 85.0,
        piping_loss_pct: float = 2.0,
        process_temp_f: Optional[float] = None,
    ) -> ExergyAnalysis:
        """
        Perform complete exergy analysis of thermal fluid system.

        Args:
            hot_temp_f: Hot supply temperature (F)
            cold_temp_f: Cold return temperature (F)
            heat_duty_btu_hr: Process heat duty (BTU/hr)
            heater_efficiency_pct: Heater 1st law efficiency (%)
            piping_loss_pct: Piping heat loss (%)
            process_temp_f: Process temperature (F) - defaults to cold_temp

        Returns:
            ExergyAnalysis with complete 2nd law analysis
        """
        self._calculation_count += 1

        # Default process temperature to cold return temp
        if process_temp_f is None:
            process_temp_f = cold_temp_f

        # Validate temperatures
        if hot_temp_f <= self.reference_temp_f:
            raise ValueError(
                f"Hot temperature {hot_temp_f}F must exceed reference {self.reference_temp_f}F"
            )
        if cold_temp_f <= self.reference_temp_f:
            logger.warning(
                f"Cold temperature {cold_temp_f}F at or below reference {self.reference_temp_f}F"
            )

        # Convert to Rankine
        t_hot_r = hot_temp_f + R_RANKINE_OFFSET
        t_cold_r = cold_temp_f + R_RANKINE_OFFSET
        t_process_r = process_temp_f + R_RANKINE_OFFSET
        t0_r = self.reference_temp_r

        # Calculate Carnot efficiency for heat source
        # For heating, source is heater/flame, sink is environment
        # Carnot = 1 - T_cold / T_hot
        carnot_efficiency = 1.0 - t0_r / t_hot_r
        carnot_pct = carnot_efficiency * 100

        # Log mean temperature for exergy calculation
        delta_t = hot_temp_f - cold_temp_f
        if abs(delta_t) < 1.0:
            t_lm_r = t_hot_r
        else:
            t_lm_r = (t_hot_r - t_cold_r) / math.log(t_hot_r / t_cold_r)

        # 1st Law (energy) efficiency - heater efficiency
        first_law_efficiency = heater_efficiency_pct / 100.0

        # Fuel/heat input required
        fuel_input_btu_hr = heat_duty_btu_hr / first_law_efficiency

        # Exergy input from fuel (chemical exergy approximation)
        # For combustion, exergy/energy ratio is approximately 1.04-1.06
        exergy_fuel_ratio = 1.05
        exergy_input = fuel_input_btu_hr * exergy_fuel_ratio * carnot_efficiency

        # Exergy transferred to thermal fluid
        # Exergy = Q * (1 - T0/T_lm)
        exergy_factor_fluid = 1.0 - t0_r / t_lm_r
        exergy_to_fluid = heat_duty_btu_hr * exergy_factor_fluid

        # Exergy delivered to process
        # Consider piping losses
        net_duty = heat_duty_btu_hr * (1.0 - piping_loss_pct / 100.0)
        exergy_factor_process = 1.0 - t0_r / t_process_r
        exergy_to_process = net_duty * exergy_factor_process

        # Exergy destruction in heater
        # Due to combustion irreversibility and finite temperature difference
        heater_destruction = exergy_input - exergy_to_fluid

        # Exergy destruction in piping (heat loss to environment)
        piping_loss_btu_hr = heat_duty_btu_hr * piping_loss_pct / 100.0
        # Exergy lost = Q_loss * (1 - T0/T_avg)
        t_avg_r = (t_hot_r + t_cold_r) / 2
        piping_destruction = piping_loss_btu_hr * (1.0 - t0_r / t_avg_r)

        # Exergy destruction in heat exchanger (finite delta-T)
        hx_destruction = exergy_to_fluid - exergy_to_process - piping_destruction

        # Total exergy destruction
        total_destruction = heater_destruction + piping_destruction + hx_destruction
        total_destruction = max(0, total_destruction)

        # 2nd Law (exergy) efficiency
        if exergy_input > 0:
            exergy_efficiency = exergy_to_process / exergy_input
        else:
            exergy_efficiency = 0.0

        exergy_efficiency_pct = exergy_efficiency * 100.0

        # Calculate destruction percentages
        heater_dest_pct = heater_destruction / exergy_input * 100 if exergy_input > 0 else 0
        piping_dest_pct = piping_destruction / exergy_input * 100 if exergy_input > 0 else 0
        hx_dest_pct = hx_destruction / exergy_input * 100 if exergy_input > 0 else 0

        # Log mean temperature ratio
        log_mean_temp_ratio = t_lm_r / t0_r

        return ExergyAnalysis(
            exergy_efficiency_pct=round(exergy_efficiency_pct, 2),
            first_law_efficiency_pct=round(heater_efficiency_pct, 2),
            exergy_input_btu_hr=round(exergy_input, 0),
            exergy_output_btu_hr=round(exergy_to_process, 0),
            exergy_destruction_btu_hr=round(total_destruction, 0),
            carnot_efficiency_pct=round(carnot_pct, 2),
            log_mean_temp_ratio=round(log_mean_temp_ratio, 4),
            heater_destruction_pct=round(heater_dest_pct, 2),
            piping_destruction_pct=round(piping_dest_pct, 2),
            mixing_destruction_pct=round(hx_dest_pct, 2),
            reference_temperature_f=self.reference_temp_f,
            calculation_method="SECOND_LAW_AVAILABILITY",
        )

    def analyze_heat_transfer(
        self,
        hot_inlet_temp_f: float,
        hot_outlet_temp_f: float,
        cold_inlet_temp_f: float,
        cold_outlet_temp_f: float,
        heat_duty_btu_hr: float,
    ) -> Dict[str, Any]:
        """
        Analyze exergy destruction in a heat exchanger.

        Args:
            hot_inlet_temp_f: Hot side inlet temperature (F)
            hot_outlet_temp_f: Hot side outlet temperature (F)
            cold_inlet_temp_f: Cold side inlet temperature (F)
            cold_outlet_temp_f: Cold side outlet temperature (F)
            heat_duty_btu_hr: Heat duty (BTU/hr)

        Returns:
            Dictionary with exergy analysis results
        """
        self._calculation_count += 1

        # Convert temperatures to Rankine
        t_hi_r = hot_inlet_temp_f + R_RANKINE_OFFSET
        t_ho_r = hot_outlet_temp_f + R_RANKINE_OFFSET
        t_ci_r = cold_inlet_temp_f + R_RANKINE_OFFSET
        t_co_r = cold_outlet_temp_f + R_RANKINE_OFFSET
        t0_r = self.reference_temp_r

        # Exergy change on hot side (decrease)
        # Delta_Ex_hot = Q * (1 - T0 / T_lm_hot)
        if abs(t_hi_r - t_ho_r) > 0.1:
            t_lm_hot_r = (t_hi_r - t_ho_r) / math.log(t_hi_r / t_ho_r)
        else:
            t_lm_hot_r = t_hi_r

        exergy_factor_hot = 1.0 - t0_r / t_lm_hot_r
        exergy_decrease_hot = heat_duty_btu_hr * exergy_factor_hot

        # Exergy change on cold side (increase)
        if abs(t_co_r - t_ci_r) > 0.1:
            t_lm_cold_r = (t_co_r - t_ci_r) / math.log(t_co_r / t_ci_r)
        else:
            t_lm_cold_r = t_co_r

        exergy_factor_cold = 1.0 - t0_r / t_lm_cold_r
        exergy_increase_cold = heat_duty_btu_hr * exergy_factor_cold

        # Exergy destruction (irreversibility)
        exergy_destruction = exergy_decrease_hot - exergy_increase_cold
        exergy_destruction = max(0, exergy_destruction)

        # Heat exchanger exergy efficiency
        if exergy_decrease_hot > 0:
            hx_exergy_efficiency = exergy_increase_cold / exergy_decrease_hot
        else:
            hx_exergy_efficiency = 0.0

        # Number of transfer units based on exergy
        # NTU_ex = (Ex_out - Ex_in) / (Ex_max - Ex_in)
        exergy_max = heat_duty_btu_hr * (1.0 - t0_r / t_hi_r)
        if exergy_max > 0:
            ntu_exergy = exergy_increase_cold / exergy_max
        else:
            ntu_exergy = 0.0

        return {
            "exergy_decrease_hot_btu_hr": round(exergy_decrease_hot, 0),
            "exergy_increase_cold_btu_hr": round(exergy_increase_cold, 0),
            "exergy_destruction_btu_hr": round(exergy_destruction, 0),
            "hx_exergy_efficiency_pct": round(hx_exergy_efficiency * 100, 2),
            "exergy_ntu": round(ntu_exergy, 4),
            "t_lm_hot_r": round(t_lm_hot_r, 2),
            "t_lm_cold_r": round(t_lm_cold_r, 2),
            "exergy_factor_hot": round(exergy_factor_hot, 4),
            "exergy_factor_cold": round(exergy_factor_cold, 4),
        }

    def calculate_stream_exergy(
        self,
        temperature_f: float,
        mass_flow_lb_hr: float,
        specific_heat_btu_lb_f: float,
    ) -> float:
        """
        Calculate exergy rate of a thermal fluid stream.

        Exergy rate = m_dot * Cp * [(T - T0) - T0 * ln(T/T0)]

        Args:
            temperature_f: Stream temperature (F)
            mass_flow_lb_hr: Mass flow rate (lb/hr)
            specific_heat_btu_lb_f: Specific heat (BTU/lb-F)

        Returns:
            Exergy rate (BTU/hr)
        """
        t_r = temperature_f + R_RANKINE_OFFSET
        t0_r = self.reference_temp_r

        # Specific exergy (BTU/lb)
        # ex = Cp * [(T - T0) - T0 * ln(T/T0)]
        if t_r > 0 and t0_r > 0:
            specific_exergy = specific_heat_btu_lb_f * (
                (t_r - t0_r) - t0_r * math.log(t_r / t0_r)
            )
        else:
            specific_exergy = 0.0

        # Exergy rate
        exergy_rate = mass_flow_lb_hr * specific_exergy

        return max(0.0, exergy_rate)

    def calculate_heat_exergy(
        self,
        heat_rate_btu_hr: float,
        temperature_f: float,
    ) -> float:
        """
        Calculate exergy content of heat at given temperature.

        Exergy = Q * (1 - T0/T)

        Args:
            heat_rate_btu_hr: Heat transfer rate (BTU/hr)
            temperature_f: Temperature of heat transfer (F)

        Returns:
            Exergy rate (BTU/hr)
        """
        t_r = temperature_f + R_RANKINE_OFFSET
        t0_r = self.reference_temp_r

        if t_r > t0_r:
            exergy_factor = 1.0 - t0_r / t_r
            exergy_rate = heat_rate_btu_hr * exergy_factor
        else:
            # Below dead state - different treatment
            exergy_rate = 0.0

        return exergy_rate

    def calculate_carnot_efficiency(
        self,
        hot_temp_f: float,
        cold_temp_f: Optional[float] = None,
    ) -> float:
        """
        Calculate Carnot efficiency between two temperatures.

        eta_carnot = 1 - T_cold / T_hot

        Args:
            hot_temp_f: Hot reservoir temperature (F)
            cold_temp_f: Cold reservoir temperature (F) - defaults to reference

        Returns:
            Carnot efficiency (0-1)
        """
        if cold_temp_f is None:
            cold_temp_f = self.reference_temp_f

        t_hot_r = hot_temp_f + R_RANKINE_OFFSET
        t_cold_r = cold_temp_f + R_RANKINE_OFFSET

        if t_hot_r <= 0:
            return 0.0

        carnot = 1.0 - t_cold_r / t_hot_r
        return max(0.0, min(1.0, carnot))

    def calculate_rational_efficiency(
        self,
        actual_work_btu_hr: float,
        exergy_input_btu_hr: float,
    ) -> float:
        """
        Calculate rational (2nd Law) efficiency.

        eta_II = W_actual / Ex_input

        Args:
            actual_work_btu_hr: Actual work/useful output (BTU/hr)
            exergy_input_btu_hr: Exergy input (BTU/hr)

        Returns:
            Rational efficiency (0-1)
        """
        if exergy_input_btu_hr <= 0:
            return 0.0

        efficiency = actual_work_btu_hr / exergy_input_btu_hr
        return max(0.0, min(1.0, efficiency))

    def analyze_improvement_potential(
        self,
        current_exergy_efficiency_pct: float,
        destruction_breakdown: ExergyDestructionBreakdown,
        fuel_cost_usd_mmbtu: float = 8.0,
        operating_hours_per_year: int = 8000,
    ) -> List[Dict[str, Any]]:
        """
        Analyze improvement potential based on exergy destruction.

        Args:
            current_exergy_efficiency_pct: Current 2nd law efficiency (%)
            destruction_breakdown: Breakdown of exergy destruction
            fuel_cost_usd_mmbtu: Fuel cost ($/MMBTU)
            operating_hours_per_year: Annual operating hours

        Returns:
            List of improvement recommendations with savings estimates
        """
        recommendations = []

        total_destruction = destruction_breakdown.total_destruction_btu_hr

        # Heater improvement potential
        if destruction_breakdown.heater_destruction_pct > 15:
            # Potential for better burner, air preheating, etc.
            potential_reduction = destruction_breakdown.heater_destruction_btu_hr * 0.20
            annual_savings_btu = potential_reduction * operating_hours_per_year
            annual_savings_usd = annual_savings_btu / 1_000_000 * fuel_cost_usd_mmbtu

            recommendations.append({
                "category": "heater",
                "title": "Improve Heater Exergy Efficiency",
                "description": (
                    "Consider air preheating or regenerative burners to reduce "
                    "combustion irreversibility"
                ),
                "current_destruction_pct": destruction_breakdown.heater_destruction_pct,
                "potential_reduction_btu_hr": round(potential_reduction, 0),
                "estimated_annual_savings_usd": round(annual_savings_usd, 0),
                "implementation_difficulty": "medium",
            })

        # Heat exchanger improvement potential
        if destruction_breakdown.heat_exchanger_destruction_pct > 10:
            potential_reduction = (
                destruction_breakdown.heat_exchanger_destruction_btu_hr * 0.30
            )
            annual_savings_btu = potential_reduction * operating_hours_per_year
            annual_savings_usd = annual_savings_btu / 1_000_000 * fuel_cost_usd_mmbtu

            recommendations.append({
                "category": "heat_exchanger",
                "title": "Reduce Heat Exchanger Irreversibility",
                "description": (
                    "Optimize temperature approach, consider additional "
                    "heat exchange area or heat integration"
                ),
                "current_destruction_pct": destruction_breakdown.heat_exchanger_destruction_pct,
                "potential_reduction_btu_hr": round(potential_reduction, 0),
                "estimated_annual_savings_usd": round(annual_savings_usd, 0),
                "implementation_difficulty": "medium",
            })

        # Piping improvement potential
        if destruction_breakdown.piping_destruction_pct > 3:
            potential_reduction = destruction_breakdown.piping_destruction_btu_hr * 0.50
            annual_savings_btu = potential_reduction * operating_hours_per_year
            annual_savings_usd = annual_savings_btu / 1_000_000 * fuel_cost_usd_mmbtu

            recommendations.append({
                "category": "piping",
                "title": "Reduce Piping Heat Losses",
                "description": (
                    "Improve pipe insulation, repair damaged sections, "
                    "consider trace heating optimization"
                ),
                "current_destruction_pct": destruction_breakdown.piping_destruction_pct,
                "potential_reduction_btu_hr": round(potential_reduction, 0),
                "estimated_annual_savings_usd": round(annual_savings_usd, 0),
                "implementation_difficulty": "low",
            })

        # Sort by savings potential
        recommendations.sort(
            key=lambda x: x.get("estimated_annual_savings_usd", 0),
            reverse=True
        )

        return recommendations

    @property
    def calculation_count(self) -> int:
        """Get total calculation count."""
        return self._calculation_count


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def calculate_exergy_efficiency(
    hot_temp_f: float,
    cold_temp_f: float,
    heat_duty_btu_hr: float,
    heater_efficiency_pct: float = 85.0,
    reference_temp_f: float = 77.0,
) -> float:
    """
    Quick calculation of exergy efficiency.

    Args:
        hot_temp_f: Hot supply temperature (F)
        cold_temp_f: Cold return temperature (F)
        heat_duty_btu_hr: Heat duty (BTU/hr)
        heater_efficiency_pct: 1st law heater efficiency (%)
        reference_temp_f: Dead state reference temperature (F)

    Returns:
        Exergy efficiency (%)
    """
    analyzer = ExergyAnalyzer(reference_temp_f=reference_temp_f)
    result = analyzer.analyze_system(
        hot_temp_f=hot_temp_f,
        cold_temp_f=cold_temp_f,
        heat_duty_btu_hr=heat_duty_btu_hr,
        heater_efficiency_pct=heater_efficiency_pct,
    )
    return result.exergy_efficiency_pct


def calculate_carnot_limit(
    hot_temp_f: float,
    cold_temp_f: float = 77.0,
) -> float:
    """
    Calculate Carnot efficiency limit.

    Args:
        hot_temp_f: Hot reservoir temperature (F)
        cold_temp_f: Cold reservoir temperature (F)

    Returns:
        Carnot efficiency (%)
    """
    t_hot_r = hot_temp_f + R_RANKINE_OFFSET
    t_cold_r = cold_temp_f + R_RANKINE_OFFSET

    carnot = (1.0 - t_cold_r / t_hot_r) * 100
    return max(0.0, carnot)
