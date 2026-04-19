"""
GL-017 CONDENSYNC - Efficiency Calculator

Zero-hallucination, deterministic calculations for condenser efficiency
analysis and optimization following HEI Standards and ASME PTC 12.2.

This module provides:
- Condenser thermal efficiency
- Heat recovery efficiency
- Cooling water temperature rise optimization
- Approach temperature calculation
- Condenser Performance Index (CPI)
- Energy savings from optimization
- Cost-benefit analysis calculations

Standards Reference:
- HEI Standards for Steam Surface Condensers (11th Edition)
- ASME PTC 12.2 - Steam Surface Condensers Performance Test Code
- EPRI Heat Rate Improvement Guidelines

Author: GL-CalculatorEngineer
Version: 1.0.0
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import math

from .provenance import ProvenanceTracker, ProvenanceRecord


# =============================================================================
# CONSTANTS AND REFERENCE DATA
# =============================================================================

# Typical heat rate impact factors (kJ/kWh per mmHg backpressure)
HEAT_RATE_IMPACT_KJ_KWH_PER_MMHG = 25.0

# Condenser Performance Index reference values
CPI_REFERENCE_VALUES = {
    "excellent": 1.00,
    "good": 0.95,
    "average": 0.85,
    "poor": 0.75,
    "critical": 0.65,
}

# Typical electricity costs (USD/MWh) for reference
DEFAULT_ELECTRICITY_COST_USD_MWH = 50.0

# Typical cooling water pump power (kW per 1000 m3/hr)
CW_PUMP_POWER_KW_PER_1000M3HR = 150.0

# Carbon emission factor for electricity (kg CO2/MWh)
# Based on average grid mix
CARBON_EMISSION_FACTOR_KG_CO2_MWH = 400.0


class PerformanceRating(Enum):
    """Condenser performance rating categories."""
    EXCELLENT = "Excellent"
    GOOD = "Good"
    AVERAGE = "Average"
    POOR = "Poor"
    CRITICAL = "Critical"


# =============================================================================
# INPUT/OUTPUT DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class EfficiencyInput:
    """
    Input parameters for efficiency calculations.

    Attributes:
        steam_temp_c: Steam saturation temperature (C)
        cw_inlet_temp_c: Cooling water inlet temperature (C)
        cw_outlet_temp_c: Cooling water outlet temperature (C)
        cw_flow_rate_m3_hr: Cooling water flow rate (m3/hr)
        heat_duty_mw: Condenser heat duty (MW)
        turbine_output_mw: Turbine electrical output (MW)
        design_backpressure_mmhg: Design backpressure (mmHg abs)
        actual_backpressure_mmhg: Actual backpressure (mmHg abs)
        design_u_value_w_m2k: Design U-value (W/m2-K)
        actual_u_value_w_m2k: Actual U-value (W/m2-K)
        heat_transfer_area_m2: Heat transfer area (m2)
        electricity_cost_usd_mwh: Electricity cost (USD/MWh)
        operating_hours_per_year: Annual operating hours
    """
    steam_temp_c: float
    cw_inlet_temp_c: float
    cw_outlet_temp_c: float
    cw_flow_rate_m3_hr: float
    heat_duty_mw: float
    turbine_output_mw: float
    design_backpressure_mmhg: float
    actual_backpressure_mmhg: float
    design_u_value_w_m2k: float
    actual_u_value_w_m2k: float
    heat_transfer_area_m2: float
    electricity_cost_usd_mwh: float = DEFAULT_ELECTRICITY_COST_USD_MWH
    operating_hours_per_year: int = 8000


@dataclass(frozen=True)
class EfficiencyOutput:
    """
    Output results from efficiency calculations.

    Attributes:
        thermal_efficiency_pct: Condenser thermal efficiency (%)
        heat_recovery_efficiency_pct: Heat recovery efficiency (%)
        cw_temp_rise_c: Cooling water temperature rise (C)
        approach_temp_c: Approach temperature (C)
        ttd_c: Terminal Temperature Difference (C)
        itd_c: Initial Temperature Difference (C)
        cpi: Condenser Performance Index (0-1)
        performance_rating: Performance category
        cleanliness_factor: Cleanliness factor (0-1)
        heat_rate_deviation_kj_kwh: Heat rate deviation from design (kJ/kWh)
        efficiency_loss_mw: Power loss due to inefficiency (MW)
        annual_energy_loss_mwh: Annual energy loss (MWh)
        annual_cost_loss_usd: Annual cost of inefficiency (USD)
        annual_carbon_penalty_tonnes: Additional CO2 emissions (tonnes/year)
        potential_savings_mw: Potential power recovery (MW)
        potential_annual_savings_usd: Potential annual savings (USD)
    """
    thermal_efficiency_pct: float
    heat_recovery_efficiency_pct: float
    cw_temp_rise_c: float
    approach_temp_c: float
    ttd_c: float
    itd_c: float
    cpi: float
    performance_rating: str
    cleanliness_factor: float
    heat_rate_deviation_kj_kwh: float
    efficiency_loss_mw: float
    annual_energy_loss_mwh: float
    annual_cost_loss_usd: float
    annual_carbon_penalty_tonnes: float
    potential_savings_mw: float
    potential_annual_savings_usd: float


# =============================================================================
# EFFICIENCY CALCULATOR CLASS
# =============================================================================

class EfficiencyCalculator:
    """
    Zero-hallucination efficiency calculator for steam condensers.

    Implements deterministic calculations following HEI Standards and
    EPRI guidelines for condenser performance analysis. All calculations
    produce bit-perfect reproducible results with complete provenance.

    Guarantees:
    - DETERMINISTIC: Same input always produces same output
    - REPRODUCIBLE: SHA-256 verified calculation chain
    - AUDITABLE: Complete step-by-step provenance trail
    - ZERO HALLUCINATION: No LLM in calculation path

    Example:
        >>> calculator = EfficiencyCalculator()
        >>> inputs = EfficiencyInput(
        ...     steam_temp_c=40.0,
        ...     cw_inlet_temp_c=25.0,
        ...     cw_outlet_temp_c=35.0,
        ...     cw_flow_rate_m3_hr=50000.0,
        ...     heat_duty_mw=200.0,
        ...     turbine_output_mw=300.0,
        ...     design_backpressure_mmhg=50.8,
        ...     actual_backpressure_mmhg=55.0,
        ...     design_u_value_w_m2k=3500.0,
        ...     actual_u_value_w_m2k=3000.0,
        ...     heat_transfer_area_m2=10000.0
        ... )
        >>> result, provenance = calculator.calculate(inputs)
        >>> print(f"CPI: {result.cpi:.3f}")
    """

    VERSION = "1.0.0"
    NAME = "EfficiencyCalculator"

    def __init__(self):
        """Initialize the efficiency calculator."""
        self._tracker: Optional[ProvenanceTracker] = None

    def calculate(
        self,
        inputs: EfficiencyInput
    ) -> Tuple[EfficiencyOutput, ProvenanceRecord]:
        """
        Perform complete efficiency analysis.

        Args:
            inputs: EfficiencyInput with all required parameters

        Returns:
            Tuple of (EfficiencyOutput, ProvenanceRecord)

        Raises:
            ValueError: If inputs are invalid or out of range
        """
        # Initialize provenance tracking
        self._tracker = ProvenanceTracker(
            calculator_name=self.NAME,
            calculator_version=self.VERSION,
            metadata={
                "standards": ["HEI Standards", "ASME PTC 12.2", "EPRI Guidelines"],
                "domain": "Condenser Efficiency Analysis"
            }
        )

        # Set inputs for provenance
        input_dict = {
            "steam_temp_c": inputs.steam_temp_c,
            "cw_inlet_temp_c": inputs.cw_inlet_temp_c,
            "cw_outlet_temp_c": inputs.cw_outlet_temp_c,
            "cw_flow_rate_m3_hr": inputs.cw_flow_rate_m3_hr,
            "heat_duty_mw": inputs.heat_duty_mw,
            "turbine_output_mw": inputs.turbine_output_mw,
            "design_backpressure_mmhg": inputs.design_backpressure_mmhg,
            "actual_backpressure_mmhg": inputs.actual_backpressure_mmhg,
            "design_u_value_w_m2k": inputs.design_u_value_w_m2k,
            "actual_u_value_w_m2k": inputs.actual_u_value_w_m2k,
            "heat_transfer_area_m2": inputs.heat_transfer_area_m2,
            "electricity_cost_usd_mwh": inputs.electricity_cost_usd_mwh,
            "operating_hours_per_year": inputs.operating_hours_per_year
        }
        self._tracker.set_inputs(input_dict)

        # Validate inputs
        self._validate_inputs(inputs)

        # Step 1: Calculate cooling water temperature rise
        cw_temp_rise = self._calculate_cw_temp_rise(
            inputs.cw_inlet_temp_c,
            inputs.cw_outlet_temp_c
        )

        # Step 2: Calculate approach temperature
        approach_temp = self._calculate_approach_temp(
            inputs.steam_temp_c,
            inputs.cw_inlet_temp_c
        )

        # Step 3: Calculate TTD and ITD
        ttd = self._calculate_ttd(
            inputs.steam_temp_c,
            inputs.cw_outlet_temp_c
        )
        itd = self._calculate_itd(
            inputs.steam_temp_c,
            inputs.cw_inlet_temp_c
        )

        # Step 4: Calculate thermal efficiency
        thermal_efficiency = self._calculate_thermal_efficiency(
            inputs.steam_temp_c,
            inputs.cw_inlet_temp_c,
            inputs.cw_outlet_temp_c
        )

        # Step 5: Calculate heat recovery efficiency
        heat_recovery_efficiency = self._calculate_heat_recovery_efficiency(
            inputs.heat_duty_mw,
            inputs.cw_flow_rate_m3_hr,
            cw_temp_rise
        )

        # Step 6: Calculate cleanliness factor
        cleanliness_factor = self._calculate_cleanliness_factor(
            inputs.actual_u_value_w_m2k,
            inputs.design_u_value_w_m2k
        )

        # Step 7: Calculate Condenser Performance Index (CPI)
        cpi = self._calculate_cpi(
            cleanliness_factor,
            ttd,
            inputs.actual_backpressure_mmhg,
            inputs.design_backpressure_mmhg
        )

        # Step 8: Determine performance rating
        performance_rating = self._determine_performance_rating(cpi)

        # Step 9: Calculate heat rate deviation
        heat_rate_deviation = self._calculate_heat_rate_deviation(
            inputs.actual_backpressure_mmhg,
            inputs.design_backpressure_mmhg
        )

        # Step 10: Calculate efficiency loss
        efficiency_loss_mw = self._calculate_efficiency_loss(
            heat_rate_deviation,
            inputs.turbine_output_mw
        )

        # Step 11: Calculate annual energy loss
        annual_energy_loss = self._calculate_annual_energy_loss(
            efficiency_loss_mw,
            inputs.operating_hours_per_year
        )

        # Step 12: Calculate annual cost loss
        annual_cost_loss = self._calculate_annual_cost_loss(
            annual_energy_loss,
            inputs.electricity_cost_usd_mwh
        )

        # Step 13: Calculate carbon penalty
        carbon_penalty = self._calculate_carbon_penalty(annual_energy_loss)

        # Step 14: Calculate potential savings
        potential_savings_mw, potential_annual_savings = self._calculate_potential_savings(
            cleanliness_factor,
            inputs.turbine_output_mw,
            inputs.actual_backpressure_mmhg,
            inputs.design_backpressure_mmhg,
            inputs.electricity_cost_usd_mwh,
            inputs.operating_hours_per_year
        )

        # Create output
        output = EfficiencyOutput(
            thermal_efficiency_pct=round(thermal_efficiency, 2),
            heat_recovery_efficiency_pct=round(heat_recovery_efficiency, 2),
            cw_temp_rise_c=round(cw_temp_rise, 2),
            approach_temp_c=round(approach_temp, 2),
            ttd_c=round(ttd, 2),
            itd_c=round(itd, 2),
            cpi=round(cpi, 4),
            performance_rating=performance_rating,
            cleanliness_factor=round(cleanliness_factor, 4),
            heat_rate_deviation_kj_kwh=round(heat_rate_deviation, 2),
            efficiency_loss_mw=round(efficiency_loss_mw, 3),
            annual_energy_loss_mwh=round(annual_energy_loss, 1),
            annual_cost_loss_usd=round(annual_cost_loss, 0),
            annual_carbon_penalty_tonnes=round(carbon_penalty, 1),
            potential_savings_mw=round(potential_savings_mw, 3),
            potential_annual_savings_usd=round(potential_annual_savings, 0)
        )

        # Set outputs and finalize provenance
        self._tracker.set_outputs({
            "thermal_efficiency_pct": output.thermal_efficiency_pct,
            "heat_recovery_efficiency_pct": output.heat_recovery_efficiency_pct,
            "cw_temp_rise_c": output.cw_temp_rise_c,
            "approach_temp_c": output.approach_temp_c,
            "ttd_c": output.ttd_c,
            "itd_c": output.itd_c,
            "cpi": output.cpi,
            "performance_rating": output.performance_rating,
            "cleanliness_factor": output.cleanliness_factor,
            "heat_rate_deviation_kj_kwh": output.heat_rate_deviation_kj_kwh,
            "efficiency_loss_mw": output.efficiency_loss_mw,
            "annual_energy_loss_mwh": output.annual_energy_loss_mwh,
            "annual_cost_loss_usd": output.annual_cost_loss_usd,
            "annual_carbon_penalty_tonnes": output.annual_carbon_penalty_tonnes,
            "potential_savings_mw": output.potential_savings_mw,
            "potential_annual_savings_usd": output.potential_annual_savings_usd
        })

        provenance = self._tracker.finalize()
        return output, provenance

    def _validate_inputs(self, inputs: EfficiencyInput) -> None:
        """
        Validate input parameters.

        Raises:
            ValueError: If any input is invalid
        """
        if inputs.steam_temp_c < 20 or inputs.steam_temp_c > 60:
            raise ValueError(
                f"Steam temperature {inputs.steam_temp_c}C out of range (20-60C)"
            )

        if inputs.cw_inlet_temp_c < 0 or inputs.cw_inlet_temp_c > 45:
            raise ValueError(
                f"CW inlet temp {inputs.cw_inlet_temp_c}C out of range (0-45C)"
            )

        if inputs.cw_outlet_temp_c <= inputs.cw_inlet_temp_c:
            raise ValueError(
                "CW outlet temp must be greater than inlet temp"
            )

        if inputs.steam_temp_c <= inputs.cw_outlet_temp_c:
            raise ValueError(
                "Steam temp must be greater than CW outlet temp"
            )

        if inputs.heat_duty_mw <= 0:
            raise ValueError("Heat duty must be positive")

        if inputs.turbine_output_mw <= 0:
            raise ValueError("Turbine output must be positive")

        if inputs.actual_u_value_w_m2k <= 0:
            raise ValueError("Actual U-value must be positive")

        if inputs.design_u_value_w_m2k <= 0:
            raise ValueError("Design U-value must be positive")

    def _calculate_cw_temp_rise(
        self,
        inlet_temp_c: float,
        outlet_temp_c: float
    ) -> float:
        """
        Calculate cooling water temperature rise.

        Formula:
            dT_cw = T_out - T_in

        Args:
            inlet_temp_c: CW inlet temperature (C)
            outlet_temp_c: CW outlet temperature (C)

        Returns:
            Temperature rise (C)
        """
        temp_rise = outlet_temp_c - inlet_temp_c

        self._tracker.add_step(
            step_number=1,
            description="Calculate cooling water temperature rise",
            operation="subtract",
            inputs={
                "cw_outlet_temp_c": outlet_temp_c,
                "cw_inlet_temp_c": inlet_temp_c
            },
            output_value=temp_rise,
            output_name="cw_temp_rise_c",
            formula="dT_cw = T_out - T_in"
        )

        return temp_rise

    def _calculate_approach_temp(
        self,
        steam_temp_c: float,
        cw_inlet_c: float
    ) -> float:
        """
        Calculate approach temperature (ITD - Initial Temperature Difference).

        This is the temperature difference between steam and cooling
        water at the inlet (hot end).

        Formula:
            T_approach = T_steam - T_cw_in

        Args:
            steam_temp_c: Steam temperature (C)
            cw_inlet_c: CW inlet temperature (C)

        Returns:
            Approach temperature (C)
        """
        approach = steam_temp_c - cw_inlet_c

        self._tracker.add_step(
            step_number=2,
            description="Calculate approach temperature",
            operation="subtract",
            inputs={
                "steam_temp_c": steam_temp_c,
                "cw_inlet_temp_c": cw_inlet_c
            },
            output_value=approach,
            output_name="approach_temp_c",
            formula="T_approach = T_steam - T_cw_in"
        )

        return approach

    def _calculate_ttd(
        self,
        steam_temp_c: float,
        cw_outlet_c: float
    ) -> float:
        """
        Calculate Terminal Temperature Difference (TTD).

        TTD is the temperature difference at the cold end of the
        condenser - a key performance indicator per HEI Standards.

        Formula:
            TTD = T_steam - T_cw_out

        Args:
            steam_temp_c: Steam temperature (C)
            cw_outlet_c: CW outlet temperature (C)

        Returns:
            TTD (C)
        """
        ttd = steam_temp_c - cw_outlet_c

        self._tracker.add_step(
            step_number=3,
            description="Calculate Terminal Temperature Difference (TTD)",
            operation="subtract",
            inputs={
                "steam_temp_c": steam_temp_c,
                "cw_outlet_temp_c": cw_outlet_c
            },
            output_value=ttd,
            output_name="ttd_c",
            formula="TTD = T_steam - T_cw_out"
        )

        return ttd

    def _calculate_itd(
        self,
        steam_temp_c: float,
        cw_inlet_c: float
    ) -> float:
        """
        Calculate Initial Temperature Difference (ITD).

        ITD is the temperature difference at the hot end.

        Formula:
            ITD = T_steam - T_cw_in

        Args:
            steam_temp_c: Steam temperature (C)
            cw_inlet_c: CW inlet temperature (C)

        Returns:
            ITD (C)
        """
        itd = steam_temp_c - cw_inlet_c

        self._tracker.add_step(
            step_number=4,
            description="Calculate Initial Temperature Difference (ITD)",
            operation="subtract",
            inputs={
                "steam_temp_c": steam_temp_c,
                "cw_inlet_temp_c": cw_inlet_c
            },
            output_value=itd,
            output_name="itd_c",
            formula="ITD = T_steam - T_cw_in"
        )

        return itd

    def _calculate_thermal_efficiency(
        self,
        steam_temp_c: float,
        cw_inlet_c: float,
        cw_outlet_c: float
    ) -> float:
        """
        Calculate condenser thermal efficiency (effectiveness).

        For a condenser (one fluid at constant temperature):
            eta = (T_cw_out - T_cw_in) / (T_steam - T_cw_in) * 100

        This represents the ratio of actual heat transfer to the
        maximum possible heat transfer.

        Args:
            steam_temp_c: Steam temperature (C)
            cw_inlet_c: CW inlet temperature (C)
            cw_outlet_c: CW outlet temperature (C)

        Returns:
            Thermal efficiency (%)
        """
        actual_rise = cw_outlet_c - cw_inlet_c
        max_possible = steam_temp_c - cw_inlet_c

        efficiency = (actual_rise / max_possible) * 100.0

        self._tracker.add_step(
            step_number=5,
            description="Calculate condenser thermal efficiency",
            operation="efficiency_ratio",
            inputs={
                "actual_cw_rise_c": actual_rise,
                "max_possible_rise_c": max_possible
            },
            output_value=efficiency,
            output_name="thermal_efficiency_pct",
            formula="eta = (T_out - T_in) / (T_steam - T_in) * 100"
        )

        return efficiency

    def _calculate_heat_recovery_efficiency(
        self,
        heat_duty_mw: float,
        cw_flow_m3_hr: float,
        cw_temp_rise_c: float
    ) -> float:
        """
        Calculate heat recovery efficiency.

        Compares actual heat transfer to theoretical maximum based
        on cooling water heat capacity.

        Formula:
            eta_recovery = Q_actual / Q_theoretical * 100

        Where Q_theoretical = m_dot * Cp * dT_max

        Args:
            heat_duty_mw: Actual heat duty (MW)
            cw_flow_m3_hr: CW flow rate (m3/hr)
            cw_temp_rise_c: CW temperature rise (C)

        Returns:
            Heat recovery efficiency (%)
        """
        # Convert flow to kg/s (assuming water density ~995 kg/m3)
        density = 995.0
        cp = 4180.0  # J/kg-K

        cw_flow_kg_s = (cw_flow_m3_hr * density) / 3600.0

        # Theoretical heat capacity
        # Using a reference max dT of 15C as typical design limit
        max_temp_rise = 15.0
        theoretical_capacity_w = cw_flow_kg_s * cp * max_temp_rise
        theoretical_capacity_mw = theoretical_capacity_w / 1_000_000.0

        efficiency = (heat_duty_mw / theoretical_capacity_mw) * 100.0

        # Cap at 100%
        efficiency = min(efficiency, 100.0)

        self._tracker.add_step(
            step_number=6,
            description="Calculate heat recovery efficiency",
            operation="heat_recovery_ratio",
            inputs={
                "heat_duty_mw": heat_duty_mw,
                "cw_flow_kg_s": cw_flow_kg_s,
                "max_temp_rise_c": max_temp_rise,
                "theoretical_capacity_mw": theoretical_capacity_mw
            },
            output_value=efficiency,
            output_name="heat_recovery_efficiency_pct",
            formula="eta_rec = Q_actual / (m_dot * Cp * dT_max) * 100"
        )

        return efficiency

    def _calculate_cleanliness_factor(
        self,
        actual_u: float,
        design_u: float
    ) -> float:
        """
        Calculate cleanliness factor.

        Formula:
            CF = U_actual / U_design

        Per HEI Standards, CF < 0.85 typically indicates need for cleaning.

        Args:
            actual_u: Actual U-value (W/m2-K)
            design_u: Design U-value (W/m2-K)

        Returns:
            Cleanliness factor (0-1)
        """
        cf = actual_u / design_u

        # Cap at 1.0 (cannot be cleaner than design)
        cf = min(cf, 1.0)

        self._tracker.add_step(
            step_number=7,
            description="Calculate cleanliness factor",
            operation="divide",
            inputs={
                "actual_u_value_w_m2k": actual_u,
                "design_u_value_w_m2k": design_u
            },
            output_value=cf,
            output_name="cleanliness_factor",
            formula="CF = U_actual / U_design"
        )

        return cf

    def _calculate_cpi(
        self,
        cleanliness_factor: float,
        ttd_c: float,
        actual_bp_mmhg: float,
        design_bp_mmhg: float
    ) -> float:
        """
        Calculate Condenser Performance Index (CPI).

        CPI is a composite index combining multiple performance factors:
        - Cleanliness factor (weighted 40%)
        - TTD performance (weighted 30%)
        - Backpressure performance (weighted 30%)

        Formula:
            CPI = 0.4 * CF + 0.3 * TTD_factor + 0.3 * BP_factor

        Args:
            cleanliness_factor: Cleanliness factor (0-1)
            ttd_c: Terminal Temperature Difference (C)
            actual_bp_mmhg: Actual backpressure (mmHg abs)
            design_bp_mmhg: Design backpressure (mmHg abs)

        Returns:
            CPI (0-1)
        """
        # TTD factor: lower TTD is better
        # Reference TTD: 3C (excellent), 8C (poor)
        ttd_factor = max(0, 1.0 - (ttd_c - 3.0) / 5.0)
        ttd_factor = min(ttd_factor, 1.0)

        # Backpressure factor: lower is better
        bp_ratio = design_bp_mmhg / actual_bp_mmhg
        bp_factor = min(bp_ratio, 1.0)

        # Composite CPI
        cpi = (0.4 * cleanliness_factor +
               0.3 * ttd_factor +
               0.3 * bp_factor)

        self._tracker.add_step(
            step_number=8,
            description="Calculate Condenser Performance Index (CPI)",
            operation="weighted_average",
            inputs={
                "cleanliness_factor": cleanliness_factor,
                "ttd_c": ttd_c,
                "ttd_factor": ttd_factor,
                "actual_bp_mmhg": actual_bp_mmhg,
                "design_bp_mmhg": design_bp_mmhg,
                "bp_factor": bp_factor
            },
            output_value=cpi,
            output_name="cpi",
            formula="CPI = 0.4*CF + 0.3*TTD_f + 0.3*BP_f"
        )

        return cpi

    def _determine_performance_rating(self, cpi: float) -> str:
        """
        Determine performance rating category from CPI.

        Rating categories:
        - Excellent: CPI >= 0.95
        - Good: 0.85 <= CPI < 0.95
        - Average: 0.75 <= CPI < 0.85
        - Poor: 0.65 <= CPI < 0.75
        - Critical: CPI < 0.65

        Args:
            cpi: Condenser Performance Index

        Returns:
            Performance rating string
        """
        if cpi >= 0.95:
            rating = PerformanceRating.EXCELLENT
        elif cpi >= 0.85:
            rating = PerformanceRating.GOOD
        elif cpi >= 0.75:
            rating = PerformanceRating.AVERAGE
        elif cpi >= 0.65:
            rating = PerformanceRating.POOR
        else:
            rating = PerformanceRating.CRITICAL

        self._tracker.add_step(
            step_number=9,
            description="Determine performance rating category",
            operation="threshold_classification",
            inputs={
                "cpi": cpi,
                "thresholds": {
                    "excellent": 0.95,
                    "good": 0.85,
                    "average": 0.75,
                    "poor": 0.65
                }
            },
            output_value=rating.value,
            output_name="performance_rating",
            formula="Rating based on CPI thresholds"
        )

        return rating.value

    def _calculate_heat_rate_deviation(
        self,
        actual_bp_mmhg: float,
        design_bp_mmhg: float
    ) -> float:
        """
        Calculate heat rate deviation from design.

        Higher backpressure increases heat rate (worse efficiency).
        Typical impact: ~25 kJ/kWh per mmHg deviation.

        Formula:
            HR_dev = (P_actual - P_design) * HR_impact_factor

        Args:
            actual_bp_mmhg: Actual backpressure (mmHg abs)
            design_bp_mmhg: Design backpressure (mmHg abs)

        Returns:
            Heat rate deviation (kJ/kWh)
        """
        bp_deviation = actual_bp_mmhg - design_bp_mmhg
        hr_deviation = bp_deviation * HEAT_RATE_IMPACT_KJ_KWH_PER_MMHG

        # Only positive deviation (worse performance)
        hr_deviation = max(0, hr_deviation)

        self._tracker.add_step(
            step_number=10,
            description="Calculate heat rate deviation from design",
            operation="heat_rate_impact",
            inputs={
                "actual_bp_mmhg": actual_bp_mmhg,
                "design_bp_mmhg": design_bp_mmhg,
                "bp_deviation_mmhg": bp_deviation,
                "hr_impact_factor": HEAT_RATE_IMPACT_KJ_KWH_PER_MMHG
            },
            output_value=hr_deviation,
            output_name="heat_rate_deviation_kj_kwh",
            formula="HR_dev = dP * 25 kJ/kWh/mmHg"
        )

        return hr_deviation

    def _calculate_efficiency_loss(
        self,
        heat_rate_deviation_kj_kwh: float,
        turbine_output_mw: float
    ) -> float:
        """
        Calculate power output loss due to heat rate deviation.

        Formula:
            P_loss = P_output * (HR_dev / HR_base)

        Where HR_base is approximately 10,000 kJ/kWh for typical plant.

        Args:
            heat_rate_deviation_kj_kwh: Heat rate deviation (kJ/kWh)
            turbine_output_mw: Turbine output (MW)

        Returns:
            Power loss (MW)
        """
        # Base heat rate (typical value)
        base_heat_rate = 10000.0  # kJ/kWh

        # Efficiency loss fraction
        loss_fraction = heat_rate_deviation_kj_kwh / base_heat_rate

        # Power loss
        power_loss = turbine_output_mw * loss_fraction

        self._tracker.add_step(
            step_number=11,
            description="Calculate power output loss",
            operation="efficiency_loss",
            inputs={
                "heat_rate_deviation_kj_kwh": heat_rate_deviation_kj_kwh,
                "base_heat_rate_kj_kwh": base_heat_rate,
                "turbine_output_mw": turbine_output_mw,
                "loss_fraction": loss_fraction
            },
            output_value=power_loss,
            output_name="efficiency_loss_mw",
            formula="P_loss = P_out * (HR_dev / HR_base)"
        )

        return power_loss

    def _calculate_annual_energy_loss(
        self,
        power_loss_mw: float,
        operating_hours: int
    ) -> float:
        """
        Calculate annual energy loss.

        Formula:
            E_loss = P_loss * hours

        Args:
            power_loss_mw: Continuous power loss (MW)
            operating_hours: Annual operating hours

        Returns:
            Annual energy loss (MWh)
        """
        energy_loss = power_loss_mw * operating_hours

        self._tracker.add_step(
            step_number=12,
            description="Calculate annual energy loss",
            operation="multiply",
            inputs={
                "power_loss_mw": power_loss_mw,
                "operating_hours": operating_hours
            },
            output_value=energy_loss,
            output_name="annual_energy_loss_mwh",
            formula="E_loss = P_loss * hours"
        )

        return energy_loss

    def _calculate_annual_cost_loss(
        self,
        annual_energy_loss_mwh: float,
        electricity_cost_usd_mwh: float
    ) -> float:
        """
        Calculate annual cost of efficiency loss.

        Formula:
            Cost = E_loss * price

        Args:
            annual_energy_loss_mwh: Annual energy loss (MWh)
            electricity_cost_usd_mwh: Electricity price (USD/MWh)

        Returns:
            Annual cost loss (USD)
        """
        cost_loss = annual_energy_loss_mwh * electricity_cost_usd_mwh

        self._tracker.add_step(
            step_number=13,
            description="Calculate annual cost of efficiency loss",
            operation="multiply",
            inputs={
                "annual_energy_loss_mwh": annual_energy_loss_mwh,
                "electricity_cost_usd_mwh": electricity_cost_usd_mwh
            },
            output_value=cost_loss,
            output_name="annual_cost_loss_usd",
            formula="Cost = E_loss * price"
        )

        return cost_loss

    def _calculate_carbon_penalty(
        self,
        annual_energy_loss_mwh: float
    ) -> float:
        """
        Calculate additional CO2 emissions due to inefficiency.

        The energy lost must be generated elsewhere, typically from
        grid mix which has associated carbon emissions.

        Formula:
            CO2 = E_loss * emission_factor

        Args:
            annual_energy_loss_mwh: Annual energy loss (MWh)

        Returns:
            Additional CO2 emissions (tonnes/year)
        """
        # Convert kg to tonnes
        carbon_kg = annual_energy_loss_mwh * CARBON_EMISSION_FACTOR_KG_CO2_MWH
        carbon_tonnes = carbon_kg / 1000.0

        self._tracker.add_step(
            step_number=14,
            description="Calculate carbon penalty from inefficiency",
            operation="carbon_calculation",
            inputs={
                "annual_energy_loss_mwh": annual_energy_loss_mwh,
                "emission_factor_kg_co2_mwh": CARBON_EMISSION_FACTOR_KG_CO2_MWH
            },
            output_value=carbon_tonnes,
            output_name="annual_carbon_penalty_tonnes",
            formula="CO2 = E_loss * EF / 1000"
        )

        return carbon_tonnes

    def _calculate_potential_savings(
        self,
        cleanliness_factor: float,
        turbine_output_mw: float,
        actual_bp_mmhg: float,
        design_bp_mmhg: float,
        electricity_cost: float,
        operating_hours: int
    ) -> Tuple[float, float]:
        """
        Calculate potential savings if condenser restored to design.

        Estimates the power recovery and cost savings from:
        - Restoring cleanliness factor to 1.0
        - Achieving design backpressure

        Args:
            cleanliness_factor: Current cleanliness factor
            turbine_output_mw: Turbine output (MW)
            actual_bp_mmhg: Current backpressure
            design_bp_mmhg: Design backpressure
            electricity_cost: Electricity price (USD/MWh)
            operating_hours: Annual operating hours

        Returns:
            Tuple of (potential power recovery MW, annual savings USD)
        """
        # Potential backpressure improvement
        bp_improvement_mmhg = max(0, actual_bp_mmhg - design_bp_mmhg)

        # Potential heat rate improvement
        hr_improvement = bp_improvement_mmhg * HEAT_RATE_IMPACT_KJ_KWH_PER_MMHG

        # Base heat rate
        base_heat_rate = 10000.0

        # Potential power recovery
        recovery_fraction = hr_improvement / base_heat_rate
        potential_power_mw = turbine_output_mw * recovery_fraction

        # Annual savings
        annual_energy_mwh = potential_power_mw * operating_hours
        annual_savings = annual_energy_mwh * electricity_cost

        self._tracker.add_step(
            step_number=15,
            description="Calculate potential savings from optimization",
            operation="savings_calculation",
            inputs={
                "cleanliness_factor": cleanliness_factor,
                "bp_improvement_mmhg": bp_improvement_mmhg,
                "hr_improvement_kj_kwh": hr_improvement,
                "recovery_fraction": recovery_fraction,
                "turbine_output_mw": turbine_output_mw
            },
            output_value=potential_power_mw,
            output_name="potential_savings_mw",
            formula="P_save = P_out * (HR_imp / HR_base)"
        )

        return potential_power_mw, annual_savings


# =============================================================================
# STANDALONE CALCULATION FUNCTIONS
# =============================================================================

def calculate_cw_temperature_rise(
    heat_duty_mw: float,
    cw_flow_rate_m3_hr: float
) -> float:
    """
    Calculate cooling water temperature rise from heat duty.

    Formula:
        dT = Q / (m_dot * Cp)

    Args:
        heat_duty_mw: Heat duty (MW)
        cw_flow_rate_m3_hr: CW flow rate (m3/hr)

    Returns:
        Temperature rise (C)
    """
    # Water properties
    density = 995.0  # kg/m3
    cp = 4180.0  # J/kg-K

    # Convert units
    heat_duty_w = heat_duty_mw * 1_000_000
    cw_flow_kg_s = (cw_flow_rate_m3_hr * density) / 3600.0

    # Calculate temperature rise
    temp_rise = heat_duty_w / (cw_flow_kg_s * cp)

    return temp_rise


def calculate_optimal_cw_flow(
    heat_duty_mw: float,
    target_temp_rise_c: float
) -> float:
    """
    Calculate optimal CW flow rate for target temperature rise.

    Formula:
        m_dot = Q / (Cp * dT)

    Args:
        heat_duty_mw: Heat duty (MW)
        target_temp_rise_c: Target CW temperature rise (C)

    Returns:
        Required CW flow rate (m3/hr)
    """
    # Water properties
    density = 995.0  # kg/m3
    cp = 4180.0  # J/kg-K

    # Convert units
    heat_duty_w = heat_duty_mw * 1_000_000

    # Calculate required mass flow
    cw_flow_kg_s = heat_duty_w / (cp * target_temp_rise_c)

    # Convert to m3/hr
    cw_flow_m3_hr = (cw_flow_kg_s / density) * 3600.0

    return cw_flow_m3_hr


def calculate_cw_pumping_power(
    flow_rate_m3_hr: float,
    head_m: float,
    pump_efficiency: float = 0.8
) -> float:
    """
    Calculate cooling water pumping power requirement.

    Formula:
        P = (rho * g * Q * H) / eta

    Args:
        flow_rate_m3_hr: CW flow rate (m3/hr)
        head_m: Pump head (m)
        pump_efficiency: Pump efficiency (default 0.8)

    Returns:
        Pumping power (kW)
    """
    # Water density
    density = 995.0  # kg/m3
    g = 9.81  # m/s2

    # Convert flow to m3/s
    flow_m3_s = flow_rate_m3_hr / 3600.0

    # Hydraulic power
    hydraulic_power_w = density * g * flow_m3_s * head_m

    # Shaft power (accounting for efficiency)
    shaft_power_w = hydraulic_power_w / pump_efficiency

    return shaft_power_w / 1000.0  # Convert to kW


def calculate_payback_period(
    investment_cost_usd: float,
    annual_savings_usd: float
) -> float:
    """
    Calculate simple payback period for efficiency improvement.

    Formula:
        Payback = Investment / Annual_Savings

    Args:
        investment_cost_usd: Initial investment (USD)
        annual_savings_usd: Annual energy savings (USD)

    Returns:
        Payback period (years)
    """
    if annual_savings_usd <= 0:
        return float('inf')

    return investment_cost_usd / annual_savings_usd


def calculate_npv(
    investment_cost_usd: float,
    annual_savings_usd: float,
    discount_rate: float,
    project_life_years: int
) -> float:
    """
    Calculate Net Present Value of efficiency improvement.

    Formula:
        NPV = -Investment + Sum(Savings / (1+r)^t)

    Args:
        investment_cost_usd: Initial investment (USD)
        annual_savings_usd: Annual savings (USD)
        discount_rate: Discount rate (e.g., 0.08 for 8%)
        project_life_years: Project life (years)

    Returns:
        NPV (USD)
    """
    npv = -investment_cost_usd

    for year in range(1, project_life_years + 1):
        discount_factor = (1 + discount_rate) ** year
        npv += annual_savings_usd / discount_factor

    return npv
