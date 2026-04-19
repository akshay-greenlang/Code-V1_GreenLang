"""
GL-017 CONDENSYNC - Condenser Performance Calculator

Zero-hallucination, deterministic calculations for comprehensive condenser
performance analysis following HEI Standards for Steam Surface Condensers.

This module provides:
- Overall heat transfer coefficient (U-value) calculation with HEI corrections
- Cleanliness factor trending with statistical analysis
- Terminal Temperature Difference (TTD) calculation and analysis
- Condenser duty calculation (rated vs actual)
- Vacuum optimization (backpressure vs efficiency trade-off)
- LMTD for condenser (special case with isothermal phase change)
- HEI (Heat Exchange Institute) standard correction factors
- Performance deviation analysis

Standards Reference:
- HEI Standards for Steam Surface Condensers (11th Edition)
- ASME PTC 12.2 - Steam Surface Condensers Performance Test Code
- EPRI Heat Rate Improvement Guidelines
- EPRI Condenser In-Leakage Guideline

Author: GL-CalculatorEngineer
Version: 1.0.0
"""

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import math

from .provenance import ProvenanceTracker, ProvenanceRecord


# =============================================================================
# CONSTANTS AND REFERENCE DATA (HEI STANDARDS)
# =============================================================================

# HEI heat transfer coefficients base values (W/m2-K)
# For admiralty brass tubes at standard conditions
HEI_BASE_U_VALUES = {
    "18_BWG_25.4mm": 3407.0,  # 18 BWG, 1" OD
    "20_BWG_25.4mm": 3521.0,  # 20 BWG, 1" OD
    "22_BWG_25.4mm": 3635.0,  # 22 BWG, 1" OD
    "18_BWG_19.1mm": 4086.0,  # 18 BWG, 3/4" OD
    "20_BWG_19.1mm": 4200.0,  # 20 BWG, 3/4" OD
    "22_BWG_19.1mm": 4314.0,  # 22 BWG, 3/4" OD
}

# HEI tube material correction factors (Fm)
HEI_MATERIAL_FACTORS = {
    "admiralty_brass": 1.00,
    "aluminum_brass": 0.97,
    "aluminum_bronze": 0.95,
    "arsenical_copper": 1.04,
    "copper_iron_194": 0.96,
    "copper_nickel_90_10": 0.87,
    "copper_nickel_80_20": 0.83,
    "copper_nickel_70_30": 0.79,
    "stainless_steel_304": 0.61,
    "stainless_steel_316": 0.61,
    "stainless_steel_317": 0.61,
    "titanium": 0.71,
    "sea_cure": 0.71,
}

# HEI inlet water temperature correction factors (Ft)
# Format: {temp_F: factor}
HEI_TEMP_CORRECTION = {
    40: 0.670,
    50: 0.765,
    60: 0.855,
    70: 0.940,
    80: 1.000,
    85: 1.025,
    90: 1.040,
    95: 1.050,
    100: 1.055,
    110: 1.060,
}

# HEI velocity correction factors (Fv)
# Format: {velocity_fps: factor}
HEI_VELOCITY_CORRECTION = {
    3.0: 0.750,
    4.0: 0.852,
    5.0: 0.937,
    6.0: 1.000,
    7.0: 1.058,
    8.0: 1.100,
    9.0: 1.137,
    10.0: 1.165,
}

# HEI cleanliness factor thresholds
HEI_CLEANLINESS_THRESHOLDS = {
    "excellent": 0.95,
    "good": 0.90,
    "acceptable": 0.85,
    "marginal": 0.80,
    "poor": 0.75,
    "critical": 0.70,
}

# Turbine backpressure correction factors (kJ/kWh per mmHg)
HEAT_RATE_CORRECTION_PER_MMHG = {
    "subcritical": 12.5,
    "supercritical": 15.0,
    "combined_cycle": 8.0,
}

# Steam saturation properties (temperature in C, pressure in kPa abs)
STEAM_SATURATION = {
    30: 4.243,
    32: 4.754,
    34: 5.319,
    36: 5.942,
    38: 6.628,
    40: 7.381,
    42: 8.205,
    44: 9.107,
    46: 10.09,
    48: 11.16,
    50: 12.34,
    52: 13.61,
    54: 15.00,
    56: 16.51,
    58: 18.15,
    60: 19.93,
}


class PerformanceStatus(Enum):
    """Performance status categories."""
    OPTIMAL = "Optimal"
    ACCEPTABLE = "Acceptable"
    DEGRADED = "Degraded"
    CRITICAL = "Critical"


class VacuumUnit(Enum):
    """Vacuum measurement units."""
    MBAR_ABS = "mbar_abs"
    MMHG_ABS = "mmHg_abs"
    KPA_ABS = "kPa_abs"
    INHG_ABS = "inHg_abs"


# =============================================================================
# INPUT/OUTPUT DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class CondenserPerformanceInput:
    """
    Input parameters for condenser performance calculations.

    Attributes:
        steam_flow_kg_s: Steam flow to condenser (kg/s)
        steam_temp_c: Steam saturation temperature (Celsius)
        condenser_pressure_kpa: Condenser absolute pressure (kPa)
        cw_inlet_temp_c: Cooling water inlet temperature (Celsius)
        cw_outlet_temp_c: Cooling water outlet temperature (Celsius)
        cw_flow_rate_kg_s: Cooling water mass flow rate (kg/s)
        heat_transfer_area_m2: Total heat transfer surface area (m2)
        tube_od_mm: Tube outside diameter (mm)
        tube_id_mm: Tube inside diameter (mm)
        tube_length_m: Effective tube length (m)
        tube_material: Tube material name
        tube_gauge: Tube gauge (e.g., "18_BWG")
        num_tubes: Total number of tubes
        num_passes: Number of cooling water passes
        design_u_value_w_m2k: Design overall heat transfer coefficient (W/m2-K)
        design_duty_mw: Design heat duty (MW)
        design_ttd_c: Design Terminal Temperature Difference (C)
        design_cw_velocity_m_s: Design cooling water velocity (m/s)
        turbine_type: Turbine type for heat rate calculations
        electricity_price_usd_mwh: Electricity price (USD/MWh)
        historical_cf_data: Optional list of historical CF values for trending
    """
    steam_flow_kg_s: float
    steam_temp_c: float
    condenser_pressure_kpa: float
    cw_inlet_temp_c: float
    cw_outlet_temp_c: float
    cw_flow_rate_kg_s: float
    heat_transfer_area_m2: float
    tube_od_mm: float
    tube_id_mm: float
    tube_length_m: float
    tube_material: str
    tube_gauge: str
    num_tubes: int
    num_passes: int
    design_u_value_w_m2k: float
    design_duty_mw: float
    design_ttd_c: float
    design_cw_velocity_m_s: float
    turbine_type: str = "subcritical"
    electricity_price_usd_mwh: float = 50.0
    historical_cf_data: Optional[List[float]] = None


@dataclass(frozen=True)
class CondenserPerformanceOutput:
    """
    Output results from condenser performance calculations.

    Attributes:
        actual_duty_mw: Actual heat duty (MW)
        duty_deviation_pct: Deviation from design duty (%)
        lmtd_c: Log Mean Temperature Difference (C)
        ttd_c: Terminal Temperature Difference (C)
        itd_c: Initial Temperature Difference (C)
        ttd_deviation_c: TTD deviation from design (C)
        cw_temp_rise_c: Cooling water temperature rise (C)
        actual_u_value_w_m2k: Actual U-value (W/m2-K)
        hei_corrected_u_value_w_m2k: HEI corrected design U-value (W/m2-K)
        cleanliness_factor: Cleanliness factor (0-1)
        cf_status: Cleanliness factor status
        cf_trend_slope: CF trend slope (per day, if historical data provided)
        cf_days_to_threshold: Predicted days to CF threshold
        cw_velocity_m_s: Actual cooling water velocity (m/s)
        velocity_deviation_pct: Velocity deviation from design (%)
        hei_material_factor: HEI material correction factor
        hei_temp_factor: HEI temperature correction factor
        hei_velocity_factor: HEI velocity correction factor
        backpressure_mmhg: Condenser backpressure (mmHg abs)
        optimal_backpressure_mmhg: Optimal backpressure (mmHg abs)
        backpressure_penalty_kw: Power loss from excess backpressure (kW)
        heat_rate_impact_kj_kwh: Heat rate impact (kJ/kWh)
        annual_cost_impact_usd: Annual cost impact (USD)
        performance_status: Overall performance status
        improvement_potential_pct: Potential improvement with cleaning (%)
    """
    actual_duty_mw: float
    duty_deviation_pct: float
    lmtd_c: float
    ttd_c: float
    itd_c: float
    ttd_deviation_c: float
    cw_temp_rise_c: float
    actual_u_value_w_m2k: float
    hei_corrected_u_value_w_m2k: float
    cleanliness_factor: float
    cf_status: str
    cf_trend_slope: float
    cf_days_to_threshold: int
    cw_velocity_m_s: float
    velocity_deviation_pct: float
    hei_material_factor: float
    hei_temp_factor: float
    hei_velocity_factor: float
    backpressure_mmhg: float
    optimal_backpressure_mmhg: float
    backpressure_penalty_kw: float
    heat_rate_impact_kj_kwh: float
    annual_cost_impact_usd: float
    performance_status: str
    improvement_potential_pct: float


# =============================================================================
# CONDENSER PERFORMANCE CALCULATOR CLASS
# =============================================================================

class CondenserPerformanceCalculator:
    """
    Zero-hallucination condenser performance calculator.

    Implements deterministic calculations following HEI Standards for
    Steam Surface Condensers. All calculations produce bit-perfect
    reproducible results with complete provenance tracking.

    Guarantees:
    - DETERMINISTIC: Same input always produces same output
    - REPRODUCIBLE: SHA-256 verified calculation chain
    - AUDITABLE: Complete step-by-step provenance trail
    - ZERO HALLUCINATION: No LLM in calculation path

    Example:
        >>> calculator = CondenserPerformanceCalculator()
        >>> inputs = CondenserPerformanceInput(
        ...     steam_flow_kg_s=150.0,
        ...     steam_temp_c=38.0,
        ...     condenser_pressure_kpa=6.5,
        ...     cw_inlet_temp_c=25.0,
        ...     cw_outlet_temp_c=35.0,
        ...     cw_flow_rate_kg_s=8000.0,
        ...     heat_transfer_area_m2=12000.0,
        ...     tube_od_mm=25.4,
        ...     tube_id_mm=22.9,
        ...     tube_length_m=12.0,
        ...     tube_material="titanium",
        ...     tube_gauge="22_BWG",
        ...     num_tubes=8000,
        ...     num_passes=2,
        ...     design_u_value_w_m2k=2800.0,
        ...     design_duty_mw=350.0,
        ...     design_ttd_c=3.0,
        ...     design_cw_velocity_m_s=2.2
        ... )
        >>> result, provenance = calculator.calculate(inputs)
        >>> print(f"Cleanliness Factor: {result.cleanliness_factor:.4f}")
    """

    VERSION = "1.0.0"
    NAME = "CondenserPerformanceCalculator"

    # Physical constants
    LATENT_HEAT_STEAM_KJ_KG = 2257.0  # At ~100C, varies slightly with pressure
    WATER_CP_KJ_KGK = 4.18
    MMHG_PER_KPA = 7.50062

    def __init__(self):
        """Initialize the condenser performance calculator."""
        self._tracker: Optional[ProvenanceTracker] = None

    def calculate(
        self,
        inputs: CondenserPerformanceInput
    ) -> Tuple[CondenserPerformanceOutput, ProvenanceRecord]:
        """
        Perform complete condenser performance analysis.

        Args:
            inputs: CondenserPerformanceInput with all required parameters

        Returns:
            Tuple of (CondenserPerformanceOutput, ProvenanceRecord)

        Raises:
            ValueError: If inputs are invalid or out of range
        """
        # Initialize provenance tracking
        self._tracker = ProvenanceTracker(
            calculator_name=self.NAME,
            calculator_version=self.VERSION,
            metadata={
                "standards": ["HEI Standards 11th Ed", "ASME PTC 12.2", "EPRI Guidelines"],
                "domain": "Steam Condenser Performance Analysis"
            }
        )

        # Convert inputs to dictionary for provenance
        input_dict = self._inputs_to_dict(inputs)
        self._tracker.set_inputs(input_dict)

        # Validate inputs
        self._validate_inputs(inputs)

        # Step 1: Calculate actual heat duty
        actual_duty_mw = self._calculate_actual_duty(
            inputs.cw_flow_rate_kg_s,
            inputs.cw_inlet_temp_c,
            inputs.cw_outlet_temp_c
        )

        # Step 2: Calculate duty deviation
        duty_deviation_pct = self._calculate_deviation(
            actual_duty_mw,
            inputs.design_duty_mw
        )

        # Step 3: Calculate temperature differences
        lmtd_c = self._calculate_condenser_lmtd(
            inputs.steam_temp_c,
            inputs.cw_inlet_temp_c,
            inputs.cw_outlet_temp_c
        )

        ttd_c = self._calculate_ttd(inputs.steam_temp_c, inputs.cw_outlet_temp_c)
        itd_c = self._calculate_itd(inputs.steam_temp_c, inputs.cw_inlet_temp_c)
        ttd_deviation_c = ttd_c - inputs.design_ttd_c
        cw_temp_rise_c = inputs.cw_outlet_temp_c - inputs.cw_inlet_temp_c

        # Step 4: Calculate actual U-value
        actual_u_value = self._calculate_actual_u_value(
            actual_duty_mw * 1e6,  # Convert to W
            inputs.heat_transfer_area_m2,
            lmtd_c
        )

        # Step 5: Calculate HEI correction factors
        hei_material_factor = self._get_hei_material_factor(inputs.tube_material)
        hei_temp_factor = self._get_hei_temp_factor(inputs.cw_inlet_temp_c)

        cw_velocity = self._calculate_cw_velocity(
            inputs.cw_flow_rate_kg_s,
            inputs.tube_id_mm,
            inputs.num_tubes,
            inputs.num_passes
        )
        hei_velocity_factor = self._get_hei_velocity_factor(cw_velocity)

        # Step 6: Calculate HEI corrected design U-value
        hei_corrected_u = self._calculate_hei_corrected_u(
            inputs.design_u_value_w_m2k,
            hei_material_factor,
            hei_temp_factor,
            hei_velocity_factor
        )

        # Step 7: Calculate cleanliness factor
        cleanliness_factor = self._calculate_cleanliness_factor(
            actual_u_value,
            hei_corrected_u
        )

        cf_status = self._get_cf_status(cleanliness_factor)

        # Step 8: Analyze CF trend if historical data available
        cf_trend_slope, cf_days_to_threshold = self._analyze_cf_trend(
            cleanliness_factor,
            inputs.historical_cf_data
        )

        # Step 9: Calculate velocity deviation
        velocity_deviation_pct = self._calculate_deviation(
            cw_velocity,
            inputs.design_cw_velocity_m_s
        )

        # Step 10: Calculate vacuum/backpressure analysis
        backpressure_mmhg = inputs.condenser_pressure_kpa * self.MMHG_PER_KPA

        optimal_backpressure = self._calculate_optimal_backpressure(
            inputs.cw_inlet_temp_c,
            cleanliness_factor
        )

        backpressure_penalty, heat_rate_impact = self._calculate_backpressure_penalty(
            backpressure_mmhg,
            optimal_backpressure,
            inputs.steam_flow_kg_s,
            inputs.turbine_type
        )

        # Step 11: Calculate annual cost impact
        annual_cost_impact = self._calculate_annual_cost_impact(
            backpressure_penalty,
            inputs.electricity_price_usd_mwh
        )

        # Step 12: Determine overall performance status
        performance_status = self._determine_performance_status(
            cleanliness_factor,
            ttd_deviation_c,
            duty_deviation_pct
        )

        # Step 13: Calculate improvement potential
        improvement_potential = self._calculate_improvement_potential(
            cleanliness_factor,
            backpressure_penalty
        )

        # Create output
        output = CondenserPerformanceOutput(
            actual_duty_mw=round(actual_duty_mw, 3),
            duty_deviation_pct=round(duty_deviation_pct, 2),
            lmtd_c=round(lmtd_c, 3),
            ttd_c=round(ttd_c, 3),
            itd_c=round(itd_c, 3),
            ttd_deviation_c=round(ttd_deviation_c, 3),
            cw_temp_rise_c=round(cw_temp_rise_c, 3),
            actual_u_value_w_m2k=round(actual_u_value, 1),
            hei_corrected_u_value_w_m2k=round(hei_corrected_u, 1),
            cleanliness_factor=round(cleanliness_factor, 4),
            cf_status=cf_status,
            cf_trend_slope=round(cf_trend_slope, 6),
            cf_days_to_threshold=cf_days_to_threshold,
            cw_velocity_m_s=round(cw_velocity, 3),
            velocity_deviation_pct=round(velocity_deviation_pct, 2),
            hei_material_factor=round(hei_material_factor, 3),
            hei_temp_factor=round(hei_temp_factor, 3),
            hei_velocity_factor=round(hei_velocity_factor, 3),
            backpressure_mmhg=round(backpressure_mmhg, 2),
            optimal_backpressure_mmhg=round(optimal_backpressure, 2),
            backpressure_penalty_kw=round(backpressure_penalty, 1),
            heat_rate_impact_kj_kwh=round(heat_rate_impact, 2),
            annual_cost_impact_usd=round(annual_cost_impact, 0),
            performance_status=performance_status,
            improvement_potential_pct=round(improvement_potential, 2)
        )

        # Set outputs and finalize provenance
        self._tracker.set_outputs(self._output_to_dict(output))
        provenance = self._tracker.finalize()

        return output, provenance

    def _inputs_to_dict(self, inputs: CondenserPerformanceInput) -> Dict:
        """Convert input dataclass to dictionary."""
        return {
            "steam_flow_kg_s": inputs.steam_flow_kg_s,
            "steam_temp_c": inputs.steam_temp_c,
            "condenser_pressure_kpa": inputs.condenser_pressure_kpa,
            "cw_inlet_temp_c": inputs.cw_inlet_temp_c,
            "cw_outlet_temp_c": inputs.cw_outlet_temp_c,
            "cw_flow_rate_kg_s": inputs.cw_flow_rate_kg_s,
            "heat_transfer_area_m2": inputs.heat_transfer_area_m2,
            "tube_od_mm": inputs.tube_od_mm,
            "tube_id_mm": inputs.tube_id_mm,
            "tube_length_m": inputs.tube_length_m,
            "tube_material": inputs.tube_material,
            "tube_gauge": inputs.tube_gauge,
            "num_tubes": inputs.num_tubes,
            "num_passes": inputs.num_passes,
            "design_u_value_w_m2k": inputs.design_u_value_w_m2k,
            "design_duty_mw": inputs.design_duty_mw,
            "design_ttd_c": inputs.design_ttd_c,
            "design_cw_velocity_m_s": inputs.design_cw_velocity_m_s,
            "turbine_type": inputs.turbine_type,
            "electricity_price_usd_mwh": inputs.electricity_price_usd_mwh,
            "historical_cf_data": inputs.historical_cf_data
        }

    def _output_to_dict(self, output: CondenserPerformanceOutput) -> Dict:
        """Convert output dataclass to dictionary."""
        return {
            "actual_duty_mw": output.actual_duty_mw,
            "duty_deviation_pct": output.duty_deviation_pct,
            "lmtd_c": output.lmtd_c,
            "ttd_c": output.ttd_c,
            "itd_c": output.itd_c,
            "ttd_deviation_c": output.ttd_deviation_c,
            "cw_temp_rise_c": output.cw_temp_rise_c,
            "actual_u_value_w_m2k": output.actual_u_value_w_m2k,
            "hei_corrected_u_value_w_m2k": output.hei_corrected_u_value_w_m2k,
            "cleanliness_factor": output.cleanliness_factor,
            "cf_status": output.cf_status,
            "cf_trend_slope": output.cf_trend_slope,
            "cf_days_to_threshold": output.cf_days_to_threshold,
            "cw_velocity_m_s": output.cw_velocity_m_s,
            "velocity_deviation_pct": output.velocity_deviation_pct,
            "hei_material_factor": output.hei_material_factor,
            "hei_temp_factor": output.hei_temp_factor,
            "hei_velocity_factor": output.hei_velocity_factor,
            "backpressure_mmhg": output.backpressure_mmhg,
            "optimal_backpressure_mmhg": output.optimal_backpressure_mmhg,
            "backpressure_penalty_kw": output.backpressure_penalty_kw,
            "heat_rate_impact_kj_kwh": output.heat_rate_impact_kj_kwh,
            "annual_cost_impact_usd": output.annual_cost_impact_usd,
            "performance_status": output.performance_status,
            "improvement_potential_pct": output.improvement_potential_pct
        }

    def _validate_inputs(self, inputs: CondenserPerformanceInput) -> None:
        """Validate input parameters."""
        # Temperature validations
        if inputs.steam_temp_c < 25 or inputs.steam_temp_c > 60:
            raise ValueError(
                f"Steam temperature {inputs.steam_temp_c}C out of typical range (25-60C)"
            )

        if inputs.cw_inlet_temp_c < 0 or inputs.cw_inlet_temp_c > 45:
            raise ValueError(
                f"CW inlet temp {inputs.cw_inlet_temp_c}C out of range (0-45C)"
            )

        if inputs.cw_outlet_temp_c <= inputs.cw_inlet_temp_c:
            raise ValueError("CW outlet temp must be greater than inlet temp")

        if inputs.steam_temp_c <= inputs.cw_outlet_temp_c:
            raise ValueError("Steam temp must be greater than CW outlet temp")

        # Flow and geometry validations
        if inputs.cw_flow_rate_kg_s <= 0:
            raise ValueError("CW flow rate must be positive")

        if inputs.tube_id_mm >= inputs.tube_od_mm:
            raise ValueError("Tube ID must be less than OD")

        if inputs.design_duty_mw <= 0:
            raise ValueError("Design duty must be positive")

    def _calculate_actual_duty(
        self,
        flow_rate_kg_s: float,
        inlet_temp_c: float,
        outlet_temp_c: float
    ) -> float:
        """
        Calculate actual heat duty from cooling water energy balance.

        Formula:
            Q = m_dot * Cp * (T_out - T_in)

        Args:
            flow_rate_kg_s: Mass flow rate (kg/s)
            inlet_temp_c: Inlet temperature (C)
            outlet_temp_c: Outlet temperature (C)

        Returns:
            Heat duty in MW
        """
        delta_t = outlet_temp_c - inlet_temp_c
        duty_kw = flow_rate_kg_s * self.WATER_CP_KJ_KGK * delta_t
        duty_mw = duty_kw / 1000.0

        self._tracker.add_step(
            step_number=1,
            description="Calculate actual heat duty from CW energy balance",
            operation="multiply",
            inputs={
                "cw_flow_rate_kg_s": flow_rate_kg_s,
                "cp_kj_kgk": self.WATER_CP_KJ_KGK,
                "delta_t_c": delta_t
            },
            output_value=duty_mw,
            output_name="actual_duty_mw",
            formula="Q = m_dot * Cp * dT"
        )

        return duty_mw

    def _calculate_deviation(self, actual: float, design: float) -> float:
        """Calculate percentage deviation from design value."""
        if design == 0:
            return 0.0
        return ((actual - design) / design) * 100.0

    def _calculate_condenser_lmtd(
        self,
        steam_temp_c: float,
        cw_inlet_c: float,
        cw_outlet_c: float
    ) -> float:
        """
        Calculate LMTD for condenser (isothermal steam side).

        For condenser with constant steam temperature:
            dT1 = T_steam - T_cw_outlet (cold end, TTD)
            dT2 = T_steam - T_cw_inlet  (hot end, ITD)

        Formula:
            LMTD = (dT2 - dT1) / ln(dT2/dT1)

        Args:
            steam_temp_c: Steam saturation temperature (C)
            cw_inlet_c: Cooling water inlet temperature (C)
            cw_outlet_c: Cooling water outlet temperature (C)

        Returns:
            LMTD in Celsius/Kelvin
        """
        dt1 = steam_temp_c - cw_outlet_c  # TTD (smaller)
        dt2 = steam_temp_c - cw_inlet_c   # ITD (larger)

        if dt1 <= 0 or dt2 <= 0:
            raise ValueError("Invalid temperature approach - must be positive")

        # Handle edge case where dt1 equals dt2
        if abs(dt2 - dt1) < 0.001:
            lmtd = dt1
        else:
            lmtd = (dt2 - dt1) / math.log(dt2 / dt1)

        self._tracker.add_step(
            step_number=3,
            description="Calculate condenser LMTD (isothermal steam)",
            operation="lmtd_formula",
            inputs={
                "steam_temp_c": steam_temp_c,
                "cw_inlet_c": cw_inlet_c,
                "cw_outlet_c": cw_outlet_c,
                "dt1_ttd": dt1,
                "dt2_itd": dt2
            },
            output_value=lmtd,
            output_name="lmtd_c",
            formula="LMTD = (ITD - TTD) / ln(ITD/TTD)"
        )

        return lmtd

    def _calculate_ttd(self, steam_temp_c: float, cw_outlet_c: float) -> float:
        """Calculate Terminal Temperature Difference."""
        return steam_temp_c - cw_outlet_c

    def _calculate_itd(self, steam_temp_c: float, cw_inlet_c: float) -> float:
        """Calculate Initial Temperature Difference."""
        return steam_temp_c - cw_inlet_c

    def _calculate_actual_u_value(
        self,
        duty_w: float,
        area_m2: float,
        lmtd_c: float
    ) -> float:
        """
        Calculate actual overall heat transfer coefficient.

        Formula:
            U = Q / (A * LMTD)

        Args:
            duty_w: Heat duty (W)
            area_m2: Heat transfer area (m2)
            lmtd_c: Log Mean Temperature Difference (C)

        Returns:
            U-value (W/m2-K)
        """
        u_value = duty_w / (area_m2 * lmtd_c)

        self._tracker.add_step(
            step_number=4,
            description="Calculate actual U-value",
            operation="divide",
            inputs={
                "duty_w": duty_w,
                "area_m2": area_m2,
                "lmtd_c": lmtd_c
            },
            output_value=u_value,
            output_name="actual_u_value_w_m2k",
            formula="U = Q / (A * LMTD)"
        )

        return u_value

    def _get_hei_material_factor(self, tube_material: str) -> float:
        """Get HEI tube material correction factor."""
        material_lower = tube_material.lower().replace(" ", "_").replace("-", "_")

        # Try exact match first
        if material_lower in HEI_MATERIAL_FACTORS:
            return HEI_MATERIAL_FACTORS[material_lower]

        # Try partial matches
        for key, value in HEI_MATERIAL_FACTORS.items():
            if key in material_lower or material_lower in key:
                return value

        # Default to admiralty brass
        return 1.0

    def _get_hei_temp_factor(self, inlet_temp_c: float) -> float:
        """
        Get HEI inlet water temperature correction factor.

        Uses linear interpolation between tabulated values.
        """
        # Convert C to F for HEI table lookup
        inlet_temp_f = inlet_temp_c * 9/5 + 32

        temps = sorted(HEI_TEMP_CORRECTION.keys())

        if inlet_temp_f <= temps[0]:
            return HEI_TEMP_CORRECTION[temps[0]]
        if inlet_temp_f >= temps[-1]:
            return HEI_TEMP_CORRECTION[temps[-1]]

        # Linear interpolation
        for i in range(len(temps) - 1):
            if temps[i] <= inlet_temp_f <= temps[i + 1]:
                t1, t2 = temps[i], temps[i + 1]
                f1 = HEI_TEMP_CORRECTION[t1]
                f2 = HEI_TEMP_CORRECTION[t2]
                return f1 + (f2 - f1) * (inlet_temp_f - t1) / (t2 - t1)

        return 1.0

    def _calculate_cw_velocity(
        self,
        flow_rate_kg_s: float,
        tube_id_mm: float,
        num_tubes: int,
        num_passes: int
    ) -> float:
        """Calculate cooling water velocity in tubes."""
        # Water density at typical temperatures
        density = 995.0  # kg/m3

        # Volumetric flow rate
        vol_flow_m3_s = flow_rate_kg_s / density

        # Tube flow area
        tube_id_m = tube_id_mm / 1000.0
        tube_area_m2 = math.pi * (tube_id_m / 2) ** 2

        # Tubes per pass
        tubes_per_pass = num_tubes / num_passes

        # Total flow area
        total_flow_area = tubes_per_pass * tube_area_m2

        velocity = vol_flow_m3_s / total_flow_area

        return velocity

    def _get_hei_velocity_factor(self, velocity_m_s: float) -> float:
        """
        Get HEI velocity correction factor.

        Uses linear interpolation between tabulated values.
        """
        # Convert m/s to fps for HEI table
        velocity_fps = velocity_m_s * 3.28084

        velocities = sorted(HEI_VELOCITY_CORRECTION.keys())

        if velocity_fps <= velocities[0]:
            return HEI_VELOCITY_CORRECTION[velocities[0]]
        if velocity_fps >= velocities[-1]:
            return HEI_VELOCITY_CORRECTION[velocities[-1]]

        # Linear interpolation
        for i in range(len(velocities) - 1):
            if velocities[i] <= velocity_fps <= velocities[i + 1]:
                v1, v2 = velocities[i], velocities[i + 1]
                f1 = HEI_VELOCITY_CORRECTION[v1]
                f2 = HEI_VELOCITY_CORRECTION[v2]
                return f1 + (f2 - f1) * (velocity_fps - v1) / (v2 - v1)

        return 1.0

    def _calculate_hei_corrected_u(
        self,
        base_u: float,
        material_factor: float,
        temp_factor: float,
        velocity_factor: float
    ) -> float:
        """
        Calculate HEI corrected design U-value.

        Formula:
            U_corrected = U_base * Fm * Ft * Fv

        Args:
            base_u: Base design U-value (W/m2-K)
            material_factor: Material correction factor (Fm)
            temp_factor: Temperature correction factor (Ft)
            velocity_factor: Velocity correction factor (Fv)

        Returns:
            Corrected U-value (W/m2-K)
        """
        corrected_u = base_u * material_factor * temp_factor * velocity_factor

        self._tracker.add_step(
            step_number=6,
            description="Calculate HEI corrected design U-value",
            operation="multiply",
            inputs={
                "base_u_value_w_m2k": base_u,
                "material_factor_fm": material_factor,
                "temp_factor_ft": temp_factor,
                "velocity_factor_fv": velocity_factor
            },
            output_value=corrected_u,
            output_name="hei_corrected_u_value_w_m2k",
            formula="U_corrected = U_base * Fm * Ft * Fv"
        )

        return corrected_u

    def _calculate_cleanliness_factor(
        self,
        actual_u: float,
        design_u: float
    ) -> float:
        """
        Calculate cleanliness factor per HEI Standards.

        Formula:
            CF = U_actual / U_design

        A CF of 1.0 indicates clean tubes at design conditions.
        HEI recommends cleaning when CF falls below 0.85.

        Args:
            actual_u: Actual U-value (W/m2-K)
            design_u: Design (or HEI corrected) U-value (W/m2-K)

        Returns:
            Cleanliness factor (0-1, typically 0.6-1.0)
        """
        cf = actual_u / design_u
        cf = min(cf, 1.0)  # Cap at 1.0

        self._tracker.add_step(
            step_number=7,
            description="Calculate cleanliness factor per HEI Standards",
            operation="divide",
            inputs={
                "actual_u_value_w_m2k": actual_u,
                "design_u_value_w_m2k": design_u
            },
            output_value=cf,
            output_name="cleanliness_factor",
            formula="CF = U_actual / U_design (HEI method)"
        )

        return cf

    def _get_cf_status(self, cf: float) -> str:
        """Determine CF status category based on HEI thresholds."""
        if cf >= HEI_CLEANLINESS_THRESHOLDS["excellent"]:
            return "Excellent"
        elif cf >= HEI_CLEANLINESS_THRESHOLDS["good"]:
            return "Good"
        elif cf >= HEI_CLEANLINESS_THRESHOLDS["acceptable"]:
            return "Acceptable"
        elif cf >= HEI_CLEANLINESS_THRESHOLDS["marginal"]:
            return "Marginal"
        elif cf >= HEI_CLEANLINESS_THRESHOLDS["poor"]:
            return "Poor"
        else:
            return "Critical"

    def _analyze_cf_trend(
        self,
        current_cf: float,
        historical_data: Optional[List[float]]
    ) -> Tuple[float, int]:
        """
        Analyze cleanliness factor trend from historical data.

        Uses linear regression to determine CF degradation rate
        and predict days until threshold is reached.

        Args:
            current_cf: Current cleanliness factor
            historical_data: List of historical CF values (oldest to newest)

        Returns:
            Tuple of (slope per day, days to threshold)
        """
        threshold = HEI_CLEANLINESS_THRESHOLDS["acceptable"]  # 0.85

        if not historical_data or len(historical_data) < 2:
            # Default estimate based on typical fouling rates
            # Assume 0.001 CF loss per day as conservative estimate
            slope = -0.001
            if current_cf > threshold:
                days_to_threshold = int((current_cf - threshold) / abs(slope))
            else:
                days_to_threshold = 0

            self._tracker.add_step(
                step_number=8,
                description="Estimate CF trend (no historical data)",
                operation="default_estimate",
                inputs={
                    "current_cf": current_cf,
                    "threshold": threshold,
                    "estimated_slope": slope
                },
                output_value={"slope": slope, "days": days_to_threshold},
                output_name="cf_trend_analysis",
                formula="Estimated from typical fouling rates"
            )

            return slope, days_to_threshold

        # Linear regression on historical data
        # Assume data points are daily
        n = len(historical_data)
        x_sum = sum(range(n))
        y_sum = sum(historical_data)
        xy_sum = sum(i * y for i, y in enumerate(historical_data))
        x2_sum = sum(i * i for i in range(n))

        # Slope calculation (least squares)
        denominator = n * x2_sum - x_sum * x_sum
        if denominator == 0:
            slope = 0.0
        else:
            slope = (n * xy_sum - x_sum * y_sum) / denominator

        # Days to threshold
        if slope >= 0:
            # Not degrading
            days_to_threshold = 365  # Cap at 1 year
        elif current_cf <= threshold:
            days_to_threshold = 0
        else:
            days_to_threshold = int((current_cf - threshold) / abs(slope))
            days_to_threshold = min(days_to_threshold, 365)

        self._tracker.add_step(
            step_number=8,
            description="Analyze CF trend from historical data",
            operation="linear_regression",
            inputs={
                "data_points": n,
                "current_cf": current_cf,
                "threshold": threshold
            },
            output_value={"slope": slope, "days": days_to_threshold},
            output_name="cf_trend_analysis",
            formula="Linear regression: slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)"
        )

        return slope, days_to_threshold

    def _calculate_optimal_backpressure(
        self,
        cw_inlet_temp_c: float,
        cleanliness_factor: float
    ) -> float:
        """
        Calculate optimal condenser backpressure.

        Optimal backpressure depends on:
        - Cooling water inlet temperature
        - Condenser cleanliness

        Rule of thumb: Steam temp = CW inlet + TTD (typically 3-5C for clean condenser)
        Then convert steam temp to saturation pressure.

        Args:
            cw_inlet_temp_c: Cooling water inlet temperature (C)
            cleanliness_factor: Current cleanliness factor

        Returns:
            Optimal backpressure (mmHg abs)
        """
        # Achievable TTD depends on cleanliness
        base_ttd = 3.0  # Clean condenser TTD
        cf_penalty = (1.0 - cleanliness_factor) * 5.0  # Additional TTD due to fouling
        achievable_ttd = base_ttd + cf_penalty

        # Calculate typical CW temperature rise
        cw_rise = 8.0  # Typical value

        # Steam temperature for optimal operation
        steam_temp = cw_inlet_temp_c + cw_rise + achievable_ttd

        # Get saturation pressure (interpolate from table)
        pressure_kpa = self._get_saturation_pressure(steam_temp)
        optimal_backpressure = pressure_kpa * self.MMHG_PER_KPA

        self._tracker.add_step(
            step_number=10,
            description="Calculate optimal condenser backpressure",
            operation="saturation_lookup",
            inputs={
                "cw_inlet_temp_c": cw_inlet_temp_c,
                "cleanliness_factor": cleanliness_factor,
                "achievable_ttd_c": achievable_ttd,
                "steam_temp_c": steam_temp
            },
            output_value=optimal_backpressure,
            output_name="optimal_backpressure_mmhg",
            formula="T_steam = T_cw_in + Rise + TTD, then saturation pressure lookup"
        )

        return optimal_backpressure

    def _get_saturation_pressure(self, temp_c: float) -> float:
        """Get saturation pressure at given temperature using interpolation."""
        temps = sorted(STEAM_SATURATION.keys())

        if temp_c <= temps[0]:
            return STEAM_SATURATION[temps[0]]
        if temp_c >= temps[-1]:
            return STEAM_SATURATION[temps[-1]]

        for i in range(len(temps) - 1):
            if temps[i] <= temp_c <= temps[i + 1]:
                t1, t2 = temps[i], temps[i + 1]
                p1 = STEAM_SATURATION[t1]
                p2 = STEAM_SATURATION[t2]
                return p1 + (p2 - p1) * (temp_c - t1) / (t2 - t1)

        return STEAM_SATURATION[temps[0]]

    def _calculate_backpressure_penalty(
        self,
        actual_bp: float,
        optimal_bp: float,
        steam_flow_kg_s: float,
        turbine_type: str
    ) -> Tuple[float, float]:
        """
        Calculate power penalty from excess backpressure.

        Args:
            actual_bp: Actual backpressure (mmHg)
            optimal_bp: Optimal backpressure (mmHg)
            steam_flow_kg_s: Steam flow rate (kg/s)
            turbine_type: Turbine type for heat rate correction

        Returns:
            Tuple of (power penalty kW, heat rate impact kJ/kWh)
        """
        excess_bp = max(0, actual_bp - optimal_bp)

        # Get heat rate correction factor
        hr_correction = HEAT_RATE_CORRECTION_PER_MMHG.get(
            turbine_type.lower(),
            HEAT_RATE_CORRECTION_PER_MMHG["subcritical"]
        )

        heat_rate_impact = excess_bp * hr_correction

        # Estimate power output for penalty calculation
        # Typical: ~1 MW per kg/s of steam at LP exhaust
        estimated_power_mw = steam_flow_kg_s * 0.3

        # Power penalty (approximately proportional to heat rate change)
        # Heat rate increase of 1% reduces output by ~1%
        base_heat_rate = 9000  # kJ/kWh typical
        penalty_fraction = heat_rate_impact / base_heat_rate
        power_penalty_kw = estimated_power_mw * 1000 * penalty_fraction

        self._tracker.add_step(
            step_number=11,
            description="Calculate backpressure penalty",
            operation="penalty_calculation",
            inputs={
                "actual_backpressure_mmhg": actual_bp,
                "optimal_backpressure_mmhg": optimal_bp,
                "excess_backpressure_mmhg": excess_bp,
                "heat_rate_correction_kj_kwh_per_mmhg": hr_correction,
                "turbine_type": turbine_type
            },
            output_value={"penalty_kw": power_penalty_kw, "heat_rate_impact": heat_rate_impact},
            output_name="backpressure_penalty",
            formula="HR_impact = excess_BP * correction_factor"
        )

        return power_penalty_kw, heat_rate_impact

    def _calculate_annual_cost_impact(
        self,
        power_penalty_kw: float,
        electricity_price_usd_mwh: float
    ) -> float:
        """Calculate annual cost impact from performance degradation."""
        hours_per_year = 8000  # Typical capacity factor ~91%
        annual_cost = (power_penalty_kw / 1000) * electricity_price_usd_mwh * hours_per_year

        self._tracker.add_step(
            step_number=12,
            description="Calculate annual cost impact",
            operation="multiply",
            inputs={
                "power_penalty_kw": power_penalty_kw,
                "electricity_price_usd_mwh": electricity_price_usd_mwh,
                "operating_hours": hours_per_year
            },
            output_value=annual_cost,
            output_name="annual_cost_impact_usd",
            formula="Cost = (Penalty_MW) * Price * Hours"
        )

        return annual_cost

    def _determine_performance_status(
        self,
        cf: float,
        ttd_deviation_c: float,
        duty_deviation_pct: float
    ) -> str:
        """Determine overall performance status."""
        if cf >= 0.95 and abs(ttd_deviation_c) <= 1.0 and abs(duty_deviation_pct) <= 5:
            return PerformanceStatus.OPTIMAL.value
        elif cf >= 0.85 and abs(ttd_deviation_c) <= 2.0 and abs(duty_deviation_pct) <= 10:
            return PerformanceStatus.ACCEPTABLE.value
        elif cf >= 0.75:
            return PerformanceStatus.DEGRADED.value
        else:
            return PerformanceStatus.CRITICAL.value

    def _calculate_improvement_potential(
        self,
        current_cf: float,
        current_penalty_kw: float
    ) -> float:
        """
        Calculate potential improvement from cleaning.

        Assumes cleaning can restore CF to 0.95.
        """
        if current_cf >= 0.95:
            return 0.0

        target_cf = 0.95
        cf_improvement = target_cf - current_cf

        # Improvement is approximately proportional to CF improvement
        improvement_pct = (cf_improvement / current_cf) * 100

        self._tracker.add_step(
            step_number=14,
            description="Calculate improvement potential from cleaning",
            operation="improvement_estimate",
            inputs={
                "current_cf": current_cf,
                "target_cf": target_cf,
                "cf_improvement": cf_improvement
            },
            output_value=improvement_pct,
            output_name="improvement_potential_pct",
            formula="Improvement = (target_CF - current_CF) / current_CF * 100"
        )

        return improvement_pct


# =============================================================================
# STANDALONE CALCULATION FUNCTIONS
# =============================================================================

def calculate_hei_u_value(
    tube_gauge: str,
    tube_od_mm: float,
    material: str,
    inlet_temp_c: float,
    velocity_m_s: float
) -> float:
    """
    Calculate HEI design U-value for given conditions.

    Args:
        tube_gauge: Tube gauge (e.g., "18_BWG")
        tube_od_mm: Tube outside diameter (mm)
        material: Tube material
        inlet_temp_c: CW inlet temperature (C)
        velocity_m_s: CW velocity (m/s)

    Returns:
        HEI design U-value (W/m2-K)
    """
    # Get base U-value
    key = f"{tube_gauge}_{tube_od_mm:.1f}mm"
    base_u = HEI_BASE_U_VALUES.get(key, 3500.0)

    # Get correction factors
    material_lower = material.lower().replace(" ", "_")
    fm = HEI_MATERIAL_FACTORS.get(material_lower, 1.0)

    # Temperature factor
    inlet_temp_f = inlet_temp_c * 9/5 + 32
    temps = sorted(HEI_TEMP_CORRECTION.keys())
    if inlet_temp_f <= temps[0]:
        ft = HEI_TEMP_CORRECTION[temps[0]]
    elif inlet_temp_f >= temps[-1]:
        ft = HEI_TEMP_CORRECTION[temps[-1]]
    else:
        for i in range(len(temps) - 1):
            if temps[i] <= inlet_temp_f <= temps[i + 1]:
                t1, t2 = temps[i], temps[i + 1]
                f1, f2 = HEI_TEMP_CORRECTION[t1], HEI_TEMP_CORRECTION[t2]
                ft = f1 + (f2 - f1) * (inlet_temp_f - t1) / (t2 - t1)
                break

    # Velocity factor
    velocity_fps = velocity_m_s * 3.28084
    velocities = sorted(HEI_VELOCITY_CORRECTION.keys())
    if velocity_fps <= velocities[0]:
        fv = HEI_VELOCITY_CORRECTION[velocities[0]]
    elif velocity_fps >= velocities[-1]:
        fv = HEI_VELOCITY_CORRECTION[velocities[-1]]
    else:
        for i in range(len(velocities) - 1):
            if velocities[i] <= velocity_fps <= velocities[i + 1]:
                v1, v2 = velocities[i], velocities[i + 1]
                f1, f2 = HEI_VELOCITY_CORRECTION[v1], HEI_VELOCITY_CORRECTION[v2]
                fv = f1 + (f2 - f1) * (velocity_fps - v1) / (v2 - v1)
                break

    return base_u * fm * ft * fv


def calculate_condenser_duty(
    steam_flow_kg_s: float,
    steam_quality: float = 1.0,
    latent_heat_kj_kg: float = 2257.0
) -> float:
    """
    Calculate condenser design duty from steam flow.

    Args:
        steam_flow_kg_s: Steam flow rate (kg/s)
        steam_quality: Steam quality at inlet (0-1)
        latent_heat_kj_kg: Latent heat of vaporization (kJ/kg)

    Returns:
        Condenser duty (MW)
    """
    duty_kw = steam_flow_kg_s * steam_quality * latent_heat_kj_kg
    return duty_kw / 1000.0


def calculate_vacuum_from_steam_temp(steam_temp_c: float) -> float:
    """
    Calculate condenser vacuum from steam saturation temperature.

    Args:
        steam_temp_c: Steam saturation temperature (C)

    Returns:
        Absolute pressure (kPa)
    """
    temps = sorted(STEAM_SATURATION.keys())

    if steam_temp_c <= temps[0]:
        return STEAM_SATURATION[temps[0]]
    if steam_temp_c >= temps[-1]:
        return STEAM_SATURATION[temps[-1]]

    for i in range(len(temps) - 1):
        if temps[i] <= steam_temp_c <= temps[i + 1]:
            t1, t2 = temps[i], temps[i + 1]
            p1, p2 = STEAM_SATURATION[t1], STEAM_SATURATION[t2]
            return p1 + (p2 - p1) * (steam_temp_c - t1) / (t2 - t1)

    return STEAM_SATURATION[temps[0]]


def calculate_steam_temp_from_vacuum(pressure_kpa: float) -> float:
    """
    Calculate steam saturation temperature from condenser vacuum.

    Args:
        pressure_kpa: Absolute pressure (kPa)

    Returns:
        Steam saturation temperature (C)
    """
    # Inverse lookup from STEAM_SATURATION table
    for temp in sorted(STEAM_SATURATION.keys()):
        if STEAM_SATURATION[temp] >= pressure_kpa:
            if temp == sorted(STEAM_SATURATION.keys())[0]:
                return float(temp)

            # Interpolate
            prev_temp = temp - 2  # Assuming 2C increments
            if prev_temp in STEAM_SATURATION:
                p1 = STEAM_SATURATION[prev_temp]
                p2 = STEAM_SATURATION[temp]
                return prev_temp + (temp - prev_temp) * (pressure_kpa - p1) / (p2 - p1)
            return float(temp)

    return float(sorted(STEAM_SATURATION.keys())[-1])
