"""
GL-017 CONDENSYNC - Vacuum Calculator

Zero-hallucination, deterministic calculations for condenser vacuum
optimization following HEI Standards and ASME PTC 12.2.

This module provides:
- Condenser vacuum pressure optimization (mmHg abs)
- Saturation temperature at vacuum
- Air in-leakage rate calculation
- Non-condensable gas accumulation
- Steam jet air ejector (SJAE) capacity
- Vacuum pump sizing
- Backpressure impact on turbine efficiency

Standards Reference:
- HEI Standards for Steam Surface Condensers (11th Edition)
- ASME PTC 12.2 - Steam Surface Condensers Performance Test Code
- Heat Exchange Institute Air Removal Equipment Standards

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

# Standard atmospheric pressure
ATMOSPHERIC_PRESSURE_MBAR = 1013.25
ATMOSPHERIC_PRESSURE_MMHG = 760.0
ATMOSPHERIC_PRESSURE_KPA = 101.325
ATMOSPHERIC_PRESSURE_PSIA = 14.696

# Steam saturation properties (temperature C: pressure kPa)
# Based on IAPWS-IF97 steam tables
STEAM_SATURATION_PRESSURE = {
    20: 2.339,
    25: 3.169,
    30: 4.246,
    32: 4.759,
    34: 5.324,
    36: 5.947,
    38: 6.632,
    40: 7.384,
    42: 8.208,
    44: 9.111,
    46: 10.099,
    48: 11.177,
    50: 12.352,
    55: 15.758,
    60: 19.940,
}

# Steam saturation properties (pressure kPa: temperature C)
STEAM_SATURATION_TEMP = {v: k for k, v in STEAM_SATURATION_PRESSURE.items()}

# HEI Standard air leakage rates by condenser size (kg/hr)
# Based on shell volume
HEI_AIR_LEAKAGE_RATES = {
    "small": {"volume_m3_max": 100, "base_rate_kg_hr": 5.0},
    "medium": {"volume_m3_max": 500, "base_rate_kg_hr": 15.0},
    "large": {"volume_m3_max": 2000, "base_rate_kg_hr": 45.0},
    "very_large": {"volume_m3_max": float("inf"), "base_rate_kg_hr": 90.0},
}

# Turbine backpressure correction factors
# Format: {backpressure_mmhg: efficiency_loss_percent}
TURBINE_BACKPRESSURE_IMPACT = {
    25.4: 0.0,   # Design point (1" Hg abs)
    38.1: 0.5,   # 1.5" Hg abs
    50.8: 1.2,   # 2" Hg abs
    63.5: 2.0,   # 2.5" Hg abs
    76.2: 3.0,   # 3" Hg abs
    88.9: 4.2,   # 3.5" Hg abs
    101.6: 5.5,  # 4" Hg abs
    127.0: 8.5,  # 5" Hg abs
}


class VacuumUnit(Enum):
    """Unit systems for vacuum measurement."""
    MMHG_ABS = "mmHg_abs"
    MMHG_VAC = "mmHg_vacuum"
    MBAR_ABS = "mbar_abs"
    KPA_ABS = "kPa_abs"
    PSIA = "psia"
    INHG_ABS = "inHg_abs"


# =============================================================================
# INPUT/OUTPUT DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class VacuumInput:
    """
    Input parameters for vacuum calculations.

    Attributes:
        steam_temp_c: Steam saturation temperature in condenser (C)
        cw_inlet_temp_c: Cooling water inlet temperature (C)
        heat_duty_mw: Condenser heat duty (MW)
        shell_volume_m3: Condenser shell volume (m3)
        num_expansion_joints: Number of expansion joints
        num_valve_stems: Number of valve stems
        turbine_exhaust_area_m2: Turbine exhaust annulus area (m2)
        design_backpressure_mmhg: Design backpressure (mmHg abs)
        air_ejector_capacity_kg_hr: Air ejector design capacity (kg/hr)
    """
    steam_temp_c: float
    cw_inlet_temp_c: float
    heat_duty_mw: float
    shell_volume_m3: float
    num_expansion_joints: int
    num_valve_stems: int
    turbine_exhaust_area_m2: float
    design_backpressure_mmhg: float
    air_ejector_capacity_kg_hr: float


@dataclass(frozen=True)
class VacuumOutput:
    """
    Output results from vacuum calculations.

    Attributes:
        condenser_pressure_kpa: Condenser absolute pressure (kPa)
        condenser_pressure_mmhg: Condenser absolute pressure (mmHg abs)
        condenser_vacuum_mmhg: Condenser vacuum (mmHg below atm)
        saturation_temp_c: Saturation temperature at pressure (C)
        expected_air_leakage_kg_hr: Expected air in-leakage (kg/hr)
        ncg_accumulation_rate_kg_hr: Non-condensable gas rate (kg/hr)
        required_ejector_capacity_kg_hr: Required air removal capacity (kg/hr)
        ejector_utilization_pct: Air ejector capacity utilization (%)
        backpressure_deviation_mmhg: Deviation from design (mmHg)
        turbine_efficiency_loss_pct: Estimated turbine efficiency loss (%)
        optimal_vacuum_mmhg: Calculated optimal vacuum (mmHg abs)
        vacuum_margin_mmhg: Margin to optimal vacuum (mmHg)
    """
    condenser_pressure_kpa: float
    condenser_pressure_mmhg: float
    condenser_vacuum_mmhg: float
    saturation_temp_c: float
    expected_air_leakage_kg_hr: float
    ncg_accumulation_rate_kg_hr: float
    required_ejector_capacity_kg_hr: float
    ejector_utilization_pct: float
    backpressure_deviation_mmhg: float
    turbine_efficiency_loss_pct: float
    optimal_vacuum_mmhg: float
    vacuum_margin_mmhg: float


# =============================================================================
# VACUUM CALCULATOR CLASS
# =============================================================================

class VacuumCalculator:
    """
    Zero-hallucination vacuum calculator for steam condensers.

    Implements deterministic calculations following HEI Standards for
    vacuum system analysis and optimization. All calculations produce
    bit-perfect reproducible results with complete provenance tracking.

    Guarantees:
    - DETERMINISTIC: Same input always produces same output
    - REPRODUCIBLE: SHA-256 verified calculation chain
    - AUDITABLE: Complete step-by-step provenance trail
    - ZERO HALLUCINATION: No LLM in calculation path

    Example:
        >>> calculator = VacuumCalculator()
        >>> inputs = VacuumInput(
        ...     steam_temp_c=38.0,
        ...     cw_inlet_temp_c=25.0,
        ...     heat_duty_mw=200.0,
        ...     shell_volume_m3=500.0,
        ...     num_expansion_joints=4,
        ...     num_valve_stems=20,
        ...     turbine_exhaust_area_m2=15.0,
        ...     design_backpressure_mmhg=50.8,
        ...     air_ejector_capacity_kg_hr=50.0
        ... )
        >>> result, provenance = calculator.calculate(inputs)
        >>> print(f"Vacuum: {result.condenser_vacuum_mmhg:.1f} mmHg")
    """

    VERSION = "1.0.0"
    NAME = "VacuumCalculator"

    def __init__(self):
        """Initialize the vacuum calculator."""
        self._tracker: Optional[ProvenanceTracker] = None

    def calculate(
        self,
        inputs: VacuumInput
    ) -> Tuple[VacuumOutput, ProvenanceRecord]:
        """
        Perform complete vacuum system analysis.

        Args:
            inputs: VacuumInput with all required parameters

        Returns:
            Tuple of (VacuumOutput, ProvenanceRecord)

        Raises:
            ValueError: If inputs are invalid or out of range
        """
        # Initialize provenance tracking
        self._tracker = ProvenanceTracker(
            calculator_name=self.NAME,
            calculator_version=self.VERSION,
            metadata={
                "standards": ["HEI Standards", "ASME PTC 12.2"],
                "domain": "Condenser Vacuum Systems"
            }
        )

        # Set inputs for provenance
        input_dict = {
            "steam_temp_c": inputs.steam_temp_c,
            "cw_inlet_temp_c": inputs.cw_inlet_temp_c,
            "heat_duty_mw": inputs.heat_duty_mw,
            "shell_volume_m3": inputs.shell_volume_m3,
            "num_expansion_joints": inputs.num_expansion_joints,
            "num_valve_stems": inputs.num_valve_stems,
            "turbine_exhaust_area_m2": inputs.turbine_exhaust_area_m2,
            "design_backpressure_mmhg": inputs.design_backpressure_mmhg,
            "air_ejector_capacity_kg_hr": inputs.air_ejector_capacity_kg_hr
        }
        self._tracker.set_inputs(input_dict)

        # Validate inputs
        self._validate_inputs(inputs)

        # Step 1: Calculate condenser pressure from steam temperature
        pressure_kpa = self._calculate_saturation_pressure(inputs.steam_temp_c)
        pressure_mmhg = self._convert_kpa_to_mmhg(pressure_kpa)

        # Step 2: Calculate vacuum (below atmospheric)
        vacuum_mmhg = self._calculate_vacuum(pressure_mmhg)

        # Step 3: Calculate expected air in-leakage
        air_leakage = self._calculate_air_leakage(
            inputs.shell_volume_m3,
            inputs.num_expansion_joints,
            inputs.num_valve_stems,
            inputs.turbine_exhaust_area_m2
        )

        # Step 4: Calculate NCG accumulation rate
        ncg_rate = self._calculate_ncg_accumulation(
            air_leakage,
            inputs.heat_duty_mw
        )

        # Step 5: Calculate required ejector capacity
        required_capacity = self._calculate_required_ejector_capacity(
            ncg_rate,
            pressure_mmhg
        )

        # Step 6: Calculate ejector utilization
        ejector_utilization = self._calculate_ejector_utilization(
            required_capacity,
            inputs.air_ejector_capacity_kg_hr
        )

        # Step 7: Calculate backpressure deviation
        bp_deviation = self._calculate_backpressure_deviation(
            pressure_mmhg,
            inputs.design_backpressure_mmhg
        )

        # Step 8: Calculate turbine efficiency impact
        efficiency_loss = self._calculate_turbine_efficiency_loss(
            pressure_mmhg,
            inputs.design_backpressure_mmhg
        )

        # Step 9: Calculate optimal vacuum
        optimal_vacuum = self._calculate_optimal_vacuum(
            inputs.cw_inlet_temp_c,
            inputs.heat_duty_mw
        )

        # Step 10: Calculate vacuum margin
        vacuum_margin = self._calculate_vacuum_margin(
            pressure_mmhg,
            optimal_vacuum
        )

        # Create output
        output = VacuumOutput(
            condenser_pressure_kpa=round(pressure_kpa, 3),
            condenser_pressure_mmhg=round(pressure_mmhg, 2),
            condenser_vacuum_mmhg=round(vacuum_mmhg, 2),
            saturation_temp_c=round(inputs.steam_temp_c, 2),
            expected_air_leakage_kg_hr=round(air_leakage, 2),
            ncg_accumulation_rate_kg_hr=round(ncg_rate, 2),
            required_ejector_capacity_kg_hr=round(required_capacity, 2),
            ejector_utilization_pct=round(ejector_utilization, 1),
            backpressure_deviation_mmhg=round(bp_deviation, 2),
            turbine_efficiency_loss_pct=round(efficiency_loss, 3),
            optimal_vacuum_mmhg=round(optimal_vacuum, 2),
            vacuum_margin_mmhg=round(vacuum_margin, 2)
        )

        # Set outputs and finalize provenance
        self._tracker.set_outputs({
            "condenser_pressure_kpa": output.condenser_pressure_kpa,
            "condenser_pressure_mmhg": output.condenser_pressure_mmhg,
            "condenser_vacuum_mmhg": output.condenser_vacuum_mmhg,
            "saturation_temp_c": output.saturation_temp_c,
            "expected_air_leakage_kg_hr": output.expected_air_leakage_kg_hr,
            "ncg_accumulation_rate_kg_hr": output.ncg_accumulation_rate_kg_hr,
            "required_ejector_capacity_kg_hr": output.required_ejector_capacity_kg_hr,
            "ejector_utilization_pct": output.ejector_utilization_pct,
            "backpressure_deviation_mmhg": output.backpressure_deviation_mmhg,
            "turbine_efficiency_loss_pct": output.turbine_efficiency_loss_pct,
            "optimal_vacuum_mmhg": output.optimal_vacuum_mmhg,
            "vacuum_margin_mmhg": output.vacuum_margin_mmhg
        })

        provenance = self._tracker.finalize()
        return output, provenance

    def _validate_inputs(self, inputs: VacuumInput) -> None:
        """
        Validate input parameters.

        Raises:
            ValueError: If any input is invalid
        """
        if inputs.steam_temp_c < 20 or inputs.steam_temp_c > 60:
            raise ValueError(
                f"Steam temperature {inputs.steam_temp_c}C out of typical "
                f"condenser range (20-60C)"
            )

        if inputs.cw_inlet_temp_c < 0 or inputs.cw_inlet_temp_c > 45:
            raise ValueError(
                f"CW inlet temp {inputs.cw_inlet_temp_c}C out of range (0-45C)"
            )

        if inputs.steam_temp_c <= inputs.cw_inlet_temp_c:
            raise ValueError(
                "Steam temp must be greater than CW inlet temp"
            )

        if inputs.heat_duty_mw <= 0:
            raise ValueError("Heat duty must be positive")

        if inputs.shell_volume_m3 <= 0:
            raise ValueError("Shell volume must be positive")

        if inputs.design_backpressure_mmhg <= 0:
            raise ValueError("Design backpressure must be positive")

        if inputs.air_ejector_capacity_kg_hr <= 0:
            raise ValueError("Air ejector capacity must be positive")

    def _calculate_saturation_pressure(self, temp_c: float) -> float:
        """
        Calculate steam saturation pressure at given temperature.

        Uses Antoine equation for water:
            log10(P) = A - B / (C + T)

        Where P is in mmHg and T is in Celsius.
        Antoine coefficients for water (20-60C range):
            A = 8.07131, B = 1730.63, C = 233.426

        Args:
            temp_c: Temperature in Celsius

        Returns:
            Saturation pressure in kPa
        """
        # Antoine equation coefficients for water
        A = 8.07131
        B = 1730.63
        C = 233.426

        # Calculate pressure in mmHg
        log_p_mmhg = A - B / (C + temp_c)
        p_mmhg = 10 ** log_p_mmhg

        # Convert to kPa
        p_kpa = p_mmhg * 0.133322

        self._tracker.add_step(
            step_number=1,
            description="Calculate saturation pressure from temperature (Antoine equation)",
            operation="antoine_equation",
            inputs={
                "temp_c": temp_c,
                "antoine_A": A,
                "antoine_B": B,
                "antoine_C": C
            },
            output_value=p_kpa,
            output_name="pressure_kpa",
            formula="log10(P) = A - B/(C + T)"
        )

        return p_kpa

    def _convert_kpa_to_mmhg(self, pressure_kpa: float) -> float:
        """
        Convert pressure from kPa to mmHg absolute.

        Conversion: 1 kPa = 7.50062 mmHg

        Args:
            pressure_kpa: Pressure in kPa

        Returns:
            Pressure in mmHg absolute
        """
        conversion_factor = 7.50062
        pressure_mmhg = pressure_kpa * conversion_factor

        self._tracker.add_step(
            step_number=2,
            description="Convert pressure from kPa to mmHg",
            operation="unit_conversion",
            inputs={
                "pressure_kpa": pressure_kpa,
                "conversion_factor": conversion_factor
            },
            output_value=pressure_mmhg,
            output_name="pressure_mmhg",
            formula="P_mmHg = P_kPa * 7.50062"
        )

        return pressure_mmhg

    def _calculate_vacuum(self, pressure_mmhg_abs: float) -> float:
        """
        Calculate vacuum (pressure below atmospheric).

        Formula:
            Vacuum = P_atm - P_abs

        Args:
            pressure_mmhg_abs: Absolute pressure in mmHg

        Returns:
            Vacuum in mmHg (below atmospheric)
        """
        vacuum = ATMOSPHERIC_PRESSURE_MMHG - pressure_mmhg_abs

        self._tracker.add_step(
            step_number=3,
            description="Calculate vacuum (below atmospheric pressure)",
            operation="subtract",
            inputs={
                "atmospheric_pressure_mmhg": ATMOSPHERIC_PRESSURE_MMHG,
                "absolute_pressure_mmhg": pressure_mmhg_abs
            },
            output_value=vacuum,
            output_name="vacuum_mmhg",
            formula="Vacuum = P_atm - P_abs"
        )

        return vacuum

    def _calculate_air_leakage(
        self,
        shell_volume_m3: float,
        num_expansion_joints: int,
        num_valve_stems: int,
        exhaust_area_m2: float
    ) -> float:
        """
        Calculate expected air in-leakage per HEI Standards.

        HEI formula components:
        - Base leakage from shell (function of volume)
        - Leakage from expansion joints
        - Leakage from valve stems
        - Leakage from turbine exhaust seals

        Args:
            shell_volume_m3: Condenser shell volume (m3)
            num_expansion_joints: Number of expansion joints
            num_valve_stems: Number of valve stems
            exhaust_area_m2: Turbine exhaust area (m2)

        Returns:
            Expected air leakage (kg/hr)
        """
        # Determine base leakage from shell volume
        base_rate = 0.0
        for size, data in HEI_AIR_LEAKAGE_RATES.items():
            if shell_volume_m3 <= data["volume_m3_max"]:
                base_rate = data["base_rate_kg_hr"]
                break

        # Additional leakage sources (HEI typical values)
        expansion_joint_leakage = num_expansion_joints * 0.5  # kg/hr per joint
        valve_stem_leakage = num_valve_stems * 0.1  # kg/hr per stem
        exhaust_seal_leakage = exhaust_area_m2 * 0.3  # kg/hr per m2

        total_leakage = (
            base_rate +
            expansion_joint_leakage +
            valve_stem_leakage +
            exhaust_seal_leakage
        )

        self._tracker.add_step(
            step_number=4,
            description="Calculate expected air in-leakage (HEI method)",
            operation="air_leakage_sum",
            inputs={
                "shell_volume_m3": shell_volume_m3,
                "base_rate_kg_hr": base_rate,
                "num_expansion_joints": num_expansion_joints,
                "expansion_joint_leakage_kg_hr": expansion_joint_leakage,
                "num_valve_stems": num_valve_stems,
                "valve_stem_leakage_kg_hr": valve_stem_leakage,
                "exhaust_area_m2": exhaust_area_m2,
                "exhaust_seal_leakage_kg_hr": exhaust_seal_leakage
            },
            output_value=total_leakage,
            output_name="air_leakage_kg_hr",
            formula="L_total = L_base + L_joints + L_valves + L_exhaust"
        )

        return total_leakage

    def _calculate_ncg_accumulation(
        self,
        air_leakage_kg_hr: float,
        heat_duty_mw: float
    ) -> float:
        """
        Calculate non-condensable gas (NCG) accumulation rate.

        NCG sources:
        - Air in-leakage (primary source)
        - Dissolved gases from makeup water
        - Chemical treatment gases

        Empirical correlation:
            NCG_rate = Air_leakage * (1 + 0.1 * Q_MW / 100)

        Args:
            air_leakage_kg_hr: Air leakage rate (kg/hr)
            heat_duty_mw: Condenser heat duty (MW)

        Returns:
            NCG accumulation rate (kg/hr)
        """
        # Correction factor for dissolved gases (increases with load)
        dissolved_gas_factor = 1.0 + 0.1 * heat_duty_mw / 100.0

        ncg_rate = air_leakage_kg_hr * dissolved_gas_factor

        self._tracker.add_step(
            step_number=5,
            description="Calculate NCG accumulation rate",
            operation="ncg_rate",
            inputs={
                "air_leakage_kg_hr": air_leakage_kg_hr,
                "heat_duty_mw": heat_duty_mw,
                "dissolved_gas_factor": dissolved_gas_factor
            },
            output_value=ncg_rate,
            output_name="ncg_rate_kg_hr",
            formula="NCG = L_air * (1 + 0.1 * Q/100)"
        )

        return ncg_rate

    def _calculate_required_ejector_capacity(
        self,
        ncg_rate_kg_hr: float,
        suction_pressure_mmhg: float
    ) -> float:
        """
        Calculate required air ejector capacity.

        The required capacity depends on:
        - NCG generation rate
        - Suction pressure (lower pressure = higher required capacity)
        - Safety margin

        HEI recommends 2x safety margin on air removal capacity.

        Args:
            ncg_rate_kg_hr: NCG accumulation rate (kg/hr)
            suction_pressure_mmhg: Suction pressure (mmHg abs)

        Returns:
            Required ejector capacity (kg/hr)
        """
        # Pressure correction factor
        # At lower vacuum, ejector capacity decreases
        # Reference pressure: 50.8 mmHg (2" Hg abs)
        reference_pressure = 50.8
        pressure_factor = reference_pressure / suction_pressure_mmhg

        # Apply safety margin (HEI recommends 2x)
        safety_margin = 2.0

        required_capacity = ncg_rate_kg_hr * pressure_factor * safety_margin

        self._tracker.add_step(
            step_number=6,
            description="Calculate required air ejector capacity",
            operation="ejector_sizing",
            inputs={
                "ncg_rate_kg_hr": ncg_rate_kg_hr,
                "suction_pressure_mmhg": suction_pressure_mmhg,
                "reference_pressure_mmhg": reference_pressure,
                "pressure_factor": pressure_factor,
                "safety_margin": safety_margin
            },
            output_value=required_capacity,
            output_name="required_ejector_capacity_kg_hr",
            formula="Cap_req = NCG * (P_ref/P_suction) * SF"
        )

        return required_capacity

    def _calculate_ejector_utilization(
        self,
        required_capacity: float,
        installed_capacity: float
    ) -> float:
        """
        Calculate air ejector utilization percentage.

        Formula:
            Utilization = (Required / Installed) * 100

        Args:
            required_capacity: Required capacity (kg/hr)
            installed_capacity: Installed capacity (kg/hr)

        Returns:
            Utilization percentage
        """
        utilization = (required_capacity / installed_capacity) * 100.0

        self._tracker.add_step(
            step_number=7,
            description="Calculate air ejector utilization",
            operation="percentage",
            inputs={
                "required_capacity_kg_hr": required_capacity,
                "installed_capacity_kg_hr": installed_capacity
            },
            output_value=utilization,
            output_name="ejector_utilization_pct",
            formula="Util% = (Cap_req / Cap_inst) * 100"
        )

        return utilization

    def _calculate_backpressure_deviation(
        self,
        actual_pressure_mmhg: float,
        design_pressure_mmhg: float
    ) -> float:
        """
        Calculate deviation from design backpressure.

        Positive deviation indicates higher backpressure (worse performance).
        Negative deviation indicates lower backpressure (better performance).

        Args:
            actual_pressure_mmhg: Actual condenser pressure (mmHg abs)
            design_pressure_mmhg: Design condenser pressure (mmHg abs)

        Returns:
            Deviation in mmHg (positive = worse)
        """
        deviation = actual_pressure_mmhg - design_pressure_mmhg

        self._tracker.add_step(
            step_number=8,
            description="Calculate backpressure deviation from design",
            operation="subtract",
            inputs={
                "actual_pressure_mmhg": actual_pressure_mmhg,
                "design_pressure_mmhg": design_pressure_mmhg
            },
            output_value=deviation,
            output_name="backpressure_deviation_mmhg",
            formula="Dev = P_actual - P_design"
        )

        return deviation

    def _calculate_turbine_efficiency_loss(
        self,
        actual_pressure_mmhg: float,
        design_pressure_mmhg: float
    ) -> float:
        """
        Calculate turbine efficiency loss due to elevated backpressure.

        Uses interpolation of HEI/turbine manufacturer data.
        Typical impact: ~1% efficiency loss per 12.7 mmHg (0.5" Hg)
        increase in backpressure.

        Args:
            actual_pressure_mmhg: Actual condenser pressure (mmHg abs)
            design_pressure_mmhg: Design condenser pressure (mmHg abs)

        Returns:
            Efficiency loss percentage
        """
        # Calculate pressure increase
        pressure_increase = actual_pressure_mmhg - design_pressure_mmhg

        if pressure_increase <= 0:
            # Better than design - no loss
            efficiency_loss = 0.0
        else:
            # Approximately 1% loss per 12.7 mmHg increase
            # This is a typical correlation; actual value depends on turbine
            loss_rate_per_mmhg = 1.0 / 12.7
            efficiency_loss = pressure_increase * loss_rate_per_mmhg

        self._tracker.add_step(
            step_number=9,
            description="Calculate turbine efficiency loss from backpressure",
            operation="efficiency_correlation",
            inputs={
                "actual_pressure_mmhg": actual_pressure_mmhg,
                "design_pressure_mmhg": design_pressure_mmhg,
                "pressure_increase_mmhg": pressure_increase,
                "loss_rate_per_mmhg": 1.0 / 12.7
            },
            output_value=efficiency_loss,
            output_name="efficiency_loss_pct",
            formula="Loss% = max(0, (P_act - P_des) * 0.0787)"
        )

        return efficiency_loss

    def _calculate_optimal_vacuum(
        self,
        cw_inlet_temp_c: float,
        heat_duty_mw: float
    ) -> float:
        """
        Calculate optimal condenser vacuum for given conditions.

        The optimal vacuum balances:
        - Lower pressure = higher turbine efficiency
        - Lower pressure = higher auxiliary power for air removal
        - Lower pressure = higher cooling water requirement

        Empirical correlation for optimal backpressure:
            P_opt = P_sat(T_cw + TTD_opt)

        Where TTD_opt is typically 2-5C depending on cleanliness.

        Args:
            cw_inlet_temp_c: Cooling water inlet temperature (C)
            heat_duty_mw: Condenser heat duty (MW)

        Returns:
            Optimal vacuum pressure (mmHg abs)
        """
        # Optimal TTD depends on load and cleanliness
        # Typical range: 2-5C, use 3C as baseline
        optimal_ttd = 3.0

        # Estimate CW temperature rise (typical 8-12C)
        estimated_cw_rise = 10.0

        # Optimal steam temperature
        optimal_steam_temp = cw_inlet_temp_c + estimated_cw_rise + optimal_ttd

        # Calculate corresponding pressure
        optimal_pressure_kpa = self._get_saturation_pressure_for_step(
            optimal_steam_temp
        )
        optimal_pressure_mmhg = optimal_pressure_kpa * 7.50062

        self._tracker.add_step(
            step_number=10,
            description="Calculate optimal condenser vacuum",
            operation="optimal_vacuum",
            inputs={
                "cw_inlet_temp_c": cw_inlet_temp_c,
                "heat_duty_mw": heat_duty_mw,
                "optimal_ttd_c": optimal_ttd,
                "estimated_cw_rise_c": estimated_cw_rise,
                "optimal_steam_temp_c": optimal_steam_temp,
                "optimal_pressure_kpa": optimal_pressure_kpa
            },
            output_value=optimal_pressure_mmhg,
            output_name="optimal_vacuum_mmhg",
            formula="P_opt = P_sat(T_cw + dT_cw + TTD_opt)"
        )

        return optimal_pressure_mmhg

    def _calculate_vacuum_margin(
        self,
        actual_pressure_mmhg: float,
        optimal_pressure_mmhg: float
    ) -> float:
        """
        Calculate margin between actual and optimal vacuum.

        Negative margin indicates actual pressure is higher than optimal
        (room for improvement).

        Args:
            actual_pressure_mmhg: Actual pressure (mmHg abs)
            optimal_pressure_mmhg: Optimal pressure (mmHg abs)

        Returns:
            Vacuum margin (mmHg), positive = actual better than optimal
        """
        margin = optimal_pressure_mmhg - actual_pressure_mmhg

        self._tracker.add_step(
            step_number=11,
            description="Calculate vacuum margin to optimal",
            operation="subtract",
            inputs={
                "optimal_pressure_mmhg": optimal_pressure_mmhg,
                "actual_pressure_mmhg": actual_pressure_mmhg
            },
            output_value=margin,
            output_name="vacuum_margin_mmhg",
            formula="Margin = P_opt - P_actual"
        )

        return margin

    def _get_saturation_pressure_for_step(self, temp_c: float) -> float:
        """
        Get saturation pressure without adding provenance step.

        Used for intermediate calculations that are part of larger steps.

        Args:
            temp_c: Temperature in Celsius

        Returns:
            Saturation pressure in kPa
        """
        A = 8.07131
        B = 1730.63
        C = 233.426

        log_p_mmhg = A - B / (C + temp_c)
        p_mmhg = 10 ** log_p_mmhg
        p_kpa = p_mmhg * 0.133322

        return p_kpa


# =============================================================================
# STANDALONE CALCULATION FUNCTIONS
# =============================================================================

def convert_pressure_units(
    value: float,
    from_unit: VacuumUnit,
    to_unit: VacuumUnit
) -> float:
    """
    Convert pressure between different unit systems.

    Supported units:
    - mmHg_abs: millimeters of mercury (absolute)
    - mmHg_vacuum: millimeters of mercury (vacuum, below atm)
    - mbar_abs: millibar (absolute)
    - kPa_abs: kilopascal (absolute)
    - psia: pounds per square inch (absolute)
    - inHg_abs: inches of mercury (absolute)

    Args:
        value: Pressure value to convert
        from_unit: Source unit
        to_unit: Target unit

    Returns:
        Converted pressure value
    """
    # First convert to kPa (base unit)
    if from_unit == VacuumUnit.KPA_ABS:
        kpa_value = value
    elif from_unit == VacuumUnit.MMHG_ABS:
        kpa_value = value * 0.133322
    elif from_unit == VacuumUnit.MMHG_VAC:
        kpa_value = (ATMOSPHERIC_PRESSURE_MMHG - value) * 0.133322
    elif from_unit == VacuumUnit.MBAR_ABS:
        kpa_value = value * 0.1
    elif from_unit == VacuumUnit.PSIA:
        kpa_value = value * 6.89476
    elif from_unit == VacuumUnit.INHG_ABS:
        kpa_value = value * 3.38639
    else:
        raise ValueError(f"Unknown from_unit: {from_unit}")

    # Convert from kPa to target unit
    if to_unit == VacuumUnit.KPA_ABS:
        return kpa_value
    elif to_unit == VacuumUnit.MMHG_ABS:
        return kpa_value / 0.133322
    elif to_unit == VacuumUnit.MMHG_VAC:
        return ATMOSPHERIC_PRESSURE_MMHG - (kpa_value / 0.133322)
    elif to_unit == VacuumUnit.MBAR_ABS:
        return kpa_value * 10.0
    elif to_unit == VacuumUnit.PSIA:
        return kpa_value / 6.89476
    elif to_unit == VacuumUnit.INHG_ABS:
        return kpa_value / 3.38639
    else:
        raise ValueError(f"Unknown to_unit: {to_unit}")


def calculate_saturation_temperature(pressure_kpa: float) -> float:
    """
    Calculate steam saturation temperature at given pressure.

    Uses inverted Antoine equation.

    Args:
        pressure_kpa: Absolute pressure in kPa

    Returns:
        Saturation temperature in Celsius
    """
    # Convert to mmHg for Antoine equation
    p_mmhg = pressure_kpa / 0.133322

    # Antoine coefficients
    A = 8.07131
    B = 1730.63
    C = 233.426

    # Invert Antoine equation: T = B / (A - log10(P)) - C
    temp_c = B / (A - math.log10(p_mmhg)) - C

    return temp_c


def calculate_air_density_at_vacuum(
    pressure_kpa: float,
    temperature_c: float
) -> float:
    """
    Calculate air density at vacuum conditions using ideal gas law.

    Formula:
        rho = P / (R * T)

    Where R = 287 J/(kg-K) for air

    Args:
        pressure_kpa: Absolute pressure (kPa)
        temperature_c: Temperature (C)

    Returns:
        Air density (kg/m3)
    """
    R_air = 287.0  # J/(kg-K)
    pressure_pa = pressure_kpa * 1000
    temperature_k = temperature_c + 273.15

    density = pressure_pa / (R_air * temperature_k)

    return density


def calculate_sjae_steam_consumption(
    air_capacity_kg_hr: float,
    suction_pressure_mmhg: float,
    motive_steam_pressure_bar: float
) -> float:
    """
    Calculate steam consumption for Steam Jet Air Ejector (SJAE).

    Empirical correlation based on HEI standards.

    Args:
        air_capacity_kg_hr: Air removal capacity (kg/hr)
        suction_pressure_mmhg: Suction pressure (mmHg abs)
        motive_steam_pressure_bar: Motive steam pressure (bar gauge)

    Returns:
        Steam consumption (kg/hr)
    """
    # Base steam ratio (kg steam per kg air) at reference conditions
    # Reference: 50.8 mmHg suction, 7 bar motive steam
    base_steam_ratio = 8.0

    # Suction pressure correction (lower pressure = more steam)
    pressure_correction = 50.8 / suction_pressure_mmhg

    # Motive steam pressure correction (higher pressure = less steam)
    motive_correction = 7.0 / motive_steam_pressure_bar

    actual_ratio = base_steam_ratio * pressure_correction * motive_correction

    steam_consumption = air_capacity_kg_hr * actual_ratio

    return steam_consumption


def calculate_vacuum_pump_power(
    air_capacity_m3_hr: float,
    suction_pressure_kpa: float,
    discharge_pressure_kpa: float,
    efficiency: float = 0.7
) -> float:
    """
    Calculate vacuum pump power requirement.

    Uses isothermal compression formula:
        W = (P1 * V1 * ln(P2/P1)) / eta

    Args:
        air_capacity_m3_hr: Volumetric capacity (m3/hr at suction)
        suction_pressure_kpa: Suction pressure (kPa abs)
        discharge_pressure_kpa: Discharge pressure (kPa abs)
        efficiency: Pump efficiency (default 0.7)

    Returns:
        Power requirement (kW)
    """
    # Convert to SI units
    suction_pa = suction_pressure_kpa * 1000
    discharge_pa = discharge_pressure_kpa * 1000
    volume_m3_s = air_capacity_m3_hr / 3600

    # Compression ratio
    compression_ratio = discharge_pa / suction_pa

    # Isothermal work
    work_w = suction_pa * volume_m3_s * math.log(compression_ratio)

    # Account for efficiency
    power_w = work_w / efficiency
    power_kw = power_w / 1000

    return power_kw
