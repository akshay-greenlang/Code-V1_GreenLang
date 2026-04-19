"""
GL-017 CONDENSYNC - Heat Transfer Calculator

Zero-hallucination, deterministic calculations for condenser heat transfer
analysis following HEI Standards and ASME PTC 12.2.

This module provides:
- Overall heat transfer coefficient (U-value) calculation
- Log Mean Temperature Difference (LMTD) calculation
- Heat duty calculation (Q = U x A x LMTD)
- Terminal Temperature Difference (TTD) calculation
- Cleanliness factor calculation
- Fouling resistance calculation (Rf)
- Heat transfer area effectiveness

Standards Reference:
- HEI Standards for Steam Surface Condensers (11th Edition)
- ASME PTC 12.2 - Steam Surface Condensers Performance Test Code
- TEMA (Tubular Exchanger Manufacturers Association) Standards

Author: GL-CalculatorEngineer
Version: 1.0.0
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, Optional, Tuple, Union
import math

from .provenance import ProvenanceTracker, ProvenanceRecord


# =============================================================================
# CONSTANTS AND REFERENCE DATA
# =============================================================================

# Standard tube materials and their thermal conductivity (W/m-K)
TUBE_THERMAL_CONDUCTIVITY = {
    "admiralty_brass": 111.0,
    "aluminum_brass": 100.0,
    "copper_nickel_90_10": 45.0,
    "copper_nickel_70_30": 29.0,
    "stainless_steel_304": 16.3,
    "stainless_steel_316": 16.3,
    "titanium": 21.9,
    "carbon_steel": 50.0,
}

# Standard tube gauges and wall thickness (mm)
TUBE_WALL_THICKNESS = {
    "18_BWG": 1.245,
    "20_BWG": 0.889,
    "22_BWG": 0.711,
    "24_BWG": 0.559,
}

# Water thermal properties at various temperatures
# Format: {temp_C: {"density": kg/m3, "cp": J/kg-K, "k": W/m-K, "mu": Pa-s}}
WATER_PROPERTIES = {
    15: {"density": 999.1, "cp": 4186, "k": 0.589, "mu": 0.001139},
    20: {"density": 998.2, "cp": 4182, "k": 0.598, "mu": 0.001002},
    25: {"density": 997.0, "cp": 4180, "k": 0.607, "mu": 0.000890},
    30: {"density": 995.6, "cp": 4178, "k": 0.615, "mu": 0.000798},
    35: {"density": 994.0, "cp": 4178, "k": 0.623, "mu": 0.000720},
    40: {"density": 992.2, "cp": 4179, "k": 0.631, "mu": 0.000653},
    45: {"density": 990.2, "cp": 4180, "k": 0.637, "mu": 0.000596},
    50: {"density": 988.0, "cp": 4181, "k": 0.644, "mu": 0.000547},
}

# HEI correction factors for tube count per pass
HEI_TUBE_COUNT_FACTOR = {
    1: 1.000,
    2: 0.990,
    3: 0.980,
    4: 0.970,
}


class UnitSystem(Enum):
    """Unit system for calculations."""
    SI = "SI"
    IMPERIAL = "Imperial"


# =============================================================================
# INPUT/OUTPUT DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class HeatTransferInput:
    """
    Input parameters for heat transfer calculations.

    Attributes:
        steam_temp_c: Steam saturation temperature (Celsius)
        cw_inlet_temp_c: Cooling water inlet temperature (Celsius)
        cw_outlet_temp_c: Cooling water outlet temperature (Celsius)
        cw_flow_rate_kg_s: Cooling water mass flow rate (kg/s)
        heat_transfer_area_m2: Total heat transfer surface area (m2)
        tube_od_mm: Tube outside diameter (mm)
        tube_id_mm: Tube inside diameter (mm)
        tube_length_m: Effective tube length (m)
        tube_material: Tube material name (from TUBE_THERMAL_CONDUCTIVITY)
        num_tubes: Total number of tubes
        num_passes: Number of cooling water passes
        design_u_value_w_m2k: Design overall heat transfer coefficient (W/m2-K)
    """
    steam_temp_c: float
    cw_inlet_temp_c: float
    cw_outlet_temp_c: float
    cw_flow_rate_kg_s: float
    heat_transfer_area_m2: float
    tube_od_mm: float
    tube_id_mm: float
    tube_length_m: float
    tube_material: str
    num_tubes: int
    num_passes: int
    design_u_value_w_m2k: float


@dataclass(frozen=True)
class HeatTransferOutput:
    """
    Output results from heat transfer calculations.

    Attributes:
        lmtd_k: Log Mean Temperature Difference (K)
        heat_duty_w: Heat duty (W)
        heat_duty_mw: Heat duty (MW)
        actual_u_value_w_m2k: Actual overall heat transfer coefficient (W/m2-K)
        ttd_c: Terminal Temperature Difference (Celsius)
        cleanliness_factor: Cleanliness factor (ratio to design)
        fouling_resistance_m2k_w: Fouling resistance (m2-K/W)
        effectiveness: Heat exchanger effectiveness (0-1)
        cw_velocity_m_s: Cooling water tube velocity (m/s)
        reynolds_number: Reynolds number for water flow
        nusselt_number: Nusselt number
        waterside_htc_w_m2k: Waterside heat transfer coefficient (W/m2-K)
        tubeside_htc_w_m2k: Tubeside (steam side) heat transfer coefficient (W/m2-K)
        tube_wall_resistance_m2k_w: Tube wall thermal resistance (m2-K/W)
    """
    lmtd_k: float
    heat_duty_w: float
    heat_duty_mw: float
    actual_u_value_w_m2k: float
    ttd_c: float
    cleanliness_factor: float
    fouling_resistance_m2k_w: float
    effectiveness: float
    cw_velocity_m_s: float
    reynolds_number: float
    nusselt_number: float
    waterside_htc_w_m2k: float
    tubeside_htc_w_m2k: float
    tube_wall_resistance_m2k_w: float


# =============================================================================
# HEAT TRANSFER CALCULATOR CLASS
# =============================================================================

class HeatTransferCalculator:
    """
    Zero-hallucination heat transfer calculator for steam condensers.

    Implements deterministic calculations following HEI Standards and
    ASME PTC 12.2 for surface condensers. All calculations produce
    bit-perfect reproducible results with complete provenance tracking.

    Guarantees:
    - DETERMINISTIC: Same input always produces same output
    - REPRODUCIBLE: SHA-256 verified calculation chain
    - AUDITABLE: Complete step-by-step provenance trail
    - ZERO HALLUCINATION: No LLM in calculation path

    Example:
        >>> calculator = HeatTransferCalculator()
        >>> inputs = HeatTransferInput(
        ...     steam_temp_c=40.0,
        ...     cw_inlet_temp_c=25.0,
        ...     cw_outlet_temp_c=35.0,
        ...     cw_flow_rate_kg_s=1000.0,
        ...     heat_transfer_area_m2=5000.0,
        ...     tube_od_mm=25.4,
        ...     tube_id_mm=22.9,
        ...     tube_length_m=10.0,
        ...     tube_material="admiralty_brass",
        ...     num_tubes=3000,
        ...     num_passes=2,
        ...     design_u_value_w_m2k=3500.0
        ... )
        >>> result, provenance = calculator.calculate(inputs)
        >>> print(f"Heat Duty: {result.heat_duty_mw:.2f} MW")
    """

    VERSION = "1.0.0"
    NAME = "HeatTransferCalculator"

    def __init__(self):
        """Initialize the heat transfer calculator."""
        self._tracker: Optional[ProvenanceTracker] = None

    def calculate(
        self,
        inputs: HeatTransferInput
    ) -> Tuple[HeatTransferOutput, ProvenanceRecord]:
        """
        Perform complete heat transfer analysis.

        Args:
            inputs: HeatTransferInput with all required parameters

        Returns:
            Tuple of (HeatTransferOutput, ProvenanceRecord)

        Raises:
            ValueError: If inputs are invalid or out of range
        """
        # Initialize provenance tracking
        self._tracker = ProvenanceTracker(
            calculator_name=self.NAME,
            calculator_version=self.VERSION,
            metadata={
                "standards": ["HEI Standards", "ASME PTC 12.2"],
                "domain": "Steam Condenser Heat Transfer"
            }
        )

        # Set inputs for provenance
        input_dict = {
            "steam_temp_c": inputs.steam_temp_c,
            "cw_inlet_temp_c": inputs.cw_inlet_temp_c,
            "cw_outlet_temp_c": inputs.cw_outlet_temp_c,
            "cw_flow_rate_kg_s": inputs.cw_flow_rate_kg_s,
            "heat_transfer_area_m2": inputs.heat_transfer_area_m2,
            "tube_od_mm": inputs.tube_od_mm,
            "tube_id_mm": inputs.tube_id_mm,
            "tube_length_m": inputs.tube_length_m,
            "tube_material": inputs.tube_material,
            "num_tubes": inputs.num_tubes,
            "num_passes": inputs.num_passes,
            "design_u_value_w_m2k": inputs.design_u_value_w_m2k
        }
        self._tracker.set_inputs(input_dict)

        # Validate inputs
        self._validate_inputs(inputs)

        # Step 1: Calculate LMTD
        lmtd_k = self._calculate_lmtd(
            inputs.steam_temp_c,
            inputs.cw_inlet_temp_c,
            inputs.cw_outlet_temp_c
        )

        # Step 2: Calculate heat duty from water side
        heat_duty_w = self._calculate_heat_duty_waterside(
            inputs.cw_flow_rate_kg_s,
            inputs.cw_inlet_temp_c,
            inputs.cw_outlet_temp_c
        )
        heat_duty_mw = heat_duty_w / 1_000_000.0

        # Step 3: Calculate actual U-value
        actual_u_value = self._calculate_actual_u_value(
            heat_duty_w,
            inputs.heat_transfer_area_m2,
            lmtd_k
        )

        # Step 4: Calculate TTD
        ttd_c = self._calculate_ttd(
            inputs.steam_temp_c,
            inputs.cw_outlet_temp_c
        )

        # Step 5: Calculate cleanliness factor
        cleanliness_factor = self._calculate_cleanliness_factor(
            actual_u_value,
            inputs.design_u_value_w_m2k
        )

        # Step 6: Calculate water velocity in tubes
        cw_velocity = self._calculate_water_velocity(
            inputs.cw_flow_rate_kg_s,
            inputs.tube_id_mm,
            inputs.num_tubes,
            inputs.num_passes,
            inputs.cw_inlet_temp_c,
            inputs.cw_outlet_temp_c
        )

        # Step 7: Calculate Reynolds number
        reynolds = self._calculate_reynolds_number(
            cw_velocity,
            inputs.tube_id_mm,
            inputs.cw_inlet_temp_c,
            inputs.cw_outlet_temp_c
        )

        # Step 8: Calculate Nusselt number using Dittus-Boelter
        nusselt = self._calculate_nusselt_number(reynolds)

        # Step 9: Calculate waterside heat transfer coefficient
        waterside_htc = self._calculate_waterside_htc(
            nusselt,
            inputs.tube_id_mm,
            inputs.cw_inlet_temp_c,
            inputs.cw_outlet_temp_c
        )

        # Step 10: Calculate tube wall resistance
        tube_wall_resistance = self._calculate_tube_wall_resistance(
            inputs.tube_od_mm,
            inputs.tube_id_mm,
            inputs.tube_material
        )

        # Step 11: Estimate tubeside (steam side) HTC using HEI correlation
        tubeside_htc = self._calculate_tubeside_htc(inputs.steam_temp_c)

        # Step 12: Calculate fouling resistance
        fouling_resistance = self._calculate_fouling_resistance(
            actual_u_value,
            waterside_htc,
            tubeside_htc,
            tube_wall_resistance,
            inputs.tube_od_mm,
            inputs.tube_id_mm
        )

        # Step 13: Calculate effectiveness
        effectiveness = self._calculate_effectiveness(
            inputs.steam_temp_c,
            inputs.cw_inlet_temp_c,
            inputs.cw_outlet_temp_c
        )

        # Create output
        output = HeatTransferOutput(
            lmtd_k=round(lmtd_k, 3),
            heat_duty_w=round(heat_duty_w, 1),
            heat_duty_mw=round(heat_duty_mw, 3),
            actual_u_value_w_m2k=round(actual_u_value, 1),
            ttd_c=round(ttd_c, 2),
            cleanliness_factor=round(cleanliness_factor, 4),
            fouling_resistance_m2k_w=round(fouling_resistance, 6),
            effectiveness=round(effectiveness, 4),
            cw_velocity_m_s=round(cw_velocity, 3),
            reynolds_number=round(reynolds, 0),
            nusselt_number=round(nusselt, 1),
            waterside_htc_w_m2k=round(waterside_htc, 1),
            tubeside_htc_w_m2k=round(tubeside_htc, 1),
            tube_wall_resistance_m2k_w=round(tube_wall_resistance, 7)
        )

        # Set outputs and finalize provenance
        self._tracker.set_outputs({
            "lmtd_k": output.lmtd_k,
            "heat_duty_w": output.heat_duty_w,
            "heat_duty_mw": output.heat_duty_mw,
            "actual_u_value_w_m2k": output.actual_u_value_w_m2k,
            "ttd_c": output.ttd_c,
            "cleanliness_factor": output.cleanliness_factor,
            "fouling_resistance_m2k_w": output.fouling_resistance_m2k_w,
            "effectiveness": output.effectiveness,
            "cw_velocity_m_s": output.cw_velocity_m_s,
            "reynolds_number": output.reynolds_number,
            "nusselt_number": output.nusselt_number,
            "waterside_htc_w_m2k": output.waterside_htc_w_m2k,
            "tubeside_htc_w_m2k": output.tubeside_htc_w_m2k,
            "tube_wall_resistance_m2k_w": output.tube_wall_resistance_m2k_w
        })

        provenance = self._tracker.finalize()
        return output, provenance

    def _validate_inputs(self, inputs: HeatTransferInput) -> None:
        """
        Validate input parameters.

        Raises:
            ValueError: If any input is invalid
        """
        # Temperature validations
        if inputs.steam_temp_c < 20 or inputs.steam_temp_c > 100:
            raise ValueError(
                f"Steam temperature {inputs.steam_temp_c}C out of range (20-100C)"
            )

        if inputs.cw_inlet_temp_c < 0 or inputs.cw_inlet_temp_c > 50:
            raise ValueError(
                f"CW inlet temp {inputs.cw_inlet_temp_c}C out of range (0-50C)"
            )

        if inputs.cw_outlet_temp_c <= inputs.cw_inlet_temp_c:
            raise ValueError(
                "CW outlet temp must be greater than inlet temp"
            )

        if inputs.steam_temp_c <= inputs.cw_outlet_temp_c:
            raise ValueError(
                "Steam temp must be greater than CW outlet temp"
            )

        # Flow rate validation
        if inputs.cw_flow_rate_kg_s <= 0:
            raise ValueError("CW flow rate must be positive")

        # Tube geometry validation
        if inputs.tube_id_mm >= inputs.tube_od_mm:
            raise ValueError("Tube ID must be less than OD")

        if inputs.tube_material not in TUBE_THERMAL_CONDUCTIVITY:
            raise ValueError(
                f"Unknown tube material: {inputs.tube_material}. "
                f"Valid options: {list(TUBE_THERMAL_CONDUCTIVITY.keys())}"
            )

    def _calculate_lmtd(
        self,
        steam_temp_c: float,
        cw_inlet_c: float,
        cw_outlet_c: float
    ) -> float:
        """
        Calculate Log Mean Temperature Difference (LMTD).

        Formula:
            LMTD = (dT1 - dT2) / ln(dT1/dT2)

        Where:
            dT1 = T_steam - T_cw_outlet (cold end)
            dT2 = T_steam - T_cw_inlet (hot end)

        For condenser, steam temperature is essentially constant.

        Args:
            steam_temp_c: Steam saturation temperature (C)
            cw_inlet_c: Cooling water inlet temperature (C)
            cw_outlet_c: Cooling water outlet temperature (C)

        Returns:
            LMTD in Kelvin (same as Celsius difference)
        """
        dt1 = steam_temp_c - cw_outlet_c  # Cold end (TTD)
        dt2 = steam_temp_c - cw_inlet_c   # Hot end (ITD)

        # Handle edge case where dt1 equals dt2
        if abs(dt1 - dt2) < 0.001:
            lmtd = dt1
        else:
            lmtd = (dt1 - dt2) / math.log(dt1 / dt2)

        self._tracker.add_step(
            step_number=1,
            description="Calculate Log Mean Temperature Difference (LMTD)",
            operation="lmtd_formula",
            inputs={
                "steam_temp_c": steam_temp_c,
                "cw_inlet_c": cw_inlet_c,
                "cw_outlet_c": cw_outlet_c,
                "dt1_cold_end": dt1,
                "dt2_hot_end": dt2
            },
            output_value=lmtd,
            output_name="lmtd_k",
            formula="LMTD = (dT1 - dT2) / ln(dT1/dT2)"
        )

        return lmtd

    def _calculate_heat_duty_waterside(
        self,
        flow_rate_kg_s: float,
        inlet_temp_c: float,
        outlet_temp_c: float
    ) -> float:
        """
        Calculate heat duty from water side energy balance.

        Formula:
            Q = m_dot * Cp * (T_out - T_in)

        Args:
            flow_rate_kg_s: Mass flow rate (kg/s)
            inlet_temp_c: Inlet temperature (C)
            outlet_temp_c: Outlet temperature (C)

        Returns:
            Heat duty in Watts
        """
        avg_temp = (inlet_temp_c + outlet_temp_c) / 2
        cp = self._get_water_cp(avg_temp)

        delta_t = outlet_temp_c - inlet_temp_c
        heat_duty = flow_rate_kg_s * cp * delta_t

        self._tracker.add_step(
            step_number=2,
            description="Calculate heat duty from water side energy balance",
            operation="multiply",
            inputs={
                "flow_rate_kg_s": flow_rate_kg_s,
                "cp_j_kg_k": cp,
                "delta_t_k": delta_t
            },
            output_value=heat_duty,
            output_name="heat_duty_w",
            formula="Q = m_dot * Cp * dT"
        )

        return heat_duty

    def _calculate_actual_u_value(
        self,
        heat_duty_w: float,
        area_m2: float,
        lmtd_k: float
    ) -> float:
        """
        Calculate actual overall heat transfer coefficient.

        Formula:
            U = Q / (A * LMTD)

        This is derived from the fundamental heat exchanger equation:
            Q = U * A * LMTD

        Args:
            heat_duty_w: Heat duty (W)
            area_m2: Heat transfer area (m2)
            lmtd_k: Log Mean Temperature Difference (K)

        Returns:
            Overall heat transfer coefficient (W/m2-K)
        """
        u_value = heat_duty_w / (area_m2 * lmtd_k)

        self._tracker.add_step(
            step_number=3,
            description="Calculate actual overall heat transfer coefficient (U-value)",
            operation="divide",
            inputs={
                "heat_duty_w": heat_duty_w,
                "area_m2": area_m2,
                "lmtd_k": lmtd_k
            },
            output_value=u_value,
            output_name="actual_u_value_w_m2k",
            formula="U = Q / (A * LMTD)"
        )

        return u_value

    def _calculate_ttd(
        self,
        steam_temp_c: float,
        cw_outlet_c: float
    ) -> float:
        """
        Calculate Terminal Temperature Difference (TTD).

        TTD is the temperature difference between steam and cooling
        water at the cold end of the condenser. It is a key performance
        indicator per HEI Standards.

        Formula:
            TTD = T_steam - T_cw_outlet

        Args:
            steam_temp_c: Steam saturation temperature (C)
            cw_outlet_c: Cooling water outlet temperature (C)

        Returns:
            TTD in Celsius
        """
        ttd = steam_temp_c - cw_outlet_c

        self._tracker.add_step(
            step_number=4,
            description="Calculate Terminal Temperature Difference (TTD)",
            operation="subtract",
            inputs={
                "steam_temp_c": steam_temp_c,
                "cw_outlet_c": cw_outlet_c
            },
            output_value=ttd,
            output_name="ttd_c",
            formula="TTD = T_steam - T_cw_outlet"
        )

        return ttd

    def _calculate_cleanliness_factor(
        self,
        actual_u: float,
        design_u: float
    ) -> float:
        """
        Calculate cleanliness factor (CF).

        Cleanliness factor is the ratio of actual U-value to design
        U-value. Per HEI Standards, a CF < 0.85 typically indicates
        the need for tube cleaning.

        Formula:
            CF = U_actual / U_design

        Args:
            actual_u: Actual U-value (W/m2-K)
            design_u: Design U-value (W/m2-K)

        Returns:
            Cleanliness factor (0-1, typically 0.7-1.0)
        """
        cf = actual_u / design_u

        self._tracker.add_step(
            step_number=5,
            description="Calculate cleanliness factor",
            operation="divide",
            inputs={
                "actual_u_w_m2k": actual_u,
                "design_u_w_m2k": design_u
            },
            output_value=cf,
            output_name="cleanliness_factor",
            formula="CF = U_actual / U_design"
        )

        return cf

    def _calculate_water_velocity(
        self,
        flow_rate_kg_s: float,
        tube_id_mm: float,
        num_tubes: int,
        num_passes: int,
        inlet_temp_c: float,
        outlet_temp_c: float
    ) -> float:
        """
        Calculate cooling water velocity in tubes.

        Formula:
            v = Q_vol / (N_tubes_per_pass * A_tube)
            A_tube = pi * (ID/2)^2

        Args:
            flow_rate_kg_s: Mass flow rate (kg/s)
            tube_id_mm: Tube inside diameter (mm)
            num_tubes: Total number of tubes
            num_passes: Number of passes
            inlet_temp_c: Inlet temperature (C)
            outlet_temp_c: Outlet temperature (C)

        Returns:
            Water velocity (m/s)
        """
        avg_temp = (inlet_temp_c + outlet_temp_c) / 2
        density = self._get_water_density(avg_temp)

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

        self._tracker.add_step(
            step_number=6,
            description="Calculate cooling water velocity in tubes",
            operation="velocity_calculation",
            inputs={
                "flow_rate_kg_s": flow_rate_kg_s,
                "vol_flow_m3_s": vol_flow_m3_s,
                "tube_id_mm": tube_id_mm,
                "num_tubes": num_tubes,
                "num_passes": num_passes,
                "tubes_per_pass": tubes_per_pass,
                "total_flow_area_m2": total_flow_area
            },
            output_value=velocity,
            output_name="cw_velocity_m_s",
            formula="v = Q_vol / (N_tubes/N_passes * pi * (ID/2)^2)"
        )

        return velocity

    def _calculate_reynolds_number(
        self,
        velocity_m_s: float,
        tube_id_mm: float,
        inlet_temp_c: float,
        outlet_temp_c: float
    ) -> float:
        """
        Calculate Reynolds number for tube flow.

        Formula:
            Re = (rho * v * D) / mu

        Args:
            velocity_m_s: Water velocity (m/s)
            tube_id_mm: Tube inside diameter (mm)
            inlet_temp_c: Inlet temperature (C)
            outlet_temp_c: Outlet temperature (C)

        Returns:
            Reynolds number (dimensionless)
        """
        avg_temp = (inlet_temp_c + outlet_temp_c) / 2
        density = self._get_water_density(avg_temp)
        viscosity = self._get_water_viscosity(avg_temp)

        tube_id_m = tube_id_mm / 1000.0
        reynolds = (density * velocity_m_s * tube_id_m) / viscosity

        self._tracker.add_step(
            step_number=7,
            description="Calculate Reynolds number",
            operation="reynolds_formula",
            inputs={
                "density_kg_m3": density,
                "velocity_m_s": velocity_m_s,
                "tube_id_m": tube_id_m,
                "viscosity_pa_s": viscosity
            },
            output_value=reynolds,
            output_name="reynolds_number",
            formula="Re = (rho * v * D) / mu"
        )

        return reynolds

    def _calculate_nusselt_number(self, reynolds: float) -> float:
        """
        Calculate Nusselt number using Dittus-Boelter correlation.

        For turbulent flow in tubes (Re > 10000):
            Nu = 0.023 * Re^0.8 * Pr^0.4

        For water, Prandtl number is approximately 6.0 at typical
        condenser operating temperatures.

        Args:
            reynolds: Reynolds number

        Returns:
            Nusselt number
        """
        # Prandtl number for water (typical range)
        prandtl = 6.0

        if reynolds < 10000:
            # Laminar or transitional flow - use simplified correlation
            nusselt = 3.66 + 0.0668 * (reynolds * prandtl / 100) ** 0.33
        else:
            # Dittus-Boelter correlation for turbulent flow
            nusselt = 0.023 * (reynolds ** 0.8) * (prandtl ** 0.4)

        self._tracker.add_step(
            step_number=8,
            description="Calculate Nusselt number (Dittus-Boelter)",
            operation="dittus_boelter",
            inputs={
                "reynolds": reynolds,
                "prandtl": prandtl
            },
            output_value=nusselt,
            output_name="nusselt_number",
            formula="Nu = 0.023 * Re^0.8 * Pr^0.4"
        )

        return nusselt

    def _calculate_waterside_htc(
        self,
        nusselt: float,
        tube_id_mm: float,
        inlet_temp_c: float,
        outlet_temp_c: float
    ) -> float:
        """
        Calculate waterside heat transfer coefficient.

        Formula:
            h = Nu * k / D

        Args:
            nusselt: Nusselt number
            tube_id_mm: Tube inside diameter (mm)
            inlet_temp_c: Inlet temperature (C)
            outlet_temp_c: Outlet temperature (C)

        Returns:
            Heat transfer coefficient (W/m2-K)
        """
        avg_temp = (inlet_temp_c + outlet_temp_c) / 2
        k_water = self._get_water_thermal_conductivity(avg_temp)
        tube_id_m = tube_id_mm / 1000.0

        htc = nusselt * k_water / tube_id_m

        self._tracker.add_step(
            step_number=9,
            description="Calculate waterside heat transfer coefficient",
            operation="htc_from_nusselt",
            inputs={
                "nusselt": nusselt,
                "k_water_w_mk": k_water,
                "tube_id_m": tube_id_m
            },
            output_value=htc,
            output_name="waterside_htc_w_m2k",
            formula="h_water = Nu * k / D"
        )

        return htc

    def _calculate_tube_wall_resistance(
        self,
        tube_od_mm: float,
        tube_id_mm: float,
        tube_material: str
    ) -> float:
        """
        Calculate tube wall thermal resistance.

        For cylindrical tube wall:
            R_wall = ln(OD/ID) / (2 * pi * k * L)

        Per unit area (based on OD):
            R_wall_per_area = (OD/2) * ln(OD/ID) / k

        Args:
            tube_od_mm: Tube outside diameter (mm)
            tube_id_mm: Tube inside diameter (mm)
            tube_material: Tube material name

        Returns:
            Tube wall resistance (m2-K/W)
        """
        k_tube = TUBE_THERMAL_CONDUCTIVITY[tube_material]
        od_m = tube_od_mm / 1000.0
        id_m = tube_id_mm / 1000.0

        # Resistance based on outside area
        r_wall = (od_m / 2) * math.log(od_m / id_m) / k_tube

        self._tracker.add_step(
            step_number=10,
            description="Calculate tube wall thermal resistance",
            operation="cylindrical_wall_resistance",
            inputs={
                "tube_od_m": od_m,
                "tube_id_m": id_m,
                "k_tube_w_mk": k_tube,
                "tube_material": tube_material
            },
            output_value=r_wall,
            output_name="tube_wall_resistance_m2k_w",
            formula="R_wall = (OD/2) * ln(OD/ID) / k"
        )

        return r_wall

    def _calculate_tubeside_htc(self, steam_temp_c: float) -> float:
        """
        Calculate tubeside (steam side) heat transfer coefficient.

        Uses HEI correlation for film condensation on horizontal tubes.
        Typical values range from 10,000 to 15,000 W/m2-K for steam
        condensing on clean tubes.

        Args:
            steam_temp_c: Steam saturation temperature (C)

        Returns:
            Steam-side HTC (W/m2-K)
        """
        # HEI typical correlation - steam side HTC varies with
        # condensate loading and tube arrangement
        # Base value for clean tubes with moderate loading
        base_htc = 12000.0

        # Temperature correction (higher temp = slightly higher HTC)
        temp_factor = 1.0 + 0.005 * (steam_temp_c - 40.0)
        htc = base_htc * temp_factor

        self._tracker.add_step(
            step_number=11,
            description="Calculate steam-side heat transfer coefficient (HEI method)",
            operation="hei_steamside_htc",
            inputs={
                "steam_temp_c": steam_temp_c,
                "base_htc": base_htc,
                "temp_factor": temp_factor
            },
            output_value=htc,
            output_name="tubeside_htc_w_m2k",
            formula="h_steam = h_base * (1 + 0.005 * (T - 40))"
        )

        return htc

    def _calculate_fouling_resistance(
        self,
        actual_u: float,
        waterside_htc: float,
        steamside_htc: float,
        tube_wall_r: float,
        tube_od_mm: float,
        tube_id_mm: float
    ) -> float:
        """
        Calculate fouling resistance from measured U-value.

        From the overall resistance equation:
            1/U = 1/h_steam + R_wall + R_fouling + (OD/ID)/h_water

        Solving for R_fouling:
            R_fouling = 1/U - 1/h_steam - R_wall - (OD/ID)/h_water

        Args:
            actual_u: Measured U-value (W/m2-K)
            waterside_htc: Water-side HTC (W/m2-K)
            steamside_htc: Steam-side HTC (W/m2-K)
            tube_wall_r: Tube wall resistance (m2-K/W)
            tube_od_mm: Tube OD (mm)
            tube_id_mm: Tube ID (mm)

        Returns:
            Fouling resistance (m2-K/W)
        """
        od_id_ratio = tube_od_mm / tube_id_mm

        # Total measured resistance
        total_r = 1.0 / actual_u

        # Clean resistances (referred to outside area)
        steam_r = 1.0 / steamside_htc
        water_r = od_id_ratio / waterside_htc

        # Fouling resistance
        fouling_r = total_r - steam_r - tube_wall_r - water_r

        # Fouling resistance cannot be negative
        fouling_r = max(0.0, fouling_r)

        self._tracker.add_step(
            step_number=12,
            description="Calculate fouling resistance from measured U-value",
            operation="fouling_from_overall_r",
            inputs={
                "total_resistance": total_r,
                "steam_resistance": steam_r,
                "tube_wall_resistance": tube_wall_r,
                "water_resistance": water_r,
                "od_id_ratio": od_id_ratio
            },
            output_value=fouling_r,
            output_name="fouling_resistance_m2k_w",
            formula="R_f = 1/U - 1/h_s - R_wall - (OD/ID)/h_w"
        )

        return fouling_r

    def _calculate_effectiveness(
        self,
        steam_temp_c: float,
        cw_inlet_c: float,
        cw_outlet_c: float
    ) -> float:
        """
        Calculate heat exchanger effectiveness.

        For a condenser (one fluid at constant temperature):
            effectiveness = (T_cw_out - T_cw_in) / (T_steam - T_cw_in)

        This represents the ratio of actual heat transfer to maximum
        possible heat transfer.

        Args:
            steam_temp_c: Steam temperature (C)
            cw_inlet_c: CW inlet temperature (C)
            cw_outlet_c: CW outlet temperature (C)

        Returns:
            Effectiveness (0-1)
        """
        actual_delta_t = cw_outlet_c - cw_inlet_c
        max_delta_t = steam_temp_c - cw_inlet_c
        effectiveness = actual_delta_t / max_delta_t

        self._tracker.add_step(
            step_number=13,
            description="Calculate heat exchanger effectiveness",
            operation="effectiveness_ratio",
            inputs={
                "cw_outlet_c": cw_outlet_c,
                "cw_inlet_c": cw_inlet_c,
                "steam_temp_c": steam_temp_c,
                "actual_delta_t": actual_delta_t,
                "max_delta_t": max_delta_t
            },
            output_value=effectiveness,
            output_name="effectiveness",
            formula="epsilon = (T_out - T_in) / (T_steam - T_in)"
        )

        return effectiveness

    # =========================================================================
    # WATER PROPERTY LOOKUP METHODS
    # =========================================================================

    def _get_water_property(self, temp_c: float, property_name: str) -> float:
        """
        Get water property at given temperature using linear interpolation.

        Args:
            temp_c: Temperature (C)
            property_name: Property to retrieve

        Returns:
            Property value at temperature
        """
        temps = sorted(WATER_PROPERTIES.keys())

        # Clamp to available range
        if temp_c <= temps[0]:
            return WATER_PROPERTIES[temps[0]][property_name]
        if temp_c >= temps[-1]:
            return WATER_PROPERTIES[temps[-1]][property_name]

        # Find bracketing temperatures
        for i in range(len(temps) - 1):
            if temps[i] <= temp_c <= temps[i + 1]:
                t1, t2 = temps[i], temps[i + 1]
                p1 = WATER_PROPERTIES[t1][property_name]
                p2 = WATER_PROPERTIES[t2][property_name]
                # Linear interpolation
                return p1 + (p2 - p1) * (temp_c - t1) / (t2 - t1)

        return WATER_PROPERTIES[temps[0]][property_name]

    def _get_water_density(self, temp_c: float) -> float:
        """Get water density at temperature (kg/m3)."""
        return self._get_water_property(temp_c, "density")

    def _get_water_cp(self, temp_c: float) -> float:
        """Get water specific heat at temperature (J/kg-K)."""
        return self._get_water_property(temp_c, "cp")

    def _get_water_thermal_conductivity(self, temp_c: float) -> float:
        """Get water thermal conductivity at temperature (W/m-K)."""
        return self._get_water_property(temp_c, "k")

    def _get_water_viscosity(self, temp_c: float) -> float:
        """Get water dynamic viscosity at temperature (Pa-s)."""
        return self._get_water_property(temp_c, "mu")


# =============================================================================
# STANDALONE CALCULATION FUNCTIONS
# =============================================================================

def calculate_lmtd(
    t_hot_in: float,
    t_hot_out: float,
    t_cold_in: float,
    t_cold_out: float,
    flow_arrangement: str = "counterflow"
) -> float:
    """
    Calculate Log Mean Temperature Difference for any heat exchanger.

    For counterflow:
        dT1 = T_hot_in - T_cold_out
        dT2 = T_hot_out - T_cold_in

    For parallel flow:
        dT1 = T_hot_in - T_cold_in
        dT2 = T_hot_out - T_cold_out

    Args:
        t_hot_in: Hot fluid inlet temperature
        t_hot_out: Hot fluid outlet temperature
        t_cold_in: Cold fluid inlet temperature
        t_cold_out: Cold fluid outlet temperature
        flow_arrangement: "counterflow" or "parallel"

    Returns:
        LMTD value
    """
    if flow_arrangement == "counterflow":
        dt1 = t_hot_in - t_cold_out
        dt2 = t_hot_out - t_cold_in
    else:  # parallel flow
        dt1 = t_hot_in - t_cold_in
        dt2 = t_hot_out - t_cold_out

    if dt1 <= 0 or dt2 <= 0:
        raise ValueError("Invalid temperatures: approach temps must be positive")

    if abs(dt1 - dt2) < 0.001:
        return dt1

    return (dt1 - dt2) / math.log(dt1 / dt2)


def calculate_heat_duty(
    u_value: float,
    area: float,
    lmtd: float
) -> float:
    """
    Calculate heat duty using fundamental heat exchanger equation.

    Formula: Q = U * A * LMTD

    Args:
        u_value: Overall heat transfer coefficient (W/m2-K)
        area: Heat transfer area (m2)
        lmtd: Log Mean Temperature Difference (K)

    Returns:
        Heat duty (W)
    """
    if u_value <= 0 or area <= 0 or lmtd <= 0:
        raise ValueError("All parameters must be positive")

    return u_value * area * lmtd


def calculate_ntu(
    u_value: float,
    area: float,
    c_min: float
) -> float:
    """
    Calculate Number of Transfer Units (NTU).

    Formula: NTU = U * A / C_min

    Where C_min is the minimum heat capacity rate (W/K).

    Args:
        u_value: Overall heat transfer coefficient (W/m2-K)
        area: Heat transfer area (m2)
        c_min: Minimum heat capacity rate (W/K)

    Returns:
        NTU (dimensionless)
    """
    if c_min <= 0:
        raise ValueError("C_min must be positive")

    return u_value * area / c_min
