# -*- coding: utf-8 -*-
"""
GL-007 FurnaceOptimizer - Heat Transfer Calculations

This module provides ZERO-HALLUCINATION heat transfer calculations for
industrial furnace optimization. All calculations use deterministic
engineering formulas with complete provenance tracking.

Key Calculations:
    - Radiant heat transfer (Stefan-Boltzmann)
    - Convective heat transfer
    - Log Mean Temperature Difference (LMTD)
    - Overall heat transfer coefficient
    - Wall and radiation losses
    - Fouling factor estimation

Engineering References:
    - ASME PTC 4: Fired Steam Generators
    - API 560: Fired Heaters for General Refinery Service
    - Hottel & Sarofim: Radiative Transfer
    - Kern: Process Heat Transfer

Example:
    >>> from greenlang.agents.process_heat.gl_007_furnace_optimizer.heat_transfer import (
    ...     FurnaceHeatTransfer,
    ... )
    >>> calculator = FurnaceHeatTransfer()
    >>> result = calculator.calculate_radiant_heat_transfer(
    ...     gas_temp_f=2500,
    ...     surface_temp_f=800,
    ...     area_ft2=500,
    ...     emissivity=0.85,
    ... )

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import logging
import math
import uuid

from greenlang.agents.process_heat.gl_007_furnace_optimizer.schemas import (
    HeatTransferAnalysis,
)
from greenlang.agents.process_heat.gl_007_furnace_optimizer.provenance import (
    ProvenanceTracker,
    generate_provenance_hash,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - HEAT TRANSFER ENGINEERING DATA
# =============================================================================

class HeatTransferConstants:
    """
    Heat transfer engineering constants - DETERMINISTIC.

    All values from ASME, API 560, and standard references.
    """

    # Stefan-Boltzmann constant
    STEFAN_BOLTZMANN_BTU_HR_FT2_R4 = 0.1714e-8  # Btu/hr-ft2-R^4

    # Temperature conversions
    RANKINE_OFFSET = 459.67  # R = F + 459.67

    # Typical emissivities
    EMISSIVITY_REFRACTORY = 0.85
    EMISSIVITY_STEEL_OXIDIZED = 0.80
    EMISSIVITY_STEEL_CLEAN = 0.25
    EMISSIVITY_FLAME = 0.35

    # Thermal conductivities (Btu/hr-ft-F)
    K_REFRACTORY = 0.5
    K_INSULATING_BRICK = 0.15
    K_MINERAL_WOOL = 0.03
    K_CARBON_STEEL = 26.0
    K_STAINLESS_STEEL = 9.4

    # Typical convective heat transfer coefficients (Btu/hr-ft2-F)
    HTC_NATURAL_CONVECTION_AIR = 1.0
    HTC_FORCED_CONVECTION_AIR = 5.0
    HTC_FLUE_GAS = 3.0
    HTC_PROCESS_GAS = 10.0
    HTC_BOILING_WATER = 500.0
    HTC_CONDENSING_STEAM = 1000.0

    # Fouling factors (hr-ft2-F/Btu)
    FOULING_CLEAN = 0.0001
    FOULING_LIGHT = 0.001
    FOULING_MODERATE = 0.002
    FOULING_HEAVY = 0.005
    FOULING_SEVERE = 0.01


# =============================================================================
# FURNACE HEAT TRANSFER CALCULATOR
# =============================================================================

class FurnaceHeatTransfer:
    """
    Zero-hallucination heat transfer calculator for furnaces.

    All calculations are DETERMINISTIC using engineering formulas from:
    - ASME PTC 4 (Fired Steam Generators)
    - API 560 (Fired Heaters)
    - Hottel & Sarofim (Radiative Transfer)

    Features:
    - Radiant heat transfer (Stefan-Boltzmann law)
    - Convective heat transfer
    - LMTD calculations
    - Overall HTC with fouling
    - Wall loss calculations
    - Complete SHA-256 provenance tracking

    Example:
        >>> calculator = FurnaceHeatTransfer()
        >>> q_radiant = calculator.calculate_radiant_heat_transfer(
        ...     gas_temp_f=2500,
        ...     surface_temp_f=800,
        ...     area_ft2=500,
        ... )
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        provenance_enabled: bool = True,
        precision: int = 4,
    ) -> None:
        """
        Initialize heat transfer calculator.

        Args:
            provenance_enabled: Enable SHA-256 provenance tracking
            precision: Decimal precision for calculations
        """
        self.provenance_enabled = provenance_enabled
        self.precision = precision
        self._provenance_tracker = ProvenanceTracker() if provenance_enabled else None

        logger.info(f"FurnaceHeatTransfer initialized v{self.VERSION}")

    # =========================================================================
    # RADIANT HEAT TRANSFER - DETERMINISTIC
    # =========================================================================

    def calculate_radiant_heat_transfer(
        self,
        gas_temp_f: float,
        surface_temp_f: float,
        area_ft2: float,
        emissivity: float = 0.85,
        view_factor: float = 1.0,
    ) -> Dict[str, float]:
        """
        Calculate radiant heat transfer - DETERMINISTIC.

        Uses Stefan-Boltzmann Law:
            Q = sigma * epsilon * F * A * (T_hot^4 - T_cold^4)

        Where:
            - sigma = Stefan-Boltzmann constant (0.1714e-8 Btu/hr-ft2-R^4)
            - epsilon = effective emissivity
            - F = view factor
            - A = area (ft2)
            - T = absolute temperatures (Rankine)

        Args:
            gas_temp_f: Hot gas temperature (F)
            surface_temp_f: Receiving surface temperature (F)
            area_ft2: Heat transfer area (ft2)
            emissivity: Effective emissivity (0-1)
            view_factor: Geometric view factor (0-1)

        Returns:
            Dictionary with heat transfer results
        """
        # Convert to Rankine - DETERMINISTIC
        t_gas_r = gas_temp_f + HeatTransferConstants.RANKINE_OFFSET
        t_surface_r = surface_temp_f + HeatTransferConstants.RANKINE_OFFSET

        # Stefan-Boltzmann calculation - DETERMINISTIC
        sigma = HeatTransferConstants.STEFAN_BOLTZMANN_BTU_HR_FT2_R4

        # Q = sigma * epsilon * F * A * (T_hot^4 - T_cold^4)
        q_radiant_btu_hr = (
            sigma * emissivity * view_factor * area_ft2 *
            (t_gas_r ** 4 - t_surface_r ** 4)
        )

        # Calculate equivalent radiant HTC
        delta_t = gas_temp_f - surface_temp_f
        if delta_t > 0:
            htc_radiant = q_radiant_btu_hr / (area_ft2 * delta_t)
        else:
            htc_radiant = 0.0

        # Heat flux
        heat_flux_btu_hr_ft2 = q_radiant_btu_hr / area_ft2 if area_ft2 > 0 else 0

        result = {
            "q_radiant_btu_hr": round(q_radiant_btu_hr, self.precision),
            "q_radiant_mmbtu_hr": round(q_radiant_btu_hr / 1e6, self.precision),
            "htc_radiant_btu_hr_ft2_f": round(htc_radiant, self.precision),
            "heat_flux_btu_hr_ft2": round(heat_flux_btu_hr_ft2, self.precision),
            "gas_temp_r": round(t_gas_r, 2),
            "surface_temp_r": round(t_surface_r, 2),
            "emissivity": emissivity,
            "view_factor": view_factor,
        }

        logger.debug(
            f"Radiant heat transfer: Q={q_radiant_btu_hr/1e6:.2f} MMBtu/hr, "
            f"HTC={htc_radiant:.2f} Btu/hr-ft2-F"
        )

        return result

    # =========================================================================
    # CONVECTIVE HEAT TRANSFER - DETERMINISTIC
    # =========================================================================

    def calculate_convective_heat_transfer(
        self,
        gas_temp_f: float,
        surface_temp_f: float,
        area_ft2: float,
        htc_btu_hr_ft2_f: float = 5.0,
    ) -> Dict[str, float]:
        """
        Calculate convective heat transfer - DETERMINISTIC.

        Uses Newton's Law of Cooling:
            Q = h * A * (T_gas - T_surface)

        Args:
            gas_temp_f: Hot gas temperature (F)
            surface_temp_f: Surface temperature (F)
            area_ft2: Heat transfer area (ft2)
            htc_btu_hr_ft2_f: Convective heat transfer coefficient

        Returns:
            Dictionary with heat transfer results
        """
        # Temperature difference - DETERMINISTIC
        delta_t = gas_temp_f - surface_temp_f

        # Newton's Law of Cooling - DETERMINISTIC
        q_convective_btu_hr = htc_btu_hr_ft2_f * area_ft2 * delta_t

        # Heat flux
        heat_flux_btu_hr_ft2 = q_convective_btu_hr / area_ft2 if area_ft2 > 0 else 0

        result = {
            "q_convective_btu_hr": round(q_convective_btu_hr, self.precision),
            "q_convective_mmbtu_hr": round(q_convective_btu_hr / 1e6, self.precision),
            "htc_convective_btu_hr_ft2_f": htc_btu_hr_ft2_f,
            "heat_flux_btu_hr_ft2": round(heat_flux_btu_hr_ft2, self.precision),
            "delta_t_f": round(delta_t, 2),
        }

        logger.debug(
            f"Convective heat transfer: Q={q_convective_btu_hr/1e6:.4f} MMBtu/hr"
        )

        return result

    # =========================================================================
    # LMTD CALCULATIONS - DETERMINISTIC
    # =========================================================================

    def calculate_lmtd(
        self,
        t_hot_in_f: float,
        t_hot_out_f: float,
        t_cold_in_f: float,
        t_cold_out_f: float,
        flow_arrangement: str = "counterflow",
    ) -> Dict[str, float]:
        """
        Calculate Log Mean Temperature Difference - DETERMINISTIC.

        Formula (counterflow):
            LMTD = (dT1 - dT2) / ln(dT1 / dT2)

        Where:
            - dT1 = T_hot_in - T_cold_out
            - dT2 = T_hot_out - T_cold_in

        Args:
            t_hot_in_f: Hot fluid inlet temperature (F)
            t_hot_out_f: Hot fluid outlet temperature (F)
            t_cold_in_f: Cold fluid inlet temperature (F)
            t_cold_out_f: Cold fluid outlet temperature (F)
            flow_arrangement: "counterflow" or "parallel"

        Returns:
            Dictionary with LMTD and related values
        """
        if flow_arrangement == "counterflow":
            # Counterflow: hot in vs cold out, hot out vs cold in
            dt1 = t_hot_in_f - t_cold_out_f
            dt2 = t_hot_out_f - t_cold_in_f
        else:
            # Parallel flow: hot in vs cold in, hot out vs cold out
            dt1 = t_hot_in_f - t_cold_in_f
            dt2 = t_hot_out_f - t_cold_out_f

        # Handle edge cases - DETERMINISTIC
        if dt1 <= 0 or dt2 <= 0:
            logger.warning(f"Invalid temperature crosses: dT1={dt1}, dT2={dt2}")
            return {
                "lmtd_f": 0.0,
                "dt1_f": round(dt1, 2),
                "dt2_f": round(dt2, 2),
                "valid": False,
                "error": "Temperature cross detected",
            }

        # LMTD calculation - DETERMINISTIC
        if abs(dt1 - dt2) < 0.1:
            # When dT1 ~ dT2, LMTD ~ arithmetic mean
            lmtd = (dt1 + dt2) / 2
        else:
            lmtd = (dt1 - dt2) / math.log(dt1 / dt2)

        # Approach temperatures
        approach_hot = t_hot_out_f - t_cold_in_f
        approach_cold = t_hot_in_f - t_cold_out_f

        result = {
            "lmtd_f": round(lmtd, self.precision),
            "dt1_f": round(dt1, 2),
            "dt2_f": round(dt2, 2),
            "approach_hot_end_f": round(approach_hot, 2),
            "approach_cold_end_f": round(approach_cold, 2),
            "flow_arrangement": flow_arrangement,
            "valid": True,
        }

        logger.debug(f"LMTD calculated: {lmtd:.1f}F ({flow_arrangement})")

        return result

    # =========================================================================
    # OVERALL HEAT TRANSFER COEFFICIENT - DETERMINISTIC
    # =========================================================================

    def calculate_overall_htc(
        self,
        htc_inside_btu_hr_ft2_f: float,
        htc_outside_btu_hr_ft2_f: float,
        wall_thickness_in: float = 0.25,
        wall_conductivity_btu_hr_ft_f: float = 26.0,
        fouling_inside: float = 0.001,
        fouling_outside: float = 0.001,
        area_ratio: float = 1.0,
    ) -> Dict[str, float]:
        """
        Calculate overall heat transfer coefficient - DETERMINISTIC.

        Formula (resistance analogy):
            1/U = 1/h_i + R_fi + (t/k) + R_fo + 1/h_o

        Args:
            htc_inside_btu_hr_ft2_f: Inside (process) HTC
            htc_outside_btu_hr_ft2_f: Outside (flue gas) HTC
            wall_thickness_in: Wall thickness (inches)
            wall_conductivity_btu_hr_ft_f: Wall thermal conductivity
            fouling_inside: Inside fouling factor (hr-ft2-F/Btu)
            fouling_outside: Outside fouling factor (hr-ft2-F/Btu)
            area_ratio: Outside to inside area ratio

        Returns:
            Dictionary with overall HTC and resistances
        """
        # Convert wall thickness to feet
        wall_thickness_ft = wall_thickness_in / 12.0

        # Individual resistances - DETERMINISTIC
        r_inside = 1.0 / htc_inside_btu_hr_ft2_f if htc_inside_btu_hr_ft2_f > 0 else float('inf')
        r_fouling_inside = fouling_inside
        r_wall = wall_thickness_ft / wall_conductivity_btu_hr_ft_f if wall_conductivity_btu_hr_ft_f > 0 else 0
        r_fouling_outside = fouling_outside / area_ratio
        r_outside = (1.0 / htc_outside_btu_hr_ft2_f) / area_ratio if htc_outside_btu_hr_ft2_f > 0 else float('inf')

        # Total resistance
        r_total = r_inside + r_fouling_inside + r_wall + r_fouling_outside + r_outside

        # Overall HTC - DETERMINISTIC
        u_overall = 1.0 / r_total if r_total > 0 else 0

        # Clean HTC (without fouling)
        r_clean = r_inside + r_wall + r_outside
        u_clean = 1.0 / r_clean if r_clean > 0 else 0

        # Fouling factor (overall)
        total_fouling = r_total - r_clean

        # Resistance percentages
        pct_inside = (r_inside / r_total * 100) if r_total > 0 else 0
        pct_fouling = (total_fouling / r_total * 100) if r_total > 0 else 0
        pct_wall = (r_wall / r_total * 100) if r_total > 0 else 0
        pct_outside = (r_outside / r_total * 100) if r_total > 0 else 0

        result = {
            "u_overall_btu_hr_ft2_f": round(u_overall, self.precision),
            "u_clean_btu_hr_ft2_f": round(u_clean, self.precision),
            "r_total_hr_ft2_f_btu": round(r_total, 6),
            "r_inside_hr_ft2_f_btu": round(r_inside, 6),
            "r_fouling_inside_hr_ft2_f_btu": round(r_fouling_inside, 6),
            "r_wall_hr_ft2_f_btu": round(r_wall, 6),
            "r_fouling_outside_hr_ft2_f_btu": round(r_fouling_outside, 6),
            "r_outside_hr_ft2_f_btu": round(r_outside, 6),
            "total_fouling_hr_ft2_f_btu": round(total_fouling, 6),
            "pct_inside_resistance": round(pct_inside, 1),
            "pct_fouling_resistance": round(pct_fouling, 1),
            "pct_wall_resistance": round(pct_wall, 1),
            "pct_outside_resistance": round(pct_outside, 1),
        }

        logger.debug(
            f"Overall HTC: U={u_overall:.2f} Btu/hr-ft2-F, "
            f"fouling={total_fouling:.6f} hr-ft2-F/Btu"
        )

        return result

    # =========================================================================
    # WALL HEAT LOSS CALCULATIONS - DETERMINISTIC
    # =========================================================================

    def calculate_wall_loss(
        self,
        t_inside_f: float,
        t_ambient_f: float,
        wall_area_ft2: float,
        wall_layers: List[Dict[str, float]] = None,
        htc_inside_btu_hr_ft2_f: float = 1.0,
        htc_outside_btu_hr_ft2_f: float = 1.5,
    ) -> Dict[str, float]:
        """
        Calculate heat loss through furnace walls - DETERMINISTIC.

        Uses composite wall conduction:
            Q = (T_inside - T_ambient) / R_total
            R_total = 1/h_i + sum(t/k) + 1/h_o

        Args:
            t_inside_f: Inside (hot) temperature (F)
            t_ambient_f: Ambient temperature (F)
            wall_area_ft2: Total wall area (ft2)
            wall_layers: List of wall layers [{thickness_in, conductivity}]
            htc_inside_btu_hr_ft2_f: Inside convective HTC
            htc_outside_btu_hr_ft2_f: Outside convective HTC

        Returns:
            Dictionary with wall loss calculations
        """
        # Default wall layers if not specified
        if wall_layers is None:
            wall_layers = [
                {"thickness_in": 4.5, "conductivity_btu_hr_ft_f": 0.5},   # Refractory
                {"thickness_in": 4.5, "conductivity_btu_hr_ft_f": 0.15}, # Insulating brick
            ]

        # Calculate total resistance - DETERMINISTIC
        r_inside = 1.0 / htc_inside_btu_hr_ft2_f if htc_inside_btu_hr_ft2_f > 0 else 0
        r_outside = 1.0 / htc_outside_btu_hr_ft2_f if htc_outside_btu_hr_ft2_f > 0 else 0

        r_wall = 0.0
        for layer in wall_layers:
            thickness_ft = layer.get("thickness_in", 0) / 12.0
            k = layer.get("conductivity_btu_hr_ft_f", 0.5)
            if k > 0:
                r_wall += thickness_ft / k

        r_total = r_inside + r_wall + r_outside

        # Temperature difference
        delta_t = t_inside_f - t_ambient_f

        # Heat loss - DETERMINISTIC
        q_loss_btu_hr_ft2 = delta_t / r_total if r_total > 0 else 0
        q_loss_total_btu_hr = q_loss_btu_hr_ft2 * wall_area_ft2

        # Interface temperatures
        t_interface_temps = []
        if r_total > 0:
            q_flux = q_loss_btu_hr_ft2
            current_temp = t_inside_f - q_flux * r_inside
            t_interface_temps.append(round(current_temp, 1))

            for layer in wall_layers:
                thickness_ft = layer.get("thickness_in", 0) / 12.0
                k = layer.get("conductivity_btu_hr_ft_f", 0.5)
                if k > 0:
                    r_layer = thickness_ft / k
                    current_temp = current_temp - q_flux * r_layer
                    t_interface_temps.append(round(current_temp, 1))

        # Outside wall surface temperature
        t_wall_outside_f = t_ambient_f + q_loss_btu_hr_ft2 * r_outside

        result = {
            "q_loss_btu_hr_ft2": round(q_loss_btu_hr_ft2, self.precision),
            "q_loss_total_btu_hr": round(q_loss_total_btu_hr, self.precision),
            "q_loss_total_mmbtu_hr": round(q_loss_total_btu_hr / 1e6, self.precision),
            "r_total_hr_ft2_f_btu": round(r_total, 4),
            "r_wall_hr_ft2_f_btu": round(r_wall, 4),
            "t_wall_outside_f": round(t_wall_outside_f, 1),
            "interface_temps_f": t_interface_temps,
            "delta_t_f": round(delta_t, 1),
        }

        logger.debug(
            f"Wall loss: Q={q_loss_total_btu_hr/1e6:.4f} MMBtu/hr, "
            f"T_surface={t_wall_outside_f:.0f}F"
        )

        return result

    # =========================================================================
    # COMPREHENSIVE HEAT TRANSFER ANALYSIS - DETERMINISTIC
    # =========================================================================

    def analyze_heat_transfer(
        self,
        design_duty_mmbtu_hr: float,
        actual_duty_mmbtu_hr: float,
        furnace_temp_f: float,
        flue_gas_temp_f: float,
        process_inlet_temp_f: float,
        process_outlet_temp_f: float,
        radiant_area_ft2: float,
        convective_area_ft2: float,
        design_htc_btu_hr_ft2_f: float = 8.0,
        ambient_temp_f: float = 77.0,
        wall_area_ft2: float = 1000.0,
    ) -> HeatTransferAnalysis:
        """
        Perform comprehensive heat transfer analysis - DETERMINISTIC.

        All calculations use engineering formulas with zero ML/AI.

        Args:
            design_duty_mmbtu_hr: Design heat duty (MMBtu/hr)
            actual_duty_mmbtu_hr: Current heat duty (MMBtu/hr)
            furnace_temp_f: Furnace temperature (F)
            flue_gas_temp_f: Flue gas exit temperature (F)
            process_inlet_temp_f: Process fluid inlet temperature (F)
            process_outlet_temp_f: Process fluid outlet temperature (F)
            radiant_area_ft2: Radiant section area (ft2)
            convective_area_ft2: Convective section area (ft2)
            design_htc_btu_hr_ft2_f: Design heat transfer coefficient
            ambient_temp_f: Ambient temperature (F)
            wall_area_ft2: Furnace wall area (ft2)

        Returns:
            HeatTransferAnalysis with complete results
        """
        start_time = datetime.now(timezone.utc)
        analysis_id = str(uuid.uuid4())

        # Duty ratio
        duty_ratio_pct = (actual_duty_mmbtu_hr / design_duty_mmbtu_hr * 100
                         if design_duty_mmbtu_hr > 0 else 0)

        # Calculate radiant heat transfer - DETERMINISTIC
        radiant_result = self.calculate_radiant_heat_transfer(
            gas_temp_f=furnace_temp_f,
            surface_temp_f=process_outlet_temp_f + 100,  # Estimate tube surface
            area_ft2=radiant_area_ft2,
        )
        radiant_heat_mmbtu_hr = radiant_result["q_radiant_mmbtu_hr"]

        # Calculate convective heat transfer - DETERMINISTIC
        convective_result = self.calculate_convective_heat_transfer(
            gas_temp_f=flue_gas_temp_f + 200,  # Average convection zone temp
            surface_temp_f=process_inlet_temp_f + 50,
            area_ft2=convective_area_ft2,
            htc_btu_hr_ft2_f=HeatTransferConstants.HTC_FLUE_GAS,
        )
        convective_heat_mmbtu_hr = convective_result["q_convective_mmbtu_hr"]

        # Calculate LMTD - DETERMINISTIC
        lmtd_result = self.calculate_lmtd(
            t_hot_in_f=furnace_temp_f,
            t_hot_out_f=flue_gas_temp_f,
            t_cold_in_f=process_inlet_temp_f,
            t_cold_out_f=process_outlet_temp_f,
            flow_arrangement="counterflow",
        )
        lmtd_f = lmtd_result["lmtd_f"]

        # Calculate overall HTC - DETERMINISTIC
        total_area = radiant_area_ft2 + convective_area_ft2
        if lmtd_f > 0 and total_area > 0:
            actual_htc = (actual_duty_mmbtu_hr * 1e6) / (total_area * lmtd_f)
        else:
            actual_htc = 0

        htc_ratio_pct = (actual_htc / design_htc_btu_hr_ft2_f * 100
                        if design_htc_btu_hr_ft2_f > 0 else 0)

        # Estimate fouling - DETERMINISTIC
        if design_htc_btu_hr_ft2_f > 0 and actual_htc > 0 and actual_htc < design_htc_btu_hr_ft2_f:
            fouling_factor = (1 / actual_htc) - (1 / design_htc_btu_hr_ft2_f)
        else:
            fouling_factor = 0

        # Determine fouling severity
        if fouling_factor > HeatTransferConstants.FOULING_SEVERE:
            fouling_severity = "severe"
        elif fouling_factor > HeatTransferConstants.FOULING_HEAVY:
            fouling_severity = "heavy"
        elif fouling_factor > HeatTransferConstants.FOULING_MODERATE:
            fouling_severity = "moderate"
        elif fouling_factor > HeatTransferConstants.FOULING_LIGHT:
            fouling_severity = "light"
        else:
            fouling_severity = "clean"

        # Calculate wall losses - DETERMINISTIC
        wall_result = self.calculate_wall_loss(
            t_inside_f=furnace_temp_f,
            t_ambient_f=ambient_temp_f,
            wall_area_ft2=wall_area_ft2,
        )
        wall_loss_mmbtu_hr = wall_result["q_loss_total_mmbtu_hr"]
        wall_loss_pct = (wall_loss_mmbtu_hr / actual_duty_mmbtu_hr * 100
                        if actual_duty_mmbtu_hr > 0 else 0)

        # Approach temperature
        approach_temp_f = flue_gas_temp_f - process_inlet_temp_f

        # Heat transfer effectiveness
        max_possible_duty = (furnace_temp_f - process_inlet_temp_f) / (
            furnace_temp_f - process_inlet_temp_f + 1)  # Simplified
        effectiveness_pct = (actual_duty_mmbtu_hr / design_duty_mmbtu_hr * 100
                           if design_duty_mmbtu_hr > 0 else 0)

        # Generate recommendations
        recommendations = []
        if fouling_severity in ["heavy", "severe"]:
            recommendations.append(
                f"High fouling detected ({fouling_severity}). Schedule tube cleaning."
            )
        if htc_ratio_pct < 80:
            recommendations.append(
                f"HTC degraded to {htc_ratio_pct:.0f}% of design. "
                "Check for fouling or process changes."
            )
        if wall_loss_pct > 3:
            recommendations.append(
                f"Wall losses elevated at {wall_loss_pct:.1f}%. "
                "Inspect insulation integrity."
            )
        if approach_temp_f > 200:
            recommendations.append(
                f"High approach temp ({approach_temp_f:.0f}F) indicates potential "
                "for economizer heat recovery."
            )

        # Generate provenance hash - DETERMINISTIC
        provenance_data = {
            "analysis_id": analysis_id,
            "inputs": {
                "design_duty_mmbtu_hr": design_duty_mmbtu_hr,
                "actual_duty_mmbtu_hr": actual_duty_mmbtu_hr,
                "furnace_temp_f": furnace_temp_f,
                "flue_gas_temp_f": flue_gas_temp_f,
            },
            "outputs": {
                "actual_htc": round(actual_htc, self.precision),
                "lmtd_f": lmtd_f,
                "fouling_factor": round(fouling_factor, 6),
            },
            "timestamp": start_time.isoformat(),
            "version": self.VERSION,
        }
        provenance_hash = generate_provenance_hash(provenance_data)

        # Track in provenance system
        if self._provenance_tracker:
            self._provenance_tracker.track_calculation(
                calc_type="heat_transfer_analysis",
                inputs=provenance_data["inputs"],
                outputs=provenance_data["outputs"],
                formula="Q = U * A * LMTD; LMTD = (dT1-dT2)/ln(dT1/dT2)",
                standard_references=[
                    "ASME PTC 4",
                    "API 560",
                    "Kern Process Heat Transfer",
                ],
            )

        # Create result
        result = HeatTransferAnalysis(
            analysis_id=analysis_id,
            timestamp=start_time,
            design_duty_mmbtu_hr=round(design_duty_mmbtu_hr, self.precision),
            actual_duty_mmbtu_hr=round(actual_duty_mmbtu_hr, self.precision),
            duty_ratio_pct=round(duty_ratio_pct, self.precision),
            radiant_heat_transfer_mmbtu_hr=round(radiant_heat_mmbtu_hr, self.precision),
            convective_heat_transfer_mmbtu_hr=round(convective_heat_mmbtu_hr, self.precision),
            overall_htc_btu_hr_ft2_f=round(actual_htc, self.precision),
            design_htc_btu_hr_ft2_f=round(design_htc_btu_hr_ft2_f, self.precision),
            htc_ratio_pct=round(htc_ratio_pct, self.precision),
            lmtd_f=round(lmtd_f, self.precision),
            approach_temp_f=round(approach_temp_f, self.precision),
            fouling_factor_hr_ft2_f_btu=round(fouling_factor, 6),
            fouling_severity=fouling_severity,
            wall_loss_mmbtu_hr=round(wall_loss_mmbtu_hr, self.precision),
            wall_loss_pct=round(wall_loss_pct, self.precision),
            opening_loss_mmbtu_hr=0.0,  # Not calculated in this simplified version
            heat_transfer_effectiveness_pct=round(effectiveness_pct, self.precision),
            recommendations=recommendations,
            provenance_hash=provenance_hash,
        )

        logger.info(
            f"Heat transfer analysis: duty_ratio={duty_ratio_pct:.1f}%, "
            f"htc_ratio={htc_ratio_pct:.1f}%, fouling={fouling_severity}"
        )

        return result


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_heat_transfer_calculator(
    provenance_enabled: bool = True,
    precision: int = 4,
) -> FurnaceHeatTransfer:
    """Factory function to create FurnaceHeatTransfer calculator."""
    return FurnaceHeatTransfer(
        provenance_enabled=provenance_enabled,
        precision=precision,
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "HeatTransferConstants",
    "FurnaceHeatTransfer",
    "create_heat_transfer_calculator",
]
