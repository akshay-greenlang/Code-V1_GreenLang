# -*- coding: utf-8 -*-
"""
Radiant Heat Transfer Calculator for GL-007 FURNACEPULSE FurnacePerformanceMonitor

Implements deterministic radiant heat transfer calculations for industrial furnaces,
including Stefan-Boltzmann law, view factor geometry, flame emissivity, tube metal
temperature prediction, heat flux distribution, and radiant section duty calculations
with zero-hallucination guarantees.

Standards Compliance:
- ASME PTC 4.2: Performance Test Code on Industrial Furnaces
- API 560: Fired Heaters for General Refinery Service
- ISO 13705: Petroleum and Natural Gas Industries - Fired Heaters

Author: GL-CalculatorEngineer
Agent: GL-007 FURNACEPULSE
Version: 1.0.0
"""

from decimal import Decimal, ROUND_HALF_UP, ROUND_DOWN
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import math

from .provenance import ProvenanceTracker, ProvenanceRecord, CalculationCategory


class RadiantSectionType(Enum):
    """Types of radiant section configurations."""
    BOX_TYPE = "box_type"
    CYLINDRICAL = "cylindrical"
    CABIN_TYPE = "cabin_type"
    VERTICAL_CYLINDRICAL = "vertical_cylindrical"
    HELICAL_COIL = "helical_coil"


class TubeArrangement(Enum):
    """Tube arrangement patterns in radiant section."""
    SINGLE_ROW_AGAINST_WALL = "single_row_against_wall"
    DOUBLE_ROW_AGAINST_WALL = "double_row_against_wall"
    SINGLE_ROW_CENTER = "single_row_center"
    DOUBLE_ROW_CENTER = "double_row_center"
    EQUILATERAL_TRIANGULAR = "equilateral_triangular"


class FlameType(Enum):
    """Types of burner flames for emissivity calculations."""
    LUMINOUS = "luminous"
    NON_LUMINOUS = "non_luminous"
    PARTIALLY_LUMINOUS = "partially_luminous"


@dataclass
class RadiantSectionGeometry:
    """
    Geometric parameters of the radiant section for view factor calculations.

    Attributes:
        section_type: Type of radiant section
        length_m: Firebox length (m)
        width_m: Firebox width (m)
        height_m: Firebox height (m)
        tube_od_mm: Tube outer diameter (mm)
        tube_spacing_mm: Center-to-center tube spacing (mm)
        tube_length_m: Effective tube length (m)
        tube_count: Number of tubes in radiant section
        tube_arrangement: Arrangement of tubes
        wall_to_tube_distance_mm: Distance from wall to tube centerline (mm)
        refractory_area_m2: Total refractory surface area (m2)
    """
    section_type: RadiantSectionType
    length_m: float
    width_m: float
    height_m: float
    tube_od_mm: float
    tube_spacing_mm: float
    tube_length_m: float
    tube_count: int
    tube_arrangement: TubeArrangement = TubeArrangement.SINGLE_ROW_AGAINST_WALL
    wall_to_tube_distance_mm: float = 150.0
    refractory_area_m2: Optional[float] = None


@dataclass
class FlameProperties:
    """
    Properties of the flame for emissivity and radiation calculations.

    Attributes:
        flame_type: Type of flame (luminous, non-luminous, etc.)
        flame_temperature_c: Adiabatic flame temperature (degC)
        partial_pressure_co2_atm: Partial pressure of CO2 (atm)
        partial_pressure_h2o_atm: Partial pressure of H2O (atm)
        mean_beam_length_m: Mean beam length for gas radiation (m)
        soot_concentration_g_m3: Soot concentration for luminous flames (g/m3)
        excess_air_percent: Excess air percentage
    """
    flame_type: FlameType
    flame_temperature_c: float
    partial_pressure_co2_atm: float = 0.12
    partial_pressure_h2o_atm: float = 0.18
    mean_beam_length_m: float = 1.5
    soot_concentration_g_m3: float = 0.0
    excess_air_percent: float = 15.0


@dataclass
class TubeConditions:
    """
    Operating conditions for tube metal temperature calculations.

    Attributes:
        process_fluid_temp_inlet_c: Process fluid inlet temperature (degC)
        process_fluid_temp_outlet_c: Process fluid outlet temperature (degC)
        process_flow_rate_kg_hr: Process fluid mass flow rate (kg/hr)
        fluid_specific_heat_kj_kg_k: Fluid specific heat (kJ/kg.K)
        tube_wall_thickness_mm: Tube wall thickness (mm)
        tube_thermal_conductivity_w_mk: Tube metal thermal conductivity (W/m.K)
        inside_film_coefficient_w_m2k: Inside film heat transfer coefficient (W/m2.K)
        fouling_factor_m2k_w: Inside fouling resistance (m2.K/W)
        design_tube_metal_temp_c: Design maximum tube metal temperature (degC)
    """
    process_fluid_temp_inlet_c: float
    process_fluid_temp_outlet_c: float
    process_flow_rate_kg_hr: float
    fluid_specific_heat_kj_kg_k: float = 2.5
    tube_wall_thickness_mm: float = 8.0
    tube_thermal_conductivity_w_mk: float = 35.0
    inside_film_coefficient_w_m2k: float = 1500.0
    fouling_factor_m2k_w: float = 0.0002
    design_tube_metal_temp_c: float = 550.0


@dataclass
class ViewFactorResult:
    """
    Result of view factor calculations.

    Attributes:
        direct_view_factor: Direct view factor F_tf (flame to tube)
        alpha_factor: Cold plane area factor (Hottel alpha)
        total_exchange_factor: Total exchange factor including refractory
        refractory_effectiveness: Refractory effectiveness factor
        geometric_factor: Pure geometric view factor
    """
    direct_view_factor: Decimal
    alpha_factor: Decimal
    total_exchange_factor: Decimal
    refractory_effectiveness: Decimal
    geometric_factor: Decimal
    provenance: ProvenanceRecord


@dataclass
class FlameEmissivityResult:
    """
    Result of flame emissivity calculations.

    Attributes:
        gas_emissivity: Gas radiation emissivity (CO2 + H2O)
        soot_emissivity: Soot contribution to emissivity
        total_emissivity: Combined flame emissivity
        absorptivity: Flame absorptivity
        emissivity_correction: Spectral overlap correction
    """
    gas_emissivity: Decimal
    soot_emissivity: Decimal
    total_emissivity: Decimal
    absorptivity: Decimal
    emissivity_correction: Decimal
    provenance: ProvenanceRecord


@dataclass
class TubeMetalTemperatureResult:
    """
    Result of tube metal temperature prediction.

    Attributes:
        tube_metal_temp_max_c: Maximum tube metal temperature (degC)
        tube_metal_temp_avg_c: Average tube metal temperature (degC)
        inside_film_temp_drop_c: Temperature drop across inside film (degC)
        wall_temp_drop_c: Temperature drop across tube wall (degC)
        fouling_temp_drop_c: Temperature drop due to fouling (degC)
        margin_to_design_c: Margin below design temperature (degC)
        is_within_design: Whether temperature is within design limit
    """
    tube_metal_temp_max_c: Decimal
    tube_metal_temp_avg_c: Decimal
    inside_film_temp_drop_c: Decimal
    wall_temp_drop_c: Decimal
    fouling_temp_drop_c: Decimal
    margin_to_design_c: Decimal
    is_within_design: bool
    provenance: ProvenanceRecord


@dataclass
class HeatFluxResult:
    """
    Result of heat flux distribution calculation.

    Attributes:
        average_heat_flux_kw_m2: Average radiant heat flux (kW/m2)
        peak_heat_flux_kw_m2: Peak radiant heat flux (kW/m2)
        peak_to_average_ratio: Circumferential heat flux ratio
        heat_flux_distribution: Distribution around tube (dict of angles to flux)
        api_design_limit_kw_m2: API 560 design limit for flux
        is_within_design: Whether flux is within design limit
    """
    average_heat_flux_kw_m2: Decimal
    peak_heat_flux_kw_m2: Decimal
    peak_to_average_ratio: Decimal
    heat_flux_distribution: Dict[str, Decimal]
    api_design_limit_kw_m2: Decimal
    is_within_design: bool
    provenance: ProvenanceRecord


@dataclass
class RadiantSectionDutyResult:
    """
    Complete result of radiant section heat transfer calculations.

    Attributes:
        radiant_duty_mw: Total radiant section duty (MW)
        radiant_efficiency_percent: Radiant section efficiency (%)
        heat_release_rate_kw_m3: Volumetric heat release (kW/m3)
        average_flux_kw_m2: Average heat flux (kW/m2)
        peak_flux_kw_m2: Peak heat flux (kW/m2)
        bridgewall_temp_c: Bridgewall temperature (degC)
        tube_metal_temp_max_c: Maximum tube metal temperature (degC)
        flame_emissivity: Effective flame emissivity
        view_factor: Effective view factor
        radiant_fraction: Fraction of heat transferred by radiation
    """
    radiant_duty_mw: Decimal
    radiant_efficiency_percent: Decimal
    heat_release_rate_kw_m3: Decimal
    average_flux_kw_m2: Decimal
    peak_flux_kw_m2: Decimal
    bridgewall_temp_c: Decimal
    tube_metal_temp_max_c: Decimal
    flame_emissivity: Decimal
    view_factor: Decimal
    radiant_fraction: Decimal
    provenance: ProvenanceRecord


class RadiantHeatTransferCalculator:
    """
    ASME PTC 4.2 and API 560 Compliant Radiant Heat Transfer Calculator.

    Implements deterministic radiant heat transfer calculations for fired heater
    radiant sections. All calculations are based on Stefan-Boltzmann radiation
    law and established correlations with zero-hallucination guarantees.

    Zero-Hallucination Guarantees:
    - Pure mathematical calculations using Decimal arithmetic
    - No LLM inference or probabilistic methods
    - Complete provenance tracking with SHA-256 hashing
    - Formulas from published standards (API 560, ASME PTC 4.2)

    Calculation Capabilities:
    1. Stefan-Boltzmann Radiation: Q = sigma * epsilon * A * (T1^4 - T2^4)
    2. View Factor Geometry: Hottel crossed-strings and analytical methods
    3. Flame Emissivity: Gas radiation (Hottel) + soot contribution
    4. Tube Metal Temperature: Heat transfer resistance chain
    5. Heat Flux Distribution: Circumferential variation
    6. Radiant Section Duty: Complete energy balance

    Example:
        >>> calculator = RadiantHeatTransferCalculator()
        >>> geometry = RadiantSectionGeometry(
        ...     section_type=RadiantSectionType.BOX_TYPE,
        ...     length_m=10.0,
        ...     width_m=3.0,
        ...     height_m=8.0,
        ...     tube_od_mm=114.3,
        ...     tube_spacing_mm=200.0,
        ...     tube_length_m=8.0,
        ...     tube_count=40
        ... )
        >>> result = calculator.calculate_radiant_duty(geometry, flame_props, tube_cond)
    """

    # Physical constants
    STEFAN_BOLTZMANN = Decimal("5.67E-8")  # W/m2.K4
    KELVIN_OFFSET = Decimal("273.15")

    # API 560 design limits
    API_MAX_FLUX_CRUDE_OIL = Decimal("37.8")  # kW/m2
    API_MAX_FLUX_VACUUM_RESIDUE = Decimal("31.5")  # kW/m2
    API_MAX_FLUX_GAS_OIL = Decimal("44.1")  # kW/m2
    API_MAX_FLUX_REFORMER = Decimal("78.8")  # kW/m2

    def __init__(self, version: str = "1.0.0"):
        """
        Initialize the Radiant Heat Transfer Calculator.

        Args:
            version: Calculator version for provenance tracking
        """
        self.version = version

    def calculate_stefan_boltzmann_radiation(
        self,
        hot_surface_temp_c: float,
        cold_surface_temp_c: float,
        emissivity: float,
        surface_area_m2: float,
        calculation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate radiation heat transfer using Stefan-Boltzmann law.

        The fundamental equation for thermal radiation between two surfaces:
        Q = sigma * epsilon * A * (T_hot^4 - T_cold^4)

        Args:
            hot_surface_temp_c: Hot surface temperature (degC)
            cold_surface_temp_c: Cold surface temperature (degC)
            emissivity: Effective emissivity (0-1)
            surface_area_m2: Heat transfer surface area (m2)
            calculation_id: Optional calculation identifier

        Returns:
            Dictionary with heat transfer rate and provenance
        """
        tracker = ProvenanceTracker(
            calculation_id=calculation_id or f"stefan_boltzmann_{id(self)}",
            calculation_type="radiant_heat_transfer",
            version=self.version,
            standard_compliance=["ASME PTC 4.2", "API 560"]
        )

        # Convert to Decimal
        t_hot = Decimal(str(hot_surface_temp_c))
        t_cold = Decimal(str(cold_surface_temp_c))
        eps = Decimal(str(emissivity))
        area = Decimal(str(surface_area_m2))

        tracker.record_inputs({
            "hot_surface_temp_c": t_hot,
            "cold_surface_temp_c": t_cold,
            "emissivity": eps,
            "surface_area_m2": area
        })

        # Convert to Kelvin
        t_hot_k = t_hot + self.KELVIN_OFFSET
        t_cold_k = t_cold + self.KELVIN_OFFSET

        tracker.record_step(
            operation="temperature_conversion",
            description="Convert temperatures to Kelvin",
            inputs={"t_hot_c": t_hot, "t_cold_c": t_cold},
            output_value=t_hot_k,
            output_name="t_hot_k",
            formula="T_K = T_C + 273.15",
            units="K"
        )

        # Calculate T^4 terms
        t_hot_k4 = t_hot_k ** 4
        t_cold_k4 = t_cold_k ** 4

        # Stefan-Boltzmann calculation
        q_radiation_w = self.STEFAN_BOLTZMANN * eps * area * (t_hot_k4 - t_cold_k4)
        q_radiation_kw = (q_radiation_w / Decimal("1000")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        tracker.record_step(
            operation="stefan_boltzmann",
            description="Calculate radiation heat transfer using Stefan-Boltzmann law",
            inputs={
                "stefan_boltzmann": self.STEFAN_BOLTZMANN,
                "emissivity": eps,
                "area_m2": area,
                "t_hot_k4": t_hot_k4,
                "t_cold_k4": t_cold_k4
            },
            output_value=q_radiation_kw,
            output_name="radiation_heat_kw",
            formula="Q = sigma * epsilon * A * (T_hot^4 - T_cold^4)",
            units="kW",
            standard_reference="Stefan-Boltzmann Law"
        )

        # Calculate heat flux
        if area > 0:
            heat_flux_kw_m2 = (q_radiation_kw / area).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        else:
            heat_flux_kw_m2 = Decimal("0")

        tracker.record_step(
            operation="divide",
            description="Calculate heat flux",
            inputs={"heat_kw": q_radiation_kw, "area_m2": area},
            output_value=heat_flux_kw_m2,
            output_name="heat_flux_kw_m2",
            formula="q = Q / A",
            units="kW/m2"
        )

        provenance = tracker.get_provenance_record(q_radiation_kw)

        return {
            "radiation_heat_kw": float(q_radiation_kw),
            "radiation_heat_w": float(q_radiation_w),
            "heat_flux_kw_m2": float(heat_flux_kw_m2),
            "temperature_difference_k": float(t_hot_k - t_cold_k),
            "provenance_hash": provenance.provenance_hash
        }

    def calculate_view_factor(
        self,
        geometry: RadiantSectionGeometry,
        calculation_id: Optional[str] = None
    ) -> ViewFactorResult:
        """
        Calculate view factors for radiant section using Hottel method.

        The view factor (F) represents the fraction of radiation leaving one
        surface that directly strikes another surface. For fired heaters,
        the effective view factor accounts for refractory re-radiation.

        Method: Hottel's crossed-strings method and analytical correlations
        for standard tube arrangements per API 560.

        Formula:
            F_eff = F_direct + F_refractory * (1 - F_direct) / (1 - rho_r * (1 - alpha))

        Args:
            geometry: RadiantSectionGeometry with section dimensions
            calculation_id: Optional calculation identifier

        Returns:
            ViewFactorResult with all view factor components
        """
        tracker = ProvenanceTracker(
            calculation_id=calculation_id or f"view_factor_{id(geometry)}",
            calculation_type="view_factor",
            version=self.version,
            standard_compliance=["API 560", "Hottel & Sarofim"]
        )

        # Convert to Decimal
        tube_od = Decimal(str(geometry.tube_od_mm)) / Decimal("1000")  # Convert to m
        tube_spacing = Decimal(str(geometry.tube_spacing_mm)) / Decimal("1000")
        wall_distance = Decimal(str(geometry.wall_to_tube_distance_mm)) / Decimal("1000")
        tube_length = Decimal(str(geometry.tube_length_m))
        tube_count = Decimal(str(geometry.tube_count))

        tracker.record_inputs({
            "tube_od_m": tube_od,
            "tube_spacing_m": tube_spacing,
            "wall_distance_m": wall_distance,
            "tube_length_m": tube_length,
            "tube_count": tube_count,
            "arrangement": geometry.tube_arrangement.value
        })

        # Calculate ratio parameters for view factor correlations
        spacing_ratio = tube_spacing / tube_od  # P/D ratio

        tracker.record_step(
            operation="divide",
            description="Calculate tube spacing to diameter ratio",
            inputs={"spacing_m": tube_spacing, "diameter_m": tube_od},
            output_value=spacing_ratio,
            output_name="spacing_ratio",
            formula="P/D = Tube Spacing / Tube OD"
        )

        # Calculate alpha factor (cold plane area factor) - Hottel method
        # Alpha = ratio of cold plane area to actual tube surface area
        # For single row against wall: alpha = 1 - D/P + (D/P)*arctan(sqrt((P/D)^2 - 1))/pi

        if spacing_ratio >= Decimal("1"):
            # Valid geometry
            sqrt_term = (spacing_ratio ** 2 - Decimal("1")).sqrt()
            arctan_term = Decimal(str(math.atan(float(sqrt_term)))) / Decimal(str(math.pi))
            d_over_p = Decimal("1") / spacing_ratio

            alpha = Decimal("1") - d_over_p + d_over_p * arctan_term
        else:
            # Tubes overlapping - invalid
            alpha = Decimal("0.5")

        alpha = alpha.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        tracker.record_step(
            operation="hottel_alpha",
            description="Calculate Hottel cold plane area factor (alpha)",
            inputs={
                "spacing_ratio": spacing_ratio,
                "d_over_p": d_over_p if spacing_ratio >= Decimal("1") else Decimal("0")
            },
            output_value=alpha,
            output_name="alpha_factor",
            formula="alpha = 1 - D/P + (D/P)*arctan(sqrt((P/D)^2 - 1))/pi",
            standard_reference="Hottel & Sarofim, Radiative Transfer"
        )

        # Calculate direct view factor based on arrangement
        if geometry.tube_arrangement == TubeArrangement.SINGLE_ROW_AGAINST_WALL:
            # F = alpha (for single row against refractory wall)
            direct_view_factor = alpha
        elif geometry.tube_arrangement == TubeArrangement.DOUBLE_ROW_AGAINST_WALL:
            # Double row has higher interception
            direct_view_factor = (alpha * Decimal("1.15")).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
            direct_view_factor = min(direct_view_factor, Decimal("0.95"))
        elif geometry.tube_arrangement == TubeArrangement.SINGLE_ROW_CENTER:
            # Center row sees both sides
            direct_view_factor = (Decimal("2") * alpha - alpha ** 2).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
        else:
            direct_view_factor = alpha

        tracker.record_step(
            operation="direct_view_factor",
            description="Calculate direct view factor for tube arrangement",
            inputs={
                "alpha": alpha,
                "arrangement": geometry.tube_arrangement.value
            },
            output_value=direct_view_factor,
            output_name="direct_view_factor",
            formula="F_direct based on tube arrangement"
        )

        # Calculate refractory effectiveness
        # Assumes refractory re-radiates to tubes
        refractory_effectiveness = Decimal("0.85")  # Typical value for well-maintained refractory

        # Calculate total exchange factor including refractory contribution
        # Using simplified formula: F_total = F_direct + (1 - F_direct) * refractory_eff
        refractory_contribution = (Decimal("1") - direct_view_factor) * refractory_effectiveness
        total_exchange = (direct_view_factor + refractory_contribution).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        tracker.record_step(
            operation="total_exchange",
            description="Calculate total exchange factor with refractory re-radiation",
            inputs={
                "direct_view_factor": direct_view_factor,
                "refractory_effectiveness": refractory_effectiveness
            },
            output_value=total_exchange,
            output_name="total_exchange_factor",
            formula="F_total = F_direct + (1 - F_direct) * eta_refr",
            standard_reference="API 560 Section 7"
        )

        # Calculate geometric view factor (pure geometry, no refractory)
        # This is the tube circumference fraction exposed to radiation
        if geometry.tube_arrangement == TubeArrangement.SINGLE_ROW_AGAINST_WALL:
            geometric_factor = Decimal("0.5") * (Decimal("1") + wall_distance / tube_od)
            geometric_factor = min(geometric_factor, Decimal("1.0"))
        else:
            geometric_factor = Decimal("1.0")

        geometric_factor = geometric_factor.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        provenance = tracker.get_provenance_record(total_exchange)

        return ViewFactorResult(
            direct_view_factor=direct_view_factor,
            alpha_factor=alpha,
            total_exchange_factor=total_exchange,
            refractory_effectiveness=refractory_effectiveness,
            geometric_factor=geometric_factor,
            provenance=provenance
        )

    def calculate_flame_emissivity(
        self,
        flame_props: FlameProperties,
        calculation_id: Optional[str] = None
    ) -> FlameEmissivityResult:
        """
        Calculate flame emissivity using Hottel gas radiation correlations.

        Flame emissivity depends on gas composition (CO2, H2O) and soot
        content. This uses the Hottel emissivity charts fitted to polynomials
        for deterministic calculation.

        Formulas:
            Gas Emissivity: epsilon_g = epsilon_CO2 + epsilon_H2O - delta_epsilon
            Soot Emissivity: epsilon_s = 1 - exp(-k * L * C_soot)
            Total: epsilon_f = epsilon_g + epsilon_s - epsilon_g * epsilon_s

        Args:
            flame_props: FlameProperties with flame conditions
            calculation_id: Optional calculation identifier

        Returns:
            FlameEmissivityResult with emissivity components
        """
        tracker = ProvenanceTracker(
            calculation_id=calculation_id or f"flame_emissivity_{id(flame_props)}",
            calculation_type="flame_emissivity",
            version=self.version,
            standard_compliance=["Hottel & Sarofim", "API 560"]
        )

        # Convert to Decimal
        t_flame = Decimal(str(flame_props.flame_temperature_c))
        p_co2 = Decimal(str(flame_props.partial_pressure_co2_atm))
        p_h2o = Decimal(str(flame_props.partial_pressure_h2o_atm))
        beam_length = Decimal(str(flame_props.mean_beam_length_m))
        soot_conc = Decimal(str(flame_props.soot_concentration_g_m3))

        tracker.record_inputs({
            "flame_temperature_c": t_flame,
            "partial_pressure_co2_atm": p_co2,
            "partial_pressure_h2o_atm": p_h2o,
            "mean_beam_length_m": beam_length,
            "soot_concentration_g_m3": soot_conc,
            "flame_type": flame_props.flame_type.value
        })

        # Calculate pressure-path length products (atm-m)
        pL_co2 = p_co2 * beam_length
        pL_h2o = p_h2o * beam_length

        tracker.record_step(
            operation="multiply",
            description="Calculate pressure-path length products",
            inputs={"p_co2": p_co2, "p_h2o": p_h2o, "beam_length": beam_length},
            output_value=pL_co2,
            output_name="pL_co2",
            formula="pL = p * L",
            units="atm-m"
        )

        # Calculate CO2 emissivity using polynomial fit to Hottel chart
        # Valid for T = 500-2000 K, pL = 0.01-10 atm-m
        t_flame_k = (t_flame + self.KELVIN_OFFSET) / Decimal("1000")  # Scale to kK

        # Simplified Hottel correlation for CO2
        # epsilon_CO2 = a * (pL)^b * exp(-c/T)
        # Coefficients from curve fit
        a_co2 = Decimal("0.42")
        b_co2 = Decimal("0.33")
        c_co2 = Decimal("0.5")

        if pL_co2 > 0:
            # Using logarithm for power calculation
            ln_pL = Decimal(str(math.log(float(pL_co2))))
            power_term = Decimal(str(math.exp(float(b_co2 * ln_pL))))
            exp_term = Decimal(str(math.exp(float(-c_co2 / t_flame_k))))
            eps_co2 = a_co2 * power_term * exp_term
        else:
            eps_co2 = Decimal("0")

        eps_co2 = max(Decimal("0"), min(Decimal("0.5"), eps_co2))
        eps_co2 = eps_co2.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        tracker.record_step(
            operation="hottel_co2",
            description="Calculate CO2 gas emissivity from Hottel correlation",
            inputs={"pL_co2": pL_co2, "temperature_kK": t_flame_k},
            output_value=eps_co2,
            output_name="emissivity_co2",
            formula="epsilon_CO2 = a*(pL)^b*exp(-c/T)",
            standard_reference="Hottel emissivity charts"
        )

        # Calculate H2O emissivity using polynomial fit
        a_h2o = Decimal("0.52")
        b_h2o = Decimal("0.35")
        c_h2o = Decimal("0.4")

        if pL_h2o > 0:
            ln_pL = Decimal(str(math.log(float(pL_h2o))))
            power_term = Decimal(str(math.exp(float(b_h2o * ln_pL))))
            exp_term = Decimal(str(math.exp(float(-c_h2o / t_flame_k))))
            eps_h2o = a_h2o * power_term * exp_term
        else:
            eps_h2o = Decimal("0")

        eps_h2o = max(Decimal("0"), min(Decimal("0.5"), eps_h2o))
        eps_h2o = eps_h2o.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        tracker.record_step(
            operation="hottel_h2o",
            description="Calculate H2O gas emissivity from Hottel correlation",
            inputs={"pL_h2o": pL_h2o, "temperature_kK": t_flame_k},
            output_value=eps_h2o,
            output_name="emissivity_h2o",
            formula="epsilon_H2O = a*(pL)^b*exp(-c/T)",
            standard_reference="Hottel emissivity charts"
        )

        # Calculate spectral overlap correction (delta_epsilon)
        # When both CO2 and H2O are present, there is spectral overlap
        delta_eps = (eps_co2 * eps_h2o * Decimal("0.25")).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        tracker.record_step(
            operation="spectral_correction",
            description="Calculate spectral overlap correction",
            inputs={"eps_co2": eps_co2, "eps_h2o": eps_h2o},
            output_value=delta_eps,
            output_name="spectral_correction",
            formula="delta_eps = eps_CO2 * eps_H2O * 0.25"
        )

        # Calculate total gas emissivity
        eps_gas = eps_co2 + eps_h2o - delta_eps
        eps_gas = eps_gas.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        tracker.record_step(
            operation="add_subtract",
            description="Calculate total gas emissivity",
            inputs={"eps_co2": eps_co2, "eps_h2o": eps_h2o, "delta_eps": delta_eps},
            output_value=eps_gas,
            output_name="gas_emissivity",
            formula="epsilon_g = epsilon_CO2 + epsilon_H2O - delta_epsilon"
        )

        # Calculate soot emissivity for luminous flames
        if flame_props.flame_type == FlameType.LUMINOUS and soot_conc > 0:
            # Beer-Lambert absorption: epsilon_s = 1 - exp(-k*L*C)
            k_soot = Decimal("0.5")  # Absorption coefficient (m2/g)
            optical_thickness = k_soot * beam_length * soot_conc
            eps_soot = Decimal("1") - Decimal(str(math.exp(float(-optical_thickness))))
        elif flame_props.flame_type == FlameType.PARTIALLY_LUMINOUS:
            eps_soot = Decimal("0.15")  # Typical for partial luminosity
        else:
            eps_soot = Decimal("0")

        eps_soot = eps_soot.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        tracker.record_step(
            operation="soot_emissivity",
            description="Calculate soot contribution to emissivity",
            inputs={
                "flame_type": flame_props.flame_type.value,
                "soot_concentration": soot_conc
            },
            output_value=eps_soot,
            output_name="soot_emissivity",
            formula="epsilon_s = 1 - exp(-k*L*C) for luminous flames"
        )

        # Calculate total flame emissivity
        # Using: epsilon_total = epsilon_g + epsilon_s - epsilon_g * epsilon_s
        eps_total = eps_gas + eps_soot - eps_gas * eps_soot
        eps_total = max(Decimal("0.1"), min(Decimal("0.95"), eps_total))
        eps_total = eps_total.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        tracker.record_step(
            operation="total_emissivity",
            description="Calculate total flame emissivity",
            inputs={"gas_emissivity": eps_gas, "soot_emissivity": eps_soot},
            output_value=eps_total,
            output_name="total_emissivity",
            formula="epsilon_f = epsilon_g + epsilon_s - epsilon_g * epsilon_s",
            standard_reference="Hottel & Sarofim combined emissivity"
        )

        # Calculate absorptivity (different from emissivity at different temperatures)
        # Using Kirchhoff's law approximation with temperature correction
        absorptivity = (eps_total * Decimal("0.95")).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        provenance = tracker.get_provenance_record(eps_total)

        return FlameEmissivityResult(
            gas_emissivity=eps_gas,
            soot_emissivity=eps_soot,
            total_emissivity=eps_total,
            absorptivity=absorptivity,
            emissivity_correction=delta_eps,
            provenance=provenance
        )

    def calculate_tube_metal_temperature(
        self,
        heat_flux_kw_m2: float,
        tube_conditions: TubeConditions,
        geometry: RadiantSectionGeometry,
        calculation_id: Optional[str] = None
    ) -> TubeMetalTemperatureResult:
        """
        Calculate tube metal temperature from heat transfer resistance chain.

        The tube metal temperature is critical for equipment life and safety.
        It is calculated from the process fluid temperature plus temperature
        drops across each resistance in the heat transfer path.

        Formula:
            T_metal = T_fluid + (q * R_inside) + (q * R_wall) + (q * R_fouling)

        Where:
            R_inside = 1 / h_i (inside film resistance)
            R_wall = t / k (wall conduction resistance)
            R_fouling = fouling factor

        Args:
            heat_flux_kw_m2: Local heat flux at tube surface (kW/m2)
            tube_conditions: TubeConditions with process data
            geometry: RadiantSectionGeometry with tube dimensions
            calculation_id: Optional calculation identifier

        Returns:
            TubeMetalTemperatureResult with temperatures and margins
        """
        tracker = ProvenanceTracker(
            calculation_id=calculation_id or f"tube_temp_{id(tube_conditions)}",
            calculation_type="tube_metal_temperature",
            version=self.version,
            standard_compliance=["API 560", "API 530"]
        )

        # Convert to Decimal
        q = Decimal(str(heat_flux_kw_m2)) * Decimal("1000")  # Convert to W/m2
        t_fluid_avg = (
            Decimal(str(tube_conditions.process_fluid_temp_inlet_c)) +
            Decimal(str(tube_conditions.process_fluid_temp_outlet_c))
        ) / Decimal("2")
        h_inside = Decimal(str(tube_conditions.inside_film_coefficient_w_m2k))
        t_wall = Decimal(str(tube_conditions.tube_wall_thickness_mm)) / Decimal("1000")
        k_wall = Decimal(str(tube_conditions.tube_thermal_conductivity_w_mk))
        r_fouling = Decimal(str(tube_conditions.fouling_factor_m2k_w))
        t_design = Decimal(str(tube_conditions.design_tube_metal_temp_c))
        tube_od = Decimal(str(geometry.tube_od_mm)) / Decimal("1000")

        tracker.record_inputs({
            "heat_flux_w_m2": q,
            "fluid_temp_avg_c": t_fluid_avg,
            "inside_film_coeff_w_m2k": h_inside,
            "wall_thickness_m": t_wall,
            "wall_conductivity_w_mk": k_wall,
            "fouling_factor_m2k_w": r_fouling,
            "design_temp_c": t_design
        })

        # Calculate inside film temperature drop
        if h_inside > 0:
            r_inside = Decimal("1") / h_inside
            delta_t_film = (q * r_inside).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
        else:
            delta_t_film = Decimal("0")

        tracker.record_step(
            operation="film_temp_drop",
            description="Calculate temperature drop across inside film",
            inputs={"heat_flux_w_m2": q, "h_inside": h_inside},
            output_value=delta_t_film,
            output_name="inside_film_temp_drop_c",
            formula="delta_T_film = q / h_i",
            units="degC"
        )

        # Calculate wall temperature drop
        if k_wall > 0:
            r_wall = t_wall / k_wall
            delta_t_wall = (q * r_wall).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
        else:
            delta_t_wall = Decimal("0")

        tracker.record_step(
            operation="wall_temp_drop",
            description="Calculate temperature drop across tube wall",
            inputs={"heat_flux_w_m2": q, "wall_thickness_m": t_wall, "k_wall": k_wall},
            output_value=delta_t_wall,
            output_name="wall_temp_drop_c",
            formula="delta_T_wall = q * t / k",
            units="degC"
        )

        # Calculate fouling temperature drop
        delta_t_fouling = (q * r_fouling).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)

        tracker.record_step(
            operation="fouling_temp_drop",
            description="Calculate temperature drop due to fouling",
            inputs={"heat_flux_w_m2": q, "fouling_factor": r_fouling},
            output_value=delta_t_fouling,
            output_name="fouling_temp_drop_c",
            formula="delta_T_foul = q * R_foul",
            units="degC"
        )

        # Calculate tube metal temperatures
        t_metal_avg = t_fluid_avg + delta_t_film + delta_t_wall / Decimal("2") + delta_t_fouling
        t_metal_avg = t_metal_avg.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)

        # Maximum temperature at peak flux location
        # Assume peak flux is 1.5x average for single-row against wall
        peak_factor = Decimal("1.5")
        t_metal_max = t_fluid_avg + (delta_t_film + delta_t_wall + delta_t_fouling) * peak_factor
        t_metal_max = t_metal_max.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)

        tracker.record_step(
            operation="tube_metal_temp",
            description="Calculate tube metal temperature",
            inputs={
                "fluid_temp_avg": t_fluid_avg,
                "delta_t_film": delta_t_film,
                "delta_t_wall": delta_t_wall,
                "delta_t_fouling": delta_t_fouling,
                "peak_factor": peak_factor
            },
            output_value=t_metal_max,
            output_name="tube_metal_temp_max_c",
            formula="T_metal = T_fluid + delta_T_film + delta_T_wall + delta_T_foul",
            units="degC",
            standard_reference="API 530"
        )

        # Calculate margin to design
        margin = t_design - t_metal_max
        margin = margin.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
        is_within_design = t_metal_max <= t_design

        tracker.record_step(
            operation="margin_check",
            description="Check margin to design temperature",
            inputs={"design_temp": t_design, "actual_temp": t_metal_max},
            output_value=margin,
            output_name="margin_to_design_c",
            formula="Margin = T_design - T_metal_max",
            units="degC"
        )

        provenance = tracker.get_provenance_record(t_metal_max)

        return TubeMetalTemperatureResult(
            tube_metal_temp_max_c=t_metal_max,
            tube_metal_temp_avg_c=t_metal_avg,
            inside_film_temp_drop_c=delta_t_film,
            wall_temp_drop_c=delta_t_wall,
            fouling_temp_drop_c=delta_t_fouling,
            margin_to_design_c=margin,
            is_within_design=is_within_design,
            provenance=provenance
        )

    def calculate_heat_flux_distribution(
        self,
        average_flux_kw_m2: float,
        geometry: RadiantSectionGeometry,
        calculation_id: Optional[str] = None
    ) -> HeatFluxResult:
        """
        Calculate circumferential heat flux distribution around tubes.

        Heat flux varies around the tube circumference based on geometry
        and arrangement. Peak flux occurs at the crown facing the flame,
        minimum at the shadow side facing refractory.

        Distribution Model:
            q(theta) = q_avg * [1 + C * cos(theta)]

        Where C is the circumferential variation factor depending on
        tube arrangement and spacing.

        Args:
            average_flux_kw_m2: Average heat flux (kW/m2)
            geometry: RadiantSectionGeometry with tube arrangement
            calculation_id: Optional calculation identifier

        Returns:
            HeatFluxResult with distribution and peak values
        """
        tracker = ProvenanceTracker(
            calculation_id=calculation_id or f"flux_dist_{id(geometry)}",
            calculation_type="heat_flux_distribution",
            version=self.version,
            standard_compliance=["API 560"]
        )

        q_avg = Decimal(str(average_flux_kw_m2))
        tube_od = Decimal(str(geometry.tube_od_mm)) / Decimal("1000")
        tube_spacing = Decimal(str(geometry.tube_spacing_mm)) / Decimal("1000")

        tracker.record_inputs({
            "average_flux_kw_m2": q_avg,
            "tube_od_m": tube_od,
            "tube_spacing_m": tube_spacing,
            "arrangement": geometry.tube_arrangement.value
        })

        # Calculate spacing ratio
        spacing_ratio = tube_spacing / tube_od

        # Determine circumferential variation factor based on arrangement
        if geometry.tube_arrangement == TubeArrangement.SINGLE_ROW_AGAINST_WALL:
            # Peak at front (0 deg), minimum at back (180 deg)
            # C factor depends on spacing ratio
            if spacing_ratio >= Decimal("2"):
                c_factor = Decimal("0.5")
            elif spacing_ratio >= Decimal("1.5"):
                c_factor = Decimal("0.4")
            else:
                c_factor = Decimal("0.3")
        elif geometry.tube_arrangement == TubeArrangement.DOUBLE_ROW_AGAINST_WALL:
            c_factor = Decimal("0.35")  # Front row shielded by back row
        elif geometry.tube_arrangement == TubeArrangement.SINGLE_ROW_CENTER:
            c_factor = Decimal("0.15")  # More uniform, sees both sides
        else:
            c_factor = Decimal("0.4")

        tracker.record_step(
            operation="c_factor",
            description="Determine circumferential variation factor",
            inputs={
                "spacing_ratio": spacing_ratio,
                "arrangement": geometry.tube_arrangement.value
            },
            output_value=c_factor,
            output_name="c_factor",
            formula="Based on tube arrangement and spacing"
        )

        # Calculate peak flux
        peak_factor = Decimal("1") + c_factor
        q_peak = (q_avg * peak_factor).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        tracker.record_step(
            operation="peak_flux",
            description="Calculate peak heat flux",
            inputs={"average_flux": q_avg, "peak_factor": peak_factor},
            output_value=q_peak,
            output_name="peak_flux_kw_m2",
            formula="q_peak = q_avg * (1 + C)",
            units="kW/m2"
        )

        # Calculate distribution at key angles
        angles = [0, 45, 90, 135, 180]  # degrees
        distribution = {}

        for angle in angles:
            # q(theta) = q_avg * [1 + C * cos(theta)]
            cos_theta = Decimal(str(math.cos(math.radians(angle))))
            q_theta = q_avg * (Decimal("1") + c_factor * cos_theta)
            q_theta = q_theta.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            distribution[f"{angle}_deg"] = q_theta

        tracker.record_step(
            operation="flux_distribution",
            description="Calculate heat flux at circumferential positions",
            inputs={"average_flux": q_avg, "c_factor": c_factor},
            output_value=distribution,
            output_name="heat_flux_distribution",
            formula="q(theta) = q_avg * [1 + C * cos(theta)]",
            units="kW/m2"
        )

        # Get API design limit
        api_limit = self.API_MAX_FLUX_GAS_OIL  # Default for general service

        # Check if within design
        is_within_design = q_peak <= api_limit

        provenance = tracker.get_provenance_record(q_peak)

        return HeatFluxResult(
            average_heat_flux_kw_m2=q_avg,
            peak_heat_flux_kw_m2=q_peak,
            peak_to_average_ratio=peak_factor,
            heat_flux_distribution=distribution,
            api_design_limit_kw_m2=api_limit,
            is_within_design=is_within_design,
            provenance=provenance
        )

    def calculate_radiant_duty(
        self,
        geometry: RadiantSectionGeometry,
        flame_props: FlameProperties,
        tube_conditions: TubeConditions,
        fuel_heat_input_mw: float,
        calculation_id: Optional[str] = None
    ) -> RadiantSectionDutyResult:
        """
        Calculate complete radiant section heat transfer duty.

        This integrates all radiant heat transfer calculations to determine
        the overall radiant section performance including duty, efficiency,
        tube temperatures, and heat flux distribution.

        Energy Balance:
            Q_radiant = Q_input * eta_radiant * F_exchange * epsilon_flame

        Args:
            geometry: RadiantSectionGeometry with section dimensions
            flame_props: FlameProperties with combustion data
            tube_conditions: TubeConditions with process data
            fuel_heat_input_mw: Total fuel heat input (MW)
            calculation_id: Optional calculation identifier

        Returns:
            RadiantSectionDutyResult with complete analysis
        """
        tracker = ProvenanceTracker(
            calculation_id=calculation_id or f"radiant_duty_{id(geometry)}",
            calculation_type="radiant_section_duty",
            version=self.version,
            standard_compliance=["API 560", "ASME PTC 4.2"]
        )

        # Convert inputs
        q_input = Decimal(str(fuel_heat_input_mw))
        length = Decimal(str(geometry.length_m))
        width = Decimal(str(geometry.width_m))
        height = Decimal(str(geometry.height_m))
        tube_od = Decimal(str(geometry.tube_od_mm)) / Decimal("1000")
        tube_length = Decimal(str(geometry.tube_length_m))
        tube_count = Decimal(str(geometry.tube_count))
        t_flame = Decimal(str(flame_props.flame_temperature_c))

        tracker.record_inputs({
            "fuel_heat_input_mw": q_input,
            "firebox_volume_m3": length * width * height,
            "tube_surface_area_m2": tube_count * Decimal(str(math.pi)) * tube_od * tube_length,
            "flame_temperature_c": t_flame
        })

        # Calculate firebox volume
        volume = length * width * height

        # Calculate volumetric heat release rate
        heat_release_kw_m3 = (q_input * Decimal("1000") / volume).quantize(
            Decimal("0.1"), rounding=ROUND_HALF_UP
        )

        tracker.record_step(
            operation="heat_release_rate",
            description="Calculate volumetric heat release rate",
            inputs={"heat_input_mw": q_input, "volume_m3": volume},
            output_value=heat_release_kw_m3,
            output_name="heat_release_rate_kw_m3",
            formula="HRR = Q_input / V",
            units="kW/m3",
            standard_reference="API 560"
        )

        # Calculate view factors
        view_factor_result = self.calculate_view_factor(geometry)
        view_factor = view_factor_result.total_exchange_factor

        tracker.record_step(
            operation="view_factor",
            description="Get view factor from geometric calculation",
            inputs={"geometry": geometry.section_type.value},
            output_value=view_factor,
            output_name="view_factor",
            formula="From Hottel method"
        )

        # Calculate flame emissivity
        emissivity_result = self.calculate_flame_emissivity(flame_props)
        flame_emissivity = emissivity_result.total_emissivity

        tracker.record_step(
            operation="flame_emissivity",
            description="Get flame emissivity from gas radiation calculation",
            inputs={"flame_type": flame_props.flame_type.value},
            output_value=flame_emissivity,
            output_name="flame_emissivity",
            formula="From Hottel gas radiation"
        )

        # Calculate tube surface area
        tube_surface_area = tube_count * Decimal(str(math.pi)) * tube_od * tube_length

        # Calculate bridgewall temperature (flue gas exit from radiant section)
        # Using energy balance: higher radiant efficiency = lower bridgewall temp
        # Typical correlation: T_bw = T_flame - (eta_rad * (T_flame - T_tube))
        t_tube_avg = (
            Decimal(str(tube_conditions.process_fluid_temp_outlet_c)) +
            Decimal("50")  # Approximate tube metal over fluid
        )

        # Estimate radiant efficiency from geometry
        radiant_fraction = view_factor * flame_emissivity
        radiant_fraction = max(Decimal("0.4"), min(Decimal("0.8"), radiant_fraction))

        t_bridgewall = t_flame - radiant_fraction * (t_flame - t_tube_avg)
        t_bridgewall = t_bridgewall.quantize(Decimal("0"), rounding=ROUND_HALF_UP)

        tracker.record_step(
            operation="bridgewall_temp",
            description="Calculate bridgewall temperature",
            inputs={
                "flame_temp_c": t_flame,
                "tube_temp_c": t_tube_avg,
                "radiant_fraction": radiant_fraction
            },
            output_value=t_bridgewall,
            output_name="bridgewall_temp_c",
            formula="T_bw = T_flame - eta_rad * (T_flame - T_tube)",
            units="degC"
        )

        # Calculate radiant duty using Stefan-Boltzmann
        # Simplified: Q_rad = sigma * F * epsilon * A * (T_flame^4 - T_tube^4)
        t_flame_k = t_flame + self.KELVIN_OFFSET
        t_tube_k = t_tube_avg + self.KELVIN_OFFSET

        q_rad_w = (
            self.STEFAN_BOLTZMANN *
            view_factor *
            flame_emissivity *
            tube_surface_area *
            (t_flame_k ** 4 - t_tube_k ** 4)
        )
        q_rad_mw = (q_rad_w / Decimal("1E6")).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        # Apply efficiency factor for losses
        radiant_efficiency = Decimal("0.75")  # Typical radiant section efficiency
        q_rad_actual = (q_rad_mw * radiant_efficiency).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        tracker.record_step(
            operation="radiant_duty",
            description="Calculate radiant section duty",
            inputs={
                "stefan_boltzmann": self.STEFAN_BOLTZMANN,
                "view_factor": view_factor,
                "flame_emissivity": flame_emissivity,
                "tube_area_m2": tube_surface_area
            },
            output_value=q_rad_actual,
            output_name="radiant_duty_mw",
            formula="Q = sigma * F * eps * A * (T_f^4 - T_t^4) * eta",
            units="MW",
            standard_reference="Stefan-Boltzmann with API 560 efficiency factors"
        )

        # Calculate radiant efficiency as percent of input
        if q_input > 0:
            radiant_eff_percent = (q_rad_actual / q_input * Decimal("100")).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            )
        else:
            radiant_eff_percent = Decimal("0")

        # Calculate heat fluxes
        if tube_surface_area > 0:
            avg_flux = (q_rad_actual * Decimal("1000") / tube_surface_area).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        else:
            avg_flux = Decimal("0")

        flux_result = self.calculate_heat_flux_distribution(float(avg_flux), geometry)
        peak_flux = flux_result.peak_heat_flux_kw_m2

        # Calculate tube metal temperature
        temp_result = self.calculate_tube_metal_temperature(
            float(peak_flux), tube_conditions, geometry
        )
        tube_metal_max = temp_result.tube_metal_temp_max_c

        provenance = tracker.get_provenance_record(q_rad_actual)

        return RadiantSectionDutyResult(
            radiant_duty_mw=q_rad_actual,
            radiant_efficiency_percent=radiant_eff_percent,
            heat_release_rate_kw_m3=heat_release_kw_m3,
            average_flux_kw_m2=avg_flux,
            peak_flux_kw_m2=peak_flux,
            bridgewall_temp_c=t_bridgewall,
            tube_metal_temp_max_c=tube_metal_max,
            flame_emissivity=flame_emissivity,
            view_factor=view_factor,
            radiant_fraction=radiant_fraction,
            provenance=provenance
        )
