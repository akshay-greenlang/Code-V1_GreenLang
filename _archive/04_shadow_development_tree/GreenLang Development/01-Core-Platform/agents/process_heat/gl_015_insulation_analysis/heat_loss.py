"""
GL-015 INSULSCAN - Heat Loss Calculator

Comprehensive heat loss calculations for insulated and bare surfaces
per ASTM C680. Supports cylindrical (pipes), spherical (vessels),
and flat surface geometries.

All calculations are DETERMINISTIC - zero hallucination.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging
import math

from pydantic import BaseModel

from greenlang.agents.process_heat.gl_015_insulation_analysis.schemas import (
    InsulationInput,
    InsulationLayer,
    PipeGeometry,
    VesselGeometry,
    FlatSurfaceGeometry,
    HeatLossResult,
    GeometryType,
    JacketingSpec,
)
from greenlang.agents.process_heat.gl_015_insulation_analysis.materials import (
    InsulationMaterialDatabase,
    InsulationMaterial,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

class HeatTransferConstants:
    """Heat transfer constants for calculations."""

    # Stefan-Boltzmann constant (BTU/hr-ft2-R4)
    STEFAN_BOLTZMANN = 1.714e-9

    # Air properties at ~77F
    AIR_KINEMATIC_VISCOSITY = 1.64e-4  # ft2/s
    AIR_THERMAL_CONDUCTIVITY = 0.015  # BTU/hr-ft-F
    AIR_PRANDTL_NUMBER = 0.71

    # Gravity constant
    GRAVITY = 32.174  # ft/s2

    # Surface emissivity defaults
    EMISSIVITY_ALUMINUM = 0.10
    EMISSIVITY_GALVANIZED = 0.28
    EMISSIVITY_PAINTED = 0.90
    EMISSIVITY_BARE_STEEL = 0.80
    EMISSIVITY_STAINLESS = 0.30

    # Wind effect coefficient
    WIND_COEFFICIENT = 0.4  # BTU/hr-ft2-F per mph^0.5


@dataclass
class ConvectionCoefficient:
    """Convection heat transfer coefficient result."""
    h_conv: float  # BTU/hr-ft2-F
    method: str  # Calculation method
    reynolds: Optional[float] = None
    nusselt: Optional[float] = None
    grashof: Optional[float] = None
    rayleigh: Optional[float] = None


@dataclass
class RadiationCoefficient:
    """Radiation heat transfer coefficient result."""
    h_rad: float  # BTU/hr-ft2-F
    emissivity: float
    view_factor: float = 1.0


class HeatLossCalculator:
    """
    Heat loss calculator per ASTM C680.

    Calculates heat loss from insulated and bare surfaces using
    deterministic engineering correlations. Supports:
    - Cylindrical surfaces (pipes)
    - Spherical surfaces (vessels, tanks)
    - Flat surfaces (walls, ducts)

    All calculations follow ASTM C680-19 methodology.

    Attributes:
        material_db: Insulation material database
        convergence_tol: Convergence tolerance for iterative calculations
        max_iterations: Maximum iterations for convergence

    Example:
        >>> calc = HeatLossCalculator()
        >>> result = calc.calculate_pipe_heat_loss(
        ...     inner_temp_f=350,
        ...     ambient_temp_f=77,
        ...     pipe_od_in=4.5,
        ...     pipe_length_ft=100,
        ...     insulation_layers=[...]
        ... )
    """

    def __init__(
        self,
        material_database: Optional[InsulationMaterialDatabase] = None,
        convergence_tol: float = 0.001,
        max_iterations: int = 100,
    ) -> None:
        """
        Initialize the heat loss calculator.

        Args:
            material_database: Insulation material database (creates new if None)
            convergence_tol: Convergence tolerance for surface temp iteration
            max_iterations: Maximum iterations for convergence
        """
        self.material_db = material_database or InsulationMaterialDatabase()
        self.convergence_tol = convergence_tol
        self.max_iterations = max_iterations
        self._calculation_count = 0

        logger.info("HeatLossCalculator initialized")

    def calculate_heat_loss(
        self,
        input_data: InsulationInput,
    ) -> HeatLossResult:
        """
        Calculate heat loss for given input.

        Routes to appropriate calculation based on geometry type.

        Args:
            input_data: Complete insulation analysis input

        Returns:
            HeatLossResult with detailed breakdown

        Raises:
            ValueError: If input is invalid or geometry not supported
        """
        self._calculation_count += 1
        logger.debug(f"Calculating heat loss for {input_data.item_id}")

        if input_data.geometry_type == GeometryType.PIPE:
            if input_data.pipe_geometry is None:
                raise ValueError("Pipe geometry required for pipe type")
            return self._calculate_pipe_heat_loss(input_data)

        elif input_data.geometry_type == GeometryType.VESSEL:
            if input_data.vessel_geometry is None:
                raise ValueError("Vessel geometry required for vessel type")
            return self._calculate_vessel_heat_loss(input_data)

        elif input_data.geometry_type == GeometryType.FLAT_SURFACE:
            if input_data.flat_geometry is None:
                raise ValueError("Flat geometry required for flat surface type")
            return self._calculate_flat_heat_loss(input_data)

        else:
            raise ValueError(f"Unsupported geometry type: {input_data.geometry_type}")

    def _calculate_pipe_heat_loss(
        self,
        input_data: InsulationInput,
    ) -> HeatLossResult:
        """
        Calculate heat loss from insulated pipe (cylindrical geometry).

        Uses radial heat conduction with series resistance model.
        Q = (T_inner - T_ambient) / R_total

        Args:
            input_data: Input with pipe geometry

        Returns:
            HeatLossResult
        """
        geom = input_data.pipe_geometry
        T_inner = input_data.operating_temperature_f
        T_ambient = input_data.ambient_temperature_f
        wind_speed = input_data.wind_speed_mph

        # Get pipe outer diameter
        pipe_od_in = geom.outer_diameter_in
        pipe_length_ft = geom.pipe_length_ft
        orientation = geom.orientation

        # Calculate total insulation thickness
        total_insulation_thickness = sum(
            layer.thickness_in for layer in input_data.insulation_layers
        )

        # Calculate outer radius after all insulation layers
        current_radius_in = pipe_od_in / 2
        layer_radii = [current_radius_in]

        for layer in input_data.insulation_layers:
            current_radius_in += layer.thickness_in
            layer_radii.append(current_radius_in)

        outer_radius_in = current_radius_in
        outer_diameter_in = outer_radius_in * 2

        # Get surface emissivity
        emissivity = self._get_surface_emissivity(input_data.jacketing)

        # Iterative solution for surface temperature
        T_surface = T_ambient + (T_inner - T_ambient) * 0.1  # Initial guess

        layer_temperatures = []
        layer_resistances = []

        for iteration in range(self.max_iterations):
            # Calculate surface heat transfer coefficients
            h_conv = self._calculate_convection_coefficient_cylinder(
                surface_temp_f=T_surface,
                ambient_temp_f=T_ambient,
                diameter_in=outer_diameter_in,
                wind_speed_mph=wind_speed,
                orientation=orientation,
            )

            h_rad = self._calculate_radiation_coefficient(
                surface_temp_f=T_surface,
                ambient_temp_f=T_ambient,
                emissivity=emissivity,
            )

            h_total = h_conv.h_conv + h_rad.h_rad

            # Calculate thermal resistances
            # Surface resistance (per foot of pipe length)
            outer_area_per_ft = math.pi * (outer_diameter_in / 12)  # sqft per linear ft
            R_surface = 1.0 / (h_total * outer_area_per_ft)

            # Insulation layer resistances
            R_layers = []
            T_current = T_inner
            temps = [T_inner]

            for i, layer in enumerate(input_data.insulation_layers):
                r_inner = layer_radii[i]
                r_outer = layer_radii[i + 1]

                # Get material and mean k value
                material = self.material_db.get_material(layer.material_id)
                if material is None:
                    raise ValueError(f"Unknown material: {layer.material_id}")

                # Estimate mean temperature for k-value
                T_layer_outer_est = T_current - (T_current - T_surface) * (i + 1) / len(
                    input_data.insulation_layers
                )
                k_mean = material.get_mean_thermal_conductivity(T_current, T_layer_outer_est)

                # Apply condition factor
                k_effective = k_mean * layer.condition_factor

                # Cylindrical resistance: R = ln(r2/r1) / (2*pi*k*L)
                # Per foot: R = ln(r2/r1) / (2*pi*k) * 12 (convert k from BTU-in to BTU-ft)
                R_layer = (
                    math.log(r_outer / r_inner) /
                    (2 * math.pi * (k_effective / 12))
                )
                R_layers.append(R_layer)

            R_total_insulation = sum(R_layers)
            R_total = R_total_insulation + R_surface

            # Calculate heat loss per linear foot
            Q_per_ft = (T_inner - T_ambient) / R_total if R_total > 0 else 0

            # Calculate new surface temperature
            T_surface_new = T_ambient + Q_per_ft * R_surface

            # Check convergence
            if abs(T_surface_new - T_surface) < self.convergence_tol:
                T_surface = T_surface_new
                break

            T_surface = T_surface_new

        # Calculate interface temperatures
        Q = Q_per_ft
        T_current = T_inner
        layer_temperatures = [T_inner]

        for R in R_layers:
            T_current = T_current - Q * R
            layer_temperatures.append(T_current)

        layer_temperatures.append(T_surface)

        # Total heat loss
        total_heat_loss = Q_per_ft * pipe_length_ft

        # Calculate heat loss per sqft
        outer_surface_area = math.pi * (outer_diameter_in / 12) * pipe_length_ft
        Q_per_sqft = total_heat_loss / outer_surface_area if outer_surface_area > 0 else 0

        # Calculate bare surface heat loss for comparison
        bare_heat_loss = self._calculate_bare_pipe_heat_loss(
            T_inner, T_ambient, pipe_od_in, pipe_length_ft, wind_speed, orientation
        )

        # Heat loss breakdown
        conv_fraction = h_conv.h_conv / h_total if h_total > 0 else 0.5
        rad_fraction = h_rad.h_rad / h_total if h_total > 0 else 0.5

        return HeatLossResult(
            heat_loss_btu_hr=round(total_heat_loss, 1),
            heat_loss_btu_hr_ft=round(Q_per_ft, 2),
            heat_loss_btu_hr_sqft=round(Q_per_sqft, 2),
            outer_surface_temperature_f=round(T_surface, 1),
            inner_surface_temperature_f=round(layer_temperatures[1] if len(layer_temperatures) > 1 else T_inner, 1),
            convection_heat_transfer_btu_hr=round(total_heat_loss * conv_fraction, 1),
            radiation_heat_transfer_btu_hr=round(total_heat_loss * rad_fraction, 1),
            total_thermal_resistance_hr_f_btu=round(R_total / pipe_length_ft, 6),
            layer_temperatures_f=[round(t, 1) for t in layer_temperatures],
            layer_resistances_hr_f_btu=[round(r, 6) for r in R_layers],
            bare_surface_heat_loss_btu_hr=round(bare_heat_loss, 1),
            heat_loss_reduction_pct=round(
                (1 - total_heat_loss / bare_heat_loss) * 100 if bare_heat_loss > 0 else 0, 1
            ),
            calculation_method="ASTM_C680_CYLINDRICAL",
            formula_reference="ASTM C680-19 Section 8",
        )

    def _calculate_vessel_heat_loss(
        self,
        input_data: InsulationInput,
    ) -> HeatLossResult:
        """
        Calculate heat loss from insulated vessel.

        Treats vessel as combination of cylindrical shell and heads.

        Args:
            input_data: Input with vessel geometry

        Returns:
            HeatLossResult
        """
        geom = input_data.vessel_geometry
        T_inner = input_data.operating_temperature_f
        T_ambient = input_data.ambient_temperature_f
        wind_speed = input_data.wind_speed_mph

        vessel_od_ft = geom.vessel_diameter_ft
        vessel_length_ft = geom.vessel_length_ft
        vessel_od_in = vessel_od_ft * 12

        # Calculate shell surface area
        shell_area_sqft = math.pi * vessel_od_ft * vessel_length_ft

        # Calculate head surface areas
        head_area_sqft = 0.0
        if geom.include_heads:
            if geom.head_type == "hemispherical":
                # Two hemispherical heads = one sphere
                head_area_sqft = math.pi * vessel_od_ft ** 2
            elif geom.head_type == "2:1_elliptical":
                # Approximate area of 2:1 elliptical heads
                head_area_sqft = 2 * (1.09 * math.pi * (vessel_od_ft / 2) ** 2)
            elif geom.head_type == "flat":
                head_area_sqft = 2 * (math.pi * (vessel_od_ft / 2) ** 2)
            else:  # torispherical
                head_area_sqft = 2 * (0.95 * math.pi * (vessel_od_ft / 2) ** 2)

        total_bare_area_sqft = shell_area_sqft + head_area_sqft

        # Calculate total insulation thickness
        total_insulation_thickness = sum(
            layer.thickness_in for layer in input_data.insulation_layers
        )

        # Outer diameter after insulation
        outer_od_in = vessel_od_in + 2 * total_insulation_thickness
        outer_od_ft = outer_od_in / 12

        # Outer surface area
        outer_shell_area = math.pi * outer_od_ft * vessel_length_ft
        outer_head_area = head_area_sqft * (outer_od_ft / vessel_od_ft) ** 2 if geom.include_heads else 0
        total_outer_area = outer_shell_area + outer_head_area

        # Get surface emissivity
        emissivity = self._get_surface_emissivity(input_data.jacketing)

        # Calculate using equivalent flat surface approach with area adjustment
        T_surface = T_ambient + (T_inner - T_ambient) * 0.1

        layer_resistances = []

        for iteration in range(self.max_iterations):
            # Surface coefficients
            h_conv = self._calculate_convection_coefficient_cylinder(
                surface_temp_f=T_surface,
                ambient_temp_f=T_ambient,
                diameter_in=outer_od_in,
                wind_speed_mph=wind_speed,
                orientation="horizontal",
            )

            h_rad = self._calculate_radiation_coefficient(
                surface_temp_f=T_surface,
                ambient_temp_f=T_ambient,
                emissivity=emissivity,
            )

            h_total = h_conv.h_conv + h_rad.h_rad

            # Surface resistance
            R_surface = 1.0 / (h_total * total_outer_area)

            # Insulation resistance (approximate using mean diameter)
            R_layers = []
            current_radius_in = vessel_od_in / 2

            for layer in input_data.insulation_layers:
                r_inner = current_radius_in
                r_outer = current_radius_in + layer.thickness_in
                current_radius_in = r_outer

                material = self.material_db.get_material(layer.material_id)
                if material is None:
                    raise ValueError(f"Unknown material: {layer.material_id}")

                k_mean = material.get_mean_thermal_conductivity(T_inner, T_surface)
                k_effective = k_mean * layer.condition_factor

                # Use mean area approach
                mean_radius_ft = (r_inner + r_outer) / 2 / 12
                mean_area = total_bare_area_sqft * (mean_radius_ft / (vessel_od_in / 2 / 12))

                R_layer = (layer.thickness_in / 12) / (k_effective / 12 * mean_area)
                R_layers.append(R_layer)

            R_total_insulation = sum(R_layers)
            R_total = R_total_insulation + R_surface

            # Heat loss
            Q = (T_inner - T_ambient) / R_total if R_total > 0 else 0

            # New surface temperature
            T_surface_new = T_ambient + Q * R_surface

            if abs(T_surface_new - T_surface) < self.convergence_tol:
                T_surface = T_surface_new
                layer_resistances = R_layers
                break

            T_surface = T_surface_new

        # Bare surface heat loss
        bare_heat_loss = self._calculate_bare_flat_heat_loss(
            T_inner, T_ambient, total_bare_area_sqft, wind_speed
        )

        conv_fraction = h_conv.h_conv / h_total if h_total > 0 else 0.5

        return HeatLossResult(
            heat_loss_btu_hr=round(Q, 1),
            heat_loss_btu_hr_sqft=round(Q / total_outer_area if total_outer_area > 0 else 0, 2),
            outer_surface_temperature_f=round(T_surface, 1),
            convection_heat_transfer_btu_hr=round(Q * conv_fraction, 1),
            radiation_heat_transfer_btu_hr=round(Q * (1 - conv_fraction), 1),
            total_thermal_resistance_hr_f_btu=round(R_total, 6),
            layer_resistances_hr_f_btu=[round(r, 6) for r in layer_resistances],
            bare_surface_heat_loss_btu_hr=round(bare_heat_loss, 1),
            heat_loss_reduction_pct=round(
                (1 - Q / bare_heat_loss) * 100 if bare_heat_loss > 0 else 0, 1
            ),
            calculation_method="ASTM_C680_VESSEL",
            formula_reference="ASTM C680-19 Section 8",
        )

    def _calculate_flat_heat_loss(
        self,
        input_data: InsulationInput,
    ) -> HeatLossResult:
        """
        Calculate heat loss from flat insulated surface.

        Uses simple series resistance model for planar geometry.

        Args:
            input_data: Input with flat geometry

        Returns:
            HeatLossResult
        """
        geom = input_data.flat_geometry
        T_inner = input_data.operating_temperature_f
        T_ambient = input_data.ambient_temperature_f
        wind_speed = input_data.wind_speed_mph

        surface_area_sqft = geom.surface_area_sqft
        orientation = geom.orientation

        # Get surface emissivity
        emissivity = self._get_surface_emissivity(input_data.jacketing)

        # Iterative solution
        T_surface = T_ambient + (T_inner - T_ambient) * 0.1
        layer_resistances = []

        for iteration in range(self.max_iterations):
            # Surface heat transfer coefficient
            h_conv = self._calculate_convection_coefficient_flat(
                surface_temp_f=T_surface,
                ambient_temp_f=T_ambient,
                length_ft=geom.length_ft,
                orientation=orientation,
                wind_speed_mph=wind_speed,
            )

            h_rad = self._calculate_radiation_coefficient(
                surface_temp_f=T_surface,
                ambient_temp_f=T_ambient,
                emissivity=emissivity,
            )

            h_total = h_conv.h_conv + h_rad.h_rad

            # Surface resistance
            R_surface = 1.0 / (h_total * surface_area_sqft)

            # Insulation resistances
            R_layers = []
            for layer in input_data.insulation_layers:
                material = self.material_db.get_material(layer.material_id)
                if material is None:
                    raise ValueError(f"Unknown material: {layer.material_id}")

                k_mean = material.get_mean_thermal_conductivity(T_inner, T_surface)
                k_effective = k_mean * layer.condition_factor

                # Flat surface: R = thickness / (k * A)
                R_layer = (layer.thickness_in / 12) / (k_effective / 12 * surface_area_sqft)
                R_layers.append(R_layer)

            R_total_insulation = sum(R_layers)
            R_total = R_total_insulation + R_surface

            # Heat loss
            Q = (T_inner - T_ambient) / R_total if R_total > 0 else 0

            # New surface temperature
            T_surface_new = T_ambient + Q * R_surface

            if abs(T_surface_new - T_surface) < self.convergence_tol:
                T_surface = T_surface_new
                layer_resistances = R_layers
                break

            T_surface = T_surface_new

        # Calculate interface temperatures
        layer_temperatures = [T_inner]
        T_current = T_inner
        for R in layer_resistances:
            T_current = T_current - Q * R
            layer_temperatures.append(T_current)
        layer_temperatures.append(T_surface)

        # Bare surface heat loss
        bare_heat_loss = self._calculate_bare_flat_heat_loss(
            T_inner, T_ambient, surface_area_sqft, wind_speed
        )

        conv_fraction = h_conv.h_conv / h_total if h_total > 0 else 0.5

        return HeatLossResult(
            heat_loss_btu_hr=round(Q, 1),
            heat_loss_btu_hr_sqft=round(Q / surface_area_sqft if surface_area_sqft > 0 else 0, 2),
            outer_surface_temperature_f=round(T_surface, 1),
            inner_surface_temperature_f=round(layer_temperatures[1] if len(layer_temperatures) > 1 else T_inner, 1),
            convection_heat_transfer_btu_hr=round(Q * conv_fraction, 1),
            radiation_heat_transfer_btu_hr=round(Q * (1 - conv_fraction), 1),
            total_thermal_resistance_hr_f_btu=round(R_total, 6),
            layer_temperatures_f=[round(t, 1) for t in layer_temperatures],
            layer_resistances_hr_f_btu=[round(r, 6) for r in layer_resistances],
            bare_surface_heat_loss_btu_hr=round(bare_heat_loss, 1),
            heat_loss_reduction_pct=round(
                (1 - Q / bare_heat_loss) * 100 if bare_heat_loss > 0 else 0, 1
            ),
            calculation_method="ASTM_C680_FLAT",
            formula_reference="ASTM C680-19 Section 8",
        )

    def _calculate_convection_coefficient_cylinder(
        self,
        surface_temp_f: float,
        ambient_temp_f: float,
        diameter_in: float,
        wind_speed_mph: float = 0.0,
        orientation: str = "horizontal",
    ) -> ConvectionCoefficient:
        """
        Calculate convection heat transfer coefficient for cylinder.

        Uses Churchill-Chu correlation for natural convection and
        Hilpert correlation for forced convection.

        Args:
            surface_temp_f: Surface temperature (F)
            ambient_temp_f: Ambient temperature (F)
            diameter_in: Outer diameter (inches)
            wind_speed_mph: Wind speed (mph)
            orientation: horizontal or vertical

        Returns:
            ConvectionCoefficient with h_conv
        """
        # Convert to absolute temperatures
        T_surface_R = surface_temp_f + 459.67
        T_ambient_R = ambient_temp_f + 459.67
        T_film_R = (T_surface_R + T_ambient_R) / 2
        T_film_F = T_film_R - 459.67

        # Air properties at film temperature
        beta = 1.0 / T_film_R  # Ideal gas
        nu = HeatTransferConstants.AIR_KINEMATIC_VISCOSITY * (T_film_R / 536.67) ** 1.5
        k_air = HeatTransferConstants.AIR_THERMAL_CONDUCTIVITY * (T_film_F / 77) ** 0.7
        Pr = HeatTransferConstants.AIR_PRANDTL_NUMBER

        D_ft = diameter_in / 12
        delta_T = abs(surface_temp_f - ambient_temp_f)

        if wind_speed_mph < 0.5:
            # Natural convection
            # Grashof number
            Gr = (
                HeatTransferConstants.GRAVITY * beta * delta_T * D_ft ** 3
            ) / (nu ** 2)
            Ra = Gr * Pr

            # Churchill-Chu correlation for horizontal cylinder
            if orientation == "horizontal":
                if Ra > 0:
                    Nu = (
                        0.60 + 0.387 * Ra ** (1 / 6) /
                        (1 + (0.559 / Pr) ** (9 / 16)) ** (8 / 27)
                    ) ** 2
                else:
                    Nu = 0.36
            else:  # vertical
                # For vertical cylinder, treat as vertical plate
                height_ft = D_ft * 3  # Assume length ~ 3x diameter
                Gr_v = (
                    HeatTransferConstants.GRAVITY * beta * delta_T * height_ft ** 3
                ) / (nu ** 2)
                Ra_v = Gr_v * Pr
                if Ra_v > 1e9:
                    Nu = 0.1 * Ra_v ** (1 / 3)
                else:
                    Nu = 0.59 * Ra_v ** (1 / 4) if Ra_v > 0 else 0.36

            h_conv = Nu * k_air / D_ft if D_ft > 0 else 1.0

            return ConvectionCoefficient(
                h_conv=max(h_conv, 0.5),
                method="natural_churchill_chu",
                grashof=Gr,
                rayleigh=Ra,
                nusselt=Nu,
            )

        else:
            # Forced convection (wind)
            V_fps = wind_speed_mph * 5280 / 3600  # ft/s
            Re = V_fps * D_ft / nu

            # Hilpert correlation
            if Re < 4:
                C, m = 0.989, 0.330
            elif Re < 40:
                C, m = 0.911, 0.385
            elif Re < 4000:
                C, m = 0.683, 0.466
            elif Re < 40000:
                C, m = 0.193, 0.618
            else:
                C, m = 0.027, 0.805

            Nu = C * Re ** m * Pr ** (1 / 3)
            h_forced = Nu * k_air / D_ft if D_ft > 0 else 1.0

            # Also calculate natural convection
            Gr = (
                HeatTransferConstants.GRAVITY * beta * delta_T * D_ft ** 3
            ) / (nu ** 2)
            Ra = Gr * Pr
            if Ra > 0:
                Nu_nat = (
                    0.60 + 0.387 * Ra ** (1 / 6) /
                    (1 + (0.559 / Pr) ** (9 / 16)) ** (8 / 27)
                ) ** 2
            else:
                Nu_nat = 0.36
            h_natural = Nu_nat * k_air / D_ft if D_ft > 0 else 0.5

            # Combined (root-sum-square)
            h_conv = math.sqrt(h_natural ** 2 + h_forced ** 2)

            return ConvectionCoefficient(
                h_conv=max(h_conv, 0.5),
                method="combined_hilpert",
                reynolds=Re,
                nusselt=Nu,
            )

    def _calculate_convection_coefficient_flat(
        self,
        surface_temp_f: float,
        ambient_temp_f: float,
        length_ft: float,
        orientation: str = "vertical",
        wind_speed_mph: float = 0.0,
    ) -> ConvectionCoefficient:
        """
        Calculate convection coefficient for flat surface.

        Args:
            surface_temp_f: Surface temperature (F)
            ambient_temp_f: Ambient temperature (F)
            length_ft: Characteristic length (ft)
            orientation: vertical, horizontal_up, horizontal_down
            wind_speed_mph: Wind speed (mph)

        Returns:
            ConvectionCoefficient
        """
        T_surface_R = surface_temp_f + 459.67
        T_ambient_R = ambient_temp_f + 459.67
        T_film_R = (T_surface_R + T_ambient_R) / 2
        T_film_F = T_film_R - 459.67

        beta = 1.0 / T_film_R
        nu = HeatTransferConstants.AIR_KINEMATIC_VISCOSITY * (T_film_R / 536.67) ** 1.5
        k_air = HeatTransferConstants.AIR_THERMAL_CONDUCTIVITY * (T_film_F / 77) ** 0.7
        Pr = HeatTransferConstants.AIR_PRANDTL_NUMBER

        delta_T = abs(surface_temp_f - ambient_temp_f)
        L = length_ft

        if wind_speed_mph < 0.5:
            # Natural convection
            Gr = (HeatTransferConstants.GRAVITY * beta * delta_T * L ** 3) / (nu ** 2)
            Ra = Gr * Pr

            if orientation == "vertical":
                if Ra > 1e9:
                    Nu = 0.1 * Ra ** (1 / 3)
                else:
                    Nu = 0.59 * Ra ** (1 / 4) if Ra > 0 else 0.54

            elif orientation == "horizontal_up":
                # Hot surface facing up or cold surface facing down
                if Ra > 1e7:
                    Nu = 0.15 * Ra ** (1 / 3)
                else:
                    Nu = 0.54 * Ra ** (1 / 4) if Ra > 0 else 0.54

            else:  # horizontal_down
                # Hot surface facing down or cold surface facing up
                Nu = 0.27 * Ra ** (1 / 4) if Ra > 0 else 0.27

            h_conv = Nu * k_air / L if L > 0 else 1.0

            return ConvectionCoefficient(
                h_conv=max(h_conv, 0.5),
                method="natural_flat",
                grashof=Gr,
                rayleigh=Ra,
                nusselt=Nu,
            )

        else:
            # Forced convection
            V_fps = wind_speed_mph * 5280 / 3600
            Re = V_fps * L / nu

            # Flat plate correlation
            if Re < 5e5:
                Nu = 0.664 * Re ** 0.5 * Pr ** (1 / 3)
            else:
                Nu = 0.037 * Re ** 0.8 * Pr ** (1 / 3)

            h_conv = Nu * k_air / L if L > 0 else 1.0

            return ConvectionCoefficient(
                h_conv=max(h_conv, 0.5),
                method="forced_flat",
                reynolds=Re,
                nusselt=Nu,
            )

    def _calculate_radiation_coefficient(
        self,
        surface_temp_f: float,
        ambient_temp_f: float,
        emissivity: float,
    ) -> RadiationCoefficient:
        """
        Calculate linearized radiation heat transfer coefficient.

        h_rad = epsilon * sigma * (T_s^4 - T_a^4) / (T_s - T_a)
              = epsilon * sigma * (T_s + T_a)(T_s^2 + T_a^2)

        Args:
            surface_temp_f: Surface temperature (F)
            ambient_temp_f: Ambient temperature (F)
            emissivity: Surface emissivity

        Returns:
            RadiationCoefficient
        """
        T_s = surface_temp_f + 459.67  # Rankine
        T_a = ambient_temp_f + 459.67  # Rankine

        sigma = HeatTransferConstants.STEFAN_BOLTZMANN

        if abs(T_s - T_a) < 0.1:
            h_rad = 4 * emissivity * sigma * T_s ** 3
        else:
            h_rad = emissivity * sigma * (T_s + T_a) * (T_s ** 2 + T_a ** 2)

        return RadiationCoefficient(
            h_rad=h_rad,
            emissivity=emissivity,
        )

    def _get_surface_emissivity(
        self,
        jacketing: Optional[JacketingSpec],
    ) -> float:
        """Get surface emissivity from jacketing specification."""
        if jacketing is None:
            return HeatTransferConstants.EMISSIVITY_BARE_STEEL

        return jacketing.emissivity

    def _calculate_bare_pipe_heat_loss(
        self,
        T_inner: float,
        T_ambient: float,
        pipe_od_in: float,
        pipe_length_ft: float,
        wind_speed_mph: float,
        orientation: str,
    ) -> float:
        """Calculate heat loss from bare (uninsulated) pipe."""
        T_surface = T_inner  # Bare surface

        h_conv = self._calculate_convection_coefficient_cylinder(
            T_surface, T_ambient, pipe_od_in, wind_speed_mph, orientation
        )
        h_rad = self._calculate_radiation_coefficient(
            T_surface, T_ambient, HeatTransferConstants.EMISSIVITY_BARE_STEEL
        )

        h_total = h_conv.h_conv + h_rad.h_rad
        surface_area = math.pi * (pipe_od_in / 12) * pipe_length_ft

        return h_total * surface_area * abs(T_inner - T_ambient)

    def _calculate_bare_flat_heat_loss(
        self,
        T_inner: float,
        T_ambient: float,
        surface_area_sqft: float,
        wind_speed_mph: float,
    ) -> float:
        """Calculate heat loss from bare flat surface."""
        T_surface = T_inner

        h_conv = self._calculate_convection_coefficient_flat(
            T_surface, T_ambient, math.sqrt(surface_area_sqft), "vertical", wind_speed_mph
        )
        h_rad = self._calculate_radiation_coefficient(
            T_surface, T_ambient, HeatTransferConstants.EMISSIVITY_BARE_STEEL
        )

        h_total = h_conv.h_conv + h_rad.h_rad

        return h_total * surface_area_sqft * abs(T_inner - T_ambient)

    @property
    def calculation_count(self) -> int:
        """Get total calculations performed."""
        return self._calculation_count
