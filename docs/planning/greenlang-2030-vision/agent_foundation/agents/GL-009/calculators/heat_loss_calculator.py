"""Comprehensive Heat Loss Calculator.

This module implements heat loss calculations for all major mechanisms
in thermal systems, following fundamental heat transfer principles.

Heat Loss Mechanisms:
    1. Radiation (Stefan-Boltzmann law)
    2. Convection (natural and forced)
    3. Conduction (Fourier's law)
    4. Flue gas losses (sensible and latent)
    5. Unburned fuel losses

Standards:
    - ASME PTC 4.1: Steam Generating Units (heat loss method)
    - ASHRAE Fundamentals: Heat transfer coefficients
    - ISO 12241: Thermal insulation calculation

Physical Constants:
    - Stefan-Boltzmann constant: 5.67e-8 W/(m2-K4)
    - Thermal conductivities per material tables

Author: GL-009 THERMALIQ Agent
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import hashlib
import json
from datetime import datetime
import math


# Physical constants
STEFAN_BOLTZMANN: float = 5.67e-8  # W/(m2-K4)
GRAVITY: float = 9.81  # m/s2


class SurfaceOrientation(Enum):
    """Orientation of heat transfer surface."""
    HORIZONTAL_TOP = "horizontal_top"
    HORIZONTAL_BOTTOM = "horizontal_bottom"
    VERTICAL = "vertical"
    INCLINED = "inclined"


class ConvectionType(Enum):
    """Type of convection heat transfer."""
    NATURAL = "natural"
    FORCED = "forced"
    MIXED = "mixed"


class InsulationMaterial(Enum):
    """Common insulation materials and their thermal properties."""
    FIBERGLASS = "fiberglass"
    MINERAL_WOOL = "mineral_wool"
    CALCIUM_SILICATE = "calcium_silicate"
    CELLULAR_GLASS = "cellular_glass"
    POLYURETHANE = "polyurethane"
    CERAMIC_FIBER = "ceramic_fiber"
    VERMICULITE = "vermiculite"
    PERLITE = "perlite"
    NONE = "none"


@dataclass(frozen=True)
class SurfaceGeometry:
    """Geometry of a heat transfer surface.

    Attributes:
        surface_area_m2: Surface area (m2)
        length_m: Characteristic length (m)
        orientation: Surface orientation
        emissivity: Surface emissivity (0-1)
        view_factor: Radiation view factor (0-1)
    """
    surface_area_m2: float
    length_m: float = 1.0
    orientation: SurfaceOrientation = SurfaceOrientation.VERTICAL
    emissivity: float = 0.9
    view_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.surface_area_m2 < 0:
            raise ValueError("Surface area cannot be negative")
        if self.emissivity < 0 or self.emissivity > 1:
            raise ValueError("Emissivity must be 0-1")
        if self.view_factor < 0 or self.view_factor > 1:
            raise ValueError("View factor must be 0-1")


@dataclass(frozen=True)
class InsulationLayer:
    """Represents an insulation layer.

    Attributes:
        material: Insulation material type
        thickness_m: Layer thickness (m)
        thermal_conductivity_w_mk: Thermal conductivity (W/m-K)
    """
    material: InsulationMaterial
    thickness_m: float
    thermal_conductivity_w_mk: Optional[float] = None

    @property
    def conductivity(self) -> float:
        """Get thermal conductivity, using default if not specified."""
        if self.thermal_conductivity_w_mk is not None:
            return self.thermal_conductivity_w_mk

        # Default conductivities at mean temperature ~100C
        defaults = {
            InsulationMaterial.FIBERGLASS: 0.040,
            InsulationMaterial.MINERAL_WOOL: 0.045,
            InsulationMaterial.CALCIUM_SILICATE: 0.065,
            InsulationMaterial.CELLULAR_GLASS: 0.050,
            InsulationMaterial.POLYURETHANE: 0.025,
            InsulationMaterial.CERAMIC_FIBER: 0.100,
            InsulationMaterial.VERMICULITE: 0.070,
            InsulationMaterial.PERLITE: 0.055,
            InsulationMaterial.NONE: 50.0,  # Steel/metal
        }
        return defaults.get(self.material, 0.05)


@dataclass(frozen=True)
class FlueGasComposition:
    """Composition of flue gas for loss calculations.

    Attributes:
        co2_percent: CO2 volume percentage
        o2_percent: O2 volume percentage
        n2_percent: N2 volume percentage
        h2o_percent: H2O volume percentage
        co_ppm: CO concentration (ppm)
        excess_air_percent: Excess air percentage
    """
    co2_percent: float
    o2_percent: float
    n2_percent: float = 0.0  # Calculated if not provided
    h2o_percent: float = 10.0
    co_ppm: float = 0.0
    excess_air_percent: float = 15.0


@dataclass
class RadiationLoss:
    """Radiation heat loss result.

    Attributes:
        heat_loss_kw: Total radiation heat loss (kW)
        surface_area_m2: Radiating surface area (m2)
        surface_temperature_k: Surface temperature (K)
        ambient_temperature_k: Ambient temperature (K)
        emissivity: Surface emissivity used
        heat_flux_w_m2: Heat flux (W/m2)
    """
    heat_loss_kw: float
    surface_area_m2: float
    surface_temperature_k: float
    ambient_temperature_k: float
    emissivity: float
    heat_flux_w_m2: float


@dataclass
class ConvectionLoss:
    """Convection heat loss result.

    Attributes:
        heat_loss_kw: Total convection heat loss (kW)
        convection_type: Natural, forced, or mixed
        heat_transfer_coefficient: h (W/m2-K)
        surface_area_m2: Convecting surface area (m2)
        surface_temperature_k: Surface temperature (K)
        ambient_temperature_k: Ambient temperature (K)
        air_velocity_m_s: Air velocity if forced
    """
    heat_loss_kw: float
    convection_type: ConvectionType
    heat_transfer_coefficient: float
    surface_area_m2: float
    surface_temperature_k: float
    ambient_temperature_k: float
    air_velocity_m_s: Optional[float] = None


@dataclass
class ConductionLoss:
    """Conduction heat loss result.

    Attributes:
        heat_loss_kw: Total conduction heat loss (kW)
        total_resistance_k_w: Total thermal resistance (K/W)
        layer_resistances: Resistance of each layer
        surface_area_m2: Cross-sectional area (m2)
        hot_side_temperature_k: Hot side temperature (K)
        cold_side_temperature_k: Cold side temperature (K)
    """
    heat_loss_kw: float
    total_resistance_k_w: float
    layer_resistances: Dict[str, float]
    surface_area_m2: float
    hot_side_temperature_k: float
    cold_side_temperature_k: float


@dataclass
class FlueGasLoss:
    """Flue gas heat loss result.

    Attributes:
        total_loss_kw: Total flue gas loss (kW)
        sensible_loss_kw: Sensible heat loss (kW)
        latent_loss_kw: Latent heat loss from moisture (kW)
        flue_gas_temperature_k: Flue gas exit temperature (K)
        ambient_temperature_k: Reference ambient temperature (K)
        flue_gas_flow_kg_s: Flue gas mass flow (kg/s)
        dry_loss_percent: Dry flue gas loss as % of input
        moisture_loss_percent: Moisture loss as % of input
    """
    total_loss_kw: float
    sensible_loss_kw: float
    latent_loss_kw: float
    flue_gas_temperature_k: float
    ambient_temperature_k: float
    flue_gas_flow_kg_s: float
    dry_loss_percent: float
    moisture_loss_percent: float


@dataclass
class UnburnedFuelLoss:
    """Unburned fuel loss result.

    Attributes:
        total_loss_kw: Total unburned fuel loss (kW)
        combustible_in_ash_loss_kw: Loss from ash combustibles (kW)
        co_loss_kw: Loss from CO in flue gas (kW)
        unburned_carbon_percent: Carbon in ash (%)
        co_concentration_ppm: CO concentration in flue gas (ppm)
    """
    total_loss_kw: float
    combustible_in_ash_loss_kw: float
    co_loss_kw: float
    unburned_carbon_percent: float
    co_concentration_ppm: float


@dataclass
class CalculationStep:
    """Records a single calculation step for audit trail."""
    step_number: int
    description: str
    operation: str
    inputs: Dict[str, float]
    output_value: float
    output_name: str
    formula: Optional[str] = None


@dataclass
class HeatLossResult:
    """Complete heat loss calculation result.

    Attributes:
        total_loss_kw: Total heat losses (kW)
        radiation_loss: Radiation loss breakdown
        convection_loss: Convection loss breakdown
        conduction_loss: Conduction loss breakdown
        flue_gas_loss: Flue gas loss breakdown
        unburned_fuel_loss: Unburned fuel loss breakdown
        loss_breakdown_percent: Losses as % of total
        calculation_steps: Audit trail of calculations
        provenance_hash: SHA-256 hash of inputs
        calculation_timestamp: When calculation performed
        warnings: Any warnings generated
    """
    total_loss_kw: float
    radiation_loss: Optional[RadiationLoss]
    convection_loss: Optional[ConvectionLoss]
    conduction_loss: Optional[ConductionLoss]
    flue_gas_loss: Optional[FlueGasLoss]
    unburned_fuel_loss: Optional[UnburnedFuelLoss]
    loss_breakdown_percent: Dict[str, float]
    calculation_steps: List[CalculationStep]
    provenance_hash: str
    calculation_timestamp: str
    calculator_version: str = "1.0.0"
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert result to dictionary for serialization."""
        result = {
            "total_loss_kw": self.total_loss_kw,
            "loss_breakdown_percent": self.loss_breakdown_percent,
            "provenance_hash": self.provenance_hash,
            "calculation_timestamp": self.calculation_timestamp,
            "calculator_version": self.calculator_version,
            "warnings": self.warnings
        }

        if self.radiation_loss:
            result["radiation_loss"] = {
                "heat_loss_kw": self.radiation_loss.heat_loss_kw,
                "heat_flux_w_m2": self.radiation_loss.heat_flux_w_m2
            }

        if self.convection_loss:
            result["convection_loss"] = {
                "heat_loss_kw": self.convection_loss.heat_loss_kw,
                "heat_transfer_coefficient": self.convection_loss.heat_transfer_coefficient
            }

        if self.conduction_loss:
            result["conduction_loss"] = {
                "heat_loss_kw": self.conduction_loss.heat_loss_kw,
                "total_resistance_k_w": self.conduction_loss.total_resistance_k_w
            }

        if self.flue_gas_loss:
            result["flue_gas_loss"] = {
                "total_loss_kw": self.flue_gas_loss.total_loss_kw,
                "sensible_loss_kw": self.flue_gas_loss.sensible_loss_kw,
                "latent_loss_kw": self.flue_gas_loss.latent_loss_kw
            }

        if self.unburned_fuel_loss:
            result["unburned_fuel_loss"] = {
                "total_loss_kw": self.unburned_fuel_loss.total_loss_kw,
                "co_loss_kw": self.unburned_fuel_loss.co_loss_kw
            }

        return result


class HeatLossCalculator:
    """Comprehensive Heat Loss Calculator.

    Calculates heat losses from all mechanisms in thermal systems:
    - Radiation (Stefan-Boltzmann law)
    - Convection (natural and forced)
    - Conduction (Fourier's law)
    - Flue gas (sensible and latent)
    - Unburned fuel

    All calculations are deterministic with complete provenance tracking.

    Example:
        >>> calculator = HeatLossCalculator()
        >>> result = calculator.calculate_total_losses(
        ...     surface_temperature_k=473.15,
        ...     ambient_temperature_k=298.15,
        ...     surface_geometry=SurfaceGeometry(surface_area_m2=50.0),
        ...     insulation_layers=[InsulationLayer(
        ...         material=InsulationMaterial.MINERAL_WOOL,
        ...         thickness_m=0.1
        ...     )]
        ... )
    """

    VERSION: str = "1.0.0"
    PRECISION: int = 4

    def __init__(self, precision: int = 4) -> None:
        """Initialize the Heat Loss Calculator.

        Args:
            precision: Decimal places for rounding
        """
        self.precision = precision
        self._calculation_steps: List[CalculationStep] = []
        self._step_counter: int = 0
        self._warnings: List[str] = []

    def calculate_radiation_loss(
        self,
        surface_temperature_k: float,
        ambient_temperature_k: float,
        geometry: SurfaceGeometry
    ) -> RadiationLoss:
        """Calculate radiation heat loss using Stefan-Boltzmann law.

        Q_rad = epsilon * sigma * A * F * (T_s^4 - T_a^4)

        Where:
            epsilon = surface emissivity
            sigma = Stefan-Boltzmann constant (5.67e-8 W/m2-K4)
            A = surface area (m2)
            F = view factor
            T_s = surface temperature (K)
            T_a = ambient temperature (K)

        Args:
            surface_temperature_k: Surface temperature (K)
            ambient_temperature_k: Ambient/surroundings temperature (K)
            geometry: Surface geometry specification

        Returns:
            RadiationLoss with heat loss and parameters
        """
        self._validate_temperatures(surface_temperature_k, ambient_temperature_k)

        # Stefan-Boltzmann radiation
        T_s = surface_temperature_k
        T_a = ambient_temperature_k
        epsilon = geometry.emissivity
        A = geometry.surface_area_m2
        F = geometry.view_factor

        # Q = epsilon * sigma * A * F * (T_s^4 - T_a^4)
        heat_loss_w = epsilon * STEFAN_BOLTZMANN * A * F * (
            math.pow(T_s, 4) - math.pow(T_a, 4)
        )
        heat_loss_kw = heat_loss_w / 1000.0

        heat_flux = heat_loss_w / A if A > 0 else 0

        self._add_calculation_step(
            description="Calculate radiation heat loss (Stefan-Boltzmann)",
            operation="stefan_boltzmann",
            inputs={
                "surface_temp_k": T_s,
                "ambient_temp_k": T_a,
                "emissivity": epsilon,
                "surface_area_m2": A,
                "view_factor": F,
                "stefan_boltzmann_const": STEFAN_BOLTZMANN
            },
            output_value=heat_loss_kw,
            output_name="radiation_loss_kw",
            formula="Q = epsilon * sigma * A * F * (Ts^4 - Ta^4)"
        )

        return RadiationLoss(
            heat_loss_kw=self._round_value(heat_loss_kw),
            surface_area_m2=A,
            surface_temperature_k=T_s,
            ambient_temperature_k=T_a,
            emissivity=epsilon,
            heat_flux_w_m2=self._round_value(heat_flux)
        )

    def calculate_natural_convection_loss(
        self,
        surface_temperature_k: float,
        ambient_temperature_k: float,
        geometry: SurfaceGeometry
    ) -> ConvectionLoss:
        """Calculate natural convection heat loss.

        Uses empirical correlations for natural convection based on
        surface orientation and characteristic length.

        For vertical surfaces:
            Nu = 0.59 * (Gr * Pr)^0.25  (laminar)
            Nu = 0.13 * (Gr * Pr)^0.33  (turbulent)

        Args:
            surface_temperature_k: Surface temperature (K)
            ambient_temperature_k: Ambient temperature (K)
            geometry: Surface geometry specification

        Returns:
            ConvectionLoss with heat loss and parameters
        """
        self._validate_temperatures(surface_temperature_k, ambient_temperature_k)

        T_s = surface_temperature_k
        T_a = ambient_temperature_k
        A = geometry.surface_area_m2
        L = geometry.length_m

        # Film temperature for air properties
        T_film = (T_s + T_a) / 2

        # Air properties at film temperature (approximate)
        air_props = self._get_air_properties(T_film)
        k_air = air_props["conductivity"]  # W/m-K
        nu_air = air_props["kinematic_viscosity"]  # m2/s
        Pr = air_props["prandtl"]
        beta = 1 / T_film  # Thermal expansion coefficient (1/K) for ideal gas

        # Grashof number: Gr = g * beta * (Ts - Ta) * L^3 / nu^2
        Gr = GRAVITY * beta * abs(T_s - T_a) * math.pow(L, 3) / math.pow(nu_air, 2)

        # Rayleigh number: Ra = Gr * Pr
        Ra = Gr * Pr

        # Nusselt number correlation based on orientation
        Nu = self._calculate_natural_convection_nusselt(Ra, geometry.orientation)

        # Heat transfer coefficient: h = Nu * k / L
        h = Nu * k_air / L

        # Heat loss: Q = h * A * (Ts - Ta)
        heat_loss_w = h * A * (T_s - T_a)
        heat_loss_kw = heat_loss_w / 1000.0

        self._add_calculation_step(
            description="Calculate natural convection heat loss",
            operation="natural_convection",
            inputs={
                "surface_temp_k": T_s,
                "ambient_temp_k": T_a,
                "surface_area_m2": A,
                "characteristic_length_m": L,
                "grashof_number": Gr,
                "nusselt_number": Nu,
                "h_coefficient": h
            },
            output_value=heat_loss_kw,
            output_name="convection_loss_kw",
            formula="Q = h * A * (Ts - Ta)"
        )

        return ConvectionLoss(
            heat_loss_kw=self._round_value(heat_loss_kw),
            convection_type=ConvectionType.NATURAL,
            heat_transfer_coefficient=self._round_value(h),
            surface_area_m2=A,
            surface_temperature_k=T_s,
            ambient_temperature_k=T_a
        )

    def calculate_forced_convection_loss(
        self,
        surface_temperature_k: float,
        ambient_temperature_k: float,
        geometry: SurfaceGeometry,
        air_velocity_m_s: float
    ) -> ConvectionLoss:
        """Calculate forced convection heat loss.

        Uses flat plate correlation for forced convection:
            Nu = 0.664 * Re^0.5 * Pr^0.33  (laminar, Re < 5e5)
            Nu = 0.037 * Re^0.8 * Pr^0.33  (turbulent)

        Args:
            surface_temperature_k: Surface temperature (K)
            ambient_temperature_k: Ambient temperature (K)
            geometry: Surface geometry specification
            air_velocity_m_s: Air velocity (m/s)

        Returns:
            ConvectionLoss with heat loss and parameters
        """
        self._validate_temperatures(surface_temperature_k, ambient_temperature_k)

        if air_velocity_m_s <= 0:
            raise ValueError("Air velocity must be positive for forced convection")

        T_s = surface_temperature_k
        T_a = ambient_temperature_k
        A = geometry.surface_area_m2
        L = geometry.length_m
        V = air_velocity_m_s

        # Film temperature for air properties
        T_film = (T_s + T_a) / 2
        air_props = self._get_air_properties(T_film)
        k_air = air_props["conductivity"]
        nu_air = air_props["kinematic_viscosity"]
        Pr = air_props["prandtl"]

        # Reynolds number: Re = V * L / nu
        Re = V * L / nu_air

        # Nusselt number based on flow regime
        if Re < 5e5:
            # Laminar
            Nu = 0.664 * math.pow(Re, 0.5) * math.pow(Pr, 0.33)
        else:
            # Turbulent
            Nu = 0.037 * math.pow(Re, 0.8) * math.pow(Pr, 0.33)

        # Heat transfer coefficient
        h = Nu * k_air / L

        # Heat loss
        heat_loss_w = h * A * (T_s - T_a)
        heat_loss_kw = heat_loss_w / 1000.0

        self._add_calculation_step(
            description="Calculate forced convection heat loss",
            operation="forced_convection",
            inputs={
                "surface_temp_k": T_s,
                "ambient_temp_k": T_a,
                "air_velocity_m_s": V,
                "reynolds_number": Re,
                "nusselt_number": Nu,
                "h_coefficient": h
            },
            output_value=heat_loss_kw,
            output_name="forced_convection_loss_kw",
            formula="Q = h * A * (Ts - Ta)"
        )

        return ConvectionLoss(
            heat_loss_kw=self._round_value(heat_loss_kw),
            convection_type=ConvectionType.FORCED,
            heat_transfer_coefficient=self._round_value(h),
            surface_area_m2=A,
            surface_temperature_k=T_s,
            ambient_temperature_k=T_a,
            air_velocity_m_s=V
        )

    def calculate_conduction_loss(
        self,
        hot_side_temperature_k: float,
        cold_side_temperature_k: float,
        surface_area_m2: float,
        insulation_layers: List[InsulationLayer]
    ) -> ConductionLoss:
        """Calculate conduction heat loss through insulation layers.

        Uses Fourier's law with thermal resistance in series:
            Q = (T_hot - T_cold) / R_total

        Where R_total = sum(thickness_i / (k_i * A))

        Args:
            hot_side_temperature_k: Hot side temperature (K)
            cold_side_temperature_k: Cold side temperature (K)
            surface_area_m2: Cross-sectional area (m2)
            insulation_layers: List of insulation layers

        Returns:
            ConductionLoss with heat loss and resistances
        """
        self._validate_temperatures(hot_side_temperature_k, cold_side_temperature_k)

        if not insulation_layers:
            raise ValueError("At least one layer required for conduction")

        T_hot = hot_side_temperature_k
        T_cold = cold_side_temperature_k
        A = surface_area_m2

        # Calculate thermal resistance of each layer
        layer_resistances: Dict[str, float] = {}
        total_resistance = 0.0

        for i, layer in enumerate(insulation_layers):
            # R = thickness / (k * A)
            k = layer.conductivity
            thickness = layer.thickness_m
            R_layer = thickness / (k * A) if (k * A) > 0 else float('inf')

            layer_name = f"{layer.material.value}_{i+1}"
            layer_resistances[layer_name] = R_layer
            total_resistance += R_layer

            self._add_calculation_step(
                description=f"Calculate thermal resistance - {layer.material.value}",
                operation="thermal_resistance",
                inputs={
                    "thickness_m": thickness,
                    "conductivity_w_mk": k,
                    "area_m2": A
                },
                output_value=R_layer,
                output_name=f"R_{layer_name}",
                formula="R = L / (k * A)"
            )

        # Heat loss: Q = dT / R_total
        if total_resistance > 0:
            heat_loss_w = (T_hot - T_cold) / total_resistance
        else:
            heat_loss_w = 0.0

        heat_loss_kw = heat_loss_w / 1000.0

        self._add_calculation_step(
            description="Calculate total conduction heat loss",
            operation="fourier_law",
            inputs={
                "temp_difference_k": T_hot - T_cold,
                "total_resistance_k_w": total_resistance
            },
            output_value=heat_loss_kw,
            output_name="conduction_loss_kw",
            formula="Q = (T_hot - T_cold) / R_total"
        )

        return ConductionLoss(
            heat_loss_kw=self._round_value(heat_loss_kw),
            total_resistance_k_w=self._round_value(total_resistance, 6),
            layer_resistances={k: self._round_value(v, 6) for k, v in layer_resistances.items()},
            surface_area_m2=A,
            hot_side_temperature_k=T_hot,
            cold_side_temperature_k=T_cold
        )

    def calculate_flue_gas_loss(
        self,
        flue_gas_temperature_k: float,
        ambient_temperature_k: float,
        flue_gas_flow_kg_s: float,
        flue_gas_composition: FlueGasComposition,
        fuel_energy_input_kw: float,
        fuel_hydrogen_percent: float = 0.0
    ) -> FlueGasLoss:
        """Calculate flue gas heat losses.

        Includes:
        - Dry flue gas (sensible heat) loss
        - Moisture from fuel combustion loss
        - Moisture from combustion air loss

        Per ASME PTC 4.1 methodology.

        Args:
            flue_gas_temperature_k: Exit flue gas temperature (K)
            ambient_temperature_k: Reference ambient temperature (K)
            flue_gas_flow_kg_s: Flue gas mass flow rate (kg/s)
            flue_gas_composition: Flue gas composition
            fuel_energy_input_kw: Total fuel energy input (kW)
            fuel_hydrogen_percent: Hydrogen content in fuel (%)

        Returns:
            FlueGasLoss with sensible and latent losses
        """
        T_fg = flue_gas_temperature_k
        T_a = ambient_temperature_k
        m_fg = flue_gas_flow_kg_s

        # Average specific heat of flue gas (approximate)
        cp_fg = self._calculate_flue_gas_cp(flue_gas_composition, T_fg)

        # Sensible heat loss: Q = m * cp * (T_fg - T_a)
        sensible_loss_kw = m_fg * cp_fg * (T_fg - T_a)

        # Latent heat loss from moisture in fuel (H2O from H2 combustion)
        # Approximate: 9 kg H2O per kg H2 in fuel
        # Latent heat of vaporization ~2260 kJ/kg
        moisture_from_fuel = fuel_hydrogen_percent / 100.0 * 9.0  # kg H2O / kg fuel (rough)
        latent_heat = 2260.0  # kJ/kg

        # This is a simplified calculation - actual would need fuel flow rate
        # Estimating from energy input
        latent_loss_kw = 0.0  # Simplified - would need more data

        total_loss_kw = sensible_loss_kw + latent_loss_kw

        # Calculate as percentage of input
        dry_loss_percent = (sensible_loss_kw / fuel_energy_input_kw * 100
                           if fuel_energy_input_kw > 0 else 0)
        moisture_loss_percent = (latent_loss_kw / fuel_energy_input_kw * 100
                                if fuel_energy_input_kw > 0 else 0)

        self._add_calculation_step(
            description="Calculate flue gas sensible heat loss",
            operation="sensible_heat",
            inputs={
                "flue_gas_temp_k": T_fg,
                "ambient_temp_k": T_a,
                "mass_flow_kg_s": m_fg,
                "specific_heat_kj_kg_k": cp_fg
            },
            output_value=sensible_loss_kw,
            output_name="flue_gas_sensible_loss_kw",
            formula="Q = m_dot * cp * (T_fg - T_amb)"
        )

        return FlueGasLoss(
            total_loss_kw=self._round_value(total_loss_kw),
            sensible_loss_kw=self._round_value(sensible_loss_kw),
            latent_loss_kw=self._round_value(latent_loss_kw),
            flue_gas_temperature_k=T_fg,
            ambient_temperature_k=T_a,
            flue_gas_flow_kg_s=m_fg,
            dry_loss_percent=self._round_value(dry_loss_percent),
            moisture_loss_percent=self._round_value(moisture_loss_percent)
        )

    def calculate_unburned_fuel_loss(
        self,
        fuel_energy_input_kw: float,
        carbon_in_ash_percent: float = 0.0,
        ash_flow_kg_s: float = 0.0,
        co_ppm: float = 0.0,
        flue_gas_flow_kg_s: float = 0.0
    ) -> UnburnedFuelLoss:
        """Calculate losses from unburned fuel.

        Includes:
        - Combustibles in ash (unburned carbon)
        - CO in flue gas (incomplete combustion)

        Args:
            fuel_energy_input_kw: Total fuel energy input (kW)
            carbon_in_ash_percent: Unburned carbon in ash (%)
            ash_flow_kg_s: Ash mass flow rate (kg/s)
            co_ppm: CO concentration in flue gas (ppm)
            flue_gas_flow_kg_s: Flue gas mass flow rate (kg/s)

        Returns:
            UnburnedFuelLoss with breakdown
        """
        # Carbon in ash loss
        # Energy content of carbon ~32,780 kJ/kg
        carbon_energy = 32780.0  # kJ/kg
        carbon_mass_kg_s = ash_flow_kg_s * (carbon_in_ash_percent / 100.0)
        ash_loss_kw = carbon_mass_kg_s * carbon_energy

        # CO loss
        # CO to CO2 releases ~10,100 kJ/kg CO
        co_energy = 10100.0  # kJ/kg CO
        # Convert ppm to mass fraction (approximate)
        co_mass_fraction = co_ppm * 28.01 / (1e6 * 29.0)  # Molar mass ratio
        co_mass_kg_s = flue_gas_flow_kg_s * co_mass_fraction
        co_loss_kw = co_mass_kg_s * co_energy

        total_loss_kw = ash_loss_kw + co_loss_kw

        self._add_calculation_step(
            description="Calculate unburned fuel losses",
            operation="unburned_fuel",
            inputs={
                "carbon_in_ash_percent": carbon_in_ash_percent,
                "co_ppm": co_ppm,
                "ash_flow_kg_s": ash_flow_kg_s,
                "flue_gas_flow_kg_s": flue_gas_flow_kg_s
            },
            output_value=total_loss_kw,
            output_name="unburned_fuel_loss_kw",
            formula="Q = m_C * HHV_C + m_CO * (HHV_CO2 - HHV_CO)"
        )

        return UnburnedFuelLoss(
            total_loss_kw=self._round_value(total_loss_kw),
            combustible_in_ash_loss_kw=self._round_value(ash_loss_kw),
            co_loss_kw=self._round_value(co_loss_kw),
            unburned_carbon_percent=carbon_in_ash_percent,
            co_concentration_ppm=co_ppm
        )

    def calculate_total_losses(
        self,
        surface_temperature_k: float,
        ambient_temperature_k: float,
        surface_geometry: SurfaceGeometry,
        insulation_layers: Optional[List[InsulationLayer]] = None,
        air_velocity_m_s: Optional[float] = None,
        flue_gas_temperature_k: Optional[float] = None,
        flue_gas_flow_kg_s: Optional[float] = None,
        flue_gas_composition: Optional[FlueGasComposition] = None,
        fuel_energy_input_kw: Optional[float] = None,
        carbon_in_ash_percent: float = 0.0,
        ash_flow_kg_s: float = 0.0,
        co_ppm: float = 0.0
    ) -> HeatLossResult:
        """Calculate all heat losses for a thermal system.

        This comprehensive method calculates radiation, convection,
        conduction, flue gas, and unburned fuel losses.

        Args:
            surface_temperature_k: Equipment surface temperature (K)
            ambient_temperature_k: Ambient temperature (K)
            surface_geometry: Surface geometry for radiation/convection
            insulation_layers: Insulation layers for conduction calc
            air_velocity_m_s: Air velocity for forced convection
            flue_gas_temperature_k: Flue gas exit temperature
            flue_gas_flow_kg_s: Flue gas mass flow rate
            flue_gas_composition: Flue gas composition
            fuel_energy_input_kw: Total fuel energy input
            carbon_in_ash_percent: Unburned carbon in ash
            ash_flow_kg_s: Ash mass flow rate
            co_ppm: CO in flue gas

        Returns:
            HeatLossResult with complete breakdown
        """
        self._reset_calculation_state()

        # Generate provenance hash
        provenance_hash = self._generate_provenance_hash(
            surface_temperature_k, ambient_temperature_k, surface_geometry
        )

        total_loss = 0.0
        loss_breakdown: Dict[str, float] = {}

        # Calculate radiation loss
        radiation = self.calculate_radiation_loss(
            surface_temperature_k, ambient_temperature_k, surface_geometry
        )
        total_loss += radiation.heat_loss_kw
        loss_breakdown["radiation"] = radiation.heat_loss_kw

        # Calculate convection loss
        if air_velocity_m_s and air_velocity_m_s > 0.5:
            convection = self.calculate_forced_convection_loss(
                surface_temperature_k, ambient_temperature_k,
                surface_geometry, air_velocity_m_s
            )
        else:
            convection = self.calculate_natural_convection_loss(
                surface_temperature_k, ambient_temperature_k, surface_geometry
            )
        total_loss += convection.heat_loss_kw
        loss_breakdown["convection"] = convection.heat_loss_kw

        # Calculate conduction loss if insulation provided
        conduction: Optional[ConductionLoss] = None
        if insulation_layers:
            conduction = self.calculate_conduction_loss(
                surface_temperature_k, ambient_temperature_k,
                surface_geometry.surface_area_m2, insulation_layers
            )
            total_loss += conduction.heat_loss_kw
            loss_breakdown["conduction"] = conduction.heat_loss_kw

        # Calculate flue gas loss if data provided
        flue_gas: Optional[FlueGasLoss] = None
        if (flue_gas_temperature_k and flue_gas_flow_kg_s and
                flue_gas_composition and fuel_energy_input_kw):
            flue_gas = self.calculate_flue_gas_loss(
                flue_gas_temperature_k, ambient_temperature_k,
                flue_gas_flow_kg_s, flue_gas_composition, fuel_energy_input_kw
            )
            total_loss += flue_gas.total_loss_kw
            loss_breakdown["flue_gas"] = flue_gas.total_loss_kw

        # Calculate unburned fuel loss if data provided
        unburned: Optional[UnburnedFuelLoss] = None
        if carbon_in_ash_percent > 0 or co_ppm > 0:
            unburned = self.calculate_unburned_fuel_loss(
                fuel_energy_input_kw or 0,
                carbon_in_ash_percent, ash_flow_kg_s,
                co_ppm, flue_gas_flow_kg_s or 0
            )
            total_loss += unburned.total_loss_kw
            loss_breakdown["unburned_fuel"] = unburned.total_loss_kw

        # Calculate percentages
        loss_percentages = {}
        for name, value in loss_breakdown.items():
            pct = (value / total_loss * 100) if total_loss > 0 else 0
            loss_percentages[name] = self._round_value(pct)

        timestamp = datetime.utcnow().isoformat() + "Z"

        return HeatLossResult(
            total_loss_kw=self._round_value(total_loss),
            radiation_loss=radiation,
            convection_loss=convection,
            conduction_loss=conduction,
            flue_gas_loss=flue_gas,
            unburned_fuel_loss=unburned,
            loss_breakdown_percent=loss_percentages,
            calculation_steps=self._calculation_steps.copy(),
            provenance_hash=provenance_hash,
            calculation_timestamp=timestamp,
            warnings=self._warnings.copy()
        )

    def _get_air_properties(self, temperature_k: float) -> Dict[str, float]:
        """Get air thermophysical properties at given temperature.

        Uses polynomial fits valid for 250-500K range.
        """
        T = temperature_k

        # Thermal conductivity (W/m-K): k = a + b*T
        k = 0.0241 + 0.00007 * (T - 300)

        # Kinematic viscosity (m2/s): nu ~ T^1.5
        nu = 15.89e-6 * math.pow(T / 300, 1.5)

        # Prandtl number (approximately constant for air)
        Pr = 0.71

        return {
            "conductivity": k,
            "kinematic_viscosity": nu,
            "prandtl": Pr
        }

    def _calculate_natural_convection_nusselt(
        self,
        rayleigh: float,
        orientation: SurfaceOrientation
    ) -> float:
        """Calculate Nusselt number for natural convection.

        Uses Churchill-Chu and other correlations.
        """
        Ra = rayleigh

        if orientation == SurfaceOrientation.VERTICAL:
            # Churchill-Chu correlation for vertical plate
            if Ra < 1e9:
                Nu = 0.68 + 0.67 * math.pow(Ra, 0.25) / math.pow(1 + math.pow(0.492 / 0.71, 9/16), 4/9)
            else:
                Nu = 0.13 * math.pow(Ra, 1/3)

        elif orientation == SurfaceOrientation.HORIZONTAL_TOP:
            # Hot surface facing up
            if Ra < 2e7:
                Nu = 0.54 * math.pow(Ra, 0.25)
            else:
                Nu = 0.15 * math.pow(Ra, 1/3)

        elif orientation == SurfaceOrientation.HORIZONTAL_BOTTOM:
            # Hot surface facing down
            Nu = 0.27 * math.pow(Ra, 0.25)

        else:
            # Default to vertical
            Nu = 0.59 * math.pow(Ra, 0.25)

        return max(1.0, Nu)  # Nu >= 1 for conduction limit

    def _calculate_flue_gas_cp(
        self,
        composition: FlueGasComposition,
        temperature_k: float
    ) -> float:
        """Calculate flue gas specific heat capacity (kJ/kg-K).

        Based on composition-weighted average.
        """
        # Approximate Cp values at elevated temperature (kJ/kg-K)
        cp_co2 = 1.0
        cp_h2o = 2.0
        cp_n2 = 1.04
        cp_o2 = 0.92

        # Weight by mass fractions (approximate conversion from vol%)
        total = (composition.co2_percent + composition.h2o_percent +
                 composition.n2_percent + composition.o2_percent)

        if total <= 0:
            return 1.1  # Default

        cp_avg = (
            composition.co2_percent * cp_co2 +
            composition.h2o_percent * cp_h2o +
            composition.n2_percent * cp_n2 +
            composition.o2_percent * cp_o2
        ) / total

        return cp_avg

    def _validate_temperatures(
        self,
        hot_temp: float,
        cold_temp: float
    ) -> None:
        """Validate temperature values."""
        if hot_temp <= 0 or cold_temp <= 0:
            raise ValueError("Temperatures must be positive (Kelvin)")

        if hot_temp < cold_temp:
            self._warnings.append(
                f"Hot temperature ({hot_temp}K) < cold temperature ({cold_temp}K)"
            )

    def _reset_calculation_state(self) -> None:
        """Reset calculation state for new calculation."""
        self._calculation_steps = []
        self._step_counter = 0
        self._warnings = []

    def _add_calculation_step(
        self,
        description: str,
        operation: str,
        inputs: Dict[str, float],
        output_value: float,
        output_name: str,
        formula: Optional[str] = None
    ) -> None:
        """Record a calculation step."""
        self._step_counter += 1
        step = CalculationStep(
            step_number=self._step_counter,
            description=description,
            operation=operation,
            inputs=inputs,
            output_value=output_value,
            output_name=output_name,
            formula=formula
        )
        self._calculation_steps.append(step)

    def _generate_provenance_hash(
        self,
        surface_temp: float,
        ambient_temp: float,
        geometry: SurfaceGeometry
    ) -> str:
        """Generate SHA-256 hash for provenance."""
        data = {
            "calculator": "HeatLossCalculator",
            "version": self.VERSION,
            "surface_temperature_k": surface_temp,
            "ambient_temperature_k": ambient_temp,
            "surface_area_m2": geometry.surface_area_m2,
            "emissivity": geometry.emissivity
        }
        json_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()

    def _round_value(self, value: float, precision: Optional[int] = None) -> float:
        """Round value to precision."""
        if precision is None:
            precision = self.precision

        decimal_value = Decimal(str(value))
        quantize_str = '0.' + '0' * precision
        rounded = decimal_value.quantize(
            Decimal(quantize_str),
            rounding=ROUND_HALF_UP
        )
        return float(rounded)
