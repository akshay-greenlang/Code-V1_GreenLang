"""
GL-015 INSULSCAN - Heat Loss Calculator Module

This module implements comprehensive heat loss calculations per ASTM C680
and CINI Manual standards for industrial insulation assessment.

Key Features:
- Conduction heat loss (Fourier's Law) for flat and cylindrical surfaces
- Convection heat loss (Newton's Cooling) with natural/forced modes
- Radiation heat loss (Stefan-Boltzmann Law)
- Combined heat transfer with thermal resistance networks
- Surface temperature iteration using energy balance
- Insulated vs bare comparison analysis
- Annual energy loss calculations

Reference Standards:
- ASTM C680: Standard Practice for Heat Loss from Insulated Pipe
- ASTM C1055: Standard Guide for Economic Thickness of Insulation
- 3E Plus Methodology (DOE Industrial Insulation)
- CINI Insulation Manual (Netherlands)
- VDI 2055 (German Thermal Insulation Standard)

Author: GL-CalculatorEngineer
Version: 1.0.0
License: Proprietary - GreenLang
"""

from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum, auto
import math
import hashlib
import json
from datetime import datetime, timezone
import uuid


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

# Stefan-Boltzmann constant (W/m^2.K^4)
STEFAN_BOLTZMANN: Decimal = Decimal("5.670374419e-8")

# Absolute zero offset (Celsius to Kelvin)
KELVIN_OFFSET: Decimal = Decimal("273.15")

# Pi to high precision
PI: Decimal = Decimal("3.14159265358979323846264338327950288419716939937510")

# Standard gravity (m/s^2)
GRAVITY: Decimal = Decimal("9.80665")

# Air properties at 20C and 1 atm (reference conditions)
AIR_PROPERTIES_20C: Dict[str, Decimal] = {
    "density_kg_m3": Decimal("1.204"),
    "specific_heat_j_kg_k": Decimal("1006"),
    "thermal_conductivity_w_m_k": Decimal("0.0257"),
    "dynamic_viscosity_pa_s": Decimal("1.825e-5"),
    "kinematic_viscosity_m2_s": Decimal("1.516e-5"),
    "prandtl_number": Decimal("0.713"),
    "thermal_diffusivity_m2_s": Decimal("2.12e-5"),
    "volumetric_expansion_coeff_k": Decimal("3.41e-3"),  # 1/T for ideal gas
}

# Default decimal precision
DEFAULT_DECIMAL_PRECISION: int = 6


# =============================================================================
# ENUMERATIONS
# =============================================================================

class SurfaceGeometry(Enum):
    """Surface geometry types for heat transfer calculations."""
    FLAT_HORIZONTAL_UP = auto()      # Flat horizontal surface, heat flow up
    FLAT_HORIZONTAL_DOWN = auto()    # Flat horizontal surface, heat flow down
    FLAT_VERTICAL = auto()           # Flat vertical surface
    CYLINDER_HORIZONTAL = auto()     # Horizontal cylinder (pipe)
    CYLINDER_VERTICAL = auto()       # Vertical cylinder
    SPHERE = auto()                  # Spherical surface


class ConvectionMode(Enum):
    """Convection mode types."""
    NATURAL = auto()                 # Natural (free) convection
    FORCED = auto()                  # Forced convection
    MIXED = auto()                   # Mixed convection


class InsulationCondition(Enum):
    """Insulation condition assessment levels."""
    EXCELLENT = auto()               # New or like-new condition
    GOOD = auto()                    # Minor wear, no damage
    FAIR = auto()                    # Moderate wear, some damage
    POOR = auto()                    # Significant damage, gaps
    FAILED = auto()                  # Complete failure, missing


class SurfaceMaterial(Enum):
    """Surface material types for emissivity lookup."""
    ALUMINUM_POLISHED = auto()
    ALUMINUM_OXIDIZED = auto()
    ALUMINUM_JACKETING = auto()
    STAINLESS_STEEL_POLISHED = auto()
    STAINLESS_STEEL_OXIDIZED = auto()
    GALVANIZED_STEEL_NEW = auto()
    GALVANIZED_STEEL_WEATHERED = auto()
    PAINTED_SURFACE = auto()
    CANVAS_FABRIC = auto()
    BARE_PIPE_STEEL = auto()
    BARE_PIPE_COPPER = auto()


class CalculationType(Enum):
    """Types of calculations for provenance tracking."""
    HEAT_LOSS_CONDUCTION = auto()
    HEAT_LOSS_CONVECTION = auto()
    HEAT_LOSS_RADIATION = auto()
    HEAT_LOSS_TOTAL = auto()
    SURFACE_TEMPERATURE = auto()
    ANNUAL_ENERGY_LOSS = auto()
    INSULATION_COMPARISON = auto()


# =============================================================================
# MATERIAL PROPERTY DATABASES
# =============================================================================

# Surface emissivity values by material and condition
# Source: ASHRAE Fundamentals, VDI 2055
SURFACE_EMISSIVITY: Dict[SurfaceMaterial, Decimal] = {
    SurfaceMaterial.ALUMINUM_POLISHED: Decimal("0.05"),
    SurfaceMaterial.ALUMINUM_OXIDIZED: Decimal("0.15"),
    SurfaceMaterial.ALUMINUM_JACKETING: Decimal("0.10"),
    SurfaceMaterial.STAINLESS_STEEL_POLISHED: Decimal("0.15"),
    SurfaceMaterial.STAINLESS_STEEL_OXIDIZED: Decimal("0.85"),
    SurfaceMaterial.GALVANIZED_STEEL_NEW: Decimal("0.25"),
    SurfaceMaterial.GALVANIZED_STEEL_WEATHERED: Decimal("0.45"),
    SurfaceMaterial.PAINTED_SURFACE: Decimal("0.90"),
    SurfaceMaterial.CANVAS_FABRIC: Decimal("0.85"),
    SurfaceMaterial.BARE_PIPE_STEEL: Decimal("0.80"),
    SurfaceMaterial.BARE_PIPE_COPPER: Decimal("0.70"),
}


# Thermal conductivity of common insulation materials (W/m.K)
# Values at mean temperature of 50C unless noted
# Source: ASTM C680, Manufacturer data
@dataclass(frozen=True)
class InsulationMaterial:
    """Insulation material thermal properties."""
    name: str
    thermal_conductivity_base: Decimal  # W/m.K at reference temp
    reference_temperature_c: Decimal    # Reference temperature
    temperature_coefficient: Decimal    # k increase per C
    max_service_temp_c: Decimal         # Maximum service temperature
    density_kg_m3: Decimal              # Material density
    source: str = ""

    def get_thermal_conductivity(self, mean_temp_c: Decimal) -> Decimal:
        """
        Get thermal conductivity at specified mean temperature.

        k(T) = k_ref * (1 + coefficient * (T - T_ref))
        """
        delta_t = mean_temp_c - self.reference_temperature_c
        k = self.thermal_conductivity_base * (
            Decimal("1") + self.temperature_coefficient * delta_t
        )
        return k


INSULATION_MATERIALS: Dict[str, InsulationMaterial] = {
    "mineral_wool": InsulationMaterial(
        name="Mineral Wool (Rock Wool)",
        thermal_conductivity_base=Decimal("0.040"),
        reference_temperature_c=Decimal("50"),
        temperature_coefficient=Decimal("0.0003"),
        max_service_temp_c=Decimal("650"),
        density_kg_m3=Decimal("120"),
        source="ASTM C547"
    ),
    "fiberglass": InsulationMaterial(
        name="Fiberglass (Glass Wool)",
        thermal_conductivity_base=Decimal("0.035"),
        reference_temperature_c=Decimal("50"),
        temperature_coefficient=Decimal("0.0004"),
        max_service_temp_c=Decimal("450"),
        density_kg_m3=Decimal("48"),
        source="ASTM C547"
    ),
    "calcium_silicate": InsulationMaterial(
        name="Calcium Silicate",
        thermal_conductivity_base=Decimal("0.055"),
        reference_temperature_c=Decimal("50"),
        temperature_coefficient=Decimal("0.0002"),
        max_service_temp_c=Decimal("1000"),
        density_kg_m3=Decimal("240"),
        source="ASTM C533"
    ),
    "cellular_glass": InsulationMaterial(
        name="Cellular Glass (Foamglas)",
        thermal_conductivity_base=Decimal("0.045"),
        reference_temperature_c=Decimal("24"),
        temperature_coefficient=Decimal("0.0005"),
        max_service_temp_c=Decimal("430"),
        density_kg_m3=Decimal("115"),
        source="ASTM C552"
    ),
    "perlite": InsulationMaterial(
        name="Expanded Perlite",
        thermal_conductivity_base=Decimal("0.052"),
        reference_temperature_c=Decimal("50"),
        temperature_coefficient=Decimal("0.0003"),
        max_service_temp_c=Decimal("815"),
        density_kg_m3=Decimal("160"),
        source="ASTM C610"
    ),
    "polyurethane_foam": InsulationMaterial(
        name="Polyurethane Foam",
        thermal_conductivity_base=Decimal("0.023"),
        reference_temperature_c=Decimal("24"),
        temperature_coefficient=Decimal("0.0006"),
        max_service_temp_c=Decimal("110"),
        density_kg_m3=Decimal("32"),
        source="ASTM C591"
    ),
    "phenolic_foam": InsulationMaterial(
        name="Phenolic Foam",
        thermal_conductivity_base=Decimal("0.021"),
        reference_temperature_c=Decimal("24"),
        temperature_coefficient=Decimal("0.0005"),
        max_service_temp_c=Decimal("150"),
        density_kg_m3=Decimal("40"),
        source="ASTM C1126"
    ),
    "aerogel_blanket": InsulationMaterial(
        name="Aerogel Blanket",
        thermal_conductivity_base=Decimal("0.015"),
        reference_temperature_c=Decimal("25"),
        temperature_coefficient=Decimal("0.0002"),
        max_service_temp_c=Decimal("650"),
        density_kg_m3=Decimal("150"),
        source="ASTM C1728"
    ),
    "microporous": InsulationMaterial(
        name="Microporous Insulation",
        thermal_conductivity_base=Decimal("0.020"),
        reference_temperature_c=Decimal("400"),
        temperature_coefficient=Decimal("0.00005"),
        max_service_temp_c=Decimal("1000"),
        density_kg_m3=Decimal("250"),
        source="VDI 2055"
    ),
}


# =============================================================================
# RESULT DATA CLASSES
# =============================================================================

@dataclass(frozen=True)
class CalculationStep:
    """Immutable record of a single calculation step."""
    step_number: int
    operation: str
    description: str
    inputs: Dict[str, Any]
    output_name: str
    output_value: Any
    formula: str = ""
    reference: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step_number": self.step_number,
            "operation": self.operation,
            "description": self.description,
            "inputs": {k: str(v) if isinstance(v, Decimal) else v
                      for k, v in self.inputs.items()},
            "output_name": self.output_name,
            "output_value": str(self.output_value) if isinstance(self.output_value, Decimal) else self.output_value,
            "formula": self.formula,
            "reference": self.reference,
        }


@dataclass(frozen=True)
class ConductionResult:
    """Result of conduction heat loss calculation."""
    heat_loss_w: Decimal
    heat_loss_w_per_m: Decimal          # Per unit length (pipes)
    heat_loss_w_per_m2: Decimal         # Per unit area (flat)
    thermal_resistance_k_per_w: Decimal
    mean_temperature_c: Decimal
    temperature_gradient_c: Decimal
    geometry: SurfaceGeometry
    provenance_hash: str = ""
    calculation_steps: Tuple[CalculationStep, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "heat_loss_w": str(self.heat_loss_w),
            "heat_loss_w_per_m": str(self.heat_loss_w_per_m),
            "heat_loss_w_per_m2": str(self.heat_loss_w_per_m2),
            "thermal_resistance_k_per_w": str(self.thermal_resistance_k_per_w),
            "mean_temperature_c": str(self.mean_temperature_c),
            "temperature_gradient_c": str(self.temperature_gradient_c),
            "geometry": self.geometry.name,
            "provenance_hash": self.provenance_hash,
        }


@dataclass(frozen=True)
class ConvectionResult:
    """Result of convection heat loss calculation."""
    heat_transfer_coefficient_w_m2_k: Decimal
    heat_loss_w: Decimal
    heat_loss_w_per_m: Decimal
    heat_loss_w_per_m2: Decimal
    nusselt_number: Decimal
    reynolds_number: Optional[Decimal]
    grashof_number: Optional[Decimal]
    prandtl_number: Decimal
    convection_mode: ConvectionMode
    correlation_used: str
    provenance_hash: str = ""
    calculation_steps: Tuple[CalculationStep, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "heat_transfer_coefficient_w_m2_k": str(self.heat_transfer_coefficient_w_m2_k),
            "heat_loss_w": str(self.heat_loss_w),
            "heat_loss_w_per_m": str(self.heat_loss_w_per_m),
            "heat_loss_w_per_m2": str(self.heat_loss_w_per_m2),
            "nusselt_number": str(self.nusselt_number),
            "reynolds_number": str(self.reynolds_number) if self.reynolds_number else None,
            "grashof_number": str(self.grashof_number) if self.grashof_number else None,
            "prandtl_number": str(self.prandtl_number),
            "convection_mode": self.convection_mode.name,
            "correlation_used": self.correlation_used,
            "provenance_hash": self.provenance_hash,
        }


@dataclass(frozen=True)
class RadiationResult:
    """Result of radiation heat loss calculation."""
    heat_loss_w: Decimal
    heat_loss_w_per_m: Decimal
    heat_loss_w_per_m2: Decimal
    surface_emissivity: Decimal
    view_factor: Decimal
    effective_sky_temperature_c: Optional[Decimal]
    radiative_coefficient_w_m2_k: Decimal
    provenance_hash: str = ""
    calculation_steps: Tuple[CalculationStep, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "heat_loss_w": str(self.heat_loss_w),
            "heat_loss_w_per_m": str(self.heat_loss_w_per_m),
            "heat_loss_w_per_m2": str(self.heat_loss_w_per_m2),
            "surface_emissivity": str(self.surface_emissivity),
            "view_factor": str(self.view_factor),
            "effective_sky_temperature_c": str(self.effective_sky_temperature_c) if self.effective_sky_temperature_c else None,
            "radiative_coefficient_w_m2_k": str(self.radiative_coefficient_w_m2_k),
            "provenance_hash": self.provenance_hash,
        }


@dataclass(frozen=True)
class TotalHeatLossResult:
    """Result of total combined heat loss calculation."""
    total_heat_loss_w: Decimal
    total_heat_loss_w_per_m: Decimal
    total_heat_loss_w_per_m2: Decimal
    conduction_fraction: Decimal
    convection_fraction: Decimal
    radiation_fraction: Decimal
    surface_temperature_c: Decimal
    ambient_temperature_c: Decimal
    process_temperature_c: Decimal
    overall_u_value_w_m2_k: Decimal
    total_thermal_resistance_m2_k_per_w: Decimal
    conduction_result: ConductionResult
    convection_result: ConvectionResult
    radiation_result: RadiationResult
    iterations_to_converge: int
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_heat_loss_w": str(self.total_heat_loss_w),
            "total_heat_loss_w_per_m": str(self.total_heat_loss_w_per_m),
            "total_heat_loss_w_per_m2": str(self.total_heat_loss_w_per_m2),
            "conduction_fraction": str(self.conduction_fraction),
            "convection_fraction": str(self.convection_fraction),
            "radiation_fraction": str(self.radiation_fraction),
            "surface_temperature_c": str(self.surface_temperature_c),
            "ambient_temperature_c": str(self.ambient_temperature_c),
            "process_temperature_c": str(self.process_temperature_c),
            "overall_u_value_w_m2_k": str(self.overall_u_value_w_m2_k),
            "total_thermal_resistance_m2_k_per_w": str(self.total_thermal_resistance_m2_k_per_w),
            "iterations_to_converge": self.iterations_to_converge,
            "provenance_hash": self.provenance_hash,
        }


@dataclass(frozen=True)
class SurfaceTemperatureResult:
    """Result of surface temperature calculation."""
    surface_temperature_c: Decimal
    inner_temperature_c: Decimal
    ambient_temperature_c: Decimal
    heat_flux_w_per_m2: Decimal
    iterations_to_converge: int
    convergence_error_c: Decimal
    is_converged: bool
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "surface_temperature_c": str(self.surface_temperature_c),
            "inner_temperature_c": str(self.inner_temperature_c),
            "ambient_temperature_c": str(self.ambient_temperature_c),
            "heat_flux_w_per_m2": str(self.heat_flux_w_per_m2),
            "iterations_to_converge": self.iterations_to_converge,
            "convergence_error_c": str(self.convergence_error_c),
            "is_converged": self.is_converged,
            "provenance_hash": self.provenance_hash,
        }


@dataclass(frozen=True)
class AnnualEnergyLossResult:
    """Result of annual energy loss calculation."""
    annual_heat_loss_kwh: Decimal
    annual_heat_loss_gj: Decimal
    annual_heat_loss_mmbtu: Decimal
    equivalent_fuel_m3_natural_gas: Decimal
    equivalent_fuel_liters_oil: Decimal
    equivalent_electricity_kwh: Decimal
    co2_emissions_kg: Decimal
    operating_hours_per_year: Decimal
    average_heat_loss_w: Decimal
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "annual_heat_loss_kwh": str(self.annual_heat_loss_kwh),
            "annual_heat_loss_gj": str(self.annual_heat_loss_gj),
            "annual_heat_loss_mmbtu": str(self.annual_heat_loss_mmbtu),
            "equivalent_fuel_m3_natural_gas": str(self.equivalent_fuel_m3_natural_gas),
            "equivalent_fuel_liters_oil": str(self.equivalent_fuel_liters_oil),
            "equivalent_electricity_kwh": str(self.equivalent_electricity_kwh),
            "co2_emissions_kg": str(self.co2_emissions_kg),
            "operating_hours_per_year": str(self.operating_hours_per_year),
            "average_heat_loss_w": str(self.average_heat_loss_w),
            "provenance_hash": self.provenance_hash,
        }


@dataclass(frozen=True)
class InsulationComparisonResult:
    """Result of insulated vs bare comparison."""
    heat_loss_insulated_w: Decimal
    heat_loss_bare_w: Decimal
    heat_loss_design_w: Decimal
    heat_savings_vs_bare_w: Decimal
    heat_savings_vs_bare_percent: Decimal
    efficiency_vs_design_percent: Decimal
    surface_temp_insulated_c: Decimal
    surface_temp_bare_c: Decimal
    surface_temp_design_c: Decimal
    insulation_effectiveness: Decimal
    energy_savings_annual_kwh: Decimal
    co2_savings_annual_kg: Decimal
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "heat_loss_insulated_w": str(self.heat_loss_insulated_w),
            "heat_loss_bare_w": str(self.heat_loss_bare_w),
            "heat_loss_design_w": str(self.heat_loss_design_w),
            "heat_savings_vs_bare_w": str(self.heat_savings_vs_bare_w),
            "heat_savings_vs_bare_percent": str(self.heat_savings_vs_bare_percent),
            "efficiency_vs_design_percent": str(self.efficiency_vs_design_percent),
            "surface_temp_insulated_c": str(self.surface_temp_insulated_c),
            "surface_temp_bare_c": str(self.surface_temp_bare_c),
            "surface_temp_design_c": str(self.surface_temp_design_c),
            "insulation_effectiveness": str(self.insulation_effectiveness),
            "energy_savings_annual_kwh": str(self.energy_savings_annual_kwh),
            "co2_savings_annual_kg": str(self.co2_savings_annual_kg),
            "provenance_hash": self.provenance_hash,
        }


# =============================================================================
# PROVENANCE BUILDER
# =============================================================================

class ProvenanceBuilder:
    """Builder for creating provenance records with calculation steps."""

    def __init__(self, calculation_type: CalculationType):
        """Initialize provenance builder."""
        self._record_id = str(uuid.uuid4())
        self._calculation_type = calculation_type
        self._timestamp = datetime.now(timezone.utc).isoformat()
        self._inputs: Dict[str, Any] = {}
        self._outputs: Dict[str, Any] = {}
        self._steps: List[CalculationStep] = []

    def add_input(self, name: str, value: Any) -> "ProvenanceBuilder":
        """Add an input parameter."""
        self._inputs[name] = value
        return self

    def add_output(self, name: str, value: Any) -> "ProvenanceBuilder":
        """Add an output value."""
        self._outputs[name] = value
        return self

    def add_step(
        self,
        step_number: int,
        operation: str,
        description: str,
        inputs: Dict[str, Any],
        output_name: str,
        output_value: Any,
        formula: str = "",
        reference: str = ""
    ) -> "ProvenanceBuilder":
        """Add a calculation step."""
        step = CalculationStep(
            step_number=step_number,
            operation=operation,
            description=description,
            inputs=inputs,
            output_name=output_name,
            output_value=output_value,
            formula=formula,
            reference=reference
        )
        self._steps.append(step)
        return self

    def build_hash(self) -> str:
        """Calculate SHA-256 hash of the calculation."""
        hash_data = {
            "record_id": self._record_id,
            "calculation_type": self._calculation_type.name,
            "timestamp": self._timestamp,
            "inputs": {k: str(v) if isinstance(v, Decimal) else v
                      for k, v in self._inputs.items()},
            "outputs": {k: str(v) if isinstance(v, Decimal) else v
                       for k, v in self._outputs.items()},
            "steps": [step.to_dict() for step in self._steps],
        }
        json_str = json.dumps(hash_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    def get_steps(self) -> Tuple[CalculationStep, ...]:
        """Get calculation steps as tuple."""
        return tuple(self._steps)


# =============================================================================
# HEAT LOSS CALCULATOR
# =============================================================================

class HeatLossCalculator:
    """
    Comprehensive heat loss calculator for industrial insulation.

    Implements ASTM C680, CINI Manual, and 3E Plus methodologies
    for calculating heat loss from insulated and bare surfaces.

    Features:
    - Conduction through insulation layers (Fourier's Law)
    - Natural and forced convection (empirical correlations)
    - Thermal radiation (Stefan-Boltzmann)
    - Combined heat transfer with surface temperature iteration
    - Multi-layer insulation systems
    - Annual energy loss calculations

    All calculations are:
    - DETERMINISTIC: Same inputs produce identical outputs
    - TRACEABLE: Complete provenance with SHA-256 hashing
    - STANDARDS-BASED: Per ASTM C680, VDI 2055, CINI Manual

    Example:
        >>> calc = HeatLossCalculator()
        >>> result = calc.calculate_total_heat_loss(
        ...     process_temperature_c=Decimal("150"),
        ...     ambient_temperature_c=Decimal("20"),
        ...     pipe_outer_diameter_m=Decimal("0.1"),
        ...     insulation_thickness_m=Decimal("0.05"),
        ...     insulation_material="mineral_wool",
        ...     pipe_length_m=Decimal("10"),
        ...     surface_material=SurfaceMaterial.ALUMINUM_JACKETING
        ... )
        >>> print(f"Heat loss: {result.total_heat_loss_w} W")
    """

    def __init__(
        self,
        precision: int = DEFAULT_DECIMAL_PRECISION,
        max_iterations: int = 100,
        convergence_tolerance_c: Decimal = Decimal("0.01")
    ):
        """
        Initialize Heat Loss Calculator.

        Args:
            precision: Decimal precision for calculations
            max_iterations: Maximum iterations for temperature convergence
            convergence_tolerance_c: Temperature convergence tolerance in Celsius
        """
        self._precision = precision
        self._max_iterations = max_iterations
        self._convergence_tol = convergence_tolerance_c

    # =========================================================================
    # CONDUCTION HEAT LOSS
    # =========================================================================

    def calculate_conduction_loss(
        self,
        inner_temperature_c: Union[Decimal, float, str],
        outer_temperature_c: Union[Decimal, float, str],
        geometry: SurfaceGeometry,
        insulation_material: str,
        insulation_thickness_m: Union[Decimal, float, str],
        inner_diameter_m: Optional[Union[Decimal, float, str]] = None,
        length_m: Optional[Union[Decimal, float, str]] = None,
        area_m2: Optional[Union[Decimal, float, str]] = None,
        custom_thermal_conductivity_w_m_k: Optional[Union[Decimal, float, str]] = None
    ) -> ConductionResult:
        """
        Calculate conduction heat loss through insulation.

        Implements Fourier's Law for heat conduction:
        - Flat surface: Q = k * A * dT / L
        - Cylindrical: Q = 2*pi*k*L*dT / ln(r2/r1)

        Args:
            inner_temperature_c: Temperature at inner surface (process temp)
            outer_temperature_c: Temperature at outer surface (estimated)
            geometry: Surface geometry type
            insulation_material: Material name from database
            insulation_thickness_m: Insulation thickness in meters
            inner_diameter_m: Inner diameter for cylindrical geometry
            length_m: Length for cylindrical geometry or perimeter calculation
            area_m2: Surface area for flat geometry
            custom_thermal_conductivity_w_m_k: Override material k value

        Returns:
            ConductionResult with heat loss values and provenance

        Reference:
            Fourier, J. (1822). Theorie analytique de la chaleur
            ASTM C680-14, Section 8
        """
        builder = ProvenanceBuilder(CalculationType.HEAT_LOSS_CONDUCTION)

        # Convert inputs to Decimal
        T_inner = self._to_decimal(inner_temperature_c)
        T_outer = self._to_decimal(outer_temperature_c)
        thickness = self._to_decimal(insulation_thickness_m)

        builder.add_input("inner_temperature_c", T_inner)
        builder.add_input("outer_temperature_c", T_outer)
        builder.add_input("geometry", geometry.name)
        builder.add_input("insulation_material", insulation_material)
        builder.add_input("insulation_thickness_m", thickness)

        # Step 1: Calculate mean temperature
        T_mean = (T_inner + T_outer) / Decimal("2")

        builder.add_step(
            step_number=1,
            operation="calculate",
            description="Calculate mean temperature for k lookup",
            inputs={"T_inner": T_inner, "T_outer": T_outer},
            output_name="T_mean",
            output_value=T_mean,
            formula="T_mean = (T_inner + T_outer) / 2"
        )

        # Step 2: Get thermal conductivity
        if custom_thermal_conductivity_w_m_k is not None:
            k = self._to_decimal(custom_thermal_conductivity_w_m_k)
        elif insulation_material in INSULATION_MATERIALS:
            material = INSULATION_MATERIALS[insulation_material]
            k = material.get_thermal_conductivity(T_mean)
        else:
            raise ValueError(f"Unknown insulation material: {insulation_material}")

        builder.add_step(
            step_number=2,
            operation="lookup",
            description="Get thermal conductivity at mean temperature",
            inputs={"material": insulation_material, "T_mean": T_mean},
            output_name="thermal_conductivity",
            output_value=k,
            formula="k(T) = k_ref * (1 + coeff * (T - T_ref))",
            reference="ASTM C680"
        )

        # Step 3: Calculate temperature difference
        delta_T = T_inner - T_outer

        builder.add_step(
            step_number=3,
            operation="subtract",
            description="Calculate temperature difference",
            inputs={"T_inner": T_inner, "T_outer": T_outer},
            output_name="delta_T",
            output_value=delta_T,
            formula="dT = T_inner - T_outer"
        )

        # Step 4: Calculate heat loss based on geometry
        if geometry in [SurfaceGeometry.CYLINDER_HORIZONTAL,
                       SurfaceGeometry.CYLINDER_VERTICAL]:
            # Cylindrical geometry (pipes)
            if inner_diameter_m is None or length_m is None:
                raise ValueError("Cylindrical geometry requires inner_diameter_m and length_m")

            d_inner = self._to_decimal(inner_diameter_m)
            L = self._to_decimal(length_m)
            r_inner = d_inner / Decimal("2")
            r_outer = r_inner + thickness

            builder.add_input("inner_diameter_m", d_inner)
            builder.add_input("length_m", L)

            # Q = 2*pi*k*L*dT / ln(r2/r1) - DETERMINISTIC
            ln_ratio = self._ln(r_outer / r_inner)
            Q = (Decimal("2") * PI * k * L * delta_T) / ln_ratio

            # Thermal resistance: R = ln(r2/r1) / (2*pi*k*L)
            R_thermal = ln_ratio / (Decimal("2") * PI * k * L)

            # Per unit length
            Q_per_m = Q / L

            # Per unit area (outer surface)
            A_outer = Decimal("2") * PI * r_outer * L
            Q_per_m2 = Q / A_outer

            builder.add_step(
                step_number=4,
                operation="calculate",
                description="Calculate cylindrical conduction heat loss",
                inputs={
                    "k": k, "L": L, "delta_T": delta_T,
                    "r_inner": r_inner, "r_outer": r_outer
                },
                output_name="heat_loss_w",
                output_value=Q,
                formula="Q = 2*pi*k*L*dT / ln(r2/r1)",
                reference="Fourier's Law for radial conduction"
            )

        else:
            # Flat geometry
            if area_m2 is None:
                raise ValueError("Flat geometry requires area_m2")

            A = self._to_decimal(area_m2)
            builder.add_input("area_m2", A)

            # Q = k * A * dT / L - DETERMINISTIC
            Q = (k * A * delta_T) / thickness

            # Thermal resistance: R = L / (k * A)
            R_thermal = thickness / (k * A)

            # Per unit length (if length provided)
            if length_m is not None:
                L = self._to_decimal(length_m)
                Q_per_m = Q / L
            else:
                Q_per_m = Q  # Assume unit length

            # Per unit area
            Q_per_m2 = Q / A

            builder.add_step(
                step_number=4,
                operation="calculate",
                description="Calculate flat surface conduction heat loss",
                inputs={"k": k, "A": A, "delta_T": delta_T, "L": thickness},
                output_name="heat_loss_w",
                output_value=Q,
                formula="Q = k * A * dT / L",
                reference="Fourier's Law for plane wall"
            )

        # Finalize
        builder.add_output("heat_loss_w", Q)
        builder.add_output("thermal_resistance_k_per_w", R_thermal)
        provenance_hash = builder.build_hash()

        return ConductionResult(
            heat_loss_w=self._apply_precision(Q),
            heat_loss_w_per_m=self._apply_precision(Q_per_m),
            heat_loss_w_per_m2=self._apply_precision(Q_per_m2),
            thermal_resistance_k_per_w=self._apply_precision(R_thermal, 8),
            mean_temperature_c=self._apply_precision(T_mean, 2),
            temperature_gradient_c=self._apply_precision(delta_T, 2),
            geometry=geometry,
            provenance_hash=provenance_hash,
            calculation_steps=builder.get_steps()
        )

    # =========================================================================
    # CONVECTION HEAT LOSS
    # =========================================================================

    def calculate_convection_coefficient(
        self,
        surface_temperature_c: Union[Decimal, float, str],
        ambient_temperature_c: Union[Decimal, float, str],
        geometry: SurfaceGeometry,
        characteristic_length_m: Union[Decimal, float, str],
        wind_speed_m_s: Optional[Union[Decimal, float, str]] = None,
        forced_velocity_m_s: Optional[Union[Decimal, float, str]] = None,
        is_outdoor: bool = False
    ) -> ConvectionResult:
        """
        Calculate convection heat transfer coefficient.

        Uses empirical correlations for natural and forced convection:
        - Natural convection: Churchill-Chu (vertical), Morgan (horizontal cylinder)
        - Forced convection: Hilpert correlation for cylinders

        Args:
            surface_temperature_c: Surface temperature
            ambient_temperature_c: Ambient air temperature
            geometry: Surface geometry type
            characteristic_length_m: Characteristic dimension (diameter or height)
            wind_speed_m_s: Wind speed for outdoor equipment
            forced_velocity_m_s: Forced flow velocity
            is_outdoor: Whether equipment is outdoors

        Returns:
            ConvectionResult with heat transfer coefficient

        Reference:
            Churchill, S.W. and Chu, H.H.S. (1975) - Vertical surfaces
            Morgan, V.T. (1975) - Horizontal cylinders
            ASTM C680-14, Annex A1
        """
        builder = ProvenanceBuilder(CalculationType.HEAT_LOSS_CONVECTION)

        # Convert inputs
        T_s = self._to_decimal(surface_temperature_c)
        T_amb = self._to_decimal(ambient_temperature_c)
        L_char = self._to_decimal(characteristic_length_m)

        builder.add_input("surface_temperature_c", T_s)
        builder.add_input("ambient_temperature_c", T_amb)
        builder.add_input("geometry", geometry.name)
        builder.add_input("characteristic_length_m", L_char)

        # Step 1: Calculate film temperature for air properties
        T_film = (T_s + T_amb) / Decimal("2")
        T_film_k = T_film + KELVIN_OFFSET

        builder.add_step(
            step_number=1,
            operation="calculate",
            description="Calculate film temperature",
            inputs={"T_s": T_s, "T_amb": T_amb},
            output_name="T_film",
            output_value=T_film,
            formula="T_film = (T_s + T_amb) / 2"
        )

        # Step 2: Get air properties at film temperature
        # Use temperature-corrected properties
        air_props = self._get_air_properties(T_film)
        nu = air_props["kinematic_viscosity_m2_s"]
        k_air = air_props["thermal_conductivity_w_m_k"]
        Pr = air_props["prandtl_number"]
        beta = Decimal("1") / T_film_k  # Volumetric expansion coefficient

        builder.add_step(
            step_number=2,
            operation="lookup",
            description="Get air properties at film temperature",
            inputs={"T_film": T_film},
            output_name="air_properties",
            output_value={"nu": nu, "k": k_air, "Pr": Pr, "beta": beta},
            reference="Air property correlations"
        )

        # Step 3: Calculate dimensionless numbers and determine mode
        delta_T = abs(T_s - T_amb)

        # Grashof number for natural convection
        # Gr = g * beta * dT * L^3 / nu^2 - DETERMINISTIC
        Gr = (GRAVITY * beta * delta_T * self._power(L_char, 3)) / self._power(nu, 2)

        builder.add_step(
            step_number=3,
            operation="calculate",
            description="Calculate Grashof number",
            inputs={"g": GRAVITY, "beta": beta, "dT": delta_T, "L": L_char, "nu": nu},
            output_name="Gr",
            output_value=Gr,
            formula="Gr = g * beta * dT * L^3 / nu^2"
        )

        # Rayleigh number
        Ra = Gr * Pr

        builder.add_step(
            step_number=4,
            operation="multiply",
            description="Calculate Rayleigh number",
            inputs={"Gr": Gr, "Pr": Pr},
            output_name="Ra",
            output_value=Ra,
            formula="Ra = Gr * Pr"
        )

        # Determine convection mode and velocity
        Re = None
        convection_mode = ConvectionMode.NATURAL
        correlation_used = ""

        if forced_velocity_m_s is not None:
            V = self._to_decimal(forced_velocity_m_s)
            Re = (V * L_char) / nu
            convection_mode = ConvectionMode.FORCED
            builder.add_input("forced_velocity_m_s", V)
        elif wind_speed_m_s is not None and is_outdoor:
            V = self._to_decimal(wind_speed_m_s)
            Re = (V * L_char) / nu
            # Check for mixed convection (Gr/Re^2 ~ 1)
            if Re > Decimal("0"):
                ratio = Gr / self._power(Re, 2)
                if ratio > Decimal("0.1") and ratio < Decimal("10"):
                    convection_mode = ConvectionMode.MIXED
                else:
                    convection_mode = ConvectionMode.FORCED
            builder.add_input("wind_speed_m_s", V)

        # Step 5: Calculate Nusselt number based on geometry and mode
        if convection_mode == ConvectionMode.NATURAL:
            Nu, correlation_used = self._calculate_natural_convection_nu(
                geometry, Ra, Pr, builder
            )
        elif convection_mode == ConvectionMode.FORCED:
            Nu, correlation_used = self._calculate_forced_convection_nu(
                geometry, Re, Pr, builder
            )
        else:
            # Mixed convection - combine natural and forced
            Nu_nat, _ = self._calculate_natural_convection_nu(geometry, Ra, Pr, builder)
            Nu_forced, _ = self._calculate_forced_convection_nu(geometry, Re, Pr, builder)
            # Churchill combination: Nu^n = Nu_nat^n + Nu_forced^n (n=3 typical)
            Nu = self._power(
                self._power(Nu_nat, 3) + self._power(Nu_forced, 3),
                Decimal("1") / Decimal("3")
            )
            correlation_used = "Churchill mixed convection"

        # Step 6: Calculate heat transfer coefficient
        # h = Nu * k / L - DETERMINISTIC
        h = (Nu * k_air) / L_char

        builder.add_step(
            step_number=6,
            operation="calculate",
            description="Calculate heat transfer coefficient",
            inputs={"Nu": Nu, "k_air": k_air, "L": L_char},
            output_name="h_conv",
            output_value=h,
            formula="h = Nu * k / L"
        )

        # These will be calculated in total heat loss
        Q = Decimal("0")
        Q_per_m = Decimal("0")
        Q_per_m2 = Decimal("0")

        builder.add_output("heat_transfer_coefficient", h)
        builder.add_output("nusselt_number", Nu)
        provenance_hash = builder.build_hash()

        return ConvectionResult(
            heat_transfer_coefficient_w_m2_k=self._apply_precision(h),
            heat_loss_w=Q,
            heat_loss_w_per_m=Q_per_m,
            heat_loss_w_per_m2=Q_per_m2,
            nusselt_number=self._apply_precision(Nu, 2),
            reynolds_number=self._apply_precision(Re, 0) if Re else None,
            grashof_number=self._apply_precision(Gr, 0),
            prandtl_number=self._apply_precision(Pr, 3),
            convection_mode=convection_mode,
            correlation_used=correlation_used,
            provenance_hash=provenance_hash,
            calculation_steps=builder.get_steps()
        )

    def _calculate_natural_convection_nu(
        self,
        geometry: SurfaceGeometry,
        Ra: Decimal,
        Pr: Decimal,
        builder: ProvenanceBuilder
    ) -> Tuple[Decimal, str]:
        """
        Calculate Nusselt number for natural convection.

        Returns:
            Tuple of (Nu, correlation_name)
        """
        if geometry == SurfaceGeometry.FLAT_VERTICAL:
            # Churchill-Chu correlation for vertical plates
            # Nu = (0.825 + 0.387*Ra^(1/6) / [1 + (0.492/Pr)^(9/16)]^(8/27))^2
            # Valid for all Ra
            term1 = Decimal("0.825")
            term2_num = Decimal("0.387") * self._power(Ra, Decimal("1") / Decimal("6"))
            term2_denom = self._power(
                Decimal("1") + self._power(Decimal("0.492") / Pr, Decimal("9") / Decimal("16")),
                Decimal("8") / Decimal("27")
            )
            Nu = self._power(term1 + term2_num / term2_denom, 2)
            correlation = "Churchill-Chu vertical plate"

            builder.add_step(
                step_number=5,
                operation="calculate",
                description="Calculate Nu using Churchill-Chu correlation",
                inputs={"Ra": Ra, "Pr": Pr},
                output_name="Nu",
                output_value=Nu,
                formula="Nu = (0.825 + 0.387*Ra^(1/6) / f(Pr))^2",
                reference="Churchill & Chu (1975)"
            )

        elif geometry == SurfaceGeometry.CYLINDER_HORIZONTAL:
            # Morgan correlation for horizontal cylinders
            # Nu = C * Ra^n where C and n depend on Ra range
            if Ra < Decimal("1e-2"):
                C, n = Decimal("0.675"), Decimal("0.058")
            elif Ra < Decimal("1e2"):
                C, n = Decimal("1.02"), Decimal("0.148")
            elif Ra < Decimal("1e4"):
                C, n = Decimal("0.850"), Decimal("0.188")
            elif Ra < Decimal("1e7"):
                C, n = Decimal("0.480"), Decimal("0.250")
            else:
                C, n = Decimal("0.125"), Decimal("0.333")

            Nu = C * self._power(Ra, n)
            correlation = f"Morgan horizontal cylinder (C={C}, n={n})"

            builder.add_step(
                step_number=5,
                operation="calculate",
                description="Calculate Nu using Morgan correlation",
                inputs={"Ra": Ra, "C": C, "n": n},
                output_name="Nu",
                output_value=Nu,
                formula="Nu = C * Ra^n",
                reference="Morgan (1975)"
            )

        elif geometry == SurfaceGeometry.FLAT_HORIZONTAL_UP:
            # Hot surface facing up
            # McAdams: Nu = 0.54 * Ra^0.25 for 10^4 < Ra < 10^7
            #          Nu = 0.15 * Ra^0.33 for 10^7 < Ra < 10^11
            if Ra < Decimal("1e7"):
                Nu = Decimal("0.54") * self._power(Ra, Decimal("0.25"))
            else:
                Nu = Decimal("0.15") * self._power(Ra, Decimal("0.333"))
            correlation = "McAdams horizontal hot face up"

            builder.add_step(
                step_number=5,
                operation="calculate",
                description="Calculate Nu for horizontal hot surface facing up",
                inputs={"Ra": Ra},
                output_name="Nu",
                output_value=Nu,
                reference="McAdams (1954)"
            )

        elif geometry == SurfaceGeometry.FLAT_HORIZONTAL_DOWN:
            # Hot surface facing down (restricted convection)
            # Nu = 0.27 * Ra^0.25
            Nu = Decimal("0.27") * self._power(Ra, Decimal("0.25"))
            correlation = "McAdams horizontal hot face down"

            builder.add_step(
                step_number=5,
                operation="calculate",
                description="Calculate Nu for horizontal hot surface facing down",
                inputs={"Ra": Ra},
                output_name="Nu",
                output_value=Nu,
                formula="Nu = 0.27 * Ra^0.25",
                reference="McAdams (1954)"
            )

        elif geometry == SurfaceGeometry.CYLINDER_VERTICAL:
            # Vertical cylinder - use vertical plate if D/L is large
            # Otherwise apply correction
            Nu_plate = self._power(
                Decimal("0.825") + Decimal("0.387") * self._power(Ra, Decimal("1") / Decimal("6")) /
                self._power(
                    Decimal("1") + self._power(Decimal("0.492") / Pr, Decimal("9") / Decimal("16")),
                    Decimal("8") / Decimal("27")
                ),
                2
            )
            Nu = Nu_plate  # Simplified - use plate correlation
            correlation = "Vertical cylinder (plate approximation)"

            builder.add_step(
                step_number=5,
                operation="calculate",
                description="Calculate Nu for vertical cylinder",
                inputs={"Ra": Ra, "Pr": Pr},
                output_name="Nu",
                output_value=Nu,
                reference="Churchill-Chu adapted"
            )

        elif geometry == SurfaceGeometry.SPHERE:
            # Yuge correlation for sphere
            # Nu = 2 + 0.43 * Ra^0.25
            Nu = Decimal("2") + Decimal("0.43") * self._power(Ra, Decimal("0.25"))
            correlation = "Yuge sphere"

            builder.add_step(
                step_number=5,
                operation="calculate",
                description="Calculate Nu for sphere",
                inputs={"Ra": Ra},
                output_name="Nu",
                output_value=Nu,
                formula="Nu = 2 + 0.43 * Ra^0.25",
                reference="Yuge (1960)"
            )

        else:
            # Default to vertical plate
            Nu = Decimal("0.59") * self._power(Ra, Decimal("0.25"))
            correlation = "Default natural convection"

        return Nu, correlation

    def _calculate_forced_convection_nu(
        self,
        geometry: SurfaceGeometry,
        Re: Decimal,
        Pr: Decimal,
        builder: ProvenanceBuilder
    ) -> Tuple[Decimal, str]:
        """
        Calculate Nusselt number for forced convection.

        Returns:
            Tuple of (Nu, correlation_name)
        """
        if geometry in [SurfaceGeometry.CYLINDER_HORIZONTAL, SurfaceGeometry.CYLINDER_VERTICAL]:
            # Hilpert correlation for cross-flow over cylinder
            # Nu = C * Re^m * Pr^(1/3)
            if Re < Decimal("4"):
                C, m = Decimal("0.989"), Decimal("0.330")
            elif Re < Decimal("40"):
                C, m = Decimal("0.911"), Decimal("0.385")
            elif Re < Decimal("4000"):
                C, m = Decimal("0.683"), Decimal("0.466")
            elif Re < Decimal("40000"):
                C, m = Decimal("0.193"), Decimal("0.618")
            else:
                C, m = Decimal("0.027"), Decimal("0.805")

            Nu = C * self._power(Re, m) * self._power(Pr, Decimal("0.333"))
            correlation = f"Hilpert cylinder (C={C}, m={m})"

            builder.add_step(
                step_number=5,
                operation="calculate",
                description="Calculate Nu using Hilpert correlation",
                inputs={"Re": Re, "Pr": Pr, "C": C, "m": m},
                output_name="Nu",
                output_value=Nu,
                formula="Nu = C * Re^m * Pr^(1/3)",
                reference="Hilpert (1933)"
            )

        else:
            # Flat plate correlation
            if Re < Decimal("5e5"):
                # Laminar: Nu = 0.664 * Re^0.5 * Pr^(1/3)
                Nu = Decimal("0.664") * self._power(Re, Decimal("0.5")) * self._power(Pr, Decimal("0.333"))
                correlation = "Flat plate laminar"
            else:
                # Turbulent: Nu = 0.037 * Re^0.8 * Pr^(1/3)
                Nu = Decimal("0.037") * self._power(Re, Decimal("0.8")) * self._power(Pr, Decimal("0.333"))
                correlation = "Flat plate turbulent"

            builder.add_step(
                step_number=5,
                operation="calculate",
                description="Calculate Nu for flat plate forced convection",
                inputs={"Re": Re, "Pr": Pr},
                output_name="Nu",
                output_value=Nu,
                reference="Incropera & DeWitt"
            )

        return Nu, correlation

    # =========================================================================
    # RADIATION HEAT LOSS
    # =========================================================================

    def calculate_radiation_loss(
        self,
        surface_temperature_c: Union[Decimal, float, str],
        ambient_temperature_c: Union[Decimal, float, str],
        surface_area_m2: Union[Decimal, float, str],
        surface_material: SurfaceMaterial,
        view_factor: Union[Decimal, float, str] = "1.0",
        is_outdoor: bool = False,
        sky_temperature_c: Optional[Union[Decimal, float, str]] = None,
        length_m: Optional[Union[Decimal, float, str]] = None,
        custom_emissivity: Optional[Union[Decimal, float, str]] = None
    ) -> RadiationResult:
        """
        Calculate radiation heat loss using Stefan-Boltzmann Law.

        Q_rad = epsilon * sigma * A * F * (T_s^4 - T_surr^4)

        For outdoor equipment, uses effective sky temperature which
        can be significantly lower than ambient air temperature.

        Args:
            surface_temperature_c: Surface temperature
            ambient_temperature_c: Ambient/surrounding temperature
            surface_area_m2: Radiating surface area
            surface_material: Surface material for emissivity lookup
            view_factor: Geometric view factor (0-1)
            is_outdoor: Whether equipment is outdoors
            sky_temperature_c: Sky temperature for outdoor (optional)
            length_m: Length for per-meter calculation
            custom_emissivity: Override emissivity value

        Returns:
            RadiationResult with heat loss values

        Reference:
            Stefan, J. (1879). Boltzmann, L. (1884).
            ASTM C680-14, Annex A2
        """
        builder = ProvenanceBuilder(CalculationType.HEAT_LOSS_RADIATION)

        # Convert inputs
        T_s = self._to_decimal(surface_temperature_c)
        T_amb = self._to_decimal(ambient_temperature_c)
        A = self._to_decimal(surface_area_m2)
        F = self._to_decimal(view_factor)

        builder.add_input("surface_temperature_c", T_s)
        builder.add_input("ambient_temperature_c", T_amb)
        builder.add_input("surface_area_m2", A)
        builder.add_input("view_factor", F)
        builder.add_input("surface_material", surface_material.name)

        # Step 1: Get emissivity
        if custom_emissivity is not None:
            epsilon = self._to_decimal(custom_emissivity)
        else:
            epsilon = SURFACE_EMISSIVITY.get(surface_material, Decimal("0.90"))

        builder.add_step(
            step_number=1,
            operation="lookup",
            description="Get surface emissivity",
            inputs={"material": surface_material.name},
            output_name="emissivity",
            output_value=epsilon,
            reference="ASHRAE Fundamentals"
        )

        # Step 2: Determine effective surrounding temperature
        T_surr_c: Decimal
        if is_outdoor and sky_temperature_c is not None:
            T_surr_c = self._to_decimal(sky_temperature_c)
        elif is_outdoor:
            # Estimate sky temperature using Swinbank correlation
            # T_sky = 0.0552 * T_amb^1.5 (Kelvin)
            T_amb_k = T_amb + KELVIN_OFFSET
            T_sky_k = Decimal("0.0552") * self._power(T_amb_k, Decimal("1.5"))
            T_surr_c = T_sky_k - KELVIN_OFFSET

            builder.add_step(
                step_number=2,
                operation="calculate",
                description="Estimate sky temperature (Swinbank)",
                inputs={"T_amb_K": T_amb_k},
                output_name="T_sky_c",
                output_value=T_surr_c,
                formula="T_sky = 0.0552 * T_amb^1.5",
                reference="Swinbank (1963)"
            )
        else:
            T_surr_c = T_amb

        # Step 3: Convert to Kelvin
        T_s_k = T_s + KELVIN_OFFSET
        T_surr_k = T_surr_c + KELVIN_OFFSET

        builder.add_step(
            step_number=3,
            operation="convert",
            description="Convert temperatures to Kelvin",
            inputs={"T_s_c": T_s, "T_surr_c": T_surr_c},
            output_name="temperatures_K",
            output_value={"T_s_K": T_s_k, "T_surr_K": T_surr_k}
        )

        # Step 4: Calculate radiation heat loss
        # Q = epsilon * sigma * A * F * (T_s^4 - T_surr^4) - DETERMINISTIC
        T_s_4 = self._power(T_s_k, 4)
        T_surr_4 = self._power(T_surr_k, 4)
        Q_rad = epsilon * STEFAN_BOLTZMANN * A * F * (T_s_4 - T_surr_4)

        builder.add_step(
            step_number=4,
            operation="calculate",
            description="Calculate radiation heat loss",
            inputs={
                "epsilon": epsilon,
                "sigma": STEFAN_BOLTZMANN,
                "A": A,
                "F": F,
                "T_s^4": T_s_4,
                "T_surr^4": T_surr_4
            },
            output_name="Q_rad",
            output_value=Q_rad,
            formula="Q = epsilon * sigma * A * F * (T_s^4 - T_surr^4)",
            reference="Stefan-Boltzmann Law"
        )

        # Step 5: Calculate radiative heat transfer coefficient
        # h_rad = epsilon * sigma * (T_s^2 + T_surr^2) * (T_s + T_surr)
        T_s_2 = self._power(T_s_k, 2)
        T_surr_2 = self._power(T_surr_k, 2)
        h_rad = epsilon * STEFAN_BOLTZMANN * (T_s_2 + T_surr_2) * (T_s_k + T_surr_k)

        builder.add_step(
            step_number=5,
            operation="calculate",
            description="Calculate radiative coefficient",
            inputs={"epsilon": epsilon, "T_s": T_s_k, "T_surr": T_surr_k},
            output_name="h_rad",
            output_value=h_rad,
            formula="h_rad = epsilon * sigma * (T_s^2 + T_surr^2) * (T_s + T_surr)"
        )

        # Per unit area
        Q_per_m2 = Q_rad / A

        # Per unit length
        if length_m is not None:
            L = self._to_decimal(length_m)
            Q_per_m = Q_rad / L
        else:
            Q_per_m = Q_rad

        builder.add_output("heat_loss_w", Q_rad)
        builder.add_output("h_rad", h_rad)
        provenance_hash = builder.build_hash()

        return RadiationResult(
            heat_loss_w=self._apply_precision(Q_rad),
            heat_loss_w_per_m=self._apply_precision(Q_per_m),
            heat_loss_w_per_m2=self._apply_precision(Q_per_m2),
            surface_emissivity=epsilon,
            view_factor=F,
            effective_sky_temperature_c=self._apply_precision(T_surr_c, 2) if is_outdoor else None,
            radiative_coefficient_w_m2_k=self._apply_precision(h_rad),
            provenance_hash=provenance_hash,
            calculation_steps=builder.get_steps()
        )

    # =========================================================================
    # TOTAL COMBINED HEAT LOSS
    # =========================================================================

    def calculate_total_heat_loss(
        self,
        process_temperature_c: Union[Decimal, float, str],
        ambient_temperature_c: Union[Decimal, float, str],
        pipe_outer_diameter_m: Union[Decimal, float, str],
        insulation_thickness_m: Union[Decimal, float, str],
        insulation_material: str,
        pipe_length_m: Union[Decimal, float, str],
        surface_material: SurfaceMaterial = SurfaceMaterial.ALUMINUM_JACKETING,
        wind_speed_m_s: Optional[Union[Decimal, float, str]] = None,
        is_outdoor: bool = False,
        geometry: SurfaceGeometry = SurfaceGeometry.CYLINDER_HORIZONTAL,
        internal_heat_transfer_coeff: Optional[Union[Decimal, float, str]] = None
    ) -> TotalHeatLossResult:
        """
        Calculate total combined heat loss with surface temperature iteration.

        Uses thermal resistance network and iterates to find surface
        temperature that balances conduction through insulation with
        convection and radiation from the surface.

        Q_cond = Q_conv + Q_rad (energy balance at surface)

        Args:
            process_temperature_c: Internal process temperature
            ambient_temperature_c: Ambient temperature
            pipe_outer_diameter_m: Pipe outer diameter (before insulation)
            insulation_thickness_m: Insulation thickness
            insulation_material: Insulation material name
            pipe_length_m: Pipe length
            surface_material: Jacketing material
            wind_speed_m_s: Wind speed (outdoor)
            is_outdoor: Whether outdoor installation
            geometry: Surface geometry
            internal_heat_transfer_coeff: Internal h value (optional)

        Returns:
            TotalHeatLossResult with all heat loss components

        Reference:
            ASTM C680-14, Section 8
            3E Plus DOE methodology
        """
        builder = ProvenanceBuilder(CalculationType.HEAT_LOSS_TOTAL)

        # Convert inputs
        T_proc = self._to_decimal(process_temperature_c)
        T_amb = self._to_decimal(ambient_temperature_c)
        D_pipe = self._to_decimal(pipe_outer_diameter_m)
        t_ins = self._to_decimal(insulation_thickness_m)
        L = self._to_decimal(pipe_length_m)

        builder.add_input("process_temperature_c", T_proc)
        builder.add_input("ambient_temperature_c", T_amb)
        builder.add_input("pipe_outer_diameter_m", D_pipe)
        builder.add_input("insulation_thickness_m", t_ins)
        builder.add_input("insulation_material", insulation_material)
        builder.add_input("pipe_length_m", L)

        # Calculate outer diameter (with insulation)
        D_outer = D_pipe + Decimal("2") * t_ins
        r_outer = D_outer / Decimal("2")

        # Calculate outer surface area
        if geometry in [SurfaceGeometry.CYLINDER_HORIZONTAL, SurfaceGeometry.CYLINDER_VERTICAL]:
            A_outer = Decimal("2") * PI * r_outer * L
        else:
            A_outer = Decimal("2") * PI * r_outer * L  # Simplified

        builder.add_step(
            step_number=1,
            operation="calculate",
            description="Calculate outer surface geometry",
            inputs={"D_pipe": D_pipe, "t_ins": t_ins, "L": L},
            output_name="geometry",
            output_value={"D_outer": D_outer, "A_outer": A_outer}
        )

        # Initial guess for surface temperature
        # T_s = T_amb + 0.1 * (T_proc - T_amb)
        T_s = T_amb + Decimal("0.1") * (T_proc - T_amb)

        # Iteration loop for surface temperature
        iteration = 0
        converged = False
        Q_total = Decimal("0")

        while iteration < self._max_iterations and not converged:
            iteration += 1

            # Calculate conduction through insulation
            cond_result = self.calculate_conduction_loss(
                inner_temperature_c=T_proc,
                outer_temperature_c=T_s,
                geometry=geometry,
                insulation_material=insulation_material,
                insulation_thickness_m=t_ins,
                inner_diameter_m=D_pipe,
                length_m=L
            )
            Q_cond = cond_result.heat_loss_w

            # Calculate convection from surface
            conv_result = self.calculate_convection_coefficient(
                surface_temperature_c=T_s,
                ambient_temperature_c=T_amb,
                geometry=geometry,
                characteristic_length_m=D_outer,
                wind_speed_m_s=wind_speed_m_s,
                is_outdoor=is_outdoor
            )
            h_conv = conv_result.heat_transfer_coefficient_w_m2_k
            Q_conv = h_conv * A_outer * (T_s - T_amb)

            # Calculate radiation from surface
            rad_result = self.calculate_radiation_loss(
                surface_temperature_c=T_s,
                ambient_temperature_c=T_amb,
                surface_area_m2=A_outer,
                surface_material=surface_material,
                is_outdoor=is_outdoor,
                length_m=L
            )
            Q_rad = rad_result.heat_loss_w

            # Energy balance: Q_cond should equal Q_conv + Q_rad
            Q_surface = Q_conv + Q_rad

            # Calculate new surface temperature estimate
            # Using thermal resistance approach
            h_total = h_conv + rad_result.radiative_coefficient_w_m2_k
            R_surface = Decimal("1") / (h_total * A_outer)
            R_cond = cond_result.thermal_resistance_k_per_w
            R_total = R_cond + R_surface

            Q_total = (T_proc - T_amb) / R_total
            T_s_new = T_amb + Q_total * R_surface

            # Check convergence
            error = abs(T_s_new - T_s)
            if error < self._convergence_tol:
                converged = True

            T_s = T_s_new

        builder.add_step(
            step_number=2,
            operation="iterate",
            description="Surface temperature iteration",
            inputs={"T_proc": T_proc, "T_amb": T_amb},
            output_name="T_surface",
            output_value=T_s,
            formula="Energy balance: Q_cond = Q_conv + Q_rad"
        )

        # Final calculations with converged surface temperature
        cond_final = self.calculate_conduction_loss(
            inner_temperature_c=T_proc,
            outer_temperature_c=T_s,
            geometry=geometry,
            insulation_material=insulation_material,
            insulation_thickness_m=t_ins,
            inner_diameter_m=D_pipe,
            length_m=L
        )

        conv_final = self.calculate_convection_coefficient(
            surface_temperature_c=T_s,
            ambient_temperature_c=T_amb,
            geometry=geometry,
            characteristic_length_m=D_outer,
            wind_speed_m_s=wind_speed_m_s,
            is_outdoor=is_outdoor
        )
        h_conv_final = conv_final.heat_transfer_coefficient_w_m2_k
        Q_conv_final = h_conv_final * A_outer * (T_s - T_amb)

        rad_final = self.calculate_radiation_loss(
            surface_temperature_c=T_s,
            ambient_temperature_c=T_amb,
            surface_area_m2=A_outer,
            surface_material=surface_material,
            is_outdoor=is_outdoor,
            length_m=L
        )
        Q_rad_final = rad_final.heat_loss_w

        Q_total_final = Q_conv_final + Q_rad_final

        # Calculate fractions
        if Q_total_final > Decimal("0"):
            conv_frac = Q_conv_final / Q_total_final
            rad_frac = Q_rad_final / Q_total_final
        else:
            conv_frac = Decimal("0")
            rad_frac = Decimal("0")
        cond_frac = Decimal("1")  # All heat goes through conduction

        # Calculate overall U-value
        # U = Q / (A * dT)
        if A_outer > Decimal("0") and (T_proc - T_amb) != Decimal("0"):
            U_overall = Q_total_final / (A_outer * (T_proc - T_amb))
            R_total_final = Decimal("1") / (U_overall * A_outer) if U_overall > Decimal("0") else Decimal("0")
        else:
            U_overall = Decimal("0")
            R_total_final = Decimal("0")

        # Per unit length and area
        Q_per_m = Q_total_final / L
        Q_per_m2 = Q_total_final / A_outer

        # Update convection result with actual heat loss
        conv_final_updated = ConvectionResult(
            heat_transfer_coefficient_w_m2_k=conv_final.heat_transfer_coefficient_w_m2_k,
            heat_loss_w=self._apply_precision(Q_conv_final),
            heat_loss_w_per_m=self._apply_precision(Q_conv_final / L),
            heat_loss_w_per_m2=self._apply_precision(Q_conv_final / A_outer),
            nusselt_number=conv_final.nusselt_number,
            reynolds_number=conv_final.reynolds_number,
            grashof_number=conv_final.grashof_number,
            prandtl_number=conv_final.prandtl_number,
            convection_mode=conv_final.convection_mode,
            correlation_used=conv_final.correlation_used,
            provenance_hash=conv_final.provenance_hash,
            calculation_steps=conv_final.calculation_steps
        )

        builder.add_output("total_heat_loss_w", Q_total_final)
        builder.add_output("surface_temperature_c", T_s)
        builder.add_output("iterations", iteration)
        provenance_hash = builder.build_hash()

        return TotalHeatLossResult(
            total_heat_loss_w=self._apply_precision(Q_total_final),
            total_heat_loss_w_per_m=self._apply_precision(Q_per_m),
            total_heat_loss_w_per_m2=self._apply_precision(Q_per_m2),
            conduction_fraction=self._apply_precision(cond_frac, 3),
            convection_fraction=self._apply_precision(conv_frac, 3),
            radiation_fraction=self._apply_precision(rad_frac, 3),
            surface_temperature_c=self._apply_precision(T_s, 2),
            ambient_temperature_c=self._apply_precision(T_amb, 2),
            process_temperature_c=self._apply_precision(T_proc, 2),
            overall_u_value_w_m2_k=self._apply_precision(U_overall),
            total_thermal_resistance_m2_k_per_w=self._apply_precision(R_total_final, 6),
            conduction_result=cond_final,
            convection_result=conv_final_updated,
            radiation_result=rad_final,
            iterations_to_converge=iteration,
            provenance_hash=provenance_hash
        )

    # =========================================================================
    # SURFACE TEMPERATURE CALCULATION
    # =========================================================================

    def calculate_surface_temperature(
        self,
        inner_temperature_c: Union[Decimal, float, str],
        ambient_temperature_c: Union[Decimal, float, str],
        insulation_thickness_m: Union[Decimal, float, str],
        insulation_material: str,
        geometry: SurfaceGeometry,
        inner_diameter_m: Optional[Union[Decimal, float, str]] = None,
        area_m2: Optional[Union[Decimal, float, str]] = None,
        surface_material: SurfaceMaterial = SurfaceMaterial.ALUMINUM_JACKETING,
        wind_speed_m_s: Optional[Union[Decimal, float, str]] = None,
        is_outdoor: bool = False
    ) -> SurfaceTemperatureResult:
        """
        Calculate surface temperature using energy balance iteration.

        Iterates until conduction heat loss equals convection + radiation
        heat loss from the surface.

        Args:
            inner_temperature_c: Temperature at inner surface
            ambient_temperature_c: Ambient temperature
            insulation_thickness_m: Insulation thickness
            insulation_material: Insulation material name
            geometry: Surface geometry
            inner_diameter_m: Inner diameter (cylindrical)
            area_m2: Surface area (flat)
            surface_material: Jacketing material
            wind_speed_m_s: Wind speed
            is_outdoor: Whether outdoor

        Returns:
            SurfaceTemperatureResult with converged surface temperature
        """
        builder = ProvenanceBuilder(CalculationType.SURFACE_TEMPERATURE)

        T_inner = self._to_decimal(inner_temperature_c)
        T_amb = self._to_decimal(ambient_temperature_c)
        t_ins = self._to_decimal(insulation_thickness_m)

        builder.add_input("inner_temperature_c", T_inner)
        builder.add_input("ambient_temperature_c", T_amb)
        builder.add_input("insulation_thickness_m", t_ins)

        # Calculate characteristic length
        if inner_diameter_m is not None:
            D_inner = self._to_decimal(inner_diameter_m)
            D_outer = D_inner + Decimal("2") * t_ins
            L_char = D_outer
        else:
            L_char = t_ins
            D_inner = None
            D_outer = None

        # Initial guess
        T_s = T_amb + Decimal("0.1") * (T_inner - T_amb)

        iteration = 0
        converged = False
        error = Decimal("999")

        while iteration < self._max_iterations and not converged:
            iteration += 1

            # Get thermal conductivity at mean temperature
            T_mean = (T_inner + T_s) / Decimal("2")
            if insulation_material in INSULATION_MATERIALS:
                k = INSULATION_MATERIALS[insulation_material].get_thermal_conductivity(T_mean)
            else:
                raise ValueError(f"Unknown material: {insulation_material}")

            # Calculate conduction resistance
            if D_inner is not None:
                # Cylindrical
                R_cond = self._ln(D_outer / D_inner) / (Decimal("2") * PI * k)
            else:
                # Flat (per unit area)
                R_cond = t_ins / k

            # Calculate surface heat transfer coefficient
            conv_result = self.calculate_convection_coefficient(
                surface_temperature_c=T_s,
                ambient_temperature_c=T_amb,
                geometry=geometry,
                characteristic_length_m=L_char,
                wind_speed_m_s=wind_speed_m_s,
                is_outdoor=is_outdoor
            )
            h_conv = conv_result.heat_transfer_coefficient_w_m2_k

            # Estimate radiation coefficient
            T_s_k = T_s + KELVIN_OFFSET
            T_amb_k = T_amb + KELVIN_OFFSET
            epsilon = SURFACE_EMISSIVITY.get(surface_material, Decimal("0.90"))
            h_rad = epsilon * STEFAN_BOLTZMANN * (
                self._power(T_s_k, 2) + self._power(T_amb_k, 2)
            ) * (T_s_k + T_amb_k)

            h_total = h_conv + h_rad

            # Surface resistance (per unit length for cylindrical)
            if D_outer is not None:
                R_surface = Decimal("1") / (h_total * PI * D_outer)
            else:
                R_surface = Decimal("1") / h_total

            # Total resistance
            R_total = R_cond + R_surface

            # Heat flux
            Q = (T_inner - T_amb) / R_total

            # New surface temperature
            T_s_new = T_amb + Q * R_surface

            error = abs(T_s_new - T_s)
            if error < self._convergence_tol:
                converged = True

            T_s = T_s_new

        # Final heat flux
        if D_outer is not None:
            A_surface = PI * D_outer  # Per unit length
        else:
            A_surface = Decimal("1")
        Q_flux = (T_inner - T_s) / R_cond

        builder.add_output("surface_temperature_c", T_s)
        builder.add_output("iterations", iteration)
        provenance_hash = builder.build_hash()

        return SurfaceTemperatureResult(
            surface_temperature_c=self._apply_precision(T_s, 2),
            inner_temperature_c=self._apply_precision(T_inner, 2),
            ambient_temperature_c=self._apply_precision(T_amb, 2),
            heat_flux_w_per_m2=self._apply_precision(Q_flux),
            iterations_to_converge=iteration,
            convergence_error_c=self._apply_precision(error, 4),
            is_converged=converged,
            provenance_hash=provenance_hash
        )

    # =========================================================================
    # HEAT LOSS PER METER
    # =========================================================================

    def calculate_heat_loss_per_meter(
        self,
        process_temperature_c: Union[Decimal, float, str],
        ambient_temperature_c: Union[Decimal, float, str],
        pipe_outer_diameter_m: Union[Decimal, float, str],
        insulation_thickness_m: Union[Decimal, float, str],
        insulation_material: str,
        surface_material: SurfaceMaterial = SurfaceMaterial.ALUMINUM_JACKETING,
        wind_speed_m_s: Optional[Union[Decimal, float, str]] = None,
        is_outdoor: bool = False
    ) -> Decimal:
        """
        Calculate heat loss per unit length (W/m) for pipes.

        Convenience method that returns just the W/m value.

        Args:
            process_temperature_c: Process temperature
            ambient_temperature_c: Ambient temperature
            pipe_outer_diameter_m: Pipe outer diameter
            insulation_thickness_m: Insulation thickness
            insulation_material: Insulation material
            surface_material: Jacketing material
            wind_speed_m_s: Wind speed
            is_outdoor: Outdoor installation

        Returns:
            Heat loss in W/m
        """
        result = self.calculate_total_heat_loss(
            process_temperature_c=process_temperature_c,
            ambient_temperature_c=ambient_temperature_c,
            pipe_outer_diameter_m=pipe_outer_diameter_m,
            insulation_thickness_m=insulation_thickness_m,
            insulation_material=insulation_material,
            pipe_length_m=Decimal("1"),  # Unit length
            surface_material=surface_material,
            wind_speed_m_s=wind_speed_m_s,
            is_outdoor=is_outdoor
        )
        return result.total_heat_loss_w_per_m

    # =========================================================================
    # ANNUAL ENERGY LOSS
    # =========================================================================

    def calculate_annual_energy_loss(
        self,
        heat_loss_w: Union[Decimal, float, str],
        operating_hours_per_year: Union[Decimal, float, str] = "8760",
        boiler_efficiency: Union[Decimal, float, str] = "0.85",
        fuel_type: str = "natural_gas"
    ) -> AnnualEnergyLossResult:
        """
        Calculate annual energy loss and fuel equivalents.

        Converts heat loss to annual energy consumption and
        equivalent fuel/electricity values.

        Args:
            heat_loss_w: Heat loss rate in Watts
            operating_hours_per_year: Operating hours (default 8760 = continuous)
            boiler_efficiency: Boiler/heater efficiency (0-1)
            fuel_type: Fuel type for CO2 calculation

        Returns:
            AnnualEnergyLossResult with energy values

        Reference:
            DOE 3E Plus methodology
            EPA emission factors
        """
        builder = ProvenanceBuilder(CalculationType.ANNUAL_ENERGY_LOSS)

        Q = self._to_decimal(heat_loss_w)
        hours = self._to_decimal(operating_hours_per_year)
        efficiency = self._to_decimal(boiler_efficiency)

        builder.add_input("heat_loss_w", Q)
        builder.add_input("operating_hours_per_year", hours)
        builder.add_input("boiler_efficiency", efficiency)
        builder.add_input("fuel_type", fuel_type)

        # Step 1: Calculate annual energy loss (accounting for efficiency)
        # Heat loss / efficiency = fuel energy needed
        annual_wh = Q * hours / efficiency
        annual_kwh = annual_wh / Decimal("1000")

        builder.add_step(
            step_number=1,
            operation="calculate",
            description="Calculate annual energy loss",
            inputs={"Q": Q, "hours": hours, "efficiency": efficiency},
            output_name="annual_kwh",
            output_value=annual_kwh,
            formula="E = Q * hours / efficiency"
        )

        # Step 2: Convert to other units
        # 1 kWh = 0.0036 GJ
        annual_gj = annual_kwh * Decimal("0.0036")

        # 1 kWh = 0.003412 MMBtu
        annual_mmbtu = annual_kwh * Decimal("0.003412")

        builder.add_step(
            step_number=2,
            operation="convert",
            description="Convert to GJ and MMBtu",
            inputs={"annual_kwh": annual_kwh},
            output_name="conversions",
            output_value={"gj": annual_gj, "mmbtu": annual_mmbtu}
        )

        # Step 3: Calculate fuel equivalents
        # Natural gas: ~10.5 kWh/m3 (gross CV)
        # Oil: ~10.8 kWh/liter
        fuel_ng_m3 = annual_kwh / Decimal("10.5")
        fuel_oil_l = annual_kwh / Decimal("10.8")

        builder.add_step(
            step_number=3,
            operation="calculate",
            description="Calculate fuel equivalents",
            inputs={"annual_kwh": annual_kwh},
            output_name="fuel_equivalents",
            output_value={"natural_gas_m3": fuel_ng_m3, "oil_liters": fuel_oil_l}
        )

        # Step 4: Calculate CO2 emissions
        # Natural gas: 0.184 kg CO2/kWh
        # Oil: 0.264 kg CO2/kWh
        # Electricity grid average: 0.4 kg CO2/kWh (varies by region)
        co2_factors: Dict[str, Decimal] = {
            "natural_gas": Decimal("0.184"),
            "oil": Decimal("0.264"),
            "electricity": Decimal("0.400"),
            "coal": Decimal("0.341"),
        }
        co2_factor = co2_factors.get(fuel_type, Decimal("0.200"))
        co2_kg = annual_kwh * co2_factor

        builder.add_step(
            step_number=4,
            operation="calculate",
            description="Calculate CO2 emissions",
            inputs={"annual_kwh": annual_kwh, "co2_factor": co2_factor},
            output_name="co2_kg",
            output_value=co2_kg,
            formula="CO2 = energy * emission_factor",
            reference="EPA emission factors"
        )

        builder.add_output("annual_kwh", annual_kwh)
        builder.add_output("co2_kg", co2_kg)
        provenance_hash = builder.build_hash()

        return AnnualEnergyLossResult(
            annual_heat_loss_kwh=self._apply_precision(annual_kwh, 0),
            annual_heat_loss_gj=self._apply_precision(annual_gj, 2),
            annual_heat_loss_mmbtu=self._apply_precision(annual_mmbtu, 2),
            equivalent_fuel_m3_natural_gas=self._apply_precision(fuel_ng_m3, 0),
            equivalent_fuel_liters_oil=self._apply_precision(fuel_oil_l, 0),
            equivalent_electricity_kwh=self._apply_precision(annual_kwh, 0),
            co2_emissions_kg=self._apply_precision(co2_kg, 0),
            operating_hours_per_year=hours,
            average_heat_loss_w=self._apply_precision(Q),
            provenance_hash=provenance_hash
        )

    # =========================================================================
    # INSULATED VS BARE COMPARISON
    # =========================================================================

    def compare_insulated_vs_bare(
        self,
        process_temperature_c: Union[Decimal, float, str],
        ambient_temperature_c: Union[Decimal, float, str],
        pipe_outer_diameter_m: Union[Decimal, float, str],
        current_insulation_thickness_m: Union[Decimal, float, str],
        design_insulation_thickness_m: Union[Decimal, float, str],
        insulation_material: str,
        pipe_length_m: Union[Decimal, float, str],
        surface_material: SurfaceMaterial = SurfaceMaterial.ALUMINUM_JACKETING,
        bare_surface_material: SurfaceMaterial = SurfaceMaterial.BARE_PIPE_STEEL,
        operating_hours_per_year: Union[Decimal, float, str] = "8760",
        wind_speed_m_s: Optional[Union[Decimal, float, str]] = None,
        is_outdoor: bool = False
    ) -> InsulationComparisonResult:
        """
        Compare heat loss for insulated, bare, and design conditions.

        Calculates:
        - Heat loss with current insulation
        - Heat loss if pipe were bare (no insulation)
        - Heat loss with design insulation thickness
        - Savings and efficiency metrics

        Args:
            process_temperature_c: Process temperature
            ambient_temperature_c: Ambient temperature
            pipe_outer_diameter_m: Pipe diameter
            current_insulation_thickness_m: Current insulation thickness
            design_insulation_thickness_m: Design/specified thickness
            insulation_material: Insulation material
            pipe_length_m: Pipe length
            surface_material: Current jacketing
            bare_surface_material: Bare pipe material
            operating_hours_per_year: Operating hours
            wind_speed_m_s: Wind speed
            is_outdoor: Outdoor installation

        Returns:
            InsulationComparisonResult with comparison metrics
        """
        builder = ProvenanceBuilder(CalculationType.INSULATION_COMPARISON)

        T_proc = self._to_decimal(process_temperature_c)
        T_amb = self._to_decimal(ambient_temperature_c)
        D_pipe = self._to_decimal(pipe_outer_diameter_m)
        t_current = self._to_decimal(current_insulation_thickness_m)
        t_design = self._to_decimal(design_insulation_thickness_m)
        L = self._to_decimal(pipe_length_m)
        hours = self._to_decimal(operating_hours_per_year)

        builder.add_input("process_temperature_c", T_proc)
        builder.add_input("ambient_temperature_c", T_amb)
        builder.add_input("current_thickness_m", t_current)
        builder.add_input("design_thickness_m", t_design)

        # Calculate heat loss with current insulation
        current_result = self.calculate_total_heat_loss(
            process_temperature_c=T_proc,
            ambient_temperature_c=T_amb,
            pipe_outer_diameter_m=D_pipe,
            insulation_thickness_m=t_current,
            insulation_material=insulation_material,
            pipe_length_m=L,
            surface_material=surface_material,
            wind_speed_m_s=wind_speed_m_s,
            is_outdoor=is_outdoor
        )
        Q_insulated = current_result.total_heat_loss_w
        T_s_insulated = current_result.surface_temperature_c

        builder.add_step(
            step_number=1,
            operation="calculate",
            description="Calculate heat loss with current insulation",
            inputs={"thickness": t_current},
            output_name="Q_insulated",
            output_value=Q_insulated
        )

        # Calculate heat loss if bare (no insulation)
        # Use very thin insulation to simulate bare pipe with convection/radiation
        bare_result = self._calculate_bare_pipe_heat_loss(
            process_temperature_c=T_proc,
            ambient_temperature_c=T_amb,
            pipe_outer_diameter_m=D_pipe,
            pipe_length_m=L,
            surface_material=bare_surface_material,
            wind_speed_m_s=wind_speed_m_s,
            is_outdoor=is_outdoor
        )
        Q_bare = bare_result["heat_loss_w"]
        T_s_bare = bare_result["surface_temperature_c"]

        builder.add_step(
            step_number=2,
            operation="calculate",
            description="Calculate heat loss if bare (no insulation)",
            inputs={"bare_surface": bare_surface_material.name},
            output_name="Q_bare",
            output_value=Q_bare
        )

        # Calculate heat loss with design insulation
        design_result = self.calculate_total_heat_loss(
            process_temperature_c=T_proc,
            ambient_temperature_c=T_amb,
            pipe_outer_diameter_m=D_pipe,
            insulation_thickness_m=t_design,
            insulation_material=insulation_material,
            pipe_length_m=L,
            surface_material=surface_material,
            wind_speed_m_s=wind_speed_m_s,
            is_outdoor=is_outdoor
        )
        Q_design = design_result.total_heat_loss_w
        T_s_design = design_result.surface_temperature_c

        builder.add_step(
            step_number=3,
            operation="calculate",
            description="Calculate heat loss with design insulation",
            inputs={"thickness": t_design},
            output_name="Q_design",
            output_value=Q_design
        )

        # Calculate savings and efficiency
        savings_vs_bare_w = Q_bare - Q_insulated
        if Q_bare > Decimal("0"):
            savings_vs_bare_pct = (savings_vs_bare_w / Q_bare) * Decimal("100")
        else:
            savings_vs_bare_pct = Decimal("0")

        if Q_design > Decimal("0"):
            efficiency_vs_design = (Q_design / Q_insulated) * Decimal("100")
        else:
            efficiency_vs_design = Decimal("100")

        # Insulation effectiveness
        if Q_bare > Q_design and Q_bare != Q_design:
            effectiveness = (Q_bare - Q_insulated) / (Q_bare - Q_design)
        else:
            effectiveness = Decimal("1") if Q_insulated <= Q_design else Decimal("0")

        builder.add_step(
            step_number=4,
            operation="calculate",
            description="Calculate savings and efficiency metrics",
            inputs={
                "Q_insulated": Q_insulated,
                "Q_bare": Q_bare,
                "Q_design": Q_design
            },
            output_name="metrics",
            output_value={
                "savings_pct": savings_vs_bare_pct,
                "efficiency_pct": efficiency_vs_design,
                "effectiveness": effectiveness
            }
        )

        # Annual energy savings
        annual_savings_wh = savings_vs_bare_w * hours
        annual_savings_kwh = annual_savings_wh / Decimal("1000")

        # CO2 savings (using natural gas factor)
        co2_savings_kg = annual_savings_kwh * Decimal("0.184")

        builder.add_output("annual_savings_kwh", annual_savings_kwh)
        builder.add_output("co2_savings_kg", co2_savings_kg)
        provenance_hash = builder.build_hash()

        return InsulationComparisonResult(
            heat_loss_insulated_w=self._apply_precision(Q_insulated),
            heat_loss_bare_w=self._apply_precision(Q_bare),
            heat_loss_design_w=self._apply_precision(Q_design),
            heat_savings_vs_bare_w=self._apply_precision(savings_vs_bare_w),
            heat_savings_vs_bare_percent=self._apply_precision(savings_vs_bare_pct, 2),
            efficiency_vs_design_percent=self._apply_precision(efficiency_vs_design, 2),
            surface_temp_insulated_c=self._apply_precision(T_s_insulated, 2),
            surface_temp_bare_c=self._apply_precision(T_s_bare, 2),
            surface_temp_design_c=self._apply_precision(T_s_design, 2),
            insulation_effectiveness=self._apply_precision(effectiveness, 3),
            energy_savings_annual_kwh=self._apply_precision(annual_savings_kwh, 0),
            co2_savings_annual_kg=self._apply_precision(co2_savings_kg, 0),
            provenance_hash=provenance_hash
        )

    def _calculate_bare_pipe_heat_loss(
        self,
        process_temperature_c: Union[Decimal, float, str],
        ambient_temperature_c: Union[Decimal, float, str],
        pipe_outer_diameter_m: Union[Decimal, float, str],
        pipe_length_m: Union[Decimal, float, str],
        surface_material: SurfaceMaterial,
        wind_speed_m_s: Optional[Union[Decimal, float, str]] = None,
        is_outdoor: bool = False
    ) -> Dict[str, Decimal]:
        """
        Calculate heat loss from bare (uninsulated) pipe.

        For bare pipe, surface temperature equals process temperature
        (assuming negligible pipe wall resistance).
        """
        T_proc = self._to_decimal(process_temperature_c)
        T_amb = self._to_decimal(ambient_temperature_c)
        D = self._to_decimal(pipe_outer_diameter_m)
        L = self._to_decimal(pipe_length_m)

        # Surface area
        A = PI * D * L

        # For bare pipe, T_surface ~ T_process
        T_s = T_proc

        # Calculate convection
        conv_result = self.calculate_convection_coefficient(
            surface_temperature_c=T_s,
            ambient_temperature_c=T_amb,
            geometry=SurfaceGeometry.CYLINDER_HORIZONTAL,
            characteristic_length_m=D,
            wind_speed_m_s=wind_speed_m_s,
            is_outdoor=is_outdoor
        )
        h_conv = conv_result.heat_transfer_coefficient_w_m2_k
        Q_conv = h_conv * A * (T_s - T_amb)

        # Calculate radiation
        rad_result = self.calculate_radiation_loss(
            surface_temperature_c=T_s,
            ambient_temperature_c=T_amb,
            surface_area_m2=A,
            surface_material=surface_material,
            is_outdoor=is_outdoor,
            length_m=L
        )
        Q_rad = rad_result.heat_loss_w

        Q_total = Q_conv + Q_rad

        return {
            "heat_loss_w": Q_total,
            "surface_temperature_c": T_s,
            "convection_w": Q_conv,
            "radiation_w": Q_rad
        }

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _to_decimal(self, value: Union[Decimal, float, int, str]) -> Decimal:
        """Convert value to Decimal - DETERMINISTIC."""
        if isinstance(value, Decimal):
            return value
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError) as e:
            raise ValueError(f"Cannot convert {value} to Decimal: {e}")

    def _apply_precision(
        self,
        value: Decimal,
        precision: Optional[int] = None
    ) -> Decimal:
        """Apply precision rounding - DETERMINISTIC."""
        prec = precision if precision is not None else self._precision
        if prec == 0:
            return value.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        quantize_str = "0." + "0" * prec
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    def _ln(self, x: Decimal) -> Decimal:
        """Calculate natural logarithm - DETERMINISTIC."""
        if x <= Decimal("0"):
            raise ValueError("Cannot compute ln of non-positive number")
        return Decimal(str(math.log(float(x))))

    def _exp(self, x: Decimal) -> Decimal:
        """Calculate e^x - DETERMINISTIC."""
        if x > Decimal("700"):
            raise ValueError("Exponent too large")
        if x < Decimal("-700"):
            return Decimal("0")
        return Decimal(str(math.exp(float(x))))

    def _power(self, base: Decimal, exponent: Union[Decimal, int]) -> Decimal:
        """Calculate base^exponent - DETERMINISTIC."""
        if base == Decimal("0"):
            if isinstance(exponent, Decimal):
                return Decimal("0") if exponent > Decimal("0") else Decimal("1")
            return Decimal("0") if exponent > 0 else Decimal("1")
        if isinstance(exponent, int):
            return Decimal(str(math.pow(float(base), exponent)))
        if exponent == Decimal("0"):
            return Decimal("1")
        return Decimal(str(math.pow(float(base), float(exponent))))

    def _sqrt(self, x: Decimal) -> Decimal:
        """Calculate square root - DETERMINISTIC."""
        if x < Decimal("0"):
            raise ValueError("Cannot compute sqrt of negative number")
        if x == Decimal("0"):
            return Decimal("0")
        return Decimal(str(math.sqrt(float(x))))

    def _get_air_properties(self, temperature_c: Decimal) -> Dict[str, Decimal]:
        """
        Get temperature-corrected air properties.

        Uses linear interpolation from reference conditions.
        Valid for -40C to 200C range.
        """
        T = float(temperature_c)
        T_ref = 20.0  # Reference temperature

        # Temperature correction factors (approximate)
        # Density: rho ~ 1/T (ideal gas)
        T_k = T + 273.15
        T_ref_k = T_ref + 273.15
        density_ratio = T_ref_k / T_k

        # Viscosity increases with temperature (Sutherland's law approximation)
        viscosity_ratio = self._power(
            Decimal(str(T_k / T_ref_k)),
            Decimal("0.76")
        )

        # Thermal conductivity increases with temperature
        k_ratio = self._power(
            Decimal(str(T_k / T_ref_k)),
            Decimal("0.82")
        )

        return {
            "density_kg_m3": AIR_PROPERTIES_20C["density_kg_m3"] * Decimal(str(density_ratio)),
            "specific_heat_j_kg_k": AIR_PROPERTIES_20C["specific_heat_j_kg_k"],
            "thermal_conductivity_w_m_k": AIR_PROPERTIES_20C["thermal_conductivity_w_m_k"] * k_ratio,
            "dynamic_viscosity_pa_s": AIR_PROPERTIES_20C["dynamic_viscosity_pa_s"] * viscosity_ratio,
            "kinematic_viscosity_m2_s": AIR_PROPERTIES_20C["kinematic_viscosity_m2_s"] * viscosity_ratio / Decimal(str(density_ratio)),
            "prandtl_number": AIR_PROPERTIES_20C["prandtl_number"],
            "volumetric_expansion_coeff_k": Decimal("1") / Decimal(str(T_k)),
        }

    def get_available_materials(self) -> List[str]:
        """Get list of available insulation materials."""
        return list(INSULATION_MATERIALS.keys())

    def get_material_properties(self, material_name: str) -> Optional[InsulationMaterial]:
        """Get properties for a specific insulation material."""
        return INSULATION_MATERIALS.get(material_name)

    def get_surface_emissivity(self, material: SurfaceMaterial) -> Decimal:
        """Get emissivity value for a surface material."""
        return SURFACE_EMISSIVITY.get(material, Decimal("0.90"))


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "STEFAN_BOLTZMANN",
    "KELVIN_OFFSET",
    "PI",
    "GRAVITY",
    "AIR_PROPERTIES_20C",

    # Enums
    "SurfaceGeometry",
    "ConvectionMode",
    "InsulationCondition",
    "SurfaceMaterial",
    "CalculationType",

    # Material data
    "SURFACE_EMISSIVITY",
    "InsulationMaterial",
    "INSULATION_MATERIALS",

    # Result classes
    "ConductionResult",
    "ConvectionResult",
    "RadiationResult",
    "TotalHeatLossResult",
    "SurfaceTemperatureResult",
    "AnnualEnergyLossResult",
    "InsulationComparisonResult",

    # Main calculator
    "HeatLossCalculator",
]
