"""
GL-011 FuelCraft - Heating Value Calculator

Deterministic calculations for fuel calorific values:
- LHV (Lower Heating Value) / Net Calorific Value
- HHV (Higher Heating Value) / Gross Calorific Value
- LHV/HHV conversions with hydrogen content
- Temperature-corrected density calculations
- Energy content from mass/volume

Standards:
- ASTM D240 (Heat of Combustion - Bomb Calorimeter)
- ASTM D4868 (Net Heat of Combustion)
- ISO 6976 (Natural Gas Calorific Value)
- API MPMS Chapter 14.5 (Natural Gas)
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import hashlib
import json


class HeatingValueType(Enum):
    """Type of heating value."""
    LHV = "LHV"  # Lower Heating Value (Net)
    HHV = "HHV"  # Higher Heating Value (Gross)


class FuelType(Enum):
    """Standard fuel types with default properties."""
    NATURAL_GAS = "natural_gas"
    DIESEL = "diesel"
    GASOLINE = "gasoline"
    FUEL_OIL_2 = "fuel_oil_2"
    FUEL_OIL_6 = "fuel_oil_6"
    LPG = "lpg"
    COAL_BITUMINOUS = "coal_bituminous"
    COAL_ANTHRACITE = "coal_anthracite"
    BIOMASS_WOOD = "biomass_wood"
    HYDROGEN = "hydrogen"
    METHANOL = "methanol"
    ETHANOL = "ethanol"
    BIODIESEL = "biodiesel"
    MARINE_FUEL_OIL = "marine_fuel_oil"
    HEAVY_FUEL_OIL = "heavy_fuel_oil"


@dataclass(frozen=True)
class FuelProperties:
    """
    Fuel physical and thermal properties.

    All properties at standard conditions (15C, 101.325 kPa).
    """
    fuel_type: str
    hhv_mj_kg: Decimal          # Gross heating value (MJ/kg)
    lhv_mj_kg: Decimal          # Net heating value (MJ/kg)
    density_kg_m3: Decimal      # Density at 15C (kg/m3)
    hydrogen_wt_pct: Decimal    # Hydrogen content (mass %)
    carbon_wt_pct: Decimal      # Carbon content (mass %)
    sulfur_wt_pct: Decimal      # Sulfur content (mass %)
    ash_wt_pct: Decimal         # Ash content (mass %)
    moisture_wt_pct: Decimal    # Moisture content (mass %)
    source_standard: str        # Reference standard
    effective_date: date        # Date of data
    uncertainty_pct: Decimal    # Uncertainty (%)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for hashing."""
        return {
            "fuel_type": self.fuel_type,
            "hhv_mj_kg": str(self.hhv_mj_kg),
            "lhv_mj_kg": str(self.lhv_mj_kg),
            "density_kg_m3": str(self.density_kg_m3),
            "hydrogen_wt_pct": str(self.hydrogen_wt_pct),
            "carbon_wt_pct": str(self.carbon_wt_pct),
            "sulfur_wt_pct": str(self.sulfur_wt_pct),
            "ash_wt_pct": str(self.ash_wt_pct),
            "moisture_wt_pct": str(self.moisture_wt_pct),
            "source_standard": self.source_standard,
            "effective_date": self.effective_date.isoformat(),
            "uncertainty_pct": str(self.uncertainty_pct)
        }


# Default fuel properties database
# Source: EPA AP-42, IPCC 2006 Guidelines, EIA
DEFAULT_FUEL_PROPERTIES: Dict[str, FuelProperties] = {
    "natural_gas": FuelProperties(
        fuel_type="natural_gas",
        hhv_mj_kg=Decimal("55.50"),
        lhv_mj_kg=Decimal("50.00"),
        density_kg_m3=Decimal("0.717"),  # At 15C, 101.325 kPa
        hydrogen_wt_pct=Decimal("23.0"),
        carbon_wt_pct=Decimal("75.0"),
        sulfur_wt_pct=Decimal("0.0"),
        ash_wt_pct=Decimal("0.0"),
        moisture_wt_pct=Decimal("0.0"),
        source_standard="IPCC 2006",
        effective_date=date(2006, 1, 1),
        uncertainty_pct=Decimal("2.0")
    ),
    "diesel": FuelProperties(
        fuel_type="diesel",
        hhv_mj_kg=Decimal("45.80"),
        lhv_mj_kg=Decimal("43.00"),
        density_kg_m3=Decimal("840.0"),
        hydrogen_wt_pct=Decimal("13.5"),
        carbon_wt_pct=Decimal("85.0"),
        sulfur_wt_pct=Decimal("0.05"),
        ash_wt_pct=Decimal("0.01"),
        moisture_wt_pct=Decimal("0.0"),
        source_standard="IPCC 2006",
        effective_date=date(2006, 1, 1),
        uncertainty_pct=Decimal("1.5")
    ),
    "gasoline": FuelProperties(
        fuel_type="gasoline",
        hhv_mj_kg=Decimal("47.30"),
        lhv_mj_kg=Decimal("44.00"),
        density_kg_m3=Decimal("745.0"),
        hydrogen_wt_pct=Decimal("14.0"),
        carbon_wt_pct=Decimal("85.0"),
        sulfur_wt_pct=Decimal("0.03"),
        ash_wt_pct=Decimal("0.0"),
        moisture_wt_pct=Decimal("0.0"),
        source_standard="IPCC 2006",
        effective_date=date(2006, 1, 1),
        uncertainty_pct=Decimal("1.5")
    ),
    "fuel_oil_2": FuelProperties(
        fuel_type="fuel_oil_2",
        hhv_mj_kg=Decimal("45.50"),
        lhv_mj_kg=Decimal("42.50"),
        density_kg_m3=Decimal("850.0"),
        hydrogen_wt_pct=Decimal("13.0"),
        carbon_wt_pct=Decimal("86.0"),
        sulfur_wt_pct=Decimal("0.25"),
        ash_wt_pct=Decimal("0.01"),
        moisture_wt_pct=Decimal("0.1"),
        source_standard="EPA AP-42",
        effective_date=date(2020, 1, 1),
        uncertainty_pct=Decimal("2.0")
    ),
    "fuel_oil_6": FuelProperties(
        fuel_type="fuel_oil_6",
        hhv_mj_kg=Decimal("43.00"),
        lhv_mj_kg=Decimal("40.50"),
        density_kg_m3=Decimal("990.0"),
        hydrogen_wt_pct=Decimal("11.0"),
        carbon_wt_pct=Decimal("86.0"),
        sulfur_wt_pct=Decimal("2.0"),
        ash_wt_pct=Decimal("0.05"),
        moisture_wt_pct=Decimal("0.5"),
        source_standard="EPA AP-42",
        effective_date=date(2020, 1, 1),
        uncertainty_pct=Decimal("3.0")
    ),
    "lpg": FuelProperties(
        fuel_type="lpg",
        hhv_mj_kg=Decimal("50.00"),
        lhv_mj_kg=Decimal("46.10"),
        density_kg_m3=Decimal("540.0"),  # Liquid at 15C
        hydrogen_wt_pct=Decimal("18.0"),
        carbon_wt_pct=Decimal("82.0"),
        sulfur_wt_pct=Decimal("0.0"),
        ash_wt_pct=Decimal("0.0"),
        moisture_wt_pct=Decimal("0.0"),
        source_standard="IPCC 2006",
        effective_date=date(2006, 1, 1),
        uncertainty_pct=Decimal("2.0")
    ),
    "coal_bituminous": FuelProperties(
        fuel_type="coal_bituminous",
        hhv_mj_kg=Decimal("30.00"),
        lhv_mj_kg=Decimal("28.50"),
        density_kg_m3=Decimal("1300.0"),  # Bulk density
        hydrogen_wt_pct=Decimal("5.0"),
        carbon_wt_pct=Decimal("75.0"),
        sulfur_wt_pct=Decimal("2.0"),
        ash_wt_pct=Decimal("10.0"),
        moisture_wt_pct=Decimal("8.0"),
        source_standard="IPCC 2006",
        effective_date=date(2006, 1, 1),
        uncertainty_pct=Decimal("5.0")
    ),
    "hydrogen": FuelProperties(
        fuel_type="hydrogen",
        hhv_mj_kg=Decimal("141.80"),
        lhv_mj_kg=Decimal("120.00"),
        density_kg_m3=Decimal("0.0899"),  # At 15C, 101.325 kPa
        hydrogen_wt_pct=Decimal("100.0"),
        carbon_wt_pct=Decimal("0.0"),
        sulfur_wt_pct=Decimal("0.0"),
        ash_wt_pct=Decimal("0.0"),
        moisture_wt_pct=Decimal("0.0"),
        source_standard="NIST",
        effective_date=date(2020, 1, 1),
        uncertainty_pct=Decimal("0.5")
    ),
    "marine_fuel_oil": FuelProperties(
        fuel_type="marine_fuel_oil",
        hhv_mj_kg=Decimal("42.70"),
        lhv_mj_kg=Decimal("40.20"),
        density_kg_m3=Decimal("980.0"),
        hydrogen_wt_pct=Decimal("10.5"),
        carbon_wt_pct=Decimal("87.0"),
        sulfur_wt_pct=Decimal("0.5"),  # IMO 2020 limit
        ash_wt_pct=Decimal("0.1"),
        moisture_wt_pct=Decimal("0.5"),
        source_standard="IMO MEPC.320(74)",
        effective_date=date(2020, 1, 1),
        uncertainty_pct=Decimal("3.0")
    ),
    "heavy_fuel_oil": FuelProperties(
        fuel_type="heavy_fuel_oil",
        hhv_mj_kg=Decimal("42.50"),
        lhv_mj_kg=Decimal("40.00"),
        density_kg_m3=Decimal("990.0"),
        hydrogen_wt_pct=Decimal("10.0"),
        carbon_wt_pct=Decimal("87.0"),
        sulfur_wt_pct=Decimal("3.5"),
        ash_wt_pct=Decimal("0.1"),
        moisture_wt_pct=Decimal("0.5"),
        source_standard="IMO MEPC",
        effective_date=date(2020, 1, 1),
        uncertainty_pct=Decimal("3.0")
    ),
}


@dataclass
class HeatingValueInput:
    """Input for heating value calculation."""
    fuel_type: str
    quantity: Decimal
    quantity_unit: str  # "kg", "tonne", "m3", "L", "bbl"
    observed_temperature_c: Optional[Decimal] = None
    hydrogen_content_override: Optional[Decimal] = None  # mass %


@dataclass
class HeatingValueResult:
    """
    Result of heating value calculation with provenance.
    """
    input_fuel_type: str
    input_quantity: Decimal
    input_unit: str
    hhv_mj: Decimal
    lhv_mj: Decimal
    hhv_mj_kg: Decimal
    lhv_mj_kg: Decimal
    mass_kg: Decimal
    fuel_properties: FuelProperties
    provenance_hash: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    calculation_steps: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if not self.provenance_hash:
            self.provenance_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA-256 provenance hash."""
        data = {
            "input_fuel_type": self.input_fuel_type,
            "input_quantity": str(self.input_quantity),
            "input_unit": self.input_unit,
            "hhv_mj": str(self.hhv_mj),
            "lhv_mj": str(self.lhv_mj),
            "mass_kg": str(self.mass_kg),
            "fuel_properties": self.fuel_properties.to_dict(),
            "timestamp": self.timestamp.isoformat()
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "input_fuel_type": self.input_fuel_type,
            "input_quantity": str(self.input_quantity),
            "input_unit": self.input_unit,
            "hhv_mj": str(self.hhv_mj),
            "lhv_mj": str(self.lhv_mj),
            "hhv_mj_kg": str(self.hhv_mj_kg),
            "lhv_mj_kg": str(self.lhv_mj_kg),
            "mass_kg": str(self.mass_kg),
            "fuel_properties": self.fuel_properties.to_dict(),
            "provenance_hash": self.provenance_hash,
            "timestamp": self.timestamp.isoformat(),
            "calculation_steps": self.calculation_steps
        }


class HeatingValueCalculator:
    """
    Deterministic heating value calculator.

    Provides ZERO-HALLUCINATION calculations for:
    - LHV/HHV lookup from fuel properties database
    - LHV <-> HHV conversion using hydrogen content
    - Energy content from mass/volume quantities
    - Temperature-corrected density for volume inputs

    All calculations use Decimal arithmetic with explicit rounding.
    """

    NAME: str = "HeatingValueCalculator"
    VERSION: str = "1.0.0"

    # Latent heat of vaporization for water at 25C
    # This is the difference between HHV and LHV per unit hydrogen
    # Source: NIST
    WATER_LATENT_HEAT_MJ_KG: Decimal = Decimal("2.442")  # MJ/kg water

    # Water produced per kg hydrogen: 9 kg H2O / kg H2
    WATER_PER_HYDROGEN: Decimal = Decimal("9.0")

    PRECISION: int = 6

    def __init__(
        self,
        fuel_database: Optional[Dict[str, FuelProperties]] = None
    ):
        """
        Initialize calculator.

        Args:
            fuel_database: Custom fuel properties database.
                          Defaults to DEFAULT_FUEL_PROPERTIES.
        """
        self._fuel_db = fuel_database or DEFAULT_FUEL_PROPERTIES

    def calculate(
        self,
        fuel_input: HeatingValueInput,
        precision: int = 6
    ) -> HeatingValueResult:
        """
        Calculate heating values for fuel quantity - DETERMINISTIC.

        Args:
            fuel_input: Input parameters
            precision: Output decimal places

        Returns:
            HeatingValueResult with full provenance

        Raises:
            ValueError: If fuel type not found
        """
        # Get fuel properties
        if fuel_input.fuel_type not in self._fuel_db:
            raise ValueError(f"Unknown fuel type: {fuel_input.fuel_type}")

        props = self._fuel_db[fuel_input.fuel_type]
        steps: List[Dict[str, Any]] = []

        # Step 1: Convert quantity to mass (kg)
        mass_kg = self._convert_to_mass(
            fuel_input.quantity,
            fuel_input.quantity_unit,
            props.density_kg_m3,
            fuel_input.observed_temperature_c
        )

        steps.append({
            "step": 1,
            "operation": "convert_to_mass",
            "input_quantity": str(fuel_input.quantity),
            "input_unit": fuel_input.quantity_unit,
            "output_kg": str(mass_kg),
            "density_kg_m3": str(props.density_kg_m3)
        })

        # Step 2: Get or calculate LHV/HHV
        if fuel_input.hydrogen_content_override is not None:
            # Recalculate LHV from HHV using provided hydrogen content
            hhv_mj_kg = props.hhv_mj_kg
            lhv_mj_kg = self._calculate_lhv_from_hhv(
                hhv_mj_kg,
                fuel_input.hydrogen_content_override
            )
            steps.append({
                "step": 2,
                "operation": "calculate_lhv_from_hhv",
                "hhv_mj_kg": str(hhv_mj_kg),
                "hydrogen_wt_pct": str(fuel_input.hydrogen_content_override),
                "lhv_mj_kg": str(lhv_mj_kg)
            })
        else:
            hhv_mj_kg = props.hhv_mj_kg
            lhv_mj_kg = props.lhv_mj_kg
            steps.append({
                "step": 2,
                "operation": "lookup_heating_values",
                "hhv_mj_kg": str(hhv_mj_kg),
                "lhv_mj_kg": str(lhv_mj_kg),
                "source": props.source_standard
            })

        # Step 3: Calculate total energy content
        hhv_mj = mass_kg * hhv_mj_kg
        lhv_mj = mass_kg * lhv_mj_kg

        steps.append({
            "step": 3,
            "operation": "calculate_energy",
            "mass_kg": str(mass_kg),
            "hhv_mj": str(hhv_mj),
            "lhv_mj": str(lhv_mj)
        })

        # Apply precision
        quantize_str = "0." + "0" * precision
        hhv_mj = hhv_mj.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)
        lhv_mj = lhv_mj.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)
        mass_kg = mass_kg.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

        return HeatingValueResult(
            input_fuel_type=fuel_input.fuel_type,
            input_quantity=fuel_input.quantity,
            input_unit=fuel_input.quantity_unit,
            hhv_mj=hhv_mj,
            lhv_mj=lhv_mj,
            hhv_mj_kg=hhv_mj_kg,
            lhv_mj_kg=lhv_mj_kg,
            mass_kg=mass_kg,
            fuel_properties=props,
            calculation_steps=steps
        )

    def convert_hhv_to_lhv(
        self,
        hhv_mj_kg: Union[float, Decimal],
        hydrogen_wt_pct: Union[float, Decimal]
    ) -> Decimal:
        """
        Convert HHV to LHV using hydrogen content.

        Formula: LHV = HHV - (H_wt% / 100) * 9 * 2.442 MJ/kg
        where 9 is kg water per kg hydrogen and 2.442 is latent heat.

        Args:
            hhv_mj_kg: Higher heating value (MJ/kg)
            hydrogen_wt_pct: Hydrogen content (mass %)

        Returns:
            Lower heating value (MJ/kg)
        """
        hhv = Decimal(str(hhv_mj_kg))
        h_pct = Decimal(str(hydrogen_wt_pct))

        return self._calculate_lhv_from_hhv(hhv, h_pct)

    def convert_lhv_to_hhv(
        self,
        lhv_mj_kg: Union[float, Decimal],
        hydrogen_wt_pct: Union[float, Decimal]
    ) -> Decimal:
        """
        Convert LHV to HHV using hydrogen content.

        Formula: HHV = LHV + (H_wt% / 100) * 9 * 2.442 MJ/kg

        Args:
            lhv_mj_kg: Lower heating value (MJ/kg)
            hydrogen_wt_pct: Hydrogen content (mass %)

        Returns:
            Higher heating value (MJ/kg)
        """
        lhv = Decimal(str(lhv_mj_kg))
        h_pct = Decimal(str(hydrogen_wt_pct))

        # Water produced per unit fuel
        water_per_fuel = h_pct / Decimal("100") * self.WATER_PER_HYDROGEN

        # Latent heat contribution
        latent_heat = water_per_fuel * self.WATER_LATENT_HEAT_MJ_KG

        return (lhv + latent_heat).quantize(
            Decimal("0.000001"),
            rounding=ROUND_HALF_UP
        )

    def calculate_density_at_temperature(
        self,
        density_15c_kg_m3: Union[float, Decimal],
        temperature_c: Union[float, Decimal],
        fuel_type: str = "diesel"
    ) -> Decimal:
        """
        Calculate density at observed temperature.

        Uses simplified linear thermal expansion model.
        For precise calculations, use ASTM D1250 tables.

        Args:
            density_15c_kg_m3: Density at 15C (kg/m3)
            temperature_c: Observed temperature (C)
            fuel_type: Fuel type for expansion coefficient

        Returns:
            Density at temperature (kg/m3)
        """
        rho_15 = Decimal(str(density_15c_kg_m3))
        t = Decimal(str(temperature_c))

        # Thermal expansion coefficients (per degree C)
        # Simplified values - production systems use ASTM D1250
        expansion_coefficients = {
            "natural_gas": Decimal("0.00366"),
            "lpg": Decimal("0.0015"),
            "gasoline": Decimal("0.00095"),
            "diesel": Decimal("0.00084"),
            "fuel_oil_2": Decimal("0.00080"),
            "fuel_oil_6": Decimal("0.00070"),
            "heavy_fuel_oil": Decimal("0.00065"),
            "marine_fuel_oil": Decimal("0.00068"),
        }

        alpha = expansion_coefficients.get(fuel_type, Decimal("0.00080"))
        delta_t = t - Decimal("15.0")

        # rho(T) = rho(15) / (1 + alpha * delta_T)
        rho_t = rho_15 / (Decimal("1.0") + alpha * delta_t)

        return rho_t.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

    def get_fuel_properties(self, fuel_type: str) -> FuelProperties:
        """Get fuel properties from database."""
        if fuel_type not in self._fuel_db:
            raise ValueError(f"Unknown fuel type: {fuel_type}")
        return self._fuel_db[fuel_type]

    def list_fuels(self) -> List[str]:
        """List available fuel types."""
        return list(self._fuel_db.keys())

    def _calculate_lhv_from_hhv(
        self,
        hhv_mj_kg: Decimal,
        hydrogen_wt_pct: Decimal
    ) -> Decimal:
        """
        Internal LHV calculation from HHV.

        Formula: LHV = HHV - (H_wt% / 100) * 9 * 2.442
        """
        # Water produced per unit fuel (kg water / kg fuel)
        water_per_fuel = hydrogen_wt_pct / Decimal("100") * self.WATER_PER_HYDROGEN

        # Latent heat of water vaporization (MJ/kg fuel)
        latent_heat = water_per_fuel * self.WATER_LATENT_HEAT_MJ_KG

        lhv = hhv_mj_kg - latent_heat

        return lhv.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)

    def _convert_to_mass(
        self,
        quantity: Decimal,
        unit: str,
        density_kg_m3: Decimal,
        temperature_c: Optional[Decimal] = None
    ) -> Decimal:
        """
        Convert quantity to mass in kg.

        Args:
            quantity: Input quantity
            unit: Input unit (kg, tonne, m3, L, bbl, gal)
            density_kg_m3: Fuel density at 15C
            temperature_c: Observed temperature for volume correction

        Returns:
            Mass in kg
        """
        if unit == "kg":
            return quantity
        elif unit == "tonne":
            return quantity * Decimal("1000")
        elif unit == "g":
            return quantity / Decimal("1000")
        elif unit == "lb":
            return quantity * Decimal("0.45359237")
        elif unit == "ton":
            return quantity * Decimal("907.18474")
        elif unit in ("m3", "L", "bbl", "gal"):
            # Volume -> Mass using density
            # Apply temperature correction if provided
            if temperature_c is not None:
                # Corrected density at observed temperature
                density = self.calculate_density_at_temperature(
                    density_kg_m3, temperature_c
                )
            else:
                density = density_kg_m3

            # Convert volume to m3
            if unit == "L":
                volume_m3 = quantity / Decimal("1000")
            elif unit == "bbl":
                volume_m3 = quantity * Decimal("0.158987294928")
            elif unit == "gal":
                volume_m3 = quantity * Decimal("0.003785411784")
            else:
                volume_m3 = quantity

            return volume_m3 * density
        else:
            raise ValueError(f"Unsupported unit: {unit}")
