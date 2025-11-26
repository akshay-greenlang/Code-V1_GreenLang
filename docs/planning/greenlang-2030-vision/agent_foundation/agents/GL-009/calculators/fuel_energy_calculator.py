"""Fuel Energy Calculator.

This module implements fuel energy content calculations including
Higher Heating Value (HHV), Lower Heating Value (LHV), combustion
stoichiometry, and energy calculations from fuel flow rates.

Key Calculations:
    - HHV/LHV conversion for various fuels
    - Fuel composition analysis
    - Combustion stoichiometry (air/fuel ratio)
    - Energy content from mass/volume flow rates
    - Excess air calculations

Standards:
    - ASTM D240: Heat of Combustion (Liquid Fuels)
    - ASTM D1826: Calorific Value (Gaseous Fuels)
    - ISO 6976: Natural Gas Properties

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


class FuelType(Enum):
    """Fuel type classifications."""
    NATURAL_GAS = "natural_gas"
    METHANE = "methane"
    PROPANE = "propane"
    BUTANE = "butane"
    DIESEL = "diesel"
    FUEL_OIL_2 = "fuel_oil_2"
    FUEL_OIL_6 = "fuel_oil_6"
    COAL_BITUMINOUS = "coal_bituminous"
    COAL_ANTHRACITE = "coal_anthracite"
    COAL_LIGNITE = "coal_lignite"
    BIOMASS_WOOD = "biomass_wood"
    BIOMASS_PELLETS = "biomass_pellets"
    HYDROGEN = "hydrogen"
    BIOGAS = "biogas"
    LANDFILL_GAS = "landfill_gas"
    GASOLINE = "gasoline"
    KEROSENE = "kerosene"
    LPG = "lpg"
    OTHER = "other"


class HeatingValueType(Enum):
    """Type of heating value."""
    HHV = "hhv"  # Higher Heating Value (Gross)
    LHV = "lhv"  # Lower Heating Value (Net)


class FuelState(Enum):
    """Physical state of fuel."""
    GAS = "gas"
    LIQUID = "liquid"
    SOLID = "solid"


@dataclass(frozen=True)
class FuelComposition:
    """Ultimate analysis composition of a fuel.

    All values in mass percentage (dry basis).

    Attributes:
        carbon_percent: Carbon content (%)
        hydrogen_percent: Hydrogen content (%)
        oxygen_percent: Oxygen content (%)
        nitrogen_percent: Nitrogen content (%)
        sulfur_percent: Sulfur content (%)
        ash_percent: Ash content (%)
        moisture_percent: Moisture content (% as-received)
    """
    carbon_percent: float = 0.0
    hydrogen_percent: float = 0.0
    oxygen_percent: float = 0.0
    nitrogen_percent: float = 0.0
    sulfur_percent: float = 0.0
    ash_percent: float = 0.0
    moisture_percent: float = 0.0

    def __post_init__(self) -> None:
        """Validate composition."""
        total = (self.carbon_percent + self.hydrogen_percent +
                 self.oxygen_percent + self.nitrogen_percent +
                 self.sulfur_percent + self.ash_percent)
        if total > 100.1:  # Allow small rounding error
            raise ValueError(f"Composition exceeds 100%: {total}%")

    @property
    def total_dry(self) -> float:
        """Total dry basis composition."""
        return (self.carbon_percent + self.hydrogen_percent +
                self.oxygen_percent + self.nitrogen_percent +
                self.sulfur_percent + self.ash_percent)


@dataclass
class CombustionResult:
    """Results of combustion stoichiometry calculation.

    Attributes:
        stoichiometric_air_fuel_ratio: Theoretical A/F ratio (kg air/kg fuel)
        actual_air_fuel_ratio: Actual A/F ratio with excess air
        excess_air_percent: Excess air percentage
        flue_gas_mass_kg_per_kg_fuel: Flue gas produced per kg fuel
        co2_produced_kg_per_kg_fuel: CO2 produced per kg fuel
        h2o_produced_kg_per_kg_fuel: H2O produced per kg fuel
        theoretical_o2_required_kg: O2 required per kg fuel
        theoretical_air_required_kg: Air required per kg fuel
    """
    stoichiometric_air_fuel_ratio: float
    actual_air_fuel_ratio: float
    excess_air_percent: float
    flue_gas_mass_kg_per_kg_fuel: float
    co2_produced_kg_per_kg_fuel: float
    h2o_produced_kg_per_kg_fuel: float
    theoretical_o2_required_kg: float
    theoretical_air_required_kg: float


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
class FuelEnergyResult:
    """Complete fuel energy calculation result.

    Attributes:
        fuel_type: Type of fuel
        fuel_state: Physical state of fuel
        hhv_kj_kg: Higher Heating Value (kJ/kg)
        lhv_kj_kg: Lower Heating Value (kJ/kg)
        hhv_kj_m3: Higher Heating Value (kJ/m3) for gases
        lhv_kj_m3: Lower Heating Value (kJ/m3) for gases
        energy_input_kw: Energy input rate (kW)
        mass_flow_kg_s: Mass flow rate (kg/s)
        volume_flow_m3_s: Volume flow rate (m3/s) for gases
        combustion_result: Stoichiometry results
        composition: Fuel composition used
        calculation_steps: Audit trail
        provenance_hash: SHA-256 hash
        calculation_timestamp: When calculated
        warnings: Any warnings
    """
    fuel_type: FuelType
    fuel_state: FuelState
    hhv_kj_kg: float
    lhv_kj_kg: float
    hhv_kj_m3: Optional[float]
    lhv_kj_m3: Optional[float]
    energy_input_kw: float
    mass_flow_kg_s: float
    volume_flow_m3_s: Optional[float]
    combustion_result: Optional[CombustionResult]
    composition: Optional[FuelComposition]
    calculation_steps: List[CalculationStep]
    provenance_hash: str
    calculation_timestamp: str
    calculator_version: str = "1.0.0"
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {
            "fuel_type": self.fuel_type.value,
            "fuel_state": self.fuel_state.value,
            "hhv_kj_kg": self.hhv_kj_kg,
            "lhv_kj_kg": self.lhv_kj_kg,
            "energy_input_kw": self.energy_input_kw,
            "mass_flow_kg_s": self.mass_flow_kg_s,
            "provenance_hash": self.provenance_hash,
            "calculation_timestamp": self.calculation_timestamp,
            "warnings": self.warnings
        }
        if self.hhv_kj_m3:
            result["hhv_kj_m3"] = self.hhv_kj_m3
        if self.lhv_kj_m3:
            result["lhv_kj_m3"] = self.lhv_kj_m3
        return result


class FuelEnergyCalculator:
    """Fuel Energy Content Calculator.

    Calculates fuel energy content from heating values and flow rates.
    Supports HHV/LHV calculations, combustion stoichiometry, and
    energy rate calculations.

    Fuel Database:
        Contains default HHV/LHV values for common fuels from
        authoritative sources (EPA, EIA, IPCC).

    Example:
        >>> calculator = FuelEnergyCalculator()
        >>> result = calculator.calculate_energy_from_mass_flow(
        ...     fuel_type=FuelType.NATURAL_GAS,
        ...     mass_flow_kg_s=0.1,
        ...     heating_value_type=HeatingValueType.LHV
        ... )
        >>> print(f"Energy input: {result.energy_input_kw} kW")
    """

    VERSION: str = "1.0.0"
    PRECISION: int = 4

    # Air composition for combustion calculations
    AIR_O2_MASS_FRACTION: float = 0.232  # 23.2% O2 by mass
    AIR_N2_MASS_FRACTION: float = 0.768  # 76.8% N2 by mass

    # Molecular weights (g/mol)
    MW_C: float = 12.011
    MW_H: float = 1.008
    MW_O: float = 16.00
    MW_N: float = 14.007
    MW_S: float = 32.065
    MW_CO2: float = 44.01
    MW_H2O: float = 18.015
    MW_SO2: float = 64.066
    MW_O2: float = 32.00

    # Default fuel properties database
    # Format: {fuel_type: (HHV kJ/kg, LHV kJ/kg, density kg/m3, state)}
    FUEL_DATABASE: Dict[FuelType, Tuple[float, float, float, FuelState]] = {
        FuelType.NATURAL_GAS: (55500, 50000, 0.72, FuelState.GAS),
        FuelType.METHANE: (55500, 50000, 0.656, FuelState.GAS),
        FuelType.PROPANE: (50350, 46350, 1.88, FuelState.GAS),
        FuelType.BUTANE: (49500, 45750, 2.52, FuelState.GAS),
        FuelType.DIESEL: (45600, 42800, 850, FuelState.LIQUID),
        FuelType.FUEL_OIL_2: (45500, 42500, 870, FuelState.LIQUID),
        FuelType.FUEL_OIL_6: (42500, 40000, 990, FuelState.LIQUID),
        FuelType.COAL_BITUMINOUS: (32500, 31000, 1300, FuelState.SOLID),
        FuelType.COAL_ANTHRACITE: (34000, 32500, 1500, FuelState.SOLID),
        FuelType.COAL_LIGNITE: (25000, 23500, 1200, FuelState.SOLID),
        FuelType.BIOMASS_WOOD: (20000, 18500, 600, FuelState.SOLID),
        FuelType.BIOMASS_PELLETS: (19000, 17500, 650, FuelState.SOLID),
        FuelType.HYDROGEN: (141800, 120000, 0.0899, FuelState.GAS),
        FuelType.BIOGAS: (22000, 20000, 1.2, FuelState.GAS),
        FuelType.LANDFILL_GAS: (18000, 16000, 1.1, FuelState.GAS),
        FuelType.GASOLINE: (46500, 43500, 750, FuelState.LIQUID),
        FuelType.KEROSENE: (46200, 43200, 800, FuelState.LIQUID),
        FuelType.LPG: (50000, 46000, 540, FuelState.LIQUID),
    }

    # Default compositions for common fuels
    DEFAULT_COMPOSITIONS: Dict[FuelType, FuelComposition] = {
        FuelType.NATURAL_GAS: FuelComposition(
            carbon_percent=75.0, hydrogen_percent=25.0
        ),
        FuelType.DIESEL: FuelComposition(
            carbon_percent=86.0, hydrogen_percent=14.0
        ),
        FuelType.COAL_BITUMINOUS: FuelComposition(
            carbon_percent=75.0, hydrogen_percent=5.0,
            oxygen_percent=8.0, nitrogen_percent=1.5,
            sulfur_percent=2.5, ash_percent=8.0
        ),
    }

    def __init__(self, precision: int = 4) -> None:
        """Initialize the Fuel Energy Calculator.

        Args:
            precision: Decimal places for rounding
        """
        self.precision = precision
        self._calculation_steps: List[CalculationStep] = []
        self._step_counter: int = 0
        self._warnings: List[str] = []

    def calculate_energy_from_mass_flow(
        self,
        fuel_type: FuelType,
        mass_flow_kg_s: float,
        heating_value_type: HeatingValueType = HeatingValueType.LHV,
        custom_heating_value_kj_kg: Optional[float] = None,
        excess_air_percent: float = 15.0
    ) -> FuelEnergyResult:
        """Calculate energy input from fuel mass flow rate.

        Energy = mass_flow * heating_value

        Args:
            fuel_type: Type of fuel
            mass_flow_kg_s: Mass flow rate (kg/s)
            heating_value_type: HHV or LHV
            custom_heating_value_kj_kg: Override default heating value
            excess_air_percent: Excess air for combustion calc

        Returns:
            FuelEnergyResult with energy input and breakdown
        """
        self._reset_calculation_state()

        # Get fuel properties
        if fuel_type in self.FUEL_DATABASE:
            hhv, lhv, density, state = self.FUEL_DATABASE[fuel_type]
        else:
            hhv, lhv, density, state = 45000, 42000, 850, FuelState.LIQUID
            self._warnings.append(f"Using default values for unknown fuel: {fuel_type}")

        # Use custom heating value if provided
        if custom_heating_value_kj_kg:
            if heating_value_type == HeatingValueType.HHV:
                hhv = custom_heating_value_kj_kg
            else:
                lhv = custom_heating_value_kj_kg

        # Select heating value
        heating_value = hhv if heating_value_type == HeatingValueType.HHV else lhv

        # Calculate energy input
        energy_kw = mass_flow_kg_s * heating_value  # kJ/s = kW

        self._add_calculation_step(
            description=f"Calculate energy input from mass flow ({heating_value_type.value})",
            operation="multiply",
            inputs={
                "mass_flow_kg_s": mass_flow_kg_s,
                "heating_value_kj_kg": heating_value
            },
            output_value=energy_kw,
            output_name="energy_input_kw",
            formula="Q = m_dot x HV"
        )

        # Calculate volume flow for gases
        volume_flow = None
        hhv_m3 = None
        lhv_m3 = None
        if state == FuelState.GAS and density > 0:
            volume_flow = mass_flow_kg_s / density
            hhv_m3 = hhv * density
            lhv_m3 = lhv * density

        # Calculate combustion stoichiometry if composition available
        combustion = None
        composition = self.DEFAULT_COMPOSITIONS.get(fuel_type)
        if composition:
            combustion = self.calculate_combustion_stoichiometry(
                composition, excess_air_percent
            )

        # Generate provenance
        provenance = self._generate_provenance_hash(
            fuel_type, mass_flow_kg_s, heating_value
        )
        timestamp = datetime.utcnow().isoformat() + "Z"

        return FuelEnergyResult(
            fuel_type=fuel_type,
            fuel_state=state,
            hhv_kj_kg=self._round_value(hhv),
            lhv_kj_kg=self._round_value(lhv),
            hhv_kj_m3=self._round_value(hhv_m3) if hhv_m3 else None,
            lhv_kj_m3=self._round_value(lhv_m3) if lhv_m3 else None,
            energy_input_kw=self._round_value(energy_kw),
            mass_flow_kg_s=mass_flow_kg_s,
            volume_flow_m3_s=self._round_value(volume_flow) if volume_flow else None,
            combustion_result=combustion,
            composition=composition,
            calculation_steps=self._calculation_steps.copy(),
            provenance_hash=provenance,
            calculation_timestamp=timestamp,
            warnings=self._warnings.copy()
        )

    def calculate_energy_from_volume_flow(
        self,
        fuel_type: FuelType,
        volume_flow_m3_s: float,
        heating_value_type: HeatingValueType = HeatingValueType.LHV,
        custom_density_kg_m3: Optional[float] = None,
        reference_conditions: str = "NTP"
    ) -> FuelEnergyResult:
        """Calculate energy input from gaseous fuel volume flow.

        Args:
            fuel_type: Type of gaseous fuel
            volume_flow_m3_s: Volume flow rate (m3/s)
            heating_value_type: HHV or LHV
            custom_density_kg_m3: Override default density
            reference_conditions: NTP (20C, 1atm) or STP (0C, 1atm)

        Returns:
            FuelEnergyResult with energy input
        """
        # Get fuel properties
        if fuel_type in self.FUEL_DATABASE:
            hhv, lhv, density, state = self.FUEL_DATABASE[fuel_type]
        else:
            raise ValueError(f"Fuel type not in database: {fuel_type}")

        if state != FuelState.GAS:
            self._warnings.append(f"Volume flow used for non-gas fuel: {fuel_type}")

        # Use custom density if provided
        if custom_density_kg_m3:
            density = custom_density_kg_m3

        # Convert volume to mass flow
        mass_flow_kg_s = volume_flow_m3_s * density

        # Calculate using mass flow method
        return self.calculate_energy_from_mass_flow(
            fuel_type=fuel_type,
            mass_flow_kg_s=mass_flow_kg_s,
            heating_value_type=heating_value_type
        )

    def calculate_hhv_from_composition(
        self,
        composition: FuelComposition
    ) -> float:
        """Calculate Higher Heating Value from ultimate analysis.

        Uses Dulong's formula:
            HHV = 33.83*C + 144.3*(H - O/8) + 9.42*S (MJ/kg)

        Where C, H, O, S are mass fractions.

        Args:
            composition: Fuel ultimate analysis

        Returns:
            HHV in kJ/kg
        """
        C = composition.carbon_percent / 100
        H = composition.hydrogen_percent / 100
        O = composition.oxygen_percent / 100
        S = composition.sulfur_percent / 100

        # Dulong's formula (MJ/kg)
        hhv_mj_kg = 33.83 * C + 144.3 * (H - O / 8) + 9.42 * S

        # Convert to kJ/kg
        hhv_kj_kg = hhv_mj_kg * 1000

        self._add_calculation_step(
            description="Calculate HHV from composition (Dulong formula)",
            operation="dulong_formula",
            inputs={
                "carbon_fraction": C,
                "hydrogen_fraction": H,
                "oxygen_fraction": O,
                "sulfur_fraction": S
            },
            output_value=hhv_kj_kg,
            output_name="hhv_kj_kg",
            formula="HHV = 33.83*C + 144.3*(H - O/8) + 9.42*S (MJ/kg)"
        )

        return self._round_value(hhv_kj_kg)

    def calculate_lhv_from_hhv(
        self,
        hhv_kj_kg: float,
        hydrogen_percent: float,
        moisture_percent: float = 0.0
    ) -> float:
        """Calculate LHV from HHV.

        LHV accounts for latent heat of water vapor not recovered.

        LHV = HHV - 2442 * (9*H + M)

        Where:
            2442 kJ/kg = latent heat of vaporization of water
            9*H = water from hydrogen combustion (kg H2O/kg fuel)
            M = moisture in fuel (kg H2O/kg fuel)

        Args:
            hhv_kj_kg: Higher Heating Value (kJ/kg)
            hydrogen_percent: Hydrogen content (%)
            moisture_percent: Moisture content (%)

        Returns:
            LHV in kJ/kg
        """
        H = hydrogen_percent / 100
        M = moisture_percent / 100
        latent_heat = 2442.0  # kJ/kg water

        # Water from combustion: H2 + 0.5*O2 -> H2O
        # 2 kg H produces 18 kg H2O, so 1 kg H produces 9 kg H2O
        water_from_combustion = 9 * H
        total_water = water_from_combustion + M

        lhv_kj_kg = hhv_kj_kg - latent_heat * total_water

        self._add_calculation_step(
            description="Calculate LHV from HHV",
            operation="subtract",
            inputs={
                "hhv_kj_kg": hhv_kj_kg,
                "hydrogen_percent": hydrogen_percent,
                "moisture_percent": moisture_percent,
                "latent_heat_kj_kg": latent_heat,
                "water_produced_kg": total_water
            },
            output_value=lhv_kj_kg,
            output_name="lhv_kj_kg",
            formula="LHV = HHV - 2442 x (9H + M)"
        )

        return self._round_value(lhv_kj_kg)

    def calculate_combustion_stoichiometry(
        self,
        composition: FuelComposition,
        excess_air_percent: float = 15.0
    ) -> CombustionResult:
        """Calculate combustion stoichiometry.

        Calculates theoretical and actual air requirements,
        flue gas production, and combustion products.

        Combustion reactions:
            C + O2 -> CO2
            H2 + 0.5*O2 -> H2O
            S + O2 -> SO2

        Args:
            composition: Fuel ultimate analysis
            excess_air_percent: Excess air percentage

        Returns:
            CombustionResult with stoichiometry details
        """
        # Mass fractions
        C = composition.carbon_percent / 100
        H = composition.hydrogen_percent / 100
        O_fuel = composition.oxygen_percent / 100
        S = composition.sulfur_percent / 100
        N = composition.nitrogen_percent / 100

        # Oxygen required for complete combustion (kg O2 / kg fuel)
        # C + O2 -> CO2: 12 g C needs 32 g O2
        O2_for_C = C * (self.MW_O2 / self.MW_C)

        # H2 + 0.5*O2 -> H2O: 2 g H needs 16 g O2
        O2_for_H = H * (self.MW_O / self.MW_H)

        # S + O2 -> SO2: 32 g S needs 32 g O2
        O2_for_S = S * (self.MW_O2 / self.MW_S)

        # Total O2 needed minus O2 in fuel
        theoretical_O2 = O2_for_C + O2_for_H + O2_for_S - O_fuel
        theoretical_O2 = max(0, theoretical_O2)

        # Air required (O2 is 23.2% of air by mass)
        theoretical_air = theoretical_O2 / self.AIR_O2_MASS_FRACTION

        # Stoichiometric A/F ratio
        stoich_af = theoretical_air

        # Actual air with excess
        excess_factor = 1 + excess_air_percent / 100
        actual_air = theoretical_air * excess_factor
        actual_af = actual_air

        # Combustion products (kg per kg fuel)
        co2_produced = C * (self.MW_CO2 / self.MW_C)
        h2o_produced = H * (self.MW_H2O / (2 * self.MW_H))
        so2_produced = S * (self.MW_SO2 / self.MW_S)

        # Flue gas = combustion products + excess air + N2 from air + N2 from fuel
        n2_from_air = actual_air * self.AIR_N2_MASS_FRACTION
        excess_o2 = (excess_air_percent / 100) * theoretical_O2

        flue_gas_mass = (co2_produced + h2o_produced + so2_produced +
                        n2_from_air + N + excess_o2)

        self._add_calculation_step(
            description="Calculate combustion stoichiometry",
            operation="stoichiometry",
            inputs={
                "carbon_fraction": C,
                "hydrogen_fraction": H,
                "excess_air_percent": excess_air_percent
            },
            output_value=stoich_af,
            output_name="stoichiometric_af_ratio",
            formula="A/F = (O2_required) / 0.232"
        )

        return CombustionResult(
            stoichiometric_air_fuel_ratio=self._round_value(stoich_af),
            actual_air_fuel_ratio=self._round_value(actual_af),
            excess_air_percent=excess_air_percent,
            flue_gas_mass_kg_per_kg_fuel=self._round_value(flue_gas_mass),
            co2_produced_kg_per_kg_fuel=self._round_value(co2_produced),
            h2o_produced_kg_per_kg_fuel=self._round_value(h2o_produced),
            theoretical_o2_required_kg=self._round_value(theoretical_O2),
            theoretical_air_required_kg=self._round_value(theoretical_air)
        )

    def calculate_excess_air_from_o2(
        self,
        flue_gas_o2_percent: float
    ) -> float:
        """Calculate excess air from flue gas O2 measurement.

        Excess Air (%) = O2 / (21 - O2) x 100

        Args:
            flue_gas_o2_percent: O2 in dry flue gas (%)

        Returns:
            Excess air percentage
        """
        if flue_gas_o2_percent >= 21:
            raise ValueError("O2 cannot exceed 21%")

        excess_air = (flue_gas_o2_percent / (21 - flue_gas_o2_percent)) * 100

        self._add_calculation_step(
            description="Calculate excess air from O2 measurement",
            operation="excess_air_calc",
            inputs={"flue_gas_o2_percent": flue_gas_o2_percent},
            output_value=excess_air,
            output_name="excess_air_percent",
            formula="EA = O2 / (21 - O2) x 100"
        )

        return self._round_value(excess_air)

    def get_fuel_properties(
        self,
        fuel_type: FuelType
    ) -> Dict[str, float]:
        """Get default properties for a fuel type.

        Args:
            fuel_type: Type of fuel

        Returns:
            Dictionary with HHV, LHV, density, etc.
        """
        if fuel_type in self.FUEL_DATABASE:
            hhv, lhv, density, state = self.FUEL_DATABASE[fuel_type]
            return {
                "hhv_kj_kg": hhv,
                "lhv_kj_kg": lhv,
                "density_kg_m3": density,
                "state": state.value,
                "hhv_lhv_ratio": hhv / lhv if lhv > 0 else 1.0
            }
        else:
            raise ValueError(f"Unknown fuel type: {fuel_type}")

    def convert_heating_value(
        self,
        value: float,
        from_unit: str,
        to_unit: str
    ) -> float:
        """Convert heating value between units.

        Supported units: kJ/kg, MJ/kg, BTU/lb, therm/lb, kWh/kg

        Args:
            value: Heating value to convert
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted heating value
        """
        # Conversion factors to kJ/kg
        to_kj_kg = {
            "kJ/kg": 1.0,
            "MJ/kg": 1000.0,
            "BTU/lb": 2.326,
            "kWh/kg": 3600.0,
            "kcal/kg": 4.184
        }

        if from_unit not in to_kj_kg or to_unit not in to_kj_kg:
            raise ValueError(f"Unsupported unit: {from_unit} or {to_unit}")

        # Convert to kJ/kg first, then to target
        value_kj_kg = value * to_kj_kg[from_unit]
        result = value_kj_kg / to_kj_kg[to_unit]

        return self._round_value(result)

    def _reset_calculation_state(self) -> None:
        """Reset calculation state."""
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
        fuel_type: FuelType,
        flow_rate: float,
        heating_value: float
    ) -> str:
        """Generate SHA-256 provenance hash."""
        data = {
            "calculator": "FuelEnergyCalculator",
            "version": self.VERSION,
            "fuel_type": fuel_type.value,
            "flow_rate": flow_rate,
            "heating_value": heating_value
        }
        json_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()

    def _round_value(self, value: float, precision: Optional[int] = None) -> float:
        """Round value to precision."""
        if precision is None:
            precision = self.precision

        if value is None:
            return 0.0

        decimal_value = Decimal(str(value))
        quantize_str = '0.' + '0' * precision
        rounded = decimal_value.quantize(
            Decimal(quantize_str),
            rounding=ROUND_HALF_UP
        )
        return float(rounded)
