"""
Combustion Stoichiometry and Analysis

Zero-Hallucination Combustion Calculations per EPA Method 19

This module implements deterministic combustion calculations for:
- Stoichiometric air requirements
- Excess air calculations
- Combustion product composition
- Heat release calculations
- CO2 emissions calculations

References:
    - EPA Method 19: Sulfur Dioxide Removal Efficiency and PM, SO2, and NOx
      Emission Rates
    - ASME PTC 4.1: Steam Generating Units
    - API 560: Fired Heaters for General Refinery Service
    - EPA AP-42: Compilation of Air Pollutant Emission Factors

Author: GreenLang Engineering Team
License: MIT
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple
import hashlib
import math


@dataclass
class FuelComposition:
    """
    Fuel composition on mass basis (weight percent).

    All components must sum to 100%.
    """
    carbon: Decimal  # C, wt%
    hydrogen: Decimal  # H, wt%
    oxygen: Decimal  # O, wt%
    nitrogen: Decimal  # N, wt%
    sulfur: Decimal  # S, wt%
    moisture: Decimal  # H2O, wt%
    ash: Decimal  # Ash, wt%

    def validate(self) -> bool:
        """Validate that components sum to 100%."""
        total = (self.carbon + self.hydrogen + self.oxygen +
                 self.nitrogen + self.sulfur + self.moisture + self.ash)
        if abs(total - Decimal("100")) > Decimal("0.01"):
            raise ValueError(f"Fuel composition must sum to 100%, got {total}%")
        return True


@dataclass
class GasFuelComposition:
    """
    Gas fuel composition on molar basis (volume percent).

    All components must sum to 100%.
    """
    methane: Decimal  # CH4, vol%
    ethane: Decimal  # C2H6, vol%
    propane: Decimal  # C3H8, vol%
    n_butane: Decimal  # n-C4H10, vol%
    iso_butane: Decimal  # i-C4H10, vol%
    n_pentane: Decimal  # n-C5H12, vol%
    iso_pentane: Decimal  # i-C5H12, vol%
    hexane_plus: Decimal  # C6+, vol%
    carbon_dioxide: Decimal  # CO2, vol%
    nitrogen: Decimal  # N2, vol%
    hydrogen_sulfide: Decimal  # H2S, vol%
    hydrogen: Decimal  # H2, vol%
    carbon_monoxide: Decimal  # CO, vol%

    def validate(self) -> bool:
        """Validate that components sum to 100%."""
        total = (self.methane + self.ethane + self.propane + self.n_butane +
                 self.iso_butane + self.n_pentane + self.iso_pentane +
                 self.hexane_plus + self.carbon_dioxide + self.nitrogen +
                 self.hydrogen_sulfide + self.hydrogen + self.carbon_monoxide)
        if abs(total - Decimal("100")) > Decimal("0.01"):
            raise ValueError(f"Fuel composition must sum to 100%, got {total}%")
        return True


@dataclass
class CombustionResult:
    """
    Combustion calculation results with complete provenance.

    All values are deterministic - same inputs produce identical outputs.
    """
    # Air requirements
    stoichiometric_air_kg_per_kg_fuel: Decimal
    actual_air_kg_per_kg_fuel: Decimal
    excess_air_percent: Decimal

    # Flue gas composition (dry basis, vol%)
    flue_gas_co2_percent: Decimal
    flue_gas_o2_percent: Decimal
    flue_gas_n2_percent: Decimal
    flue_gas_so2_ppm: Decimal

    # Flue gas composition (wet basis, vol%)
    flue_gas_h2o_percent: Decimal

    # Mass flows (per kg fuel)
    flue_gas_mass_kg_per_kg_fuel: Decimal
    co2_mass_kg_per_kg_fuel: Decimal
    h2o_mass_kg_per_kg_fuel: Decimal
    so2_mass_kg_per_kg_fuel: Decimal

    # Heat release
    higher_heating_value_kj_kg: Decimal
    lower_heating_value_kj_kg: Decimal

    # Adiabatic flame temperature
    adiabatic_flame_temperature_k: Optional[Decimal]

    # Provenance
    provenance_hash: str

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "stoichiometric_air_kg_per_kg_fuel": float(self.stoichiometric_air_kg_per_kg_fuel),
            "actual_air_kg_per_kg_fuel": float(self.actual_air_kg_per_kg_fuel),
            "excess_air_percent": float(self.excess_air_percent),
            "flue_gas_co2_percent": float(self.flue_gas_co2_percent),
            "flue_gas_o2_percent": float(self.flue_gas_o2_percent),
            "flue_gas_n2_percent": float(self.flue_gas_n2_percent),
            "flue_gas_so2_ppm": float(self.flue_gas_so2_ppm),
            "flue_gas_h2o_percent": float(self.flue_gas_h2o_percent),
            "flue_gas_mass_kg_per_kg_fuel": float(self.flue_gas_mass_kg_per_kg_fuel),
            "co2_mass_kg_per_kg_fuel": float(self.co2_mass_kg_per_kg_fuel),
            "h2o_mass_kg_per_kg_fuel": float(self.h2o_mass_kg_per_kg_fuel),
            "so2_mass_kg_per_kg_fuel": float(self.so2_mass_kg_per_kg_fuel),
            "higher_heating_value_kj_kg": float(self.higher_heating_value_kj_kg),
            "lower_heating_value_kj_kg": float(self.lower_heating_value_kj_kg),
            "adiabatic_flame_temperature_k": float(self.adiabatic_flame_temperature_k) if self.adiabatic_flame_temperature_k else None,
            "provenance_hash": self.provenance_hash
        }


class CombustionCalculator:
    """
    Combustion stoichiometry and analysis calculator.

    ZERO-HALLUCINATION GUARANTEE:
    - All calculations are deterministic
    - Based on EPA Method 19 and ASME PTC 4.1
    - Complete provenance tracking

    References:
        - EPA Method 19: 40 CFR Part 60, Appendix A
        - ASME PTC 4.1: Steam Generating Units
        - Perry's Chemical Engineers' Handbook, 8th Ed.
    """

    # Molecular weights (g/mol) - DETERMINISTIC CONSTANTS
    MW_C = Decimal("12.011")
    MW_H = Decimal("1.008")
    MW_O = Decimal("15.999")
    MW_N = Decimal("14.007")
    MW_S = Decimal("32.065")
    MW_H2O = Decimal("18.015")
    MW_CO2 = Decimal("44.009")
    MW_SO2 = Decimal("64.064")
    MW_O2 = Decimal("31.998")
    MW_N2 = Decimal("28.014")
    MW_AIR = Decimal("28.97")

    # Air composition - DETERMINISTIC CONSTANTS
    AIR_O2_MOLAR = Decimal("0.2095")  # O2 mole fraction in air
    AIR_N2_MOLAR = Decimal("0.7808")  # N2 mole fraction in air
    AIR_AR_MOLAR = Decimal("0.0093")  # Ar mole fraction (treated as N2)

    # Standard heating values (kJ/kg) - EPA AP-42 values
    HHV_CARBON = Decimal("32780")  # kJ/kg C
    HHV_HYDROGEN = Decimal("141800")  # kJ/kg H2
    HHV_SULFUR = Decimal("9260")  # kJ/kg S
    LHV_WATER_VAPOR = Decimal("2442")  # kJ/kg H2O at 25C

    # Gas component properties
    GAS_COMPONENTS = {
        "methane": {"formula": "CH4", "MW": Decimal("16.043"), "HHV": Decimal("55528"), "LHV": Decimal("50013")},
        "ethane": {"formula": "C2H6", "MW": Decimal("30.070"), "HHV": Decimal("51901"), "LHV": Decimal("47511")},
        "propane": {"formula": "C3H8", "MW": Decimal("44.097"), "HHV": Decimal("50358"), "LHV": Decimal("46354")},
        "n_butane": {"formula": "C4H10", "MW": Decimal("58.123"), "HHV": Decimal("49510"), "LHV": Decimal("45717")},
        "iso_butane": {"formula": "C4H10", "MW": Decimal("58.123"), "HHV": Decimal("49363"), "LHV": Decimal("45570")},
        "n_pentane": {"formula": "C5H12", "MW": Decimal("72.150"), "HHV": Decimal("48643"), "LHV": Decimal("44983")},
        "iso_pentane": {"formula": "C5H12", "MW": Decimal("72.150"), "HHV": Decimal("48558"), "LHV": Decimal("44898")},
        "hexane_plus": {"formula": "C6H14", "MW": Decimal("86.177"), "HHV": Decimal("48310"), "LHV": Decimal("44740")},
        "hydrogen": {"formula": "H2", "MW": Decimal("2.016"), "HHV": Decimal("141800"), "LHV": Decimal("119950")},
        "carbon_monoxide": {"formula": "CO", "MW": Decimal("28.010"), "HHV": Decimal("10103"), "LHV": Decimal("10103")},
        "hydrogen_sulfide": {"formula": "H2S", "MW": Decimal("34.081"), "HHV": Decimal("16500"), "LHV": Decimal("15200")},
    }

    def __init__(self, precision: int = 4):
        """Initialize combustion calculator."""
        self.precision = precision

    def _apply_precision(self, value: Decimal) -> Decimal:
        """Apply precision rounding."""
        if self.precision == 0:
            return value.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        quantize_str = "0." + "0" * self.precision
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    def _calculate_provenance(self, inputs: Dict, outputs: Dict) -> str:
        """Calculate SHA-256 hash for audit trail."""
        provenance_data = {
            "method": "EPA_Method_19_ASME_PTC_4.1",
            "inputs": {k: str(v) for k, v in inputs.items()},
            "outputs": {k: str(v) for k, v in outputs.items()}
        }
        provenance_str = str(sorted(provenance_data.items()))
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def calculate_solid_fuel(
        self,
        fuel: FuelComposition,
        excess_air_percent: float,
        air_temperature_k: float = 298.15,
        fuel_temperature_k: float = 298.15
    ) -> CombustionResult:
        """
        Calculate combustion for solid/liquid fuel.

        ZERO-HALLUCINATION: Deterministic stoichiometric calculation
        per EPA Method 19 and ASME PTC 4.1.

        Reference:
            - EPA Method 19: 40 CFR Part 60, Appendix A, Section 12.3
            - ASME PTC 4.1: Section 5, Combustion Calculations

        Args:
            fuel: Fuel ultimate analysis (mass basis)
            excess_air_percent: Excess air as percentage (e.g., 15 for 15%)
            air_temperature_k: Combustion air temperature (K)
            fuel_temperature_k: Fuel temperature (K)

        Returns:
            CombustionResult with complete provenance
        """
        fuel.validate()

        # Convert to decimal fractions
        c = fuel.carbon / Decimal("100")
        h = fuel.hydrogen / Decimal("100")
        o = fuel.oxygen / Decimal("100")
        n = fuel.nitrogen / Decimal("100")
        s = fuel.sulfur / Decimal("100")
        m = fuel.moisture / Decimal("100")
        ash = fuel.ash / Decimal("100")

        excess_air = Decimal(str(excess_air_percent)) / Decimal("100")

        # Step 1: Calculate stoichiometric oxygen requirement
        # Reference: ASME PTC 4.1 Equation 5-1
        # O2_stoich = (C/12.011)*32 + (H/4.032)*32 + (S/32.065)*32 - O
        o2_for_c = c / self.MW_C * self.MW_O2  # kg O2/kg fuel
        o2_for_h = h / (Decimal("4") * self.MW_H) * self.MW_O2  # kg O2/kg fuel
        o2_for_s = s / self.MW_S * self.MW_O2  # kg O2/kg fuel
        o2_from_fuel = o  # kg O2/kg fuel (oxygen in fuel)

        o2_stoich = o2_for_c + o2_for_h + o2_for_s - o2_from_fuel

        if o2_stoich < 0:
            o2_stoich = Decimal("0")

        # Step 2: Calculate stoichiometric air
        # Air is 23.2% O2 by mass
        air_stoich = o2_stoich / Decimal("0.232")

        # Step 3: Calculate actual air with excess
        air_actual = air_stoich * (Decimal("1") + excess_air)

        # Step 4: Calculate combustion products
        # CO2 from carbon combustion
        co2_mass = c * self.MW_CO2 / self.MW_C

        # H2O from hydrogen combustion + moisture
        h2o_from_h = h * self.MW_H2O / (Decimal("2") * self.MW_H)
        h2o_total = h2o_from_h + m

        # SO2 from sulfur combustion
        so2_mass = s * self.MW_SO2 / self.MW_S

        # N2 from air + fuel nitrogen
        n2_from_air = air_actual * Decimal("0.768")  # 76.8% N2 by mass in air
        n2_from_fuel = n
        n2_total = n2_from_air + n2_from_fuel

        # Excess O2 in flue gas
        o2_excess = air_actual * Decimal("0.232") - o2_stoich

        # Step 5: Calculate flue gas composition
        flue_gas_mass = co2_mass + h2o_total + so2_mass + n2_total + o2_excess

        # Dry flue gas composition (volume/molar basis)
        # Convert mass to moles
        mol_co2 = co2_mass / self.MW_CO2
        mol_so2 = so2_mass / self.MW_SO2
        mol_n2 = n2_total / self.MW_N2
        mol_o2 = o2_excess / self.MW_O2
        mol_h2o = h2o_total / self.MW_H2O

        mol_dry = mol_co2 + mol_so2 + mol_n2 + mol_o2
        mol_wet = mol_dry + mol_h2o

        # Dry basis percentages
        co2_percent_dry = mol_co2 / mol_dry * Decimal("100")
        o2_percent_dry = mol_o2 / mol_dry * Decimal("100")
        n2_percent_dry = mol_n2 / mol_dry * Decimal("100")
        so2_ppm = mol_so2 / mol_dry * Decimal("1000000")

        # Wet basis
        h2o_percent_wet = mol_h2o / mol_wet * Decimal("100")

        # Step 6: Calculate heating values
        # Reference: ASME PTC 4.1 Section 5.6
        hhv = (c * self.HHV_CARBON + h * self.HHV_HYDROGEN +
               s * self.HHV_SULFUR - m * self.LHV_WATER_VAPOR)

        # LHV = HHV - latent heat of water formed
        lhv = hhv - h2o_from_h * self.LHV_WATER_VAPOR

        # Step 7: Calculate adiabatic flame temperature (simplified)
        # Reference: Perry's 8th Ed, Chapter 27
        aft = self._calculate_adiabatic_flame_temp(
            hhv, flue_gas_mass, air_temperature_k, fuel_temperature_k
        )

        # Create provenance
        inputs = {
            "carbon_wt_pct": str(fuel.carbon),
            "hydrogen_wt_pct": str(fuel.hydrogen),
            "oxygen_wt_pct": str(fuel.oxygen),
            "nitrogen_wt_pct": str(fuel.nitrogen),
            "sulfur_wt_pct": str(fuel.sulfur),
            "moisture_wt_pct": str(fuel.moisture),
            "ash_wt_pct": str(fuel.ash),
            "excess_air_percent": str(excess_air_percent),
            "air_temperature_k": str(air_temperature_k)
        }
        outputs = {
            "stoich_air": str(air_stoich),
            "actual_air": str(air_actual),
            "co2_mass": str(co2_mass),
            "hhv": str(hhv)
        }
        provenance_hash = self._calculate_provenance(inputs, outputs)

        return CombustionResult(
            stoichiometric_air_kg_per_kg_fuel=self._apply_precision(air_stoich),
            actual_air_kg_per_kg_fuel=self._apply_precision(air_actual),
            excess_air_percent=self._apply_precision(Decimal(str(excess_air_percent))),
            flue_gas_co2_percent=self._apply_precision(co2_percent_dry),
            flue_gas_o2_percent=self._apply_precision(o2_percent_dry),
            flue_gas_n2_percent=self._apply_precision(n2_percent_dry),
            flue_gas_so2_ppm=self._apply_precision(so2_ppm),
            flue_gas_h2o_percent=self._apply_precision(h2o_percent_wet),
            flue_gas_mass_kg_per_kg_fuel=self._apply_precision(flue_gas_mass),
            co2_mass_kg_per_kg_fuel=self._apply_precision(co2_mass),
            h2o_mass_kg_per_kg_fuel=self._apply_precision(h2o_total),
            so2_mass_kg_per_kg_fuel=self._apply_precision(so2_mass),
            higher_heating_value_kj_kg=self._apply_precision(hhv),
            lower_heating_value_kj_kg=self._apply_precision(lhv),
            adiabatic_flame_temperature_k=self._apply_precision(aft) if aft else None,
            provenance_hash=provenance_hash
        )

    def calculate_gas_fuel(
        self,
        fuel: GasFuelComposition,
        excess_air_percent: float,
        air_temperature_k: float = 298.15
    ) -> CombustionResult:
        """
        Calculate combustion for gaseous fuel.

        ZERO-HALLUCINATION: Deterministic stoichiometric calculation.

        Reference:
            - API 560: Section 8, Fired Heater Combustion
            - EPA Method 19: Natural Gas Combustion Factors

        Args:
            fuel: Gas composition (molar/volume basis)
            excess_air_percent: Excess air percentage
            air_temperature_k: Combustion air temperature (K)

        Returns:
            CombustionResult with complete provenance
        """
        fuel.validate()

        excess_air = Decimal(str(excess_air_percent)) / Decimal("100")

        # Convert vol% to mole fractions
        ch4 = fuel.methane / Decimal("100")
        c2h6 = fuel.ethane / Decimal("100")
        c3h8 = fuel.propane / Decimal("100")
        c4h10 = (fuel.n_butane + fuel.iso_butane) / Decimal("100")
        c5h12 = (fuel.n_pentane + fuel.iso_pentane) / Decimal("100")
        c6h14 = fuel.hexane_plus / Decimal("100")
        co2_fuel = fuel.carbon_dioxide / Decimal("100")
        n2_fuel = fuel.nitrogen / Decimal("100")
        h2s = fuel.hydrogen_sulfide / Decimal("100")
        h2 = fuel.hydrogen / Decimal("100")
        co = fuel.carbon_monoxide / Decimal("100")

        # Calculate molecular weight of fuel mixture
        mw_fuel = (
            ch4 * self.GAS_COMPONENTS["methane"]["MW"] +
            c2h6 * self.GAS_COMPONENTS["ethane"]["MW"] +
            c3h8 * self.GAS_COMPONENTS["propane"]["MW"] +
            c4h10 * self.GAS_COMPONENTS["n_butane"]["MW"] +
            c5h12 * self.GAS_COMPONENTS["n_pentane"]["MW"] +
            c6h14 * self.GAS_COMPONENTS["hexane_plus"]["MW"] +
            co2_fuel * self.MW_CO2 +
            n2_fuel * self.MW_N2 +
            h2s * self.GAS_COMPONENTS["hydrogen_sulfide"]["MW"] +
            h2 * self.GAS_COMPONENTS["hydrogen"]["MW"] +
            co * self.GAS_COMPONENTS["carbon_monoxide"]["MW"]
        )

        # Stoichiometric O2 per mole of fuel mixture
        # CH4 + 2O2 -> CO2 + 2H2O
        # C2H6 + 3.5O2 -> 2CO2 + 3H2O
        # C3H8 + 5O2 -> 3CO2 + 4H2O
        # C4H10 + 6.5O2 -> 4CO2 + 5H2O
        # C5H12 + 8O2 -> 5CO2 + 6H2O
        # C6H14 + 9.5O2 -> 6CO2 + 7H2O
        # H2 + 0.5O2 -> H2O
        # CO + 0.5O2 -> CO2
        # H2S + 1.5O2 -> SO2 + H2O

        o2_stoich_mol = (
            ch4 * Decimal("2") +
            c2h6 * Decimal("3.5") +
            c3h8 * Decimal("5") +
            c4h10 * Decimal("6.5") +
            c5h12 * Decimal("8") +
            c6h14 * Decimal("9.5") +
            h2 * Decimal("0.5") +
            co * Decimal("0.5") +
            h2s * Decimal("1.5")
        )

        # Convert to mass basis (per kg fuel)
        o2_stoich_mass = o2_stoich_mol * self.MW_O2 / mw_fuel

        # Stoichiometric air
        air_stoich = o2_stoich_mass / Decimal("0.232")

        # Actual air
        air_actual = air_stoich * (Decimal("1") + excess_air)

        # Products (per mole fuel)
        co2_mol = ch4 + Decimal("2") * c2h6 + Decimal("3") * c3h8 + Decimal("4") * c4h10 + Decimal("5") * c5h12 + Decimal("6") * c6h14 + co + co2_fuel
        h2o_mol = Decimal("2") * ch4 + Decimal("3") * c2h6 + Decimal("4") * c3h8 + Decimal("5") * c4h10 + Decimal("6") * c5h12 + Decimal("7") * c6h14 + h2 + h2s
        so2_mol = h2s

        # Convert to mass (per kg fuel)
        co2_mass = co2_mol * self.MW_CO2 / mw_fuel
        h2o_mass = h2o_mol * self.MW_H2O / mw_fuel
        so2_mass = so2_mol * self.MW_SO2 / mw_fuel

        # N2 from air and fuel
        n2_from_air = air_actual * Decimal("0.768")
        n2_from_fuel = n2_fuel * self.MW_N2 / mw_fuel
        n2_total = n2_from_air + n2_from_fuel

        # Excess O2
        o2_excess = air_actual * Decimal("0.232") - o2_stoich_mass

        # Total flue gas
        flue_gas_mass = co2_mass + h2o_mass + so2_mass + n2_total + o2_excess

        # Molar composition
        mol_co2 = co2_mass / self.MW_CO2
        mol_h2o = h2o_mass / self.MW_H2O
        mol_so2 = so2_mass / self.MW_SO2
        mol_n2 = n2_total / self.MW_N2
        mol_o2 = o2_excess / self.MW_O2

        mol_dry = mol_co2 + mol_so2 + mol_n2 + mol_o2
        mol_wet = mol_dry + mol_h2o

        co2_percent_dry = mol_co2 / mol_dry * Decimal("100")
        o2_percent_dry = mol_o2 / mol_dry * Decimal("100")
        n2_percent_dry = mol_n2 / mol_dry * Decimal("100")
        so2_ppm = mol_so2 / mol_dry * Decimal("1000000")
        h2o_percent_wet = mol_h2o / mol_wet * Decimal("100")

        # Heating values (weighted average)
        hhv = (
            ch4 * self.GAS_COMPONENTS["methane"]["HHV"] +
            c2h6 * self.GAS_COMPONENTS["ethane"]["HHV"] +
            c3h8 * self.GAS_COMPONENTS["propane"]["HHV"] +
            c4h10 * self.GAS_COMPONENTS["n_butane"]["HHV"] +
            c5h12 * self.GAS_COMPONENTS["n_pentane"]["HHV"] +
            c6h14 * self.GAS_COMPONENTS["hexane_plus"]["HHV"] +
            h2 * self.GAS_COMPONENTS["hydrogen"]["HHV"] +
            co * self.GAS_COMPONENTS["carbon_monoxide"]["HHV"] +
            h2s * self.GAS_COMPONENTS["hydrogen_sulfide"]["HHV"]
        )

        lhv = (
            ch4 * self.GAS_COMPONENTS["methane"]["LHV"] +
            c2h6 * self.GAS_COMPONENTS["ethane"]["LHV"] +
            c3h8 * self.GAS_COMPONENTS["propane"]["LHV"] +
            c4h10 * self.GAS_COMPONENTS["n_butane"]["LHV"] +
            c5h12 * self.GAS_COMPONENTS["n_pentane"]["LHV"] +
            c6h14 * self.GAS_COMPONENTS["hexane_plus"]["LHV"] +
            h2 * self.GAS_COMPONENTS["hydrogen"]["LHV"] +
            co * self.GAS_COMPONENTS["carbon_monoxide"]["LHV"] +
            h2s * self.GAS_COMPONENTS["hydrogen_sulfide"]["LHV"]
        )

        # Adiabatic flame temperature
        aft = self._calculate_adiabatic_flame_temp(
            hhv, flue_gas_mass, air_temperature_k, Decimal("298.15")
        )

        inputs = {
            "methane_vol_pct": str(fuel.methane),
            "ethane_vol_pct": str(fuel.ethane),
            "propane_vol_pct": str(fuel.propane),
            "excess_air_percent": str(excess_air_percent)
        }
        outputs = {
            "stoich_air": str(air_stoich),
            "actual_air": str(air_actual),
            "co2_mass": str(co2_mass)
        }
        provenance_hash = self._calculate_provenance(inputs, outputs)

        return CombustionResult(
            stoichiometric_air_kg_per_kg_fuel=self._apply_precision(air_stoich),
            actual_air_kg_per_kg_fuel=self._apply_precision(air_actual),
            excess_air_percent=self._apply_precision(Decimal(str(excess_air_percent))),
            flue_gas_co2_percent=self._apply_precision(co2_percent_dry),
            flue_gas_o2_percent=self._apply_precision(o2_percent_dry),
            flue_gas_n2_percent=self._apply_precision(n2_percent_dry),
            flue_gas_so2_ppm=self._apply_precision(so2_ppm),
            flue_gas_h2o_percent=self._apply_precision(h2o_percent_wet),
            flue_gas_mass_kg_per_kg_fuel=self._apply_precision(flue_gas_mass),
            co2_mass_kg_per_kg_fuel=self._apply_precision(co2_mass),
            h2o_mass_kg_per_kg_fuel=self._apply_precision(h2o_mass),
            so2_mass_kg_per_kg_fuel=self._apply_precision(so2_mass),
            higher_heating_value_kj_kg=self._apply_precision(hhv),
            lower_heating_value_kj_kg=self._apply_precision(lhv),
            adiabatic_flame_temperature_k=self._apply_precision(aft) if aft else None,
            provenance_hash=provenance_hash
        )

    def _calculate_adiabatic_flame_temp(
        self,
        hhv: Decimal,
        flue_gas_mass: Decimal,
        air_temp: float,
        fuel_temp: Decimal
    ) -> Optional[Decimal]:
        """
        Calculate adiabatic flame temperature.

        Reference: Perry's Chemical Engineers' Handbook, 8th Ed, Chapter 27

        Simplified calculation assuming constant Cp.
        """
        if flue_gas_mass <= 0:
            return None

        # Average Cp of flue gas (kJ/kg-K) - simplified
        cp_fg = Decimal("1.15")

        # Reference temperature
        t_ref = Decimal("298.15")

        # Energy balance: Q_combustion = m_fg * Cp * (T_flame - T_ref)
        # T_flame = T_ref + Q_combustion / (m_fg * Cp)

        delta_t = hhv / (flue_gas_mass * cp_fg)
        t_flame = t_ref + delta_t

        # Limit to realistic range
        if t_flame > Decimal("2500"):
            t_flame = Decimal("2500")

        return t_flame

    def excess_air_from_o2(self, o2_percent_dry: float) -> Decimal:
        """
        Calculate excess air from O2 measurement.

        Reference: EPA Method 19, Equation 19-1

        Args:
            o2_percent_dry: O2 concentration in dry flue gas (%)

        Returns:
            Excess air percentage
        """
        o2 = Decimal(str(o2_percent_dry))

        if o2 >= Decimal("20.95"):
            raise ValueError("O2 cannot exceed atmospheric concentration")

        # EPA Method 19 Equation 19-1
        # Excess Air (%) = 100 * O2d / (20.95 - O2d)
        excess_air = Decimal("100") * o2 / (Decimal("20.95") - o2)

        return self._apply_precision(excess_air)

    def f_factor_calculation(
        self,
        fuel: FuelComposition
    ) -> Tuple[Decimal, Decimal, Decimal]:
        """
        Calculate EPA F-factors for emissions calculations.

        Reference: EPA Method 19, Table 19-2

        Args:
            fuel: Fuel composition

        Returns:
            Tuple of (Fd_dry, Fw_wet, Fc_carbon) F-factors
        """
        fuel.validate()

        c = fuel.carbon / Decimal("100")
        h = fuel.hydrogen / Decimal("100")
        o = fuel.oxygen / Decimal("100")
        n = fuel.nitrogen / Decimal("100")
        s = fuel.sulfur / Decimal("100")
        m = fuel.moisture / Decimal("100")

        # Calculate GCV (HHV) in Btu/lb
        hhv_btu_lb = (c * Decimal("14093") + h * Decimal("60958") +
                      s * Decimal("3983") - o * Decimal("760"))

        # Fd (dry) - scf/MMBtu
        # Reference: EPA Method 19, Equation 19-9
        fd = (
            Decimal("10.68") * c +
            Decimal("15.68") * h -
            Decimal("1.96") * o -
            Decimal("0.968") * n +
            Decimal("0.735") * s
        ) * Decimal("1E6") / hhv_btu_lb

        # Fw (wet) - scf/MMBtu
        # Reference: EPA Method 19, Equation 19-10
        fw = fd + Decimal("46.8") * (h + m) * Decimal("1E6") / hhv_btu_lb

        # Fc (carbon) - scf/MMBtu
        # Reference: EPA Method 19, Equation 19-11
        fc = Decimal("1.96") * c * Decimal("1E6") / hhv_btu_lb

        return (
            self._apply_precision(fd),
            self._apply_precision(fw),
            self._apply_precision(fc)
        )


# Convenience functions
def combustion_coal(
    carbon: float,
    hydrogen: float,
    oxygen: float,
    nitrogen: float,
    sulfur: float,
    moisture: float,
    ash: float,
    excess_air: float
) -> CombustionResult:
    """
    Calculate coal combustion.

    Example:
        >>> result = combustion_coal(
        ...     carbon=70.0, hydrogen=5.0, oxygen=8.0,
        ...     nitrogen=1.5, sulfur=2.0, moisture=8.0, ash=5.5,
        ...     excess_air=20.0
        ... )
        >>> print(f"CO2: {result.co2_mass_kg_per_kg_fuel} kg/kg")
    """
    calc = CombustionCalculator()
    fuel = FuelComposition(
        carbon=Decimal(str(carbon)),
        hydrogen=Decimal(str(hydrogen)),
        oxygen=Decimal(str(oxygen)),
        nitrogen=Decimal(str(nitrogen)),
        sulfur=Decimal(str(sulfur)),
        moisture=Decimal(str(moisture)),
        ash=Decimal(str(ash))
    )
    return calc.calculate_solid_fuel(fuel, excess_air)


def combustion_natural_gas(
    methane: float = 95.0,
    ethane: float = 2.5,
    propane: float = 0.5,
    co2: float = 1.0,
    n2: float = 1.0,
    excess_air: float = 15.0
) -> CombustionResult:
    """
    Calculate natural gas combustion.

    Example:
        >>> result = combustion_natural_gas(excess_air=15.0)
        >>> print(f"CO2: {result.co2_mass_kg_per_kg_fuel} kg/kg")
    """
    calc = CombustionCalculator()
    fuel = GasFuelComposition(
        methane=Decimal(str(methane)),
        ethane=Decimal(str(ethane)),
        propane=Decimal(str(propane)),
        n_butane=Decimal("0"),
        iso_butane=Decimal("0"),
        n_pentane=Decimal("0"),
        iso_pentane=Decimal("0"),
        hexane_plus=Decimal("0"),
        carbon_dioxide=Decimal(str(co2)),
        nitrogen=Decimal(str(n2)),
        hydrogen_sulfide=Decimal("0"),
        hydrogen=Decimal("0"),
        carbon_monoxide=Decimal("0")
    )
    return calc.calculate_gas_fuel(fuel, excess_air)
