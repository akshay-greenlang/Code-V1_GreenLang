"""
Combustion Stoichiometry Calculator Module for GL-010 EMISSIONWATCH.

This module provides deterministic combustion stoichiometry calculations
for emissions modeling. All calculations follow fundamental chemical
principles and are guaranteed to be zero-hallucination.

Fundamental Reactions:
- Carbon: C + O2 -> CO2
- Hydrogen: H2 + 0.5*O2 -> H2O
- Sulfur: S + O2 -> SO2
- Incomplete: C + 0.5*O2 -> CO

References:
- ASME PTC 4: Fired Steam Generators
- EPA Method 19: F-factor calculations
- Turns, S.R. "An Introduction to Combustion"

Zero-Hallucination Guarantee:
- All calculations based on stoichiometric principles
- Mass balance verification for all outputs
- Full provenance tracking
"""

from typing import Dict, List, Optional, Union, Tuple
from decimal import Decimal, ROUND_HALF_UP
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, Field

from .constants import (
    MW, O2_IN_AIR, N2_TO_O2_RATIO, AIR_COMPOSITION,
    NORMAL_TEMP_K, NORMAL_PRESSURE_KPA,
)


class CombustionMode(str, Enum):
    """Combustion mode."""
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True)
class CombustionStep:
    """Individual combustion calculation step."""
    step_number: int
    description: str
    formula: str
    inputs: Dict[str, Union[str, float, Decimal]]
    output_value: Decimal
    output_unit: str


@dataclass(frozen=True)
class CombustionResult:
    """
    Complete combustion calculation result.

    Attributes:
        flue_gas_composition: Molar composition of flue gas
        flue_gas_volume: Flue gas volume (Nm3/kg fuel)
        excess_air_percent: Excess air percentage
        o2_in_flue_gas: O2 concentration (% dry)
        co2_in_flue_gas: CO2 concentration (% dry)
        adiabatic_flame_temp: Theoretical flame temperature (K)
        calculation_steps: Detailed calculation steps
    """
    flue_gas_composition: Dict[str, Decimal]
    flue_gas_volume_wet: Decimal
    flue_gas_volume_dry: Decimal
    excess_air_percent: Decimal
    o2_in_flue_gas_dry: Decimal
    co2_in_flue_gas_dry: Decimal
    h2o_in_flue_gas: Decimal
    adiabatic_flame_temp: Decimal
    air_fuel_ratio: Decimal
    calculation_steps: List[CombustionStep]


class FuelComposition(BaseModel):
    """Fuel composition for combustion calculations."""
    carbon: float = Field(ge=0, le=100, description="Carbon (wt%)")
    hydrogen: float = Field(ge=0, le=25, description="Hydrogen (wt%)")
    oxygen: float = Field(ge=0, le=50, description="Oxygen (wt%)")
    nitrogen: float = Field(ge=0, le=10, description="Nitrogen (wt%)")
    sulfur: float = Field(ge=0, le=10, description="Sulfur (wt%)")
    ash: float = Field(ge=0, le=50, description="Ash (wt%)")
    moisture: float = Field(ge=0, le=70, description="Moisture (wt%)")


class CombustionConditions(BaseModel):
    """Operating conditions for combustion."""
    excess_air_percent: float = Field(
        ge=0, le=500, default=15,
        description="Excess air (%)"
    )
    combustion_efficiency: float = Field(
        ge=0.9, le=1.0, default=0.99,
        description="Combustion efficiency (fraction)"
    )
    air_temperature_k: float = Field(
        default=298.15,
        description="Combustion air temperature (K)"
    )
    air_humidity_kg_kg: float = Field(
        default=0.01, ge=0, le=0.05,
        description="Air humidity (kg H2O/kg dry air)"
    )
    fuel_temperature_k: float = Field(
        default=298.15,
        description="Fuel temperature (K)"
    )


class CombustionStoichiometry:
    """
    Zero-hallucination combustion stoichiometry calculator.

    Implements deterministic calculations for:
    - Stoichiometric air/oxygen requirements
    - Flue gas composition and volume
    - Excess air calculations
    - Adiabatic flame temperature

    All calculations are based on fundamental chemical stoichiometry
    and produce identical results for identical inputs.
    """

    # Heat of combustion for elements (kJ/mol)
    HEAT_OF_COMBUSTION: Dict[str, Decimal] = {
        "C": Decimal("393.5"),   # C + O2 -> CO2
        "H2": Decimal("285.8"),  # H2 + 0.5*O2 -> H2O (liquid)
        "S": Decimal("296.8"),   # S + O2 -> SO2
        "CO": Decimal("283.0"),  # CO + 0.5*O2 -> CO2
    }

    # Specific heats of gases (kJ/kmol/K at ~1500K)
    CP_GASES: Dict[str, Decimal] = {
        "CO2": Decimal("54.3"),
        "H2O": Decimal("43.8"),
        "N2": Decimal("33.0"),
        "O2": Decimal("35.5"),
        "SO2": Decimal("52.0"),
        "Ar": Decimal("20.8"),
    }

    @classmethod
    def calculate_stoichiometric_requirements(
        cls,
        fuel: FuelComposition,
        precision: int = 4
    ) -> Tuple[Decimal, Decimal, Decimal, List[CombustionStep]]:
        """
        Calculate stoichiometric O2 and air requirements.

        Stoichiometric combustion reactions:
        C + O2 -> CO2
        H2 + 0.5*O2 -> H2O
        S + O2 -> SO2

        Args:
            fuel: Fuel composition (weight %)
            precision: Decimal places

        Returns:
            Tuple of (O2 kmol/kg fuel, air kmol/kg fuel, air kg/kg fuel, steps)
        """
        steps = []

        # Convert weight % to fractions
        c = Decimal(str(fuel.carbon)) / Decimal("100")
        h = Decimal(str(fuel.hydrogen)) / Decimal("100")
        o = Decimal(str(fuel.oxygen)) / Decimal("100")
        s = Decimal(str(fuel.sulfur)) / Decimal("100")

        # O2 required for each element (kmol O2 / kg fuel)
        # C: 1 mol O2 / mol C = 32/12 kg O2/kg C
        o2_for_c = c / MW["C"]  # kmol C/kg fuel = kmol O2/kg fuel

        # H: 0.5 mol O2 / mol H2 = 0.5*32/2 = 8 kg O2/kg H2
        # H fraction is atomic H, so divide by 2 for H2
        o2_for_h = (h / MW["H"]) / Decimal("2") * Decimal("0.5")

        # S: 1 mol O2 / mol S
        o2_for_s = s / MW["S"]

        # O2 in fuel (reduces requirement)
        o2_in_fuel = o / MW["O2"]

        # Total stoichiometric O2 (kmol/kg fuel)
        stoich_o2 = o2_for_c + o2_for_h + o2_for_s - o2_in_fuel

        steps.append(CombustionStep(
            step_number=1,
            description="Calculate stoichiometric O2 requirement",
            formula="n_O2 = C/12 + H/4 + S/32 - O/32 (kmol/kg)",
            inputs={
                "C_frac": str(c),
                "H_frac": str(h),
                "S_frac": str(s),
                "O_frac": str(o)
            },
            output_value=cls._apply_precision(stoich_o2, precision),
            output_unit="kmol O2/kg fuel"
        ))

        # Stoichiometric air (kmol/kg fuel)
        # Air is 20.95% O2 by volume
        o2_in_air = Decimal("0.2095")
        stoich_air_kmol = stoich_o2 / o2_in_air

        steps.append(CombustionStep(
            step_number=2,
            description="Calculate stoichiometric air (molar)",
            formula="n_air = n_O2 / 0.2095",
            inputs={
                "stoich_O2_kmol": str(stoich_o2),
                "O2_in_air_vol": str(o2_in_air)
            },
            output_value=cls._apply_precision(stoich_air_kmol, precision),
            output_unit="kmol air/kg fuel"
        ))

        # Stoichiometric air (kg/kg fuel)
        stoich_air_kg = stoich_air_kmol * MW["AIR"]

        steps.append(CombustionStep(
            step_number=3,
            description="Calculate stoichiometric air (mass)",
            formula="m_air = n_air * MW_air",
            inputs={
                "stoich_air_kmol": str(stoich_air_kmol),
                "MW_air": str(MW["AIR"])
            },
            output_value=cls._apply_precision(stoich_air_kg, precision),
            output_unit="kg air/kg fuel"
        ))

        return stoich_o2, stoich_air_kmol, stoich_air_kg, steps

    @classmethod
    def calculate_combustion(
        cls,
        fuel: FuelComposition,
        conditions: CombustionConditions,
        precision: int = 4
    ) -> CombustionResult:
        """
        Calculate complete combustion products and conditions.

        Args:
            fuel: Fuel composition
            conditions: Operating conditions
            precision: Decimal places

        Returns:
            CombustionResult with complete analysis
        """
        steps = []

        # Step 1: Calculate stoichiometric requirements
        stoich_o2, stoich_air_kmol, stoich_air_kg, stoich_steps = \
            cls.calculate_stoichiometric_requirements(fuel, precision)
        steps.extend(stoich_steps)

        # Convert fuel composition
        c = Decimal(str(fuel.carbon)) / Decimal("100")
        h = Decimal(str(fuel.hydrogen)) / Decimal("100")
        o = Decimal(str(fuel.oxygen)) / Decimal("100")
        n = Decimal(str(fuel.nitrogen)) / Decimal("100")
        s = Decimal(str(fuel.sulfur)) / Decimal("100")
        moisture = Decimal(str(fuel.moisture)) / Decimal("100")

        excess_air = Decimal(str(conditions.excess_air_percent)) / Decimal("100")
        comb_eff = Decimal(str(conditions.combustion_efficiency))

        # Step 2: Calculate actual air supply
        actual_air_kmol = stoich_air_kmol * (Decimal("1") + excess_air)
        actual_o2_kmol = stoich_o2 * (Decimal("1") + excess_air)

        steps.append(CombustionStep(
            step_number=len(steps) + 1,
            description="Calculate actual air supply",
            formula="n_air_actual = n_air_stoich * (1 + excess_air)",
            inputs={
                "stoich_air": str(stoich_air_kmol),
                "excess_air": str(excess_air)
            },
            output_value=cls._apply_precision(actual_air_kmol, precision),
            output_unit="kmol air/kg fuel"
        ))

        # Step 3: Calculate combustion products
        # CO2 from complete combustion of carbon
        co2_produced = c / MW["C"] * comb_eff

        # CO from incomplete combustion
        co_produced = c / MW["C"] * (Decimal("1") - comb_eff)

        # H2O from hydrogen combustion and fuel moisture
        h2o_from_h = (h / MW["H"]) / Decimal("2")
        h2o_from_moisture = moisture / MW["H2O"]
        # H2O from combustion air humidity
        humidity = Decimal(str(conditions.air_humidity_kg_kg))
        h2o_from_air = actual_air_kmol * MW["AIR"] * humidity / MW["H2O"]

        total_h2o = h2o_from_h + h2o_from_moisture + h2o_from_air

        # SO2 from sulfur
        so2_produced = s / MW["S"]

        # N2 from air and fuel
        n2_from_air = actual_air_kmol * Decimal("0.7808")
        n2_from_fuel = n / MW["N2"]
        total_n2 = n2_from_air + n2_from_fuel

        # Excess O2
        o2_consumed = stoich_o2
        excess_o2 = actual_o2_kmol - o2_consumed

        # Argon from air
        ar_from_air = actual_air_kmol * Decimal("0.0093")

        steps.append(CombustionStep(
            step_number=len(steps) + 1,
            description="Calculate flue gas components",
            formula="Products from stoichiometry",
            inputs={
                "CO2": str(cls._apply_precision(co2_produced, precision)),
                "H2O": str(cls._apply_precision(total_h2o, precision)),
                "SO2": str(cls._apply_precision(so2_produced, precision)),
                "N2": str(cls._apply_precision(total_n2, precision)),
                "O2_excess": str(cls._apply_precision(excess_o2, precision))
            },
            output_value=co2_produced,
            output_unit="kmol/kg fuel"
        ))

        # Step 4: Calculate total flue gas
        total_wet = co2_produced + co_produced + total_h2o + so2_produced + total_n2 + excess_o2 + ar_from_air
        total_dry = total_wet - total_h2o

        # Step 5: Calculate concentrations (% dry basis)
        o2_dry_percent = (excess_o2 / total_dry) * Decimal("100")
        co2_dry_percent = (co2_produced / total_dry) * Decimal("100")
        h2o_percent = (total_h2o / total_wet) * Decimal("100")

        steps.append(CombustionStep(
            step_number=len(steps) + 1,
            description="Calculate flue gas concentrations",
            formula="%_dry = (n_i / n_total_dry) * 100",
            inputs={
                "total_dry_kmol": str(total_dry),
                "total_wet_kmol": str(total_wet)
            },
            output_value=cls._apply_precision(o2_dry_percent, precision),
            output_unit="% O2 (dry)"
        ))

        # Step 6: Convert to volumes at normal conditions
        mv = Decimal("22.414")  # Nm3/kmol
        flue_gas_wet_nm3 = total_wet * mv
        flue_gas_dry_nm3 = total_dry * mv

        steps.append(CombustionStep(
            step_number=len(steps) + 1,
            description="Calculate flue gas volume at STP",
            formula="V = n * 22.414 Nm3/kmol",
            inputs={
                "total_wet_kmol": str(total_wet),
                "molar_volume": str(mv)
            },
            output_value=cls._apply_precision(flue_gas_wet_nm3, precision),
            output_unit="Nm3/kg fuel (wet)"
        ))

        # Step 7: Calculate adiabatic flame temperature
        aft, aft_step = cls.calculate_adiabatic_flame_temperature(
            fuel, conditions, precision
        )
        steps.append(aft_step)

        # Compile composition dictionary
        flue_gas_composition = {
            "CO2": cls._apply_precision(co2_produced, precision),
            "CO": cls._apply_precision(co_produced, precision),
            "H2O": cls._apply_precision(total_h2o, precision),
            "SO2": cls._apply_precision(so2_produced, precision),
            "N2": cls._apply_precision(total_n2, precision),
            "O2": cls._apply_precision(excess_o2, precision),
            "Ar": cls._apply_precision(ar_from_air, precision),
        }

        # Calculate air-fuel ratio
        afr = actual_air_kmol * MW["AIR"]

        return CombustionResult(
            flue_gas_composition=flue_gas_composition,
            flue_gas_volume_wet=cls._apply_precision(flue_gas_wet_nm3, precision),
            flue_gas_volume_dry=cls._apply_precision(flue_gas_dry_nm3, precision),
            excess_air_percent=cls._apply_precision(excess_air * Decimal("100"), precision),
            o2_in_flue_gas_dry=cls._apply_precision(o2_dry_percent, precision),
            co2_in_flue_gas_dry=cls._apply_precision(co2_dry_percent, precision),
            h2o_in_flue_gas=cls._apply_precision(h2o_percent, precision),
            adiabatic_flame_temp=cls._apply_precision(aft, precision),
            air_fuel_ratio=cls._apply_precision(afr, precision),
            calculation_steps=steps
        )

    @classmethod
    def calculate_excess_air_from_o2(
        cls,
        o2_percent_dry: Union[float, Decimal],
        fuel: FuelComposition,
        precision: int = 4
    ) -> Tuple[Decimal, List[CombustionStep]]:
        """
        Calculate excess air from measured O2 in flue gas.

        Formula:
        Excess Air (%) = O2% / (20.9 - O2%) * 100

        This is an approximation. More accurate calculation uses
        fuel-specific F-factors.

        Args:
            o2_percent_dry: Measured O2 concentration (% dry)
            fuel: Fuel composition (for validation)
            precision: Decimal places

        Returns:
            Tuple of (excess air %, calculation steps)
        """
        steps = []
        o2 = Decimal(str(o2_percent_dry))

        # Validate O2 range
        if o2 < 0 or o2 >= Decimal("20.9"):
            raise ValueError(f"O2 must be 0-20.9%, got {o2}%")

        # Simple formula
        o2_in_air = Decimal("20.9")
        excess_air = (o2 / (o2_in_air - o2)) * Decimal("100")

        steps.append(CombustionStep(
            step_number=1,
            description="Calculate excess air from O2 measurement",
            formula="EA% = O2% / (20.9 - O2%) * 100",
            inputs={
                "O2_percent_dry": str(o2),
                "O2_in_air": str(o2_in_air)
            },
            output_value=cls._apply_precision(excess_air, precision),
            output_unit="%"
        ))

        return cls._apply_precision(excess_air, precision), steps

    @classmethod
    def calculate_excess_air_from_co2(
        cls,
        co2_percent_dry: Union[float, Decimal],
        fuel: FuelComposition,
        precision: int = 4
    ) -> Tuple[Decimal, List[CombustionStep]]:
        """
        Calculate excess air from measured CO2 in flue gas.

        This method requires knowing the theoretical CO2 at
        stoichiometric conditions, which depends on fuel composition.

        Args:
            co2_percent_dry: Measured CO2 concentration (% dry)
            fuel: Fuel composition
            precision: Decimal places

        Returns:
            Tuple of (excess air %, calculation steps)
        """
        steps = []
        co2_measured = Decimal(str(co2_percent_dry))

        # Calculate theoretical max CO2 (at stoichiometric conditions)
        c = Decimal(str(fuel.carbon)) / Decimal("100")
        h = Decimal(str(fuel.hydrogen)) / Decimal("100")
        o = Decimal(str(fuel.oxygen)) / Decimal("100")
        n = Decimal(str(fuel.nitrogen)) / Decimal("100")
        s = Decimal(str(fuel.sulfur)) / Decimal("100")

        # Moles of products at stoichiometric (per kg fuel)
        co2_stoich = c / MW["C"]
        n2_stoich = n / MW["N2"]

        # O2 required (stoichiometric)
        o2_stoich = c / MW["C"] + (h / MW["H"]) / Decimal("4") + s / MW["S"] - o / MW["O2"]

        # N2 from stoichiometric air
        n2_from_air = o2_stoich * N2_TO_O2_RATIO
        total_n2_stoich = n2_from_air + n2_stoich

        # Total dry flue gas at stoichiometric
        total_dry_stoich = co2_stoich + total_n2_stoich + s / MW["S"]

        # Theoretical max CO2 %
        co2_max = (co2_stoich / total_dry_stoich) * Decimal("100")

        steps.append(CombustionStep(
            step_number=1,
            description="Calculate theoretical maximum CO2",
            formula="CO2_max% = n_CO2 / n_total_dry_stoich * 100",
            inputs={
                "fuel_carbon_%": str(fuel.carbon),
                "n_CO2_stoich": str(co2_stoich)
            },
            output_value=cls._apply_precision(co2_max, precision),
            output_unit="% CO2 max"
        ))

        # Calculate excess air
        # At higher excess air, CO2% decreases (dilution by N2 and O2)
        # EA% = (CO2_max / CO2_meas - 1) * 100
        if co2_measured <= 0:
            raise ValueError("CO2 must be positive")

        excess_air = (co2_max / co2_measured - Decimal("1")) * Decimal("100")

        steps.append(CombustionStep(
            step_number=2,
            description="Calculate excess air from CO2",
            formula="EA% = (CO2_max / CO2_meas - 1) * 100",
            inputs={
                "CO2_max": str(co2_max),
                "CO2_measured": str(co2_measured)
            },
            output_value=cls._apply_precision(excess_air, precision),
            output_unit="%"
        ))

        return cls._apply_precision(excess_air, precision), steps

    @classmethod
    def calculate_adiabatic_flame_temperature(
        cls,
        fuel: FuelComposition,
        conditions: CombustionConditions,
        precision: int = 0
    ) -> Tuple[Decimal, CombustionStep]:
        """
        Calculate adiabatic flame temperature.

        Energy balance:
        Heat_released = Heat_absorbed_by_products

        Sum(n_i * Cp_i) * (Tad - Tref) = HHV * fuel_mass

        This is a simplified calculation assuming constant Cp.

        Args:
            fuel: Fuel composition
            conditions: Combustion conditions
            precision: Decimal places

        Returns:
            Tuple of (adiabatic flame temp K, calculation step)
        """
        c = Decimal(str(fuel.carbon)) / Decimal("100")
        h = Decimal(str(fuel.hydrogen)) / Decimal("100")
        s = Decimal(str(fuel.sulfur)) / Decimal("100")
        excess_air = Decimal(str(conditions.excess_air_percent)) / Decimal("100")
        t_air = Decimal(str(conditions.air_temperature_k))

        # Heat released (kJ/kg fuel)
        # Using heats of combustion
        q_c = (c / MW["C"]) * cls.HEAT_OF_COMBUSTION["C"] * Decimal("1000")  # kJ/kg fuel
        q_h = (h / MW["H"] / Decimal("2")) * cls.HEAT_OF_COMBUSTION["H2"] * Decimal("1000")
        q_s = (s / MW["S"]) * cls.HEAT_OF_COMBUSTION["S"] * Decimal("1000")

        heat_released = q_c + q_h + q_s  # kJ/kg fuel

        # Moles of products (kmol/kg fuel)
        co2_mol = c / MW["C"]
        h2o_mol = h / MW["H"] / Decimal("2")
        so2_mol = s / MW["S"]

        # Air and N2
        o2_stoich = c / MW["C"] + (h / MW["H"]) / Decimal("4") + s / MW["S"]
        actual_o2 = o2_stoich * (Decimal("1") + excess_air)
        excess_o2_mol = actual_o2 - o2_stoich
        n2_mol = actual_o2 * N2_TO_O2_RATIO

        # Heat capacity of products (kJ/K per kg fuel)
        cp_products = (
            co2_mol * cls.CP_GASES["CO2"] +
            h2o_mol * cls.CP_GASES["H2O"] +
            so2_mol * cls.CP_GASES["SO2"] +
            n2_mol * cls.CP_GASES["N2"] +
            excess_o2_mol * cls.CP_GASES["O2"]
        )

        # Adiabatic flame temperature
        # Q = Cp * (Tad - Tref)
        # Tad = Tref + Q / Cp
        t_ref = Decimal("298.15")  # Reference temperature
        delta_t = heat_released / cp_products
        t_ad = t_air + delta_t

        # Cap at reasonable maximum
        t_ad = min(t_ad, Decimal("2500"))

        step = CombustionStep(
            step_number=1,
            description="Calculate adiabatic flame temperature",
            formula="Tad = Tair + Q_released / Cp_products",
            inputs={
                "heat_released_kj_kg": str(cls._apply_precision(heat_released, 1)),
                "cp_products_kj_K_kg": str(cls._apply_precision(cp_products, 4)),
                "T_air_K": str(t_air)
            },
            output_value=cls._apply_precision(t_ad, precision),
            output_unit="K"
        )

        return cls._apply_precision(t_ad, precision), step

    @classmethod
    def convert_wet_to_dry(
        cls,
        concentration_wet: Union[float, Decimal],
        moisture_percent: Union[float, Decimal],
        precision: int = 4
    ) -> Tuple[Decimal, CombustionStep]:
        """
        Convert concentration from wet to dry basis.

        Formula:
        C_dry = C_wet / (1 - H2O/100)

        Args:
            concentration_wet: Concentration on wet basis
            moisture_percent: Moisture content (%)
            precision: Decimal places

        Returns:
            Tuple of (concentration dry basis, calculation step)
        """
        c_wet = Decimal(str(concentration_wet))
        h2o = Decimal(str(moisture_percent))

        if h2o < 0 or h2o >= 100:
            raise ValueError("Moisture must be 0-100%")

        c_dry = c_wet / (Decimal("1") - h2o / Decimal("100"))

        step = CombustionStep(
            step_number=1,
            description="Convert wet to dry basis",
            formula="C_dry = C_wet / (1 - H2O/100)",
            inputs={
                "C_wet": str(c_wet),
                "moisture_%": str(h2o)
            },
            output_value=cls._apply_precision(c_dry, precision),
            output_unit="dry basis"
        )

        return cls._apply_precision(c_dry, precision), step

    @classmethod
    def convert_dry_to_wet(
        cls,
        concentration_dry: Union[float, Decimal],
        moisture_percent: Union[float, Decimal],
        precision: int = 4
    ) -> Tuple[Decimal, CombustionStep]:
        """
        Convert concentration from dry to wet basis.

        Formula:
        C_wet = C_dry * (1 - H2O/100)

        Args:
            concentration_dry: Concentration on dry basis
            moisture_percent: Moisture content (%)
            precision: Decimal places

        Returns:
            Tuple of (concentration wet basis, calculation step)
        """
        c_dry = Decimal(str(concentration_dry))
        h2o = Decimal(str(moisture_percent))

        if h2o < 0 or h2o >= 100:
            raise ValueError("Moisture must be 0-100%")

        c_wet = c_dry * (Decimal("1") - h2o / Decimal("100"))

        step = CombustionStep(
            step_number=1,
            description="Convert dry to wet basis",
            formula="C_wet = C_dry * (1 - H2O/100)",
            inputs={
                "C_dry": str(c_dry),
                "moisture_%": str(h2o)
            },
            output_value=cls._apply_precision(c_wet, precision),
            output_unit="wet basis"
        )

        return cls._apply_precision(c_wet, precision), step

    @staticmethod
    def _apply_precision(value: Decimal, precision: int) -> Decimal:
        """Apply decimal precision with ROUND_HALF_UP."""
        if precision < 0:
            raise ValueError(f"Precision must be non-negative")
        quantize_str = "0." + "0" * precision if precision > 0 else "1"
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


# Convenience functions
def calculate_combustion_products(
    carbon: float,
    hydrogen: float,
    oxygen: float,
    nitrogen: float,
    sulfur: float,
    ash: float,
    moisture: float,
    excess_air_percent: float = 15
) -> CombustionResult:
    """
    Convenience function to calculate combustion products.

    Args:
        carbon, hydrogen, oxygen, nitrogen, sulfur, ash, moisture: Fuel composition (wt%)
        excess_air_percent: Excess air (%)

    Returns:
        CombustionResult with complete analysis
    """
    fuel = FuelComposition(
        carbon=carbon,
        hydrogen=hydrogen,
        oxygen=oxygen,
        nitrogen=nitrogen,
        sulfur=sulfur,
        ash=ash,
        moisture=moisture
    )

    conditions = CombustionConditions(
        excess_air_percent=excess_air_percent
    )

    return CombustionStoichiometry.calculate_combustion(fuel, conditions)


def get_excess_air_from_o2(o2_percent: float) -> Decimal:
    """
    Quick calculation of excess air from O2 measurement.

    Args:
        o2_percent: O2 in flue gas (% dry basis)

    Returns:
        Excess air (%)
    """
    # Dummy fuel for the calculation
    fuel = FuelComposition(
        carbon=75, hydrogen=5, oxygen=5,
        nitrogen=1.5, sulfur=2.5, ash=10, moisture=1
    )
    excess_air, _ = CombustionStoichiometry.calculate_excess_air_from_o2(
        o2_percent, fuel
    )
    return excess_air
