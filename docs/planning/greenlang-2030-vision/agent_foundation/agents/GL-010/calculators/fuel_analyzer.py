"""
Fuel Analyzer Module for GL-010 EMISSIONWATCH.

This module provides deterministic fuel composition analysis and property
calculations for emissions modeling. All calculations follow ASTM standards
and are guaranteed to be zero-hallucination.

Analysis Types:
- Ultimate Analysis: C, H, O, N, S, Ash, Moisture
- Proximate Analysis: Fixed Carbon, Volatile Matter, Ash, Moisture
- Heating Value: HHV, LHV, Net Calorific Value
- Stoichiometric Requirements: Air, Oxygen, Flue Gas

References:
- ASTM D3176: Standard Practice for Ultimate Analysis of Coal and Coke
- ASTM D5865: Standard Test Method for Gross Calorific Value of Coal and Coke
- ASTM D1945: Standard Test Method for Analysis of Natural Gas by Gas Chromatography
- ISO 6976: Natural gas - Calculation of calorific values

Zero-Hallucination Guarantee:
- All calculations are deterministic (same input -> same output)
- Based on published fuel property correlations
- Full provenance tracking
"""

from typing import Dict, List, Optional, Union, Tuple
from decimal import Decimal, ROUND_HALF_UP
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator

from .constants import MW, O2_IN_AIR, N2_TO_O2_RATIO


class FuelState(str, Enum):
    """Physical state of fuel."""
    GAS = "gas"
    LIQUID = "liquid"
    SOLID = "solid"


class AnalysisBasis(str, Enum):
    """Basis for fuel analysis."""
    AS_RECEIVED = "as_received"  # AR
    DRY = "dry"
    DRY_ASH_FREE = "dry_ash_free"  # DAF


@dataclass(frozen=True)
class FuelAnalysisStep:
    """Individual fuel analysis calculation step."""
    step_number: int
    description: str
    formula: str
    inputs: Dict[str, Union[str, float, Decimal]]
    output_value: Decimal
    output_unit: str


@dataclass(frozen=True)
class FuelProperties:
    """
    Complete fuel properties result.

    Attributes:
        composition: Ultimate analysis (weight %)
        heating_value_hhv: Higher heating value (MJ/kg)
        heating_value_lhv: Lower heating value (MJ/kg)
        stoich_air_ratio: Stoichiometric air-to-fuel ratio (kg/kg)
        stoich_o2_ratio: Stoichiometric O2-to-fuel ratio (kg/kg)
        flue_gas_volume: Flue gas volume at STP (Nm3/kg fuel)
        calculation_steps: Detailed calculation steps
    """
    composition: Dict[str, Decimal]
    heating_value_hhv: Decimal
    heating_value_lhv: Decimal
    stoich_air_ratio: Decimal
    stoich_o2_ratio: Decimal
    flue_gas_volume: Decimal
    calculation_steps: List[FuelAnalysisStep]


class UltimateAnalysis(BaseModel):
    """
    Ultimate analysis of fuel composition.

    All values in weight percent, must sum to 100%.
    """
    carbon: float = Field(ge=0, le=100, description="Carbon content (wt%)")
    hydrogen: float = Field(ge=0, le=25, description="Hydrogen content (wt%)")
    oxygen: float = Field(ge=0, le=50, description="Oxygen content (wt%)")
    nitrogen: float = Field(ge=0, le=10, description="Nitrogen content (wt%)")
    sulfur: float = Field(ge=0, le=10, description="Sulfur content (wt%)")
    ash: float = Field(ge=0, le=50, description="Ash content (wt%)")
    moisture: float = Field(ge=0, le=70, description="Moisture content (wt%)")
    basis: AnalysisBasis = Field(
        default=AnalysisBasis.AS_RECEIVED,
        description="Analysis basis"
    )

    @model_validator(mode='after')
    def validate_sum(self) -> 'UltimateAnalysis':
        """Validate that components sum to 100%."""
        total = (
            self.carbon + self.hydrogen + self.oxygen +
            self.nitrogen + self.sulfur + self.ash + self.moisture
        )
        if abs(total - 100.0) > 0.5:
            raise ValueError(
                f"Ultimate analysis must sum to 100%, got {total:.2f}%"
            )
        return self


class ProximateAnalysis(BaseModel):
    """
    Proximate analysis of fuel (solid fuels).

    All values in weight percent, must sum to 100%.
    """
    fixed_carbon: float = Field(ge=0, le=100, description="Fixed carbon (wt%)")
    volatile_matter: float = Field(ge=0, le=100, description="Volatile matter (wt%)")
    ash: float = Field(ge=0, le=50, description="Ash content (wt%)")
    moisture: float = Field(ge=0, le=70, description="Moisture content (wt%)")
    basis: AnalysisBasis = Field(
        default=AnalysisBasis.AS_RECEIVED,
        description="Analysis basis"
    )

    @model_validator(mode='after')
    def validate_sum(self) -> 'ProximateAnalysis':
        """Validate that components sum to 100%."""
        total = self.fixed_carbon + self.volatile_matter + self.ash + self.moisture
        if abs(total - 100.0) > 0.5:
            raise ValueError(f"Proximate analysis must sum to 100%, got {total:.2f}%")
        return self


class NaturalGasComposition(BaseModel):
    """
    Natural gas composition by mole percent.

    Common natural gas components.
    """
    methane: float = Field(ge=0, le=100, description="CH4 (mol%)")
    ethane: float = Field(ge=0, le=20, description="C2H6 (mol%)")
    propane: float = Field(ge=0, le=10, description="C3H8 (mol%)")
    n_butane: float = Field(ge=0, le=5, description="n-C4H10 (mol%)")
    i_butane: float = Field(ge=0, le=5, description="i-C4H10 (mol%)")
    pentanes_plus: float = Field(ge=0, le=5, default=0, description="C5+ (mol%)")
    nitrogen: float = Field(ge=0, le=20, default=0, description="N2 (mol%)")
    carbon_dioxide: float = Field(ge=0, le=10, default=0, description="CO2 (mol%)")
    hydrogen_sulfide: float = Field(ge=0, le=5, default=0, description="H2S (mol%)")

    @model_validator(mode='after')
    def validate_sum(self) -> 'NaturalGasComposition':
        """Validate that components sum to ~100%."""
        total = (
            self.methane + self.ethane + self.propane +
            self.n_butane + self.i_butane + self.pentanes_plus +
            self.nitrogen + self.carbon_dioxide + self.hydrogen_sulfide
        )
        if abs(total - 100.0) > 1.0:
            raise ValueError(f"Gas composition must sum to 100%, got {total:.2f}%")
        return self


class FuelAnalyzer:
    """
    Zero-hallucination fuel analyzer.

    Provides deterministic fuel property calculations including:
    - Ultimate analysis conversion between bases
    - Heating value calculations (Dulong formula)
    - Stoichiometric air requirements
    - Flue gas composition and volume

    All calculations produce identical results for identical inputs.
    """

    # Default fuel compositions (ultimate analysis, dry basis)
    DEFAULT_COMPOSITIONS: Dict[str, Dict[str, Decimal]] = {
        "natural_gas": {
            "C": Decimal("75.0"),
            "H": Decimal("24.0"),
            "O": Decimal("0"),
            "N": Decimal("0.5"),
            "S": Decimal("0"),
            "Ash": Decimal("0"),
            "H2O": Decimal("0.5"),
        },
        "propane": {
            "C": Decimal("81.7"),
            "H": Decimal("18.3"),
            "O": Decimal("0"),
            "N": Decimal("0"),
            "S": Decimal("0"),
            "Ash": Decimal("0"),
            "H2O": Decimal("0"),
        },
        "fuel_oil_no2": {
            "C": Decimal("87.0"),
            "H": Decimal("12.6"),
            "O": Decimal("0"),
            "N": Decimal("0.1"),
            "S": Decimal("0.3"),
            "Ash": Decimal("0"),
            "H2O": Decimal("0"),
        },
        "fuel_oil_no6": {
            "C": Decimal("85.0"),
            "H": Decimal("10.5"),
            "O": Decimal("0"),
            "N": Decimal("0.5"),
            "S": Decimal("2.5"),
            "Ash": Decimal("0.05"),
            "H2O": Decimal("1.45"),
        },
        "coal_bituminous": {
            "C": Decimal("75.0"),
            "H": Decimal("5.0"),
            "O": Decimal("6.0"),
            "N": Decimal("1.5"),
            "S": Decimal("2.5"),
            "Ash": Decimal("10.0"),
            "H2O": Decimal("0"),
        },
        "coal_subbituminous": {
            "C": Decimal("65.0"),
            "H": Decimal("4.5"),
            "O": Decimal("12.0"),
            "N": Decimal("1.0"),
            "S": Decimal("0.5"),
            "Ash": Decimal("8.0"),
            "H2O": Decimal("9.0"),
        },
        "coal_lignite": {
            "C": Decimal("55.0"),
            "H": Decimal("4.0"),
            "O": Decimal("15.0"),
            "N": Decimal("0.8"),
            "S": Decimal("1.2"),
            "Ash": Decimal("12.0"),
            "H2O": Decimal("12.0"),
        },
        "wood": {
            "C": Decimal("50.0"),
            "H": Decimal("6.0"),
            "O": Decimal("42.0"),
            "N": Decimal("0.5"),
            "S": Decimal("0"),
            "Ash": Decimal("1.5"),
            "H2O": Decimal("0"),
        },
        "biomass": {
            "C": Decimal("45.0"),
            "H": Decimal("5.5"),
            "O": Decimal("40.0"),
            "N": Decimal("1.0"),
            "S": Decimal("0.1"),
            "Ash": Decimal("5.0"),
            "H2O": Decimal("3.4"),
        },
    }

    # Default heating values (MJ/kg, HHV, dry basis)
    DEFAULT_HHV: Dict[str, Decimal] = {
        "natural_gas": Decimal("52.2"),
        "propane": Decimal("50.3"),
        "butane": Decimal("49.5"),
        "fuel_oil_no2": Decimal("45.5"),
        "fuel_oil_no6": Decimal("42.5"),
        "diesel": Decimal("45.5"),
        "gasoline": Decimal("46.5"),
        "coal_bituminous": Decimal("29.0"),
        "coal_subbituminous": Decimal("22.0"),
        "coal_lignite": Decimal("15.0"),
        "coal_anthracite": Decimal("32.0"),
        "wood": Decimal("20.0"),
        "biomass": Decimal("18.0"),
        "petroleum_coke": Decimal("35.0"),
    }

    @classmethod
    def analyze_fuel(
        cls,
        ultimate_analysis: UltimateAnalysis,
        measured_hhv: Optional[float] = None,
        precision: int = 4
    ) -> FuelProperties:
        """
        Perform complete fuel analysis from ultimate analysis.

        Calculates:
        - Composition on different bases
        - Heating values (HHV and LHV)
        - Stoichiometric air requirements
        - Flue gas volume

        Args:
            ultimate_analysis: Fuel ultimate analysis
            measured_hhv: Measured HHV (MJ/kg), calculates if None
            precision: Decimal places in results

        Returns:
            FuelProperties with complete analysis
        """
        steps = []

        # Convert to Decimal
        c = Decimal(str(ultimate_analysis.carbon))
        h = Decimal(str(ultimate_analysis.hydrogen))
        o = Decimal(str(ultimate_analysis.oxygen))
        n = Decimal(str(ultimate_analysis.nitrogen))
        s = Decimal(str(ultimate_analysis.sulfur))
        ash = Decimal(str(ultimate_analysis.ash))
        moisture = Decimal(str(ultimate_analysis.moisture))

        composition = {
            "C": c, "H": h, "O": o, "N": n,
            "S": s, "Ash": ash, "H2O": moisture
        }

        # Step 1: Calculate or use HHV
        if measured_hhv is not None:
            hhv = Decimal(str(measured_hhv))
            steps.append(FuelAnalysisStep(
                step_number=1,
                description="Use measured higher heating value",
                formula="HHV = measured value",
                inputs={"measured_hhv_mj_kg": str(measured_hhv)},
                output_value=hhv,
                output_unit="MJ/kg"
            ))
        else:
            # Calculate using Dulong formula
            hhv, hhv_steps = cls.calculate_hhv_dulong(
                c, h, o, s, precision
            )
            steps.extend(hhv_steps)

        # Step 2: Calculate LHV from HHV
        lhv, lhv_step = cls.calculate_lhv(hhv, h, moisture, precision)
        steps.append(lhv_step)

        # Step 3: Calculate stoichiometric air
        stoich_air, stoich_o2, air_steps = cls.calculate_stoich_air(
            c, h, o, s, precision
        )
        steps.extend(air_steps)

        # Step 4: Calculate flue gas volume
        fg_volume, fg_steps = cls.calculate_flue_gas_volume(
            c, h, o, n, s, moisture, precision
        )
        steps.extend(fg_steps)

        return FuelProperties(
            composition=composition,
            heating_value_hhv=cls._apply_precision(hhv, precision),
            heating_value_lhv=cls._apply_precision(lhv, precision),
            stoich_air_ratio=cls._apply_precision(stoich_air, precision),
            stoich_o2_ratio=cls._apply_precision(stoich_o2, precision),
            flue_gas_volume=cls._apply_precision(fg_volume, precision),
            calculation_steps=steps
        )

    @classmethod
    def calculate_hhv_dulong(
        cls,
        c: Decimal,
        h: Decimal,
        o: Decimal,
        s: Decimal,
        precision: int = 4
    ) -> Tuple[Decimal, List[FuelAnalysisStep]]:
        """
        Calculate Higher Heating Value using Dulong formula.

        Dulong Formula:
        HHV (MJ/kg) = 33.83*C + 144.3*(H - O/8) + 9.42*S

        Where C, H, O, S are weight fractions (not percentages).

        Args:
            c: Carbon content (wt%)
            h: Hydrogen content (wt%)
            o: Oxygen content (wt%)
            s: Sulfur content (wt%)
            precision: Decimal places

        Returns:
            Tuple of (HHV in MJ/kg, calculation steps)

        Reference:
            Modified Dulong formula, ASTM standards
        """
        steps = []

        # Convert percentages to fractions
        c_frac = c / Decimal("100")
        h_frac = h / Decimal("100")
        o_frac = o / Decimal("100")
        s_frac = s / Decimal("100")

        # Dulong formula coefficients
        coef_c = Decimal("33.83")
        coef_h = Decimal("144.3")
        coef_s = Decimal("9.42")

        # Calculate HHV
        # Hydrogen available for combustion = H - O/8
        h_available = h_frac - o_frac / Decimal("8")
        h_available = max(h_available, Decimal("0"))

        hhv = coef_c * c_frac + coef_h * h_available + coef_s * s_frac

        steps.append(FuelAnalysisStep(
            step_number=1,
            description="Calculate HHV using Dulong formula",
            formula="HHV = 33.83*C + 144.3*(H - O/8) + 9.42*S",
            inputs={
                "C_wt%": str(c),
                "H_wt%": str(h),
                "O_wt%": str(o),
                "S_wt%": str(s)
            },
            output_value=cls._apply_precision(hhv, precision),
            output_unit="MJ/kg"
        ))

        return hhv, steps

    @classmethod
    def calculate_lhv(
        cls,
        hhv: Decimal,
        hydrogen: Decimal,
        moisture: Decimal,
        precision: int = 4
    ) -> Tuple[Decimal, FuelAnalysisStep]:
        """
        Calculate Lower Heating Value from HHV.

        LHV accounts for latent heat of water vapor not recovered:
        - Water from combustion of hydrogen: H2 + 0.5*O2 -> H2O
        - Fuel moisture

        Formula:
        LHV = HHV - 2.442 * (9*H + M) / 100

        Where:
        - 2.442 MJ/kg is latent heat of vaporization at 25C
        - 9*H is water formed from hydrogen combustion (H2O/H2 = 18/2 = 9)
        - M is moisture content

        Args:
            hhv: Higher heating value (MJ/kg)
            hydrogen: Hydrogen content (wt%)
            moisture: Moisture content (wt%)
            precision: Decimal places

        Returns:
            Tuple of (LHV in MJ/kg, calculation step)
        """
        latent_heat = Decimal("2.442")  # MJ/kg water at 25C
        water_from_h = Decimal("9") * hydrogen  # Water formed from H
        total_water = (water_from_h + moisture) / Decimal("100")

        lhv = hhv - latent_heat * total_water

        step = FuelAnalysisStep(
            step_number=2,
            description="Calculate LHV from HHV",
            formula="LHV = HHV - 2.442 * (9*H + M) / 100",
            inputs={
                "HHV_mj_kg": str(hhv),
                "H_wt%": str(hydrogen),
                "M_wt%": str(moisture),
                "water_from_combustion": str(water_from_h)
            },
            output_value=cls._apply_precision(lhv, precision),
            output_unit="MJ/kg"
        )

        return lhv, step

    @classmethod
    def calculate_stoich_air(
        cls,
        c: Decimal,
        h: Decimal,
        o: Decimal,
        s: Decimal,
        precision: int = 4
    ) -> Tuple[Decimal, Decimal, List[FuelAnalysisStep]]:
        """
        Calculate stoichiometric air and oxygen requirements.

        Combustion reactions:
        C + O2 -> CO2          (32/12 = 2.67 kg O2/kg C)
        H2 + 0.5*O2 -> H2O     (16/2 = 8 kg O2/kg H2)
        S + O2 -> SO2          (32/32 = 1 kg O2/kg S)

        Air contains 23.2% O2 by mass.

        Args:
            c: Carbon content (wt%)
            h: Hydrogen content (wt%)
            o: Oxygen content (wt%)
            s: Sulfur content (wt%)
            precision: Decimal places

        Returns:
            Tuple of (stoich air ratio, stoich O2 ratio, steps)
        """
        steps = []

        # Convert to fractions
        c_frac = c / Decimal("100")
        h_frac = h / Decimal("100")
        o_frac = o / Decimal("100")
        s_frac = s / Decimal("100")

        # O2 requirements (kg O2 / kg fuel)
        o2_for_c = Decimal("2.667") * c_frac  # 32/12
        o2_for_h = Decimal("8.0") * h_frac    # 16/2
        o2_for_s = Decimal("1.0") * s_frac    # 32/32

        # Subtract O2 in fuel
        total_o2 = o2_for_c + o2_for_h + o2_for_s - o_frac

        steps.append(FuelAnalysisStep(
            step_number=3,
            description="Calculate stoichiometric O2 requirement",
            formula="O2 = 2.667*C + 8*H + S - O (all as fractions)",
            inputs={
                "C_frac": str(c_frac),
                "H_frac": str(h_frac),
                "S_frac": str(s_frac),
                "O_frac": str(o_frac)
            },
            output_value=cls._apply_precision(total_o2, precision),
            output_unit="kg O2/kg fuel"
        ))

        # Convert to air requirement
        # Air is 23.2% O2 by mass
        o2_in_air_mass = Decimal("0.232")
        stoich_air = total_o2 / o2_in_air_mass

        steps.append(FuelAnalysisStep(
            step_number=4,
            description="Calculate stoichiometric air requirement",
            formula="Air = O2 / 0.232",
            inputs={
                "O2_required": str(total_o2),
                "O2_in_air_mass_frac": str(o2_in_air_mass)
            },
            output_value=cls._apply_precision(stoich_air, precision),
            output_unit="kg air/kg fuel"
        ))

        return stoich_air, total_o2, steps

    @classmethod
    def calculate_flue_gas_volume(
        cls,
        c: Decimal,
        h: Decimal,
        o: Decimal,
        n: Decimal,
        s: Decimal,
        moisture: Decimal,
        precision: int = 4
    ) -> Tuple[Decimal, List[FuelAnalysisStep]]:
        """
        Calculate theoretical flue gas volume at STP.

        Flue gas components:
        - CO2 from carbon combustion
        - H2O from hydrogen and fuel moisture
        - SO2 from sulfur
        - N2 from combustion air and fuel

        Args:
            c, h, o, n, s: Composition (wt%)
            moisture: Moisture content (wt%)
            precision: Decimal places

        Returns:
            Tuple of (flue gas volume Nm3/kg fuel, steps)
        """
        steps = []

        # Convert to fractions
        c_frac = c / Decimal("100")
        h_frac = h / Decimal("100")
        o_frac = o / Decimal("100")
        n_frac = n / Decimal("100")
        s_frac = s / Decimal("100")
        m_frac = moisture / Decimal("100")

        # Molar volume at STP (Nm3/kmol)
        mv = Decimal("22.414")

        # CO2 volume (kmol/kg fuel * Nm3/kmol = Nm3/kg)
        # C + O2 -> CO2: 1 mol C -> 1 mol CO2
        co2_vol = (c_frac / MW["C"]) * mv

        # H2O volume from hydrogen combustion
        # H2 + 0.5*O2 -> H2O: 2 mol H -> 1 mol H2O
        h2o_from_h = (h_frac / MW["H"] / Decimal("2")) * mv

        # H2O from fuel moisture
        h2o_from_m = (m_frac / MW["H2O"]) * mv

        total_h2o = h2o_from_h + h2o_from_m

        # SO2 volume
        # S + O2 -> SO2: 1 mol S -> 1 mol SO2
        so2_vol = (s_frac / MW["S"]) * mv

        # N2 from combustion air
        # Stoichiometric O2 requirement (kmol/kg fuel)
        o2_kmol = (
            c_frac / MW["C"] +
            h_frac / MW["H"] / Decimal("4") +
            s_frac / MW["S"] -
            o_frac / MW["O2"]
        )
        # N2 in air = O2 * 3.76 (molar ratio)
        n2_from_air = o2_kmol * N2_TO_O2_RATIO * mv

        # N2 from fuel
        n2_from_fuel = (n_frac / MW["N2"]) * mv

        total_n2 = n2_from_air + n2_from_fuel

        # Total dry flue gas
        dry_fg = co2_vol + so2_vol + total_n2

        # Total wet flue gas
        wet_fg = dry_fg + total_h2o

        steps.append(FuelAnalysisStep(
            step_number=5,
            description="Calculate flue gas volume at STP",
            formula="V_fg = V_CO2 + V_H2O + V_SO2 + V_N2",
            inputs={
                "V_CO2_Nm3_kg": str(cls._apply_precision(co2_vol, precision)),
                "V_H2O_Nm3_kg": str(cls._apply_precision(total_h2o, precision)),
                "V_SO2_Nm3_kg": str(cls._apply_precision(so2_vol, precision)),
                "V_N2_Nm3_kg": str(cls._apply_precision(total_n2, precision))
            },
            output_value=cls._apply_precision(wet_fg, precision),
            output_unit="Nm3/kg fuel (wet)"
        ))

        return wet_fg, steps

    @classmethod
    def convert_analysis_basis(
        cls,
        analysis: UltimateAnalysis,
        target_basis: AnalysisBasis,
        precision: int = 4
    ) -> Tuple[Dict[str, Decimal], List[FuelAnalysisStep]]:
        """
        Convert ultimate analysis between different bases.

        Conversion formulas:
        - AR to Dry: X_dry = X_ar / (1 - M_ar/100)
        - AR to DAF: X_daf = X_ar / (1 - (M_ar + Ash_ar)/100)
        - Dry to AR: X_ar = X_dry * (1 - M_ar/100)
        - Dry to DAF: X_daf = X_dry / (1 - Ash_dry/100)

        Args:
            analysis: Original ultimate analysis
            target_basis: Target basis for conversion
            precision: Decimal places

        Returns:
            Tuple of (converted composition dict, steps)
        """
        steps = []
        current_basis = analysis.basis

        c = Decimal(str(analysis.carbon))
        h = Decimal(str(analysis.hydrogen))
        o = Decimal(str(analysis.oxygen))
        n = Decimal(str(analysis.nitrogen))
        s = Decimal(str(analysis.sulfur))
        ash = Decimal(str(analysis.ash))
        moisture = Decimal(str(analysis.moisture))

        if current_basis == target_basis:
            return {
                "C": c, "H": h, "O": o, "N": n,
                "S": s, "Ash": ash, "H2O": moisture
            }, steps

        # Convert to dry first if needed
        if current_basis == AnalysisBasis.AS_RECEIVED:
            if moisture > 0:
                conversion_factor = Decimal("100") / (Decimal("100") - moisture)
                c = c * conversion_factor
                h = h * conversion_factor
                o = o * conversion_factor
                n = n * conversion_factor
                s = s * conversion_factor
                ash = ash * conversion_factor
                moisture = Decimal("0")

                steps.append(FuelAnalysisStep(
                    step_number=1,
                    description="Convert from as-received to dry basis",
                    formula="X_dry = X_ar * 100 / (100 - M_ar)",
                    inputs={"moisture_ar": str(analysis.moisture)},
                    output_value=conversion_factor,
                    output_unit="conversion factor"
                ))

        # Convert to target basis
        if target_basis == AnalysisBasis.DRY:
            result = {
                "C": cls._apply_precision(c, precision),
                "H": cls._apply_precision(h, precision),
                "O": cls._apply_precision(o, precision),
                "N": cls._apply_precision(n, precision),
                "S": cls._apply_precision(s, precision),
                "Ash": cls._apply_precision(ash, precision),
                "H2O": Decimal("0")
            }

        elif target_basis == AnalysisBasis.DRY_ASH_FREE:
            if ash > 0:
                conversion_factor = Decimal("100") / (Decimal("100") - ash)
                c = c * conversion_factor
                h = h * conversion_factor
                o = o * conversion_factor
                n = n * conversion_factor
                s = s * conversion_factor

                steps.append(FuelAnalysisStep(
                    step_number=2,
                    description="Convert from dry to dry-ash-free basis",
                    formula="X_daf = X_dry * 100 / (100 - Ash_dry)",
                    inputs={"ash_dry": str(analysis.ash)},
                    output_value=conversion_factor,
                    output_unit="conversion factor"
                ))

            result = {
                "C": cls._apply_precision(c, precision),
                "H": cls._apply_precision(h, precision),
                "O": cls._apply_precision(o, precision),
                "N": cls._apply_precision(n, precision),
                "S": cls._apply_precision(s, precision),
                "Ash": Decimal("0"),
                "H2O": Decimal("0")
            }

        elif target_basis == AnalysisBasis.AS_RECEIVED:
            # Need moisture content to convert back
            orig_moisture = Decimal(str(analysis.moisture))
            if orig_moisture > 0:
                conversion_factor = (Decimal("100") - orig_moisture) / Decimal("100")
                c = c * conversion_factor
                h = h * conversion_factor
                o = o * conversion_factor
                n = n * conversion_factor
                s = s * conversion_factor
                ash = ash * conversion_factor
                moisture = orig_moisture

            result = {
                "C": cls._apply_precision(c, precision),
                "H": cls._apply_precision(h, precision),
                "O": cls._apply_precision(o, precision),
                "N": cls._apply_precision(n, precision),
                "S": cls._apply_precision(s, precision),
                "Ash": cls._apply_precision(ash, precision),
                "H2O": cls._apply_precision(moisture, precision)
            }
        else:
            result = {
                "C": c, "H": h, "O": o, "N": n,
                "S": s, "Ash": ash, "H2O": moisture
            }

        return result, steps

    @classmethod
    def analyze_natural_gas(
        cls,
        gas_composition: NaturalGasComposition,
        precision: int = 4
    ) -> FuelProperties:
        """
        Analyze natural gas from molar composition.

        Converts molar composition to mass basis and calculates
        fuel properties.

        Args:
            gas_composition: Molar composition of natural gas
            precision: Decimal places

        Returns:
            FuelProperties with complete analysis
        """
        steps = []

        # Molecular weights of components
        mw_components = {
            "CH4": Decimal("16.043"),
            "C2H6": Decimal("30.070"),
            "C3H8": Decimal("44.097"),
            "C4H10": Decimal("58.123"),
            "N2": Decimal("28.014"),
            "CO2": Decimal("44.009"),
            "H2S": Decimal("34.082"),
        }

        # Convert mole fractions to Decimal
        mol_fracs = {
            "CH4": Decimal(str(gas_composition.methane)) / Decimal("100"),
            "C2H6": Decimal(str(gas_composition.ethane)) / Decimal("100"),
            "C3H8": Decimal(str(gas_composition.propane)) / Decimal("100"),
            "C4H10": Decimal(str(gas_composition.n_butane + gas_composition.i_butane)) / Decimal("100"),
            "C5+": Decimal(str(gas_composition.pentanes_plus)) / Decimal("100"),
            "N2": Decimal(str(gas_composition.nitrogen)) / Decimal("100"),
            "CO2": Decimal(str(gas_composition.carbon_dioxide)) / Decimal("100"),
            "H2S": Decimal(str(gas_composition.hydrogen_sulfide)) / Decimal("100"),
        }

        # Calculate average molecular weight
        avg_mw = (
            mol_fracs["CH4"] * mw_components["CH4"] +
            mol_fracs["C2H6"] * mw_components["C2H6"] +
            mol_fracs["C3H8"] * mw_components["C3H8"] +
            mol_fracs["C4H10"] * mw_components["C4H10"] +
            mol_fracs["C5+"] * Decimal("72.15") +  # Approximate C5
            mol_fracs["N2"] * mw_components["N2"] +
            mol_fracs["CO2"] * mw_components["CO2"] +
            mol_fracs["H2S"] * mw_components["H2S"]
        )

        steps.append(FuelAnalysisStep(
            step_number=1,
            description="Calculate average molecular weight",
            formula="MW_avg = sum(yi * MWi)",
            inputs={"mole_fractions": str(mol_fracs)},
            output_value=cls._apply_precision(avg_mw, precision),
            output_unit="g/mol"
        ))

        # Calculate mass fractions and elemental composition
        # C atoms per molecule: CH4=1, C2H6=2, C3H8=3, C4H10=4, C5+=5
        carbon_mass = (
            mol_fracs["CH4"] * Decimal("1") +
            mol_fracs["C2H6"] * Decimal("2") +
            mol_fracs["C3H8"] * Decimal("3") +
            mol_fracs["C4H10"] * Decimal("4") +
            mol_fracs["C5+"] * Decimal("5") +
            mol_fracs["CO2"] * Decimal("1")
        ) * MW["C"] / avg_mw * Decimal("100")

        # H atoms: CH4=4, C2H6=6, C3H8=8, C4H10=10, C5+=12, H2S=2
        hydrogen_mass = (
            mol_fracs["CH4"] * Decimal("4") +
            mol_fracs["C2H6"] * Decimal("6") +
            mol_fracs["C3H8"] * Decimal("8") +
            mol_fracs["C4H10"] * Decimal("10") +
            mol_fracs["C5+"] * Decimal("12") +
            mol_fracs["H2S"] * Decimal("2")
        ) * MW["H"] / avg_mw * Decimal("100")

        # Other elements
        oxygen_mass = mol_fracs["CO2"] * Decimal("2") * MW["O"] / avg_mw * Decimal("100")
        nitrogen_mass = mol_fracs["N2"] * Decimal("2") * MW["N"] / avg_mw * Decimal("100")
        sulfur_mass = mol_fracs["H2S"] * MW["S"] / avg_mw * Decimal("100")

        composition = {
            "C": cls._apply_precision(carbon_mass, precision),
            "H": cls._apply_precision(hydrogen_mass, precision),
            "O": cls._apply_precision(oxygen_mass, precision),
            "N": cls._apply_precision(nitrogen_mass, precision),
            "S": cls._apply_precision(sulfur_mass, precision),
            "Ash": Decimal("0"),
            "H2O": Decimal("0"),
        }

        steps.append(FuelAnalysisStep(
            step_number=2,
            description="Calculate elemental composition from gas analysis",
            formula="Mass_i = (atoms_i * MW_atom / MW_avg) * 100",
            inputs={"gas_composition": "molar"},
            output_value=carbon_mass,
            output_unit="wt% C"
        ))

        # Calculate HHV from composition
        hhv, hhv_steps = cls.calculate_hhv_dulong(
            carbon_mass, hydrogen_mass, oxygen_mass, sulfur_mass, precision
        )
        # Adjust step numbers
        for step in hhv_steps:
            steps.append(FuelAnalysisStep(
                step_number=step.step_number + 2,
                description=step.description,
                formula=step.formula,
                inputs=step.inputs,
                output_value=step.output_value,
                output_unit=step.output_unit
            ))

        # Calculate LHV
        lhv, lhv_step = cls.calculate_lhv(hhv, hydrogen_mass, Decimal("0"), precision)
        steps.append(FuelAnalysisStep(
            step_number=len(steps) + 1,
            description=lhv_step.description,
            formula=lhv_step.formula,
            inputs=lhv_step.inputs,
            output_value=lhv_step.output_value,
            output_unit=lhv_step.output_unit
        ))

        # Calculate stoichiometric air
        stoich_air, stoich_o2, air_steps = cls.calculate_stoich_air(
            carbon_mass, hydrogen_mass, oxygen_mass, sulfur_mass, precision
        )
        for step in air_steps:
            steps.append(FuelAnalysisStep(
                step_number=len(steps) + 1,
                description=step.description,
                formula=step.formula,
                inputs=step.inputs,
                output_value=step.output_value,
                output_unit=step.output_unit
            ))

        # Calculate flue gas volume
        fg_volume, fg_steps = cls.calculate_flue_gas_volume(
            carbon_mass, hydrogen_mass, oxygen_mass, nitrogen_mass,
            sulfur_mass, Decimal("0"), precision
        )
        for step in fg_steps:
            steps.append(FuelAnalysisStep(
                step_number=len(steps) + 1,
                description=step.description,
                formula=step.formula,
                inputs=step.inputs,
                output_value=step.output_value,
                output_unit=step.output_unit
            ))

        return FuelProperties(
            composition=composition,
            heating_value_hhv=cls._apply_precision(hhv, precision),
            heating_value_lhv=cls._apply_precision(lhv, precision),
            stoich_air_ratio=cls._apply_precision(stoich_air, precision),
            stoich_o2_ratio=cls._apply_precision(stoich_o2, precision),
            flue_gas_volume=cls._apply_precision(fg_volume, precision),
            calculation_steps=steps
        )

    @classmethod
    def get_default_composition(
        cls,
        fuel_type: str
    ) -> Optional[Dict[str, Decimal]]:
        """Get default composition for a fuel type."""
        return cls.DEFAULT_COMPOSITIONS.get(fuel_type.lower())

    @classmethod
    def get_default_hhv(
        cls,
        fuel_type: str
    ) -> Optional[Decimal]:
        """Get default HHV for a fuel type."""
        return cls.DEFAULT_HHV.get(fuel_type.lower())

    @staticmethod
    def _apply_precision(value: Decimal, precision: int) -> Decimal:
        """Apply decimal precision with ROUND_HALF_UP."""
        if precision < 0:
            raise ValueError(f"Precision must be non-negative")
        quantize_str = "0." + "0" * precision if precision > 0 else "1"
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


# Convenience functions
def analyze_coal(
    carbon: float,
    hydrogen: float,
    oxygen: float,
    nitrogen: float,
    sulfur: float,
    ash: float,
    moisture: float
) -> FuelProperties:
    """Convenience function to analyze coal from ultimate analysis."""
    analysis = UltimateAnalysis(
        carbon=carbon,
        hydrogen=hydrogen,
        oxygen=oxygen,
        nitrogen=nitrogen,
        sulfur=sulfur,
        ash=ash,
        moisture=moisture
    )
    return FuelAnalyzer.analyze_fuel(analysis)


def get_fuel_properties(fuel_type: str) -> Optional[FuelProperties]:
    """Get default properties for a standard fuel type."""
    composition = FuelAnalyzer.get_default_composition(fuel_type)
    if composition is None:
        return None

    analysis = UltimateAnalysis(
        carbon=float(composition["C"]),
        hydrogen=float(composition["H"]),
        oxygen=float(composition["O"]),
        nitrogen=float(composition["N"]),
        sulfur=float(composition["S"]),
        ash=float(composition["Ash"]),
        moisture=float(composition["H2O"])
    )

    hhv = FuelAnalyzer.get_default_hhv(fuel_type)
    return FuelAnalyzer.analyze_fuel(analysis, measured_hhv=float(hhv) if hhv else None)
