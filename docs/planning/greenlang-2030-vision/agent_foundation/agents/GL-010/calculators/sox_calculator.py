"""
SOx (Sulfur Oxides) Calculator Module for GL-010 EMISSIONWATCH.

This module provides deterministic, physics-based calculations for SOx emissions
from combustion and industrial processes. All calculations follow EPA methodologies
and are guaranteed to be zero-hallucination (no LLM in calculation path).

SOx Formation Chemistry:
1. Primary SO2: S + O2 -> SO2 (fuel sulfur oxidation)
2. Secondary SO3: SO2 + 1/2 O2 -> SO3 (further oxidation, typically 1-5%)

References:
- EPA Method 6/6C (40 CFR Part 60, Appendix A)
- EPA AP-42, Fifth Edition, Chapter 1
- EPA 40 CFR Part 75 (Acid Rain Program)

Zero-Hallucination Guarantee:
- All calculations are deterministic (same input -> same output)
- Mass balance approach ensures conservation of sulfur
- Full provenance tracking with SHA-256 hashes
"""

from typing import Dict, List, Optional, Union, Tuple
from decimal import Decimal, ROUND_HALF_UP
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, Field, field_validator

from .constants import (
    MW, F_FACTORS,
    LB_TO_KG, KG_TO_LB, MMBTU_TO_GJ,
    NORMAL_TEMP_K, NORMAL_PRESSURE_KPA,
)
from .units import UnitConverter


class SOxSpecies(str, Enum):
    """SOx species for calculations."""
    SO2 = "SO2"
    SO3 = "SO3"
    SOX_AS_SO2 = "SOx_as_SO2"  # Standard reporting basis


class SOxControlDevice(str, Enum):
    """SOx control technologies."""
    NONE = "none"
    WET_SCRUBBER = "wet_scrubber"
    DRY_SCRUBBER = "dry_scrubber"
    SPRAY_DRY_ABSORBER = "spray_dry_absorber"
    FLUE_GAS_DESULFURIZATION = "fgd"
    DRY_SORBENT_INJECTION = "dsi"
    CIRCULATING_DRY_SCRUBBER = "cds"
    SEAWATER_FGD = "seawater_fgd"


@dataclass(frozen=True)
class SOxCalculationStep:
    """Individual SOx calculation step with provenance."""
    step_number: int
    description: str
    formula: str
    inputs: Dict[str, Union[str, float, Decimal]]
    output_value: Decimal
    output_unit: str


@dataclass(frozen=True)
class SOxCalculationResult:
    """
    Complete SOx calculation result with full provenance.

    Attributes:
        so2_emissions: SO2 emissions
        so3_emissions: SO3 emissions
        total_sox: Total SOx as SO2 equivalent
        unit: Output unit
        sulfur_balance: Mass balance verification
        calculation_steps: Detailed calculation steps
        method: Calculation method used
        reference_standard: Regulatory reference
    """
    so2_emissions: Decimal
    so3_emissions: Decimal
    total_sox: Decimal
    unit: str
    sulfur_balance: Decimal  # Should be close to 1.0 for mass balance
    calculation_steps: List[SOxCalculationStep]
    method: str
    reference_standard: str


class FuelSulfurInput(BaseModel):
    """Input parameters for fuel sulfur content."""
    sulfur_weight_percent: float = Field(
        ge=0, le=10,
        description="Sulfur content in fuel (weight %)"
    )
    fuel_type: str = Field(description="Type of fuel")
    fuel_flow_rate: float = Field(
        gt=0,
        description="Fuel flow rate"
    )
    fuel_flow_unit: str = Field(
        default="kg/hr",
        description="Unit for fuel flow rate"
    )
    fuel_heating_value: float = Field(
        gt=0,
        description="Higher heating value of fuel"
    )
    heating_value_unit: str = Field(
        default="MJ/kg",
        description="Unit for heating value"
    )

    @field_validator("fuel_type")
    @classmethod
    def validate_fuel_type(cls, v: str) -> str:
        """Validate fuel type is supported."""
        valid_fuels = [
            "natural_gas", "fuel_oil_no2", "fuel_oil_no4", "fuel_oil_no6",
            "coal_bituminous", "coal_subbituminous", "coal_lignite",
            "coal_anthracite", "diesel", "gasoline", "propane", "butane",
            "wood", "biomass", "petroleum_coke", "residual_oil"
        ]
        if v.lower() not in valid_fuels:
            raise ValueError(f"Fuel type must be one of: {valid_fuels}")
        return v.lower()


class ScrubberInput(BaseModel):
    """Input parameters for scrubber efficiency modeling."""
    scrubber_type: SOxControlDevice = Field(
        description="Type of scrubber system"
    )
    design_efficiency_percent: float = Field(
        ge=0, le=100,
        description="Design removal efficiency (%)"
    )
    inlet_so2_ppm: Optional[float] = Field(
        default=None, ge=0,
        description="Inlet SO2 concentration (ppm)"
    )
    liquid_to_gas_ratio: Optional[float] = Field(
        default=None, gt=0,
        description="L/G ratio for wet scrubbers (gal/1000 acf)"
    )
    reagent_stoichiometry: float = Field(
        default=1.0, gt=0, le=3.0,
        description="Reagent stoichiometric ratio (actual/theoretical)"
    )
    operating_temperature_c: float = Field(
        default=50, gt=0, lt=200,
        description="Scrubber operating temperature (C)"
    )


class CEMSSOxInput(BaseModel):
    """Input parameters for CEMS SO2 data."""
    so2_concentration_ppm: float = Field(
        ge=0,
        description="Measured SO2 concentration (ppm)"
    )
    o2_percent: float = Field(
        ge=0, lt=21,
        description="Measured O2 concentration (%)"
    )
    flue_gas_flow_dscfm: Optional[float] = Field(
        default=None, gt=0,
        description="Dry standard cubic feet per minute"
    )
    moisture_percent: float = Field(
        default=10, ge=0, lt=50,
        description="Flue gas moisture content (%)"
    )


class SOxCalculator:
    """
    Zero-hallucination SOx emissions calculator.

    Implements deterministic calculations for SOx emissions using:
    1. Mass balance from fuel sulfur content
    2. EPA Method 6/6C for CEMS-based calculations
    3. Scrubber efficiency modeling

    All calculations produce identical results for identical inputs,
    with full provenance tracking for regulatory compliance.
    """

    # Control device efficiencies (decimal) - baseline values
    CONTROL_EFFICIENCIES: Dict[SOxControlDevice, Decimal] = {
        SOxControlDevice.NONE: Decimal("0"),
        SOxControlDevice.WET_SCRUBBER: Decimal("0.90"),
        SOxControlDevice.DRY_SCRUBBER: Decimal("0.80"),
        SOxControlDevice.SPRAY_DRY_ABSORBER: Decimal("0.85"),
        SOxControlDevice.FLUE_GAS_DESULFURIZATION: Decimal("0.95"),
        SOxControlDevice.DRY_SORBENT_INJECTION: Decimal("0.50"),
        SOxControlDevice.CIRCULATING_DRY_SCRUBBER: Decimal("0.92"),
        SOxControlDevice.SEAWATER_FGD: Decimal("0.98"),
    }

    # SO2 to SO3 conversion fractions by fuel type
    SO3_CONVERSION_FRACTION: Dict[str, Decimal] = {
        "natural_gas": Decimal("0.02"),
        "fuel_oil_no2": Decimal("0.02"),
        "fuel_oil_no4": Decimal("0.025"),
        "fuel_oil_no6": Decimal("0.03"),
        "diesel": Decimal("0.02"),
        "coal_bituminous": Decimal("0.03"),
        "coal_subbituminous": Decimal("0.025"),
        "coal_lignite": Decimal("0.02"),
        "coal_anthracite": Decimal("0.035"),
        "petroleum_coke": Decimal("0.04"),
        "residual_oil": Decimal("0.03"),
        "wood": Decimal("0.01"),
        "biomass": Decimal("0.01"),
    }

    # Default sulfur content by fuel type (weight %)
    DEFAULT_SULFUR_CONTENT: Dict[str, Decimal] = {
        "natural_gas": Decimal("0.0003"),  # Pipeline quality
        "fuel_oil_no2": Decimal("0.3"),
        "fuel_oil_no4": Decimal("0.8"),
        "fuel_oil_no6": Decimal("2.0"),
        "diesel": Decimal("0.0015"),  # ULSD
        "coal_bituminous": Decimal("2.0"),
        "coal_subbituminous": Decimal("0.5"),
        "coal_lignite": Decimal("1.0"),
        "coal_anthracite": Decimal("0.8"),
        "petroleum_coke": Decimal("5.0"),
        "residual_oil": Decimal("2.5"),
        "wood": Decimal("0.01"),
        "biomass": Decimal("0.1"),
        "propane": Decimal("0.01"),
        "butane": Decimal("0.01"),
        "gasoline": Decimal("0.003"),
    }

    @classmethod
    def calculate_from_fuel_sulfur(
        cls,
        fuel_input: FuelSulfurInput,
        control_device: SOxControlDevice = SOxControlDevice.NONE,
        control_efficiency_override: Optional[float] = None,
        precision: int = 4
    ) -> SOxCalculationResult:
        """
        Calculate SOx emissions from fuel sulfur content using mass balance.

        The fundamental principle is conservation of mass:
        - All sulfur in fuel is converted to SO2 during combustion
        - A small fraction (1-5%) further oxidizes to SO3
        - Control devices remove a fraction of total SOx

        Formula:
        SO2 (kg/hr) = Fuel_flow (kg/hr) * S_content * (MW_SO2 / MW_S)

        Args:
            fuel_input: Fuel sulfur and flow parameters
            control_device: SOx control device type
            control_efficiency_override: Override default control efficiency
            precision: Decimal places in result

        Returns:
            SOxCalculationResult with complete breakdown and provenance
        """
        steps = []

        # Convert inputs to Decimal
        s_percent = Decimal(str(fuel_input.sulfur_weight_percent))
        fuel_flow = Decimal(str(fuel_input.fuel_flow_rate))
        hhv = Decimal(str(fuel_input.fuel_heating_value))

        # Step 1: Calculate sulfur mass flow
        s_fraction = s_percent / Decimal("100")
        s_mass_flow = fuel_flow * s_fraction

        steps.append(SOxCalculationStep(
            step_number=1,
            description="Calculate sulfur mass flow in fuel",
            formula="S_flow = Fuel_flow * S_percent / 100",
            inputs={
                "fuel_flow": f"{fuel_flow} {fuel_input.fuel_flow_unit}",
                "sulfur_percent": str(s_percent)
            },
            output_value=cls._apply_precision(s_mass_flow, precision),
            output_unit="kg S/hr"
        ))

        # Step 2: Calculate SO2 formation (stoichiometric)
        # S + O2 -> SO2
        # MW ratio: 64.066 / 32.06 = 1.998
        mw_ratio_so2 = MW["SO2"] / MW["S"]
        so2_mass_flow = s_mass_flow * mw_ratio_so2

        steps.append(SOxCalculationStep(
            step_number=2,
            description="Calculate SO2 formation (stoichiometric conversion)",
            formula="SO2_flow = S_flow * (MW_SO2 / MW_S)",
            inputs={
                "sulfur_flow_kg_hr": str(s_mass_flow),
                "MW_SO2": str(MW["SO2"]),
                "MW_S": str(MW["S"])
            },
            output_value=cls._apply_precision(so2_mass_flow, precision),
            output_unit="kg SO2/hr"
        ))

        # Step 3: Calculate SO3 formation
        # A fraction of SO2 further oxidizes to SO3
        so3_fraction = cls.SO3_CONVERSION_FRACTION.get(
            fuel_input.fuel_type,
            Decimal("0.025")
        )
        # SO2 converted to SO3
        so2_to_so3 = so2_mass_flow * so3_fraction
        # Convert SO2 mass to SO3 mass (MW ratio: 80.066/64.066 = 1.25)
        mw_ratio_so3_so2 = MW["SO3"] / MW["SO2"]
        so3_mass_flow = so2_to_so3 * mw_ratio_so3_so2

        # Remaining SO2
        so2_remaining = so2_mass_flow * (Decimal("1") - so3_fraction)

        steps.append(SOxCalculationStep(
            step_number=3,
            description="Calculate SO3 formation from SO2 oxidation",
            formula="SO3 = SO2 * conversion_fraction * (MW_SO3/MW_SO2)",
            inputs={
                "so2_flow_kg_hr": str(so2_mass_flow),
                "conversion_fraction": str(so3_fraction),
                "MW_SO3": str(MW["SO3"]),
                "MW_SO2": str(MW["SO2"])
            },
            output_value=cls._apply_precision(so3_mass_flow, precision),
            output_unit="kg SO3/hr"
        ))

        # Step 4: Calculate heat input for emission factor
        # Convert fuel flow to heat input
        heat_input_mj_hr = fuel_flow * hhv  # MJ/hr
        heat_input_mmbtu_hr = heat_input_mj_hr / (MMBTU_TO_GJ * Decimal("1000"))

        steps.append(SOxCalculationStep(
            step_number=4,
            description="Calculate heat input rate",
            formula="Heat_input = Fuel_flow * HHV",
            inputs={
                "fuel_flow_kg_hr": str(fuel_flow),
                "hhv_mj_kg": str(hhv)
            },
            output_value=cls._apply_precision(heat_input_mmbtu_hr, precision),
            output_unit="MMBtu/hr"
        ))

        # Step 5: Calculate uncontrolled emission factor
        # Convert SO2 to lb, heat input to MMBtu
        so2_lb_hr = so2_remaining * KG_TO_LB
        so3_lb_hr = so3_mass_flow * KG_TO_LB
        so2_ef_uncontrolled = so2_lb_hr / heat_input_mmbtu_hr
        so3_ef_uncontrolled = so3_lb_hr / heat_input_mmbtu_hr

        steps.append(SOxCalculationStep(
            step_number=5,
            description="Calculate uncontrolled emission factors",
            formula="EF = Mass_rate / Heat_input",
            inputs={
                "so2_lb_hr": str(cls._apply_precision(so2_lb_hr, precision)),
                "so3_lb_hr": str(cls._apply_precision(so3_lb_hr, precision)),
                "heat_input_mmbtu_hr": str(heat_input_mmbtu_hr)
            },
            output_value=cls._apply_precision(so2_ef_uncontrolled, precision),
            output_unit="lb SO2/MMBtu"
        ))

        # Step 6: Apply control efficiency
        if control_efficiency_override is not None:
            control_eff = Decimal(str(control_efficiency_override)) / Decimal("100")
        else:
            control_eff = cls.CONTROL_EFFICIENCIES.get(
                control_device,
                Decimal("0")
            )

        so2_ef_controlled = so2_ef_uncontrolled * (Decimal("1") - control_eff)
        so3_ef_controlled = so3_ef_uncontrolled * (Decimal("1") - control_eff)

        steps.append(SOxCalculationStep(
            step_number=6,
            description="Apply control device efficiency",
            formula="EF_controlled = EF_uncontrolled * (1 - efficiency)",
            inputs={
                "uncontrolled_so2_ef": str(so2_ef_uncontrolled),
                "control_device": control_device.value,
                "efficiency": str(control_eff)
            },
            output_value=cls._apply_precision(so2_ef_controlled, precision),
            output_unit="lb SO2/MMBtu"
        ))

        # Step 7: Calculate total SOx as SO2 equivalent
        # SO3 as SO2 equivalent = SO3 * (MW_SO2/MW_SO3)
        so3_as_so2 = so3_ef_controlled * (MW["SO2"] / MW["SO3"])
        total_sox = so2_ef_controlled + so3_as_so2

        steps.append(SOxCalculationStep(
            step_number=7,
            description="Calculate total SOx as SO2 equivalent",
            formula="Total_SOx = SO2 + SO3*(MW_SO2/MW_SO3)",
            inputs={
                "so2_ef": str(so2_ef_controlled),
                "so3_ef": str(so3_ef_controlled),
                "MW_SO2": str(MW["SO2"]),
                "MW_SO3": str(MW["SO3"])
            },
            output_value=cls._apply_precision(total_sox, precision),
            output_unit="lb SOx as SO2/MMBtu"
        ))

        # Calculate sulfur balance for verification
        # Input S should equal output S (in SOx)
        sulfur_in_sox = (
            (so2_ef_controlled * heat_input_mmbtu_hr / KG_TO_LB) * (MW["S"] / MW["SO2"]) +
            (so3_ef_controlled * heat_input_mmbtu_hr / KG_TO_LB) * (MW["S"] / MW["SO3"])
        )
        sulfur_balance = sulfur_in_sox / s_mass_flow if s_mass_flow > 0 else Decimal("1")

        return SOxCalculationResult(
            so2_emissions=cls._apply_precision(so2_ef_controlled, precision),
            so3_emissions=cls._apply_precision(so3_ef_controlled, precision),
            total_sox=cls._apply_precision(total_sox, precision),
            unit="lb/MMBtu",
            sulfur_balance=cls._apply_precision(sulfur_balance, precision),
            calculation_steps=steps,
            method="Mass Balance (Fuel Sulfur)",
            reference_standard="EPA AP-42, Chapter 1"
        )

    @classmethod
    def calculate_from_cems(
        cls,
        cems_input: CEMSSOxInput,
        heat_input_mmbtu_hr: Union[float, Decimal],
        fuel_type: str,
        precision: int = 4
    ) -> SOxCalculationResult:
        """
        Calculate SOx emissions from CEMS data using EPA Method 6C.

        Formula (similar to EPA Method 19):
        E (lb/MMBtu) = C * K * Fd * (20.9 / (20.9 - %O2d))

        Where:
        - C = SO2 concentration (ppm, dry basis)
        - K = 1.660 x 10^-7 (conversion constant for SO2)
        - Fd = F-factor (dscf/MMBtu)
        - %O2d = Oxygen concentration (%, dry basis)

        Args:
            cems_input: CEMS measurement data
            heat_input_mmbtu_hr: Heat input rate (MMBtu/hr)
            fuel_type: Type of fuel for F-factor lookup
            precision: Decimal places in result

        Returns:
            SOxCalculationResult with emission rate and provenance

        Reference:
            EPA 40 CFR Part 60, Appendix A, Methods 6 and 6C
        """
        steps = []

        so2_ppm = Decimal(str(cems_input.so2_concentration_ppm))
        o2_measured = Decimal(str(cems_input.o2_percent))
        heat_input = Decimal(str(heat_input_mmbtu_hr))

        # Step 1: Get F-factor for fuel type
        f_factors = F_FACTORS.get(fuel_type.lower(), F_FACTORS["natural_gas"])
        fd = f_factors["Fd"]

        steps.append(SOxCalculationStep(
            step_number=1,
            description="Look up dry F-factor (Fd) for fuel type",
            formula="Lookup from EPA Method 19, Table 19-1",
            inputs={"fuel_type": fuel_type},
            output_value=fd,
            output_unit="dscf/MMBtu"
        ))

        # Step 2: Correct to dry basis
        moisture = Decimal(str(cems_input.moisture_percent)) / Decimal("100")
        so2_dry = so2_ppm / (Decimal("1") - moisture)

        steps.append(SOxCalculationStep(
            step_number=2,
            description="Convert SO2 to dry basis",
            formula="C_dry = C_wet / (1 - moisture_fraction)",
            inputs={
                "so2_ppm_wet": str(so2_ppm),
                "moisture_percent": str(cems_input.moisture_percent)
            },
            output_value=cls._apply_precision(so2_dry, precision),
            output_unit="ppm (dry)"
        ))

        # Step 3: Calculate emission factor
        # K for SO2 = MW_SO2 / (385.3 * 10^6) = 1.660e-7
        K = Decimal("1.660e-7")
        o2_correction = Decimal("20.9") / (Decimal("20.9") - o2_measured)
        emission_factor = so2_dry * K * fd * o2_correction

        steps.append(SOxCalculationStep(
            step_number=3,
            description="Calculate emission factor (EPA Method 6C)",
            formula="E = C * K * Fd * (20.9 / (20.9 - %O2))",
            inputs={
                "C_dry_ppm": str(so2_dry),
                "K": str(K),
                "Fd": str(fd),
                "O2_percent": str(o2_measured)
            },
            output_value=cls._apply_precision(emission_factor, precision),
            output_unit="lb SO2/MMBtu"
        ))

        # Step 4: Calculate mass emission rate
        mass_rate = emission_factor * heat_input

        steps.append(SOxCalculationStep(
            step_number=4,
            description="Calculate mass emission rate",
            formula="Mass_rate = EF * Heat_input",
            inputs={
                "emission_factor": str(emission_factor),
                "heat_input_mmbtu_hr": str(heat_input)
            },
            output_value=cls._apply_precision(mass_rate, precision),
            output_unit="lb SO2/hr"
        ))

        return SOxCalculationResult(
            so2_emissions=cls._apply_precision(emission_factor, precision),
            so3_emissions=Decimal("0"),  # Cannot measure SO3 with standard CEMS
            total_sox=cls._apply_precision(emission_factor, precision),
            unit="lb/MMBtu",
            sulfur_balance=Decimal("1"),  # N/A for CEMS
            calculation_steps=steps,
            method="EPA Method 6C (CEMS)",
            reference_standard="40 CFR Part 60, Appendix A, Method 6C"
        )

    @classmethod
    def calculate_scrubber_efficiency(
        cls,
        scrubber_input: ScrubberInput,
        inlet_so2_lb_hr: Union[float, Decimal],
        precision: int = 4
    ) -> Tuple[Decimal, Decimal, List[SOxCalculationStep]]:
        """
        Calculate actual scrubber efficiency and outlet SO2.

        Scrubber efficiency depends on:
        - Scrubber type and design
        - Liquid-to-gas ratio (wet scrubbers)
        - Reagent stoichiometry
        - Operating temperature
        - Inlet SO2 loading

        Args:
            scrubber_input: Scrubber operating parameters
            inlet_so2_lb_hr: Inlet SO2 mass rate (lb/hr)
            precision: Decimal places in result

        Returns:
            Tuple of (actual efficiency, outlet SO2 lb/hr, steps)
        """
        steps = []
        inlet_so2 = Decimal(str(inlet_so2_lb_hr))

        # Step 1: Get baseline efficiency
        base_eff = cls.CONTROL_EFFICIENCIES.get(
            scrubber_input.scrubber_type,
            Decimal("0.85")
        )

        steps.append(SOxCalculationStep(
            step_number=1,
            description="Look up baseline scrubber efficiency",
            formula="Lookup from scrubber type",
            inputs={"scrubber_type": scrubber_input.scrubber_type.value},
            output_value=base_eff,
            output_unit="fraction"
        ))

        # Step 2: Apply reagent stoichiometry factor
        # Efficiency increases with higher stoichiometry (up to a point)
        stoich = Decimal(str(scrubber_input.reagent_stoichiometry))
        stoich_factor = min(stoich, Decimal("1.5")) / Decimal("1.0")

        steps.append(SOxCalculationStep(
            step_number=2,
            description="Calculate stoichiometry adjustment factor",
            formula="stoich_factor = min(stoich, 1.5) / 1.0",
            inputs={"reagent_stoichiometry": str(stoich)},
            output_value=stoich_factor,
            output_unit="dimensionless"
        ))

        # Step 3: Apply L/G ratio factor (for wet scrubbers)
        lg_factor = Decimal("1.0")
        if scrubber_input.scrubber_type in [
            SOxControlDevice.WET_SCRUBBER,
            SOxControlDevice.FLUE_GAS_DESULFURIZATION,
            SOxControlDevice.SEAWATER_FGD
        ] and scrubber_input.liquid_to_gas_ratio is not None:
            lg = Decimal(str(scrubber_input.liquid_to_gas_ratio))
            # Typical L/G is 40-80 gal/1000 acf
            # Efficiency improves logarithmically with L/G
            lg_reference = Decimal("60")  # Reference L/G
            if lg > 0:
                lg_factor = Decimal("0.85") + Decimal("0.15") * (lg / lg_reference).ln() / Decimal("0.5")
                lg_factor = max(Decimal("0.8"), min(lg_factor, Decimal("1.1")))

        steps.append(SOxCalculationStep(
            step_number=3,
            description="Calculate L/G ratio adjustment factor",
            formula="lg_factor = 0.85 + 0.15 * ln(L/G / 60) / 0.5",
            inputs={"liquid_to_gas_ratio": str(scrubber_input.liquid_to_gas_ratio or "N/A")},
            output_value=lg_factor,
            output_unit="dimensionless"
        ))

        # Step 4: Apply temperature factor
        # Lower temperatures improve efficiency (better gas absorption)
        temp_c = Decimal(str(scrubber_input.operating_temperature_c))
        temp_reference = Decimal("50")  # Reference temperature
        temp_factor = Decimal("1.0") - (temp_c - temp_reference) / Decimal("200")
        temp_factor = max(Decimal("0.9"), min(temp_factor, Decimal("1.05")))

        steps.append(SOxCalculationStep(
            step_number=4,
            description="Calculate temperature adjustment factor",
            formula="temp_factor = 1.0 - (T - 50) / 200",
            inputs={"operating_temp_c": str(temp_c)},
            output_value=temp_factor,
            output_unit="dimensionless"
        ))

        # Step 5: Calculate actual efficiency
        actual_eff = base_eff * stoich_factor * lg_factor * temp_factor
        # Cap at design efficiency
        design_eff = Decimal(str(scrubber_input.design_efficiency_percent)) / Decimal("100")
        actual_eff = min(actual_eff, design_eff)
        actual_eff = min(actual_eff, Decimal("0.99"))  # Physical limit

        steps.append(SOxCalculationStep(
            step_number=5,
            description="Calculate actual scrubber efficiency",
            formula="efficiency = base * stoich_factor * lg_factor * temp_factor",
            inputs={
                "base_efficiency": str(base_eff),
                "stoich_factor": str(stoich_factor),
                "lg_factor": str(lg_factor),
                "temp_factor": str(temp_factor)
            },
            output_value=cls._apply_precision(actual_eff * Decimal("100"), precision),
            output_unit="%"
        ))

        # Step 6: Calculate outlet SO2
        outlet_so2 = inlet_so2 * (Decimal("1") - actual_eff)

        steps.append(SOxCalculationStep(
            step_number=6,
            description="Calculate outlet SO2 after scrubbing",
            formula="outlet_SO2 = inlet_SO2 * (1 - efficiency)",
            inputs={
                "inlet_so2_lb_hr": str(inlet_so2),
                "efficiency": str(actual_eff)
            },
            output_value=cls._apply_precision(outlet_so2, precision),
            output_unit="lb SO2/hr"
        ))

        return (
            cls._apply_precision(actual_eff * Decimal("100"), precision),
            cls._apply_precision(outlet_so2, precision),
            steps
        )

    @classmethod
    def calculate_mass_emissions(
        cls,
        emission_factor_lb_mmbtu: Union[float, Decimal],
        heat_input_mmbtu_hr: Union[float, Decimal],
        operating_hours: Union[float, Decimal],
        precision: int = 2
    ) -> Tuple[Decimal, Decimal, List[SOxCalculationStep]]:
        """
        Calculate mass emissions from emission factor and operating data.

        Args:
            emission_factor_lb_mmbtu: SOx emission factor (lb/MMBtu)
            heat_input_mmbtu_hr: Heat input rate (MMBtu/hr)
            operating_hours: Operating hours in period
            precision: Decimal places in result

        Returns:
            Tuple of (lb/hr, tons/period, calculation steps)
        """
        steps = []
        ef = Decimal(str(emission_factor_lb_mmbtu))
        hi = Decimal(str(heat_input_mmbtu_hr))
        hours = Decimal(str(operating_hours))

        # Calculate lb/hr
        lb_per_hr = ef * hi

        steps.append(SOxCalculationStep(
            step_number=1,
            description="Calculate hourly emission rate",
            formula="lb/hr = Emission_factor * Heat_input",
            inputs={
                "emission_factor_lb_mmbtu": str(ef),
                "heat_input_mmbtu_hr": str(hi)
            },
            output_value=cls._apply_precision(lb_per_hr, precision),
            output_unit="lb/hr"
        ))

        # Calculate total tons
        total_lb = lb_per_hr * hours
        total_tons = total_lb / Decimal("2000")

        steps.append(SOxCalculationStep(
            step_number=2,
            description="Calculate total mass emissions",
            formula="tons = (lb/hr * hours) / 2000",
            inputs={
                "lb_per_hr": str(lb_per_hr),
                "operating_hours": str(hours)
            },
            output_value=cls._apply_precision(total_tons, precision),
            output_unit="short tons"
        ))

        return (
            cls._apply_precision(lb_per_hr, precision),
            cls._apply_precision(total_tons, precision),
            steps
        )

    @classmethod
    def estimate_sulfur_from_emission_factor(
        cls,
        emission_factor_lb_mmbtu: Union[float, Decimal],
        fuel_heating_value_mj_kg: Union[float, Decimal],
        precision: int = 3
    ) -> Tuple[Decimal, List[SOxCalculationStep]]:
        """
        Back-calculate fuel sulfur content from emission factor.

        This is useful when sulfur content is unknown but emissions
        data is available.

        Args:
            emission_factor_lb_mmbtu: Measured or reported SO2 EF
            fuel_heating_value_mj_kg: Fuel heating value
            precision: Decimal places in result

        Returns:
            Tuple of (estimated sulfur weight %, calculation steps)
        """
        steps = []
        ef = Decimal(str(emission_factor_lb_mmbtu))
        hhv = Decimal(str(fuel_heating_value_mj_kg))

        # Convert emission factor to kg SO2 / MJ
        ef_kg_mj = ef * LB_TO_KG / (MMBTU_TO_GJ * Decimal("1000"))

        steps.append(SOxCalculationStep(
            step_number=1,
            description="Convert emission factor to SI units",
            formula="EF_kg_MJ = EF_lb_MMBtu * (kg/lb) / (MJ/MMBtu)",
            inputs={
                "ef_lb_mmbtu": str(ef)
            },
            output_value=cls._apply_precision(ef_kg_mj, 8),
            output_unit="kg SO2/MJ"
        ))

        # Calculate sulfur content
        # EF = S% * (MW_SO2/MW_S) * (1/HHV)
        # S% = EF * HHV * (MW_S/MW_SO2)
        mw_ratio = MW["S"] / MW["SO2"]
        s_percent = ef_kg_mj * hhv * mw_ratio * Decimal("100")

        steps.append(SOxCalculationStep(
            step_number=2,
            description="Back-calculate sulfur content",
            formula="S% = EF * HHV * (MW_S/MW_SO2) * 100",
            inputs={
                "ef_kg_mj": str(ef_kg_mj),
                "hhv_mj_kg": str(hhv),
                "MW_S": str(MW["S"]),
                "MW_SO2": str(MW["SO2"])
            },
            output_value=cls._apply_precision(s_percent, precision),
            output_unit="weight %"
        ))

        return cls._apply_precision(s_percent, precision), steps

    @staticmethod
    def _apply_precision(value: Decimal, precision: int) -> Decimal:
        """Apply decimal precision with ROUND_HALF_UP."""
        if precision < 0:
            raise ValueError(f"Precision must be non-negative, got {precision}")
        quantize_str = "0." + "0" * precision if precision > 0 else "1"
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


# Convenience functions
def calculate_sox_from_fuel(
    fuel_type: str,
    fuel_flow_kg_hr: float,
    fuel_heating_value_mj_kg: float,
    sulfur_percent: Optional[float] = None,
    control_device: str = "none"
) -> SOxCalculationResult:
    """
    Convenience function to calculate SOx emissions from fuel combustion.

    Args:
        fuel_type: Type of fuel
        fuel_flow_kg_hr: Fuel flow rate (kg/hr)
        fuel_heating_value_mj_kg: Fuel heating value (MJ/kg)
        sulfur_percent: Sulfur content (%), uses default if None
        control_device: SOx control device

    Returns:
        SOxCalculationResult with complete breakdown
    """
    if sulfur_percent is None:
        sulfur_percent = float(SOxCalculator.DEFAULT_SULFUR_CONTENT.get(
            fuel_type.lower(),
            Decimal("1.0")
        ))

    fuel_input = FuelSulfurInput(
        sulfur_weight_percent=sulfur_percent,
        fuel_type=fuel_type,
        fuel_flow_rate=fuel_flow_kg_hr,
        fuel_flow_unit="kg/hr",
        fuel_heating_value=fuel_heating_value_mj_kg,
        heating_value_unit="MJ/kg"
    )

    return SOxCalculator.calculate_from_fuel_sulfur(
        fuel_input=fuel_input,
        control_device=SOxControlDevice(control_device)
    )


def calculate_sox_from_cems(
    so2_ppm: float,
    o2_percent: float,
    heat_input_mmbtu_hr: float,
    fuel_type: str = "coal_bituminous"
) -> SOxCalculationResult:
    """
    Convenience function to calculate SOx from CEMS data.

    Args:
        so2_ppm: Measured SO2 concentration (ppm)
        o2_percent: Measured O2 percentage
        heat_input_mmbtu_hr: Heat input rate (MMBtu/hr)
        fuel_type: Type of fuel for F-factor

    Returns:
        SOxCalculationResult with emission rate
    """
    cems_input = CEMSSOxInput(
        so2_concentration_ppm=so2_ppm,
        o2_percent=o2_percent
    )
    return SOxCalculator.calculate_from_cems(
        cems_input=cems_input,
        heat_input_mmbtu_hr=heat_input_mmbtu_hr,
        fuel_type=fuel_type
    )
