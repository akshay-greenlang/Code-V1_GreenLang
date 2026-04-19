# -*- coding: utf-8 -*-
"""
GL-MRV-WST-002: Incineration MRV Agent
=======================================

Calculates emissions from waste incineration and waste-to-energy facilities.
Supports CO2, CH4, and N2O emissions from various waste types.

Key Features:
- IPCC 2006/2019 incineration methodology
- Waste composition-based fossil/biogenic carbon split
- Energy recovery credit calculations
- Support for mass burn, RDF, and fluidized bed technologies
- EU ETS compliant reporting

Zero-Hallucination Guarantees:
- All calculations use IPCC deterministic formulas
- Emission factors from published regulatory sources
- SHA-256 provenance hash for complete audit trail

Reference Standards:
- IPCC 2006 Guidelines Volume 5, Chapter 5 (Incineration)
- EU ETS Monitoring and Reporting Regulation
- EPA AP-42 Emission Factors

Author: GreenLang Framework Team
Version: 1.0.0
"""

from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional
import logging

from pydantic import BaseModel, Field, field_validator

from greenlang.agents.mrv.waste.base import (
    BaseWasteMRVAgent,
    WasteMRVInput,
    WasteMRVOutput,
    WasteType,
    TreatmentMethod,
    EmissionScope,
    DataQualityTier,
    CalculationMethod,
    EmissionFactor,
    CalculationStep,
    GWP_AR6_100,
)

logger = logging.getLogger(__name__)


# =============================================================================
# INCINERATION CONSTANTS
# =============================================================================

class IncineratorType(str):
    """Types of waste incinerators."""
    MASS_BURN = "mass_burn"
    MODULAR = "modular"
    RDF = "rdf"  # Refuse Derived Fuel
    FLUIDIZED_BED = "fluidized_bed"
    ROTARY_KILN = "rotary_kiln"
    GRATE = "grate"


# Fossil carbon content by waste type (fraction of total carbon)
FOSSIL_CARBON_FRACTION: Dict[str, Decimal] = {
    WasteType.PLASTIC.value: Decimal("1.0"),  # 100% fossil
    WasteType.RUBBER.value: Decimal("0.8"),
    WasteType.TEXTILES.value: Decimal("0.4"),  # Mix of synthetic and natural
    WasteType.MUNICIPAL_SOLID_WASTE.value: Decimal("0.33"),  # Mixed waste
    WasteType.FOOD_WASTE.value: Decimal("0"),  # 100% biogenic
    WasteType.YARD_WASTE.value: Decimal("0"),
    WasteType.PAPER.value: Decimal("0"),
    WasteType.CARDBOARD.value: Decimal("0"),
    WasteType.WOOD.value: Decimal("0"),
    WasteType.HAZARDOUS.value: Decimal("0.5"),
    WasteType.MEDICAL.value: Decimal("0.4"),
    WasteType.E_WASTE.value: Decimal("0.6"),
}

# Total carbon content by waste type (fraction of dry weight)
TOTAL_CARBON_CONTENT: Dict[str, Decimal] = {
    WasteType.PLASTIC.value: Decimal("0.75"),
    WasteType.RUBBER.value: Decimal("0.67"),
    WasteType.TEXTILES.value: Decimal("0.50"),
    WasteType.MUNICIPAL_SOLID_WASTE.value: Decimal("0.28"),
    WasteType.FOOD_WASTE.value: Decimal("0.38"),
    WasteType.YARD_WASTE.value: Decimal("0.46"),
    WasteType.PAPER.value: Decimal("0.46"),
    WasteType.CARDBOARD.value: Decimal("0.44"),
    WasteType.WOOD.value: Decimal("0.50"),
    WasteType.HAZARDOUS.value: Decimal("0.30"),
    WasteType.MEDICAL.value: Decimal("0.35"),
    WasteType.E_WASTE.value: Decimal("0.20"),
}

# Combustion efficiency by incinerator type
COMBUSTION_EFFICIENCY: Dict[str, Decimal] = {
    "mass_burn": Decimal("0.995"),
    "modular": Decimal("0.990"),
    "rdf": Decimal("0.998"),
    "fluidized_bed": Decimal("0.998"),
    "rotary_kiln": Decimal("0.995"),
    "grate": Decimal("0.995"),
    "default": Decimal("0.995"),
}

# N2O emission factors (kg N2O/tonne waste)
N2O_FACTORS: Dict[str, Decimal] = {
    "continuous": Decimal("0.05"),  # Continuous operation
    "semi_continuous": Decimal("0.06"),
    "batch": Decimal("0.10"),
    "default": Decimal("0.06"),
}

# CH4 emission factors (kg CH4/tonne waste)
CH4_FACTORS: Dict[str, Decimal] = {
    "continuous": Decimal("0.0"),  # Near zero for proper combustion
    "semi_continuous": Decimal("0.5"),
    "batch": Decimal("6.5"),
    "default": Decimal("0.2"),
}


# =============================================================================
# INPUT/OUTPUT MODELS
# =============================================================================

class WasteComponent(BaseModel):
    """Individual waste component for incineration."""
    waste_type: WasteType = Field(..., description="Type of waste")
    mass_tonnes: Decimal = Field(..., gt=0, description="Mass in metric tonnes")
    moisture_content: Decimal = Field(
        Decimal("0.2"), ge=0, le=1, description="Moisture content as fraction"
    )
    lower_heating_value_mj_kg: Optional[Decimal] = Field(
        None, gt=0, description="Lower heating value (MJ/kg)"
    )

    class Config:
        use_enum_values = True


class IncinerationInput(WasteMRVInput):
    """Input model for Incineration MRV Agent."""

    # Facility characteristics
    incinerator_type: str = Field(
        "mass_burn", description="Type of incinerator"
    )
    operation_mode: str = Field(
        "continuous", description="Operation mode (continuous/semi_continuous/batch)"
    )

    # Waste components
    waste_components: List[WasteComponent] = Field(
        default_factory=list, description="Waste components incinerated"
    )

    # Simplified input (if not using components)
    total_waste_tonnes: Optional[Decimal] = Field(
        None, gt=0, description="Total waste incinerated"
    )
    default_waste_type: WasteType = Field(
        WasteType.MUNICIPAL_SOLID_WASTE, description="Default waste type"
    )

    # Energy recovery
    has_energy_recovery: bool = Field(True, description="Whether facility has energy recovery")
    electricity_generated_mwh: Optional[Decimal] = Field(
        None, ge=0, description="Electricity generated"
    )
    heat_recovered_gj: Optional[Decimal] = Field(
        None, ge=0, description="Heat recovered"
    )
    grid_emission_factor_kg_co2_per_kwh: Decimal = Field(
        Decimal("0.4"), ge=0, description="Grid emission factor for avoided emissions"
    )

    # Combustion parameters
    combustion_efficiency: Optional[Decimal] = Field(
        None, ge=0.9, le=1, description="Combustion efficiency override"
    )
    oxidation_factor: Decimal = Field(
        Decimal("1.0"), ge=0.9, le=1, description="Carbon oxidation factor"
    )

    @field_validator("incinerator_type")
    @classmethod
    def validate_incinerator_type(cls, v: str) -> str:
        """Validate incinerator type."""
        valid_types = ["mass_burn", "modular", "rdf", "fluidized_bed", "rotary_kiln", "grate"]
        if v.lower() not in valid_types:
            return "mass_burn"
        return v.lower()


class IncinerationOutput(WasteMRVOutput):
    """Output model for Incineration MRV Agent."""

    # Incineration-specific outputs
    total_waste_incinerated_tonnes: Decimal = Field(
        Decimal("0"), description="Total waste incinerated"
    )

    # Carbon breakdown
    total_carbon_kg: Decimal = Field(
        Decimal("0"), description="Total carbon in waste"
    )
    fossil_carbon_kg: Decimal = Field(
        Decimal("0"), description="Fossil carbon content"
    )
    biogenic_carbon_kg: Decimal = Field(
        Decimal("0"), description="Biogenic carbon content"
    )

    # CO2 breakdown
    co2_fossil_kg: Decimal = Field(
        Decimal("0"), description="CO2 from fossil sources"
    )
    co2_biogenic_kg: Decimal = Field(
        Decimal("0"), description="CO2 from biogenic sources (carbon neutral)"
    )

    # Energy
    electricity_generated_mwh: Decimal = Field(
        Decimal("0"), description="Electricity generated"
    )
    heat_recovered_gj: Decimal = Field(
        Decimal("0"), description="Heat recovered"
    )
    avoided_emissions_kg_co2e: Decimal = Field(
        Decimal("0"), description="Avoided emissions from energy recovery"
    )

    # Net emissions
    net_emissions_kg_co2e: Decimal = Field(
        Decimal("0"), description="Net emissions (gross - avoided)"
    )


# =============================================================================
# INCINERATION MRV AGENT
# =============================================================================

class IncinerationMRVAgent(BaseWasteMRVAgent[IncinerationInput, IncinerationOutput]):
    """
    GL-MRV-WST-002: Waste Incineration MRV Agent

    Calculates emissions from waste incineration and waste-to-energy facilities.

    Calculation Approach:
    1. Determine waste composition and total carbon content
    2. Split carbon into fossil and biogenic fractions
    3. Calculate CO2 from carbon oxidation
    4. Add N2O and CH4 emissions based on combustion conditions
    5. Calculate energy recovery credits (avoided emissions)
    6. Report net emissions (gross - avoided)

    Key Formula (IPCC):
        CO2 = Sum(SW * CF * FCF * OF * 44/12)

    Where:
        SW = Solid waste incinerated (tonnes)
        CF = Carbon fraction
        FCF = Fossil carbon fraction
        OF = Oxidation factor
        44/12 = Molecular weight ratio CO2/C

    Example:
        >>> agent = IncinerationMRVAgent()
        >>> input_data = IncinerationInput(
        ...     organization_id="ORG001",
        ...     reporting_year=2024,
        ...     total_waste_tonnes=Decimal("50000"),
        ...     has_energy_recovery=True,
        ...     electricity_generated_mwh=Decimal("15000"),
        ... )
        >>> result = agent.calculate(input_data)
    """

    AGENT_ID = "GL-MRV-WST-002"
    AGENT_NAME = "Incineration MRV Agent"
    AGENT_VERSION = "1.0.0"
    TREATMENT_METHOD = TreatmentMethod.INCINERATION
    DEFAULT_SCOPE = EmissionScope.SCOPE_1

    def __init__(self):
        """Initialize Incineration MRV Agent."""
        super().__init__()
        self._fossil_carbon = FOSSIL_CARBON_FRACTION
        self._total_carbon = TOTAL_CARBON_CONTENT
        self._combustion_eff = COMBUSTION_EFFICIENCY
        self._n2o_factors = N2O_FACTORS
        self._ch4_factors = CH4_FACTORS
        self.logger.info(f"Initialized {self.AGENT_ID}: {self.AGENT_NAME}")

    def calculate(self, input_data: IncinerationInput) -> IncinerationOutput:
        """
        Calculate incineration emissions.

        Args:
            input_data: Incineration input data

        Returns:
            IncinerationOutput with emissions and audit trail
        """
        start_time = datetime.now(timezone.utc)
        steps: List[CalculationStep] = []
        emission_factors: List[EmissionFactor] = []
        warnings: List[str] = []

        # Step 1: Initialize
        steps.append(CalculationStep(
            step_number=1,
            description="Initialize incineration emissions calculation",
            formula="N/A",
            inputs={
                "incinerator_type": input_data.incinerator_type,
                "operation_mode": input_data.operation_mode,
                "has_energy_recovery": input_data.has_energy_recovery,
            },
            output="Initialization complete",
        ))

        # Step 2: Collect waste data
        components = self._collect_waste_data(input_data)
        total_waste = sum(c.mass_tonnes for c in components)

        steps.append(CalculationStep(
            step_number=2,
            description="Aggregate waste components",
            formula="total_waste = sum(component_mass)",
            inputs={"num_components": len(components)},
            output=f"{total_waste} tonnes",
        ))

        # Step 3: Calculate carbon content
        total_carbon_kg = Decimal("0")
        fossil_carbon_kg = Decimal("0")
        biogenic_carbon_kg = Decimal("0")

        for component in components:
            # Get dry mass
            dry_mass = component.mass_tonnes * (Decimal("1") - component.moisture_content)
            dry_mass_kg = dry_mass * Decimal("1000")

            # Get carbon content
            carbon_fraction = self._total_carbon.get(
                component.waste_type.value,
                self._total_carbon[WasteType.MUNICIPAL_SOLID_WASTE.value]
            )
            carbon_kg = dry_mass_kg * carbon_fraction

            # Split fossil/biogenic
            fossil_fraction = self._fossil_carbon.get(
                component.waste_type.value,
                self._fossil_carbon[WasteType.MUNICIPAL_SOLID_WASTE.value]
            )
            fossil_kg = carbon_kg * fossil_fraction
            biogenic_kg = carbon_kg * (Decimal("1") - fossil_fraction)

            total_carbon_kg += carbon_kg
            fossil_carbon_kg += fossil_kg
            biogenic_carbon_kg += biogenic_kg

        total_carbon_kg = total_carbon_kg.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        fossil_carbon_kg = fossil_carbon_kg.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        biogenic_carbon_kg = biogenic_carbon_kg.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        steps.append(CalculationStep(
            step_number=3,
            description="Calculate carbon content (fossil vs biogenic)",
            formula="carbon_kg = dry_mass_kg * carbon_fraction",
            inputs={
                "total_waste_tonnes": str(total_waste),
                "avg_carbon_fraction": str(
                    (total_carbon_kg / (total_waste * Decimal("1000"))).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    ) if total_waste > 0 else Decimal("0")
                ),
            },
            output=f"Total: {total_carbon_kg} kg, Fossil: {fossil_carbon_kg} kg, Biogenic: {biogenic_carbon_kg} kg",
        ))

        # Step 4: Calculate CO2 emissions
        combustion_eff = input_data.combustion_efficiency or self._combustion_eff.get(
            input_data.incinerator_type, self._combustion_eff["default"]
        )
        ox_factor = input_data.oxidation_factor

        # CO2 from fossil carbon (44/12 = 3.667)
        co2_fossil_kg = (fossil_carbon_kg * combustion_eff * ox_factor * Decimal("3.667")).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        # CO2 from biogenic carbon (reported but carbon neutral)
        co2_biogenic_kg = (biogenic_carbon_kg * combustion_eff * ox_factor * Decimal("3.667")).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        steps.append(CalculationStep(
            step_number=4,
            description="Calculate CO2 emissions from carbon oxidation",
            formula="CO2 = carbon_kg * combustion_eff * ox_factor * (44/12)",
            inputs={
                "fossil_carbon_kg": str(fossil_carbon_kg),
                "combustion_efficiency": str(combustion_eff),
                "oxidation_factor": str(ox_factor),
            },
            output=f"CO2_fossil: {co2_fossil_kg} kg, CO2_biogenic: {co2_biogenic_kg} kg",
        ))

        # Record emission factor
        ef_incineration = EmissionFactor(
            factor_id=f"ipcc_incineration_{input_data.incinerator_type}",
            factor_value=Decimal("3.667"),  # CO2/C ratio
            factor_unit="kg CO2/kg C",
            source="IPCC",
            source_uri="https://www.ipcc-nggip.iges.or.jp/public/2019rf/vol5.html",
            version="2019",
            last_updated="2019-05-01",
            uncertainty_pct=5.0,
            data_quality_tier=DataQualityTier.TIER_2,
            geographic_scope="global",
            treatment_method=TreatmentMethod.INCINERATION,
        )
        emission_factors.append(ef_incineration)

        # Step 5: Calculate N2O emissions
        n2o_ef = self._n2o_factors.get(input_data.operation_mode, self._n2o_factors["default"])
        n2o_kg = (total_waste * n2o_ef).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        n2o_co2e_kg = (n2o_kg * GWP_AR6_100["N2O"]).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        steps.append(CalculationStep(
            step_number=5,
            description="Calculate N2O emissions",
            formula="N2O = waste_tonnes * N2O_EF",
            inputs={
                "waste_tonnes": str(total_waste),
                "n2o_ef_kg_per_tonne": str(n2o_ef),
                "gwp_n2o": str(GWP_AR6_100["N2O"]),
            },
            output=f"N2O: {n2o_kg} kg = {n2o_co2e_kg} kg CO2e",
        ))

        # Step 6: Calculate CH4 emissions
        ch4_ef = self._ch4_factors.get(input_data.operation_mode, self._ch4_factors["default"])
        ch4_kg = (total_waste * ch4_ef).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        ch4_co2e_kg = (ch4_kg * GWP_AR6_100["CH4"]).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        steps.append(CalculationStep(
            step_number=6,
            description="Calculate CH4 emissions",
            formula="CH4 = waste_tonnes * CH4_EF",
            inputs={
                "waste_tonnes": str(total_waste),
                "ch4_ef_kg_per_tonne": str(ch4_ef),
                "gwp_ch4": str(GWP_AR6_100["CH4"]),
            },
            output=f"CH4: {ch4_kg} kg = {ch4_co2e_kg} kg CO2e",
        ))

        # Step 7: Calculate total gross emissions
        total_gross_kg = co2_fossil_kg + n2o_co2e_kg + ch4_co2e_kg

        steps.append(CalculationStep(
            step_number=7,
            description="Calculate total gross emissions",
            formula="gross_emissions = CO2_fossil + N2O_CO2e + CH4_CO2e",
            inputs={
                "co2_fossil_kg": str(co2_fossil_kg),
                "n2o_co2e_kg": str(n2o_co2e_kg),
                "ch4_co2e_kg": str(ch4_co2e_kg),
            },
            output=f"{total_gross_kg} kg CO2e",
        ))

        # Step 8: Calculate avoided emissions from energy recovery
        avoided_emissions_kg = Decimal("0")
        electricity_mwh = input_data.electricity_generated_mwh or Decimal("0")
        heat_gj = input_data.heat_recovered_gj or Decimal("0")

        if input_data.has_energy_recovery:
            # Electricity: MWh * 1000 kWh/MWh * grid EF
            electricity_avoided = (
                electricity_mwh * Decimal("1000") * input_data.grid_emission_factor_kg_co2_per_kwh
            )

            # Heat: assume 50 kg CO2/GJ replaced
            heat_avoided = heat_gj * Decimal("50")

            avoided_emissions_kg = (electricity_avoided + heat_avoided).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )

        steps.append(CalculationStep(
            step_number=8,
            description="Calculate avoided emissions from energy recovery",
            formula="avoided = electricity_mwh * 1000 * grid_ef + heat_gj * 50",
            inputs={
                "electricity_mwh": str(electricity_mwh),
                "heat_gj": str(heat_gj),
                "grid_ef": str(input_data.grid_emission_factor_kg_co2_per_kwh),
            },
            output=f"{avoided_emissions_kg} kg CO2e avoided",
        ))

        # Step 9: Calculate net emissions
        net_emissions_kg = (total_gross_kg - avoided_emissions_kg).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )
        if net_emissions_kg < Decimal("0"):
            net_emissions_kg = Decimal("0")
            warnings.append("Net emissions adjusted to zero (avoided > gross)")

        steps.append(CalculationStep(
            step_number=9,
            description="Calculate net emissions",
            formula="net_emissions = gross_emissions - avoided_emissions",
            inputs={
                "gross_emissions_kg": str(total_gross_kg),
                "avoided_emissions_kg": str(avoided_emissions_kg),
            },
            output=f"{net_emissions_kg} kg CO2e net",
        ))

        # Create activity summary
        activity_summary = {
            "organization_id": input_data.organization_id,
            "facility_id": input_data.facility_id,
            "reporting_year": input_data.reporting_year,
            "incinerator_type": input_data.incinerator_type,
            "operation_mode": input_data.operation_mode,
            "total_waste_tonnes": str(total_waste),
            "has_energy_recovery": input_data.has_energy_recovery,
        }

        # Build output
        output = IncinerationOutput(
            agent_id=self.AGENT_ID,
            agent_version=self.AGENT_VERSION,
            total_emissions_kg_co2e=total_gross_kg,
            total_emissions_mt_co2e=self._kg_to_metric_tons(total_gross_kg),
            co2_kg=co2_fossil_kg,
            ch4_kg=ch4_kg,
            n2o_kg=n2o_kg,
            ch4_generated_kg=ch4_kg,
            ch4_captured_kg=Decimal("0"),
            ch4_flared_kg=Decimal("0"),
            ch4_utilized_kg=Decimal("0"),
            ch4_emitted_kg=ch4_kg,
            scope=EmissionScope.SCOPE_1,
            calculation_steps=steps,
            provenance_hash="",
            data_quality_tier=DataQualityTier.TIER_2,
            calculation_timestamp=datetime.now(timezone.utc),
            calculation_duration_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
            emission_factors_used=emission_factors,
            activity_data_summary=activity_summary,
            warnings=warnings,
            # Incineration-specific fields
            total_waste_incinerated_tonnes=total_waste,
            total_carbon_kg=total_carbon_kg,
            fossil_carbon_kg=fossil_carbon_kg,
            biogenic_carbon_kg=biogenic_carbon_kg,
            co2_fossil_kg=co2_fossil_kg,
            co2_biogenic_kg=co2_biogenic_kg,
            electricity_generated_mwh=electricity_mwh,
            heat_recovered_gj=heat_gj,
            avoided_emissions_kg_co2e=avoided_emissions_kg,
            net_emissions_kg_co2e=net_emissions_kg,
        )

        output.provenance_hash = self._generate_provenance_hash(
            input_data=activity_summary,
            output_data={
                "total_emissions_kg_co2e": total_gross_kg,
                "net_emissions_kg_co2e": net_emissions_kg,
            },
            steps=steps,
        )

        return output

    def _collect_waste_data(self, input_data: IncinerationInput) -> List[WasteComponent]:
        """Collect and normalize waste component data."""
        components = list(input_data.waste_components)

        if not components and input_data.total_waste_tonnes:
            components.append(WasteComponent(
                waste_type=input_data.default_waste_type,
                mass_tonnes=input_data.total_waste_tonnes,
            ))

        return components
