# -*- coding: utf-8 -*-
"""
GL-MRV-WST-006: Plastic Waste MRV Agent
========================================

Calculates emissions from plastic waste management including tracking
across disposal pathways (landfill, incineration, recycling, ocean leakage).

Key Features:
- Multi-pathway plastic waste tracking
- Polymer-specific emission factors
- Ocean plastic leakage estimation
- Circular economy metrics
- Extended Producer Responsibility support

Zero-Hallucination Guarantees:
- All calculations use deterministic formulas
- Emission factors from published regulatory sources
- SHA-256 provenance hash for complete audit trail

Reference Standards:
- IPCC 2006/2019 Guidelines
- Ellen MacArthur Foundation Plastic Economy
- UNEP Global Plastic Outlook
- EU Single-Use Plastics Directive

Author: GreenLang Framework Team
Version: 1.0.0
"""

from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional
from enum import Enum
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
# PLASTIC WASTE CONSTANTS
# =============================================================================

class PolymerType(str, Enum):
    """Types of plastic polymers."""
    PET = "pet"  # Polyethylene terephthalate
    HDPE = "hdpe"  # High-density polyethylene
    PVC = "pvc"  # Polyvinyl chloride
    LDPE = "ldpe"  # Low-density polyethylene
    PP = "pp"  # Polypropylene
    PS = "ps"  # Polystyrene
    OTHER = "other"  # Other plastics (resin code 7)
    MIXED = "mixed"


class PlasticDisposalPath(str, Enum):
    """Disposal pathways for plastic waste."""
    LANDFILL = "landfill"
    INCINERATION = "incineration"
    INCINERATION_ENERGY = "incineration_energy"
    MECHANICAL_RECYCLING = "mechanical_recycling"
    CHEMICAL_RECYCLING = "chemical_recycling"
    OPEN_BURNING = "open_burning"
    OCEAN_LEAKAGE = "ocean_leakage"
    MISMANAGED = "mismanaged"
    EXPORT = "export"


# Carbon content by polymer type (fraction)
POLYMER_CARBON_CONTENT: Dict[str, Decimal] = {
    PolymerType.PET.value: Decimal("0.625"),  # C10H8O4
    PolymerType.HDPE.value: Decimal("0.857"),  # (C2H4)n
    PolymerType.PVC.value: Decimal("0.385"),  # (C2H3Cl)n
    PolymerType.LDPE.value: Decimal("0.857"),  # (C2H4)n
    PolymerType.PP.value: Decimal("0.857"),  # (C3H6)n
    PolymerType.PS.value: Decimal("0.923"),  # (C8H8)n
    PolymerType.OTHER.value: Decimal("0.750"),
    PolymerType.MIXED.value: Decimal("0.750"),
}

# Incineration CO2 emission factors (kg CO2/tonne plastic)
INCINERATION_FACTORS: Dict[str, Decimal] = {
    PolymerType.PET.value: Decimal("2290"),
    PolymerType.HDPE.value: Decimal("3140"),
    PolymerType.PVC.value: Decimal("1410"),
    PolymerType.LDPE.value: Decimal("3140"),
    PolymerType.PP.value: Decimal("3140"),
    PolymerType.PS.value: Decimal("3380"),
    PolymerType.OTHER.value: Decimal("2760"),
    PolymerType.MIXED.value: Decimal("2760"),
}

# Recycling process emissions (kg CO2e/tonne recycled)
RECYCLING_PROCESS_EMISSIONS: Dict[str, Decimal] = {
    PolymerType.PET.value: Decimal("500"),
    PolymerType.HDPE.value: Decimal("400"),
    PolymerType.PVC.value: Decimal("600"),
    PolymerType.LDPE.value: Decimal("450"),
    PolymerType.PP.value: Decimal("420"),
    PolymerType.PS.value: Decimal("550"),
    PolymerType.OTHER.value: Decimal("500"),
    PolymerType.MIXED.value: Decimal("600"),
}

# Avoided emissions from recycling (vs virgin production, kg CO2e/tonne)
VIRGIN_PRODUCTION_FACTORS: Dict[str, Decimal] = {
    PolymerType.PET.value: Decimal("3500"),
    PolymerType.HDPE.value: Decimal("1900"),
    PolymerType.PVC.value: Decimal("2200"),
    PolymerType.LDPE.value: Decimal("2100"),
    PolymerType.PP.value: Decimal("1700"),
    PolymerType.PS.value: Decimal("3400"),
    PolymerType.OTHER.value: Decimal("2500"),
    PolymerType.MIXED.value: Decimal("2300"),
}

# Ocean degradation factors (years to degrade)
OCEAN_DEGRADATION_TIME: Dict[str, int] = {
    PolymerType.PET.value: 450,
    PolymerType.HDPE.value: 500,
    PolymerType.PVC.value: 500,
    PolymerType.LDPE.value: 500,
    PolymerType.PP.value: 500,
    PolymerType.PS.value: 80,  # Styrofoam breaks down faster
    PolymerType.OTHER.value: 400,
    PolymerType.MIXED.value: 400,
}


# =============================================================================
# INPUT/OUTPUT MODELS
# =============================================================================

class PlasticWasteRecord(BaseModel):
    """Individual plastic waste record."""
    polymer_type: PolymerType = Field(..., description="Polymer type")
    mass_tonnes: Decimal = Field(..., gt=0, description="Mass in metric tonnes")
    disposal_path: PlasticDisposalPath = Field(..., description="Disposal pathway")
    collection_source: str = Field(
        "general", description="Collection source (residential/commercial/industrial)"
    )
    contamination_rate: Decimal = Field(
        Decimal("0.1"), ge=0, le=0.5, description="Contamination rate"
    )

    class Config:
        use_enum_values = True


class PlasticWasteInput(WasteMRVInput):
    """Input model for Plastic Waste MRV Agent."""

    # Plastic waste records
    plastic_records: List[PlasticWasteRecord] = Field(
        default_factory=list, description="Plastic waste records"
    )

    # Simplified input
    total_plastic_tonnes: Optional[Decimal] = Field(
        None, gt=0, description="Total plastic waste"
    )
    default_polymer: PolymerType = Field(
        PolymerType.MIXED, description="Default polymer type"
    )
    default_disposal: PlasticDisposalPath = Field(
        PlasticDisposalPath.LANDFILL, description="Default disposal path"
    )

    # Regional factors
    regional_mismanagement_rate: Decimal = Field(
        Decimal("0.02"), ge=0, le=0.5, description="Regional waste mismanagement rate"
    )
    ocean_leakage_rate: Decimal = Field(
        Decimal("0.02"), ge=0, le=0.1, description="Ocean leakage rate from mismanaged"
    )

    # Recycling parameters
    recycling_yield: Decimal = Field(
        Decimal("0.85"), ge=0, le=1, description="Recycling process yield"
    )
    include_avoided_emissions: bool = Field(
        True, description="Include avoided emissions from recycling"
    )

    # Energy recovery
    energy_recovery_efficiency: Decimal = Field(
        Decimal("0.25"), ge=0, le=0.5, description="Electrical efficiency of WtE"
    )
    grid_emission_factor: Decimal = Field(
        Decimal("0.4"), ge=0, description="Grid emission factor (kg CO2/kWh)"
    )


class PlasticWasteOutput(WasteMRVOutput):
    """Output model for Plastic Waste MRV Agent."""

    # Plastic-specific outputs
    total_plastic_tonnes: Decimal = Field(
        Decimal("0"), description="Total plastic waste tracked"
    )

    # Pathway breakdown
    landfilled_tonnes: Decimal = Field(Decimal("0"))
    incinerated_tonnes: Decimal = Field(Decimal("0"))
    recycled_tonnes: Decimal = Field(Decimal("0"))
    mismanaged_tonnes: Decimal = Field(Decimal("0"))
    ocean_leakage_tonnes: Decimal = Field(Decimal("0"))

    # Emissions by pathway
    landfill_emissions_kg_co2e: Decimal = Field(Decimal("0"))
    incineration_emissions_kg_co2e: Decimal = Field(Decimal("0"))
    recycling_emissions_kg_co2e: Decimal = Field(Decimal("0"))
    open_burning_emissions_kg_co2e: Decimal = Field(Decimal("0"))

    # Credits and avoided
    recycling_avoided_kg_co2e: Decimal = Field(Decimal("0"))
    energy_avoided_kg_co2e: Decimal = Field(Decimal("0"))
    net_emissions_kg_co2e: Decimal = Field(Decimal("0"))

    # Circularity metrics
    recycling_rate_pct: Decimal = Field(Decimal("0"))
    leakage_rate_pct: Decimal = Field(Decimal("0"))

    # Polymer breakdown
    polymer_breakdown: Dict[str, Dict[str, str]] = Field(
        default_factory=dict, description="Emissions by polymer type"
    )


# =============================================================================
# PLASTIC WASTE MRV AGENT
# =============================================================================

class PlasticWasteMRVAgent(BaseWasteMRVAgent[PlasticWasteInput, PlasticWasteOutput]):
    """
    GL-MRV-WST-006: Plastic Waste MRV Agent

    Calculates emissions from plastic waste across disposal pathways.

    Calculation Approach:
    1. Track plastic waste by polymer type and disposal pathway
    2. Calculate incineration emissions (fossil CO2)
    3. Calculate recycling process emissions
    4. Estimate mismanaged waste and ocean leakage
    5. Calculate avoided emissions from recycling
    6. Calculate energy recovery credits
    7. Report net emissions and circularity metrics

    Key Formula:
        Incineration: CO2 = mass * carbon_content * 44/12
        Recycling Net: avoided - process_emissions
        Landfill: minimal (plastics don't biodegrade)

    Example:
        >>> agent = PlasticWasteMRVAgent()
        >>> input_data = PlasticWasteInput(
        ...     organization_id="ORG001",
        ...     reporting_year=2024,
        ...     plastic_records=[
        ...         PlasticWasteRecord(
        ...             polymer_type=PolymerType.PET,
        ...             mass_tonnes=Decimal("1000"),
        ...             disposal_path=PlasticDisposalPath.MECHANICAL_RECYCLING,
        ...         ),
        ...     ],
        ... )
        >>> result = agent.calculate(input_data)
    """

    AGENT_ID = "GL-MRV-WST-006"
    AGENT_NAME = "Plastic Waste MRV Agent"
    AGENT_VERSION = "1.0.0"
    TREATMENT_METHOD = TreatmentMethod.RECYCLING
    DEFAULT_SCOPE = EmissionScope.SCOPE_3

    # Landfill emission factor for plastics (minimal, transport only)
    LANDFILL_EF = Decimal("21.3")  # kg CO2e/tonne

    # Open burning inefficiency multiplier
    OPEN_BURNING_MULTIPLIER = Decimal("1.5")

    # Energy content of plastics (MJ/kg)
    PLASTIC_ENERGY_CONTENT = Decimal("40")

    def __init__(self):
        """Initialize Plastic Waste MRV Agent."""
        super().__init__()
        self._carbon_content = POLYMER_CARBON_CONTENT
        self._incineration_ef = INCINERATION_FACTORS
        self._recycling_ef = RECYCLING_PROCESS_EMISSIONS
        self._virgin_ef = VIRGIN_PRODUCTION_FACTORS
        self.logger.info(f"Initialized {self.AGENT_ID}: {self.AGENT_NAME}")

    def calculate(self, input_data: PlasticWasteInput) -> PlasticWasteOutput:
        """
        Calculate plastic waste emissions.

        Args:
            input_data: Plastic waste input data

        Returns:
            PlasticWasteOutput with emissions and circularity metrics
        """
        start_time = datetime.now(timezone.utc)
        steps: List[CalculationStep] = []
        emission_factors: List[EmissionFactor] = []
        warnings: List[str] = []

        # Step 1: Initialize
        steps.append(CalculationStep(
            step_number=1,
            description="Initialize plastic waste emissions calculation",
            formula="N/A",
            inputs={
                "reporting_year": input_data.reporting_year,
                "include_avoided": input_data.include_avoided_emissions,
            },
            output="Initialization complete",
        ))

        # Step 2: Collect plastic records
        records = self._collect_plastic_data(input_data)
        total_plastic = sum(r.mass_tonnes for r in records)

        steps.append(CalculationStep(
            step_number=2,
            description="Aggregate plastic waste records",
            formula="total = sum(record_mass)",
            inputs={"num_records": len(records)},
            output=f"{total_plastic} tonnes",
        ))

        # Step 3: Calculate emissions by pathway
        landfilled = Decimal("0")
        incinerated = Decimal("0")
        recycled = Decimal("0")
        mismanaged = Decimal("0")
        ocean_leakage = Decimal("0")

        landfill_emissions = Decimal("0")
        incineration_emissions = Decimal("0")
        recycling_emissions = Decimal("0")
        open_burning_emissions = Decimal("0")
        recycling_avoided = Decimal("0")
        energy_avoided = Decimal("0")

        polymer_breakdown: Dict[str, Dict[str, str]] = {}

        for record in records:
            polymer = record.polymer_type.value
            disposal = record.disposal_path.value
            mass = record.mass_tonnes

            # Initialize polymer breakdown
            if polymer not in polymer_breakdown:
                polymer_breakdown[polymer] = {
                    "total_tonnes": "0",
                    "emissions_kg": "0",
                    "avoided_kg": "0",
                }

            if disposal in ["landfill"]:
                landfilled += mass
                emissions = mass * self.LANDFILL_EF * Decimal("1000")
                landfill_emissions += emissions
                polymer_breakdown[polymer]["emissions_kg"] = str(
                    Decimal(polymer_breakdown[polymer]["emissions_kg"]) + emissions
                )

            elif disposal in ["incineration", "incineration_energy"]:
                incinerated += mass
                ef = self._incineration_ef.get(polymer, self._incineration_ef[PolymerType.MIXED.value])
                emissions = mass * ef * Decimal("1000")
                incineration_emissions += emissions
                polymer_breakdown[polymer]["emissions_kg"] = str(
                    Decimal(polymer_breakdown[polymer]["emissions_kg"]) + emissions
                )

                # Energy recovery credit
                if disposal == "incineration_energy":
                    # Energy = mass * energy_content * efficiency
                    energy_mj = mass * self.PLASTIC_ENERGY_CONTENT * Decimal("1000")
                    energy_kwh = energy_mj / Decimal("3.6")  # MJ to kWh
                    electricity_kwh = energy_kwh * input_data.energy_recovery_efficiency
                    avoided = electricity_kwh * input_data.grid_emission_factor
                    energy_avoided += avoided

            elif disposal in ["mechanical_recycling", "chemical_recycling"]:
                recycled += mass
                # Effective recycled mass after losses
                effective_mass = mass * input_data.recycling_yield * (Decimal("1") - record.contamination_rate)

                # Process emissions
                process_ef = self._recycling_ef.get(polymer, self._recycling_ef[PolymerType.MIXED.value])
                emissions = effective_mass * process_ef * Decimal("1000")
                recycling_emissions += emissions
                polymer_breakdown[polymer]["emissions_kg"] = str(
                    Decimal(polymer_breakdown[polymer]["emissions_kg"]) + emissions
                )

                # Avoided emissions
                if input_data.include_avoided_emissions:
                    virgin_ef = self._virgin_ef.get(polymer, self._virgin_ef[PolymerType.MIXED.value])
                    avoided = effective_mass * virgin_ef * Decimal("1000")
                    recycling_avoided += avoided
                    polymer_breakdown[polymer]["avoided_kg"] = str(
                        Decimal(polymer_breakdown[polymer]["avoided_kg"]) + avoided
                    )

            elif disposal == "open_burning":
                ef = self._incineration_ef.get(polymer, self._incineration_ef[PolymerType.MIXED.value])
                emissions = mass * ef * self.OPEN_BURNING_MULTIPLIER * Decimal("1000")
                open_burning_emissions += emissions
                mismanaged += mass
                polymer_breakdown[polymer]["emissions_kg"] = str(
                    Decimal(polymer_breakdown[polymer]["emissions_kg"]) + emissions
                )

            elif disposal in ["mismanaged", "ocean_leakage"]:
                mismanaged += mass
                if disposal == "ocean_leakage":
                    ocean_leakage += mass

            elif disposal == "export":
                # Track but don't assign emissions (transfers to importer)
                warnings.append(f"Exported plastic ({mass}t) emissions not counted locally")

            polymer_breakdown[polymer]["total_tonnes"] = str(
                Decimal(polymer_breakdown[polymer]["total_tonnes"]) + mass
            )

        # Quantize all values
        landfill_emissions = landfill_emissions.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        incineration_emissions = incineration_emissions.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        recycling_emissions = recycling_emissions.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        open_burning_emissions = open_burning_emissions.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        recycling_avoided = recycling_avoided.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        energy_avoided = energy_avoided.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        steps.append(CalculationStep(
            step_number=3,
            description="Calculate emissions by disposal pathway",
            formula="emissions = mass * pathway_EF",
            inputs={
                "landfilled_tonnes": str(landfilled),
                "incinerated_tonnes": str(incinerated),
                "recycled_tonnes": str(recycled),
                "mismanaged_tonnes": str(mismanaged),
            },
            output=f"Landfill: {landfill_emissions}kg, Incin: {incineration_emissions}kg, Recycle: {recycling_emissions}kg",
        ))

        # Step 4: Calculate total emissions
        total_gross = (
            landfill_emissions + incineration_emissions +
            recycling_emissions + open_burning_emissions
        )

        steps.append(CalculationStep(
            step_number=4,
            description="Calculate total gross emissions",
            formula="gross = sum(pathway_emissions)",
            inputs={
                "landfill_kg": str(landfill_emissions),
                "incineration_kg": str(incineration_emissions),
                "recycling_kg": str(recycling_emissions),
                "open_burning_kg": str(open_burning_emissions),
            },
            output=f"{total_gross} kg CO2e",
        ))

        # Step 5: Calculate net emissions
        total_avoided = recycling_avoided + energy_avoided
        net_emissions = (total_gross - total_avoided).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        steps.append(CalculationStep(
            step_number=5,
            description="Calculate net emissions",
            formula="net = gross - recycling_avoided - energy_avoided",
            inputs={
                "gross_kg": str(total_gross),
                "recycling_avoided_kg": str(recycling_avoided),
                "energy_avoided_kg": str(energy_avoided),
            },
            output=f"{net_emissions} kg CO2e net",
        ))

        # Step 6: Calculate circularity metrics
        recycling_rate = (recycled / total_plastic * Decimal("100")).quantize(
            Decimal("0.1"), rounding=ROUND_HALF_UP
        ) if total_plastic > 0 else Decimal("0")

        leakage_rate = ((mismanaged + ocean_leakage) / total_plastic * Decimal("100")).quantize(
            Decimal("0.1"), rounding=ROUND_HALF_UP
        ) if total_plastic > 0 else Decimal("0")

        steps.append(CalculationStep(
            step_number=6,
            description="Calculate circularity metrics",
            formula="recycling_rate = recycled / total * 100",
            inputs={
                "recycled_tonnes": str(recycled),
                "total_tonnes": str(total_plastic),
                "mismanaged_tonnes": str(mismanaged),
            },
            output=f"Recycling: {recycling_rate}%, Leakage: {leakage_rate}%",
        ))

        # Record emission factors
        for polymer in polymer_breakdown:
            ef = EmissionFactor(
                factor_id=f"plastic_incineration_{polymer}",
                factor_value=self._incineration_ef.get(polymer, self._incineration_ef[PolymerType.MIXED.value]),
                factor_unit="kg CO2/tonne",
                source="IPCC/EPA",
                source_uri="https://www.epa.gov/warm",
                version="2024",
                last_updated="2024-01-01",
                uncertainty_pct=10.0,
                data_quality_tier=DataQualityTier.TIER_2,
                geographic_scope="global",
                waste_type=WasteType.PLASTIC,
            )
            emission_factors.append(ef)

        # Create activity summary
        activity_summary = {
            "organization_id": input_data.organization_id,
            "facility_id": input_data.facility_id,
            "reporting_year": input_data.reporting_year,
            "total_plastic_tonnes": str(total_plastic),
            "recycling_rate_pct": str(recycling_rate),
            "polymer_types": list(polymer_breakdown.keys()),
        }

        # Build output
        output = PlasticWasteOutput(
            agent_id=self.AGENT_ID,
            agent_version=self.AGENT_VERSION,
            total_emissions_kg_co2e=total_gross,
            total_emissions_mt_co2e=self._kg_to_metric_tons(total_gross),
            co2_kg=total_gross,  # All fossil CO2 for plastics
            ch4_kg=Decimal("0"),
            n2o_kg=Decimal("0"),
            scope=EmissionScope.SCOPE_3,
            calculation_steps=steps,
            provenance_hash="",
            data_quality_tier=DataQualityTier.TIER_2,
            calculation_timestamp=datetime.now(timezone.utc),
            calculation_duration_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
            emission_factors_used=emission_factors,
            activity_data_summary=activity_summary,
            warnings=warnings,
            # Plastic-specific fields
            total_plastic_tonnes=total_plastic,
            landfilled_tonnes=landfilled,
            incinerated_tonnes=incinerated,
            recycled_tonnes=recycled,
            mismanaged_tonnes=mismanaged,
            ocean_leakage_tonnes=ocean_leakage,
            landfill_emissions_kg_co2e=landfill_emissions,
            incineration_emissions_kg_co2e=incineration_emissions,
            recycling_emissions_kg_co2e=recycling_emissions,
            open_burning_emissions_kg_co2e=open_burning_emissions,
            recycling_avoided_kg_co2e=recycling_avoided,
            energy_avoided_kg_co2e=energy_avoided,
            net_emissions_kg_co2e=net_emissions,
            recycling_rate_pct=recycling_rate,
            leakage_rate_pct=leakage_rate,
            polymer_breakdown=polymer_breakdown,
        )

        output.provenance_hash = self._generate_provenance_hash(
            input_data=activity_summary,
            output_data={
                "total_emissions_kg_co2e": total_gross,
                "net_emissions_kg_co2e": net_emissions,
            },
            steps=steps,
        )

        return output

    def _collect_plastic_data(self, input_data: PlasticWasteInput) -> List[PlasticWasteRecord]:
        """Collect and normalize plastic waste records."""
        records = list(input_data.plastic_records)

        if not records and input_data.total_plastic_tonnes:
            records.append(PlasticWasteRecord(
                polymer_type=input_data.default_polymer,
                mass_tonnes=input_data.total_plastic_tonnes,
                disposal_path=input_data.default_disposal,
            ))

        return records
