# -*- coding: utf-8 -*-
"""
GL-MRV-WST-003: Recycling MRV Agent
====================================

Calculates emissions and avoided emissions from recycling activities.
Supports both processing emissions and avoided virgin production credits.

Key Features:
- Process emissions from recycling operations
- Avoided emissions credits from virgin material displacement
- Material-specific recycling efficiency factors
- Support for closed-loop and open-loop recycling
- Quality factor adjustments for downcycling

Zero-Hallucination Guarantees:
- All calculations use deterministic formulas
- Emission factors from EPA WARM, DEFRA, and GHG Protocol
- SHA-256 provenance hash for complete audit trail

Reference Standards:
- EPA WARM Model (Waste Reduction Model)
- GHG Protocol Product Standard
- ISO 14044 Life Cycle Assessment
- DEFRA GHG Conversion Factors

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
    RECYCLING_CREDITS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# RECYCLING CONSTANTS
# =============================================================================

class RecyclingType(str, Enum):
    """Types of recycling processes."""
    CLOSED_LOOP = "closed_loop"  # Material recycled back to same product
    OPEN_LOOP = "open_loop"  # Material recycled to different product
    DOWNCYCLING = "downcycling"  # Material recycled to lower value product
    UPCYCLING = "upcycling"  # Material recycled to higher value product


# Process emissions from recycling (kg CO2e per tonne processed)
RECYCLING_PROCESS_EMISSIONS: Dict[str, Decimal] = {
    WasteType.PAPER.value: Decimal("50"),
    WasteType.CARDBOARD.value: Decimal("45"),
    WasteType.PLASTIC.value: Decimal("200"),
    WasteType.METAL.value: Decimal("150"),
    WasteType.GLASS.value: Decimal("25"),
    WasteType.TEXTILES.value: Decimal("100"),
    WasteType.E_WASTE.value: Decimal("300"),
    WasteType.WOOD.value: Decimal("30"),
    WasteType.RUBBER.value: Decimal("180"),
    "default": Decimal("100"),
}

# Avoided emissions per tonne recycled (compared to virgin production)
AVOIDED_VIRGIN_EMISSIONS: Dict[str, Decimal] = {
    WasteType.PAPER.value: Decimal("730"),
    WasteType.CARDBOARD.value: Decimal("565"),
    WasteType.PLASTIC.value: Decimal("1640"),
    WasteType.METAL.value: Decimal("2820"),  # Aluminum
    WasteType.GLASS.value: Decimal("340"),
    WasteType.TEXTILES.value: Decimal("2230"),
    WasteType.E_WASTE.value: Decimal("2800"),
    WasteType.WOOD.value: Decimal("546"),
    WasteType.RUBBER.value: Decimal("850"),
    "default": Decimal("500"),
}

# Material loss rates during recycling (fraction lost)
MATERIAL_LOSS_RATES: Dict[str, Decimal] = {
    WasteType.PAPER.value: Decimal("0.10"),
    WasteType.CARDBOARD.value: Decimal("0.08"),
    WasteType.PLASTIC.value: Decimal("0.15"),
    WasteType.METAL.value: Decimal("0.05"),
    WasteType.GLASS.value: Decimal("0.03"),
    WasteType.TEXTILES.value: Decimal("0.20"),
    WasteType.E_WASTE.value: Decimal("0.25"),
    WasteType.WOOD.value: Decimal("0.12"),
    "default": Decimal("0.15"),
}

# Quality degradation factors (1.0 = no degradation)
QUALITY_FACTORS: Dict[str, Dict[str, Decimal]] = {
    RecyclingType.CLOSED_LOOP.value: {"factor": Decimal("1.0"), "description": "No quality loss"},
    RecyclingType.OPEN_LOOP.value: {"factor": Decimal("0.9"), "description": "Minor quality reduction"},
    RecyclingType.DOWNCYCLING.value: {"factor": Decimal("0.6"), "description": "Significant quality reduction"},
    RecyclingType.UPCYCLING.value: {"factor": Decimal("1.2"), "description": "Quality improvement"},
}


# =============================================================================
# INPUT/OUTPUT MODELS
# =============================================================================

class RecycledMaterial(BaseModel):
    """Individual recycled material record."""
    waste_type: WasteType = Field(..., description="Type of material")
    input_tonnes: Decimal = Field(..., gt=0, description="Input mass to recycling")
    recycling_type: RecyclingType = Field(
        RecyclingType.OPEN_LOOP, description="Type of recycling process"
    )
    recycling_yield: Optional[Decimal] = Field(
        None, ge=0, le=1, description="Recycling yield override"
    )
    quality_factor: Optional[Decimal] = Field(
        None, ge=0, le=2, description="Quality factor override"
    )
    contamination_rate: Decimal = Field(
        Decimal("0.05"), ge=0, le=0.5, description="Contamination rate"
    )

    class Config:
        use_enum_values = True


class RecyclingInput(WasteMRVInput):
    """Input model for Recycling MRV Agent."""

    # Recycled materials
    recycled_materials: List[RecycledMaterial] = Field(
        default_factory=list, description="Materials recycled"
    )

    # Simplified input
    total_recycled_tonnes: Optional[Decimal] = Field(
        None, gt=0, description="Total material recycled"
    )
    default_material_type: WasteType = Field(
        WasteType.MIXED, description="Default material type"
    )

    # Facility characteristics
    facility_type: str = Field(
        "mrf", description="Facility type (mrf/specialized/informal)"
    )

    # Process energy
    electricity_kwh_per_tonne: Decimal = Field(
        Decimal("50"), ge=0, description="Electricity consumption per tonne"
    )
    grid_emission_factor: Decimal = Field(
        Decimal("0.4"), ge=0, description="Grid emission factor (kg CO2e/kWh)"
    )

    # Credits configuration
    include_avoided_emissions: bool = Field(
        True, description="Whether to calculate avoided emissions credits"
    )
    allocation_method: str = Field(
        "substitution", description="Allocation method for credits"
    )


class RecyclingOutput(WasteMRVOutput):
    """Output model for Recycling MRV Agent."""

    # Recycling-specific outputs
    total_input_tonnes: Decimal = Field(
        Decimal("0"), description="Total material input"
    )
    total_recycled_output_tonnes: Decimal = Field(
        Decimal("0"), description="Recycled material output"
    )
    recycling_efficiency: Decimal = Field(
        Decimal("0"), description="Overall recycling efficiency"
    )

    # Emissions breakdown
    process_emissions_kg_co2e: Decimal = Field(
        Decimal("0"), description="Emissions from recycling process"
    )
    transport_emissions_kg_co2e: Decimal = Field(
        Decimal("0"), description="Transport emissions"
    )
    energy_emissions_kg_co2e: Decimal = Field(
        Decimal("0"), description="Energy consumption emissions"
    )

    # Credits
    avoided_emissions_kg_co2e: Decimal = Field(
        Decimal("0"), description="Avoided emissions from virgin displacement"
    )
    net_emissions_kg_co2e: Decimal = Field(
        Decimal("0"), description="Net emissions (process - avoided)"
    )

    # Material breakdown
    material_breakdown: Dict[str, Dict[str, str]] = Field(
        default_factory=dict, description="Breakdown by material type"
    )


# =============================================================================
# RECYCLING MRV AGENT
# =============================================================================

class RecyclingMRVAgent(BaseWasteMRVAgent[RecyclingInput, RecyclingOutput]):
    """
    GL-MRV-WST-003: Recycling MRV Agent

    Calculates emissions and avoided emissions from recycling activities.

    Calculation Approach:
    1. Determine input material quantities and types
    2. Calculate material losses and recycled output
    3. Calculate process emissions (energy, transport)
    4. Calculate avoided emissions from virgin material displacement
    5. Apply quality factors and allocation methods
    6. Report net emissions (may be negative for emission credits)

    Key Formula:
        Net = Process_Emissions - (Recycled_Output * Virgin_EF * Quality_Factor)

    Example:
        >>> agent = RecyclingMRVAgent()
        >>> input_data = RecyclingInput(
        ...     organization_id="ORG001",
        ...     reporting_year=2024,
        ...     recycled_materials=[
        ...         RecycledMaterial(waste_type=WasteType.PAPER, input_tonnes=Decimal("1000")),
        ...         RecycledMaterial(waste_type=WasteType.PLASTIC, input_tonnes=Decimal("500")),
        ...     ],
        ...     include_avoided_emissions=True,
        ... )
        >>> result = agent.calculate(input_data)
    """

    AGENT_ID = "GL-MRV-WST-003"
    AGENT_NAME = "Recycling MRV Agent"
    AGENT_VERSION = "1.0.0"
    TREATMENT_METHOD = TreatmentMethod.RECYCLING
    DEFAULT_SCOPE = EmissionScope.SCOPE_3  # Typically reported in Scope 3

    def __init__(self):
        """Initialize Recycling MRV Agent."""
        super().__init__()
        self._process_emissions = RECYCLING_PROCESS_EMISSIONS
        self._avoided_emissions = AVOIDED_VIRGIN_EMISSIONS
        self._loss_rates = MATERIAL_LOSS_RATES
        self._quality_factors = QUALITY_FACTORS
        self.logger.info(f"Initialized {self.AGENT_ID}: {self.AGENT_NAME}")

    def calculate(self, input_data: RecyclingInput) -> RecyclingOutput:
        """
        Calculate recycling emissions and credits.

        Args:
            input_data: Recycling input data

        Returns:
            RecyclingOutput with emissions and credits
        """
        start_time = datetime.now(timezone.utc)
        steps: List[CalculationStep] = []
        emission_factors: List[EmissionFactor] = []
        warnings: List[str] = []

        # Step 1: Initialize
        steps.append(CalculationStep(
            step_number=1,
            description="Initialize recycling emissions calculation",
            formula="N/A",
            inputs={
                "facility_type": input_data.facility_type,
                "include_avoided_emissions": input_data.include_avoided_emissions,
                "allocation_method": input_data.allocation_method,
            },
            output="Initialization complete",
        ))

        # Step 2: Collect material data
        materials = self._collect_material_data(input_data)
        total_input = sum(m.input_tonnes for m in materials)

        steps.append(CalculationStep(
            step_number=2,
            description="Aggregate recycled materials",
            formula="total_input = sum(material_input)",
            inputs={"num_materials": len(materials)},
            output=f"{total_input} tonnes",
        ))

        # Step 3: Calculate recycling yields and outputs
        total_output = Decimal("0")
        material_breakdown: Dict[str, Dict[str, str]] = {}

        for material in materials:
            # Get loss rate
            loss_rate = material.recycling_yield
            if loss_rate is None:
                base_loss = self._loss_rates.get(
                    material.waste_type.value,
                    self._loss_rates["default"]
                )
                # Add contamination losses
                loss_rate = Decimal("1") - base_loss - material.contamination_rate

            if loss_rate < Decimal("0"):
                loss_rate = Decimal("0")
                warnings.append(f"High contamination for {material.waste_type.value}")

            output_tonnes = (material.input_tonnes * loss_rate).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
            total_output += output_tonnes

            material_breakdown[material.waste_type.value] = {
                "input_tonnes": str(material.input_tonnes),
                "output_tonnes": str(output_tonnes),
                "yield": str(loss_rate),
            }

        recycling_efficiency = (total_output / total_input).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        ) if total_input > 0 else Decimal("0")

        steps.append(CalculationStep(
            step_number=3,
            description="Calculate recycling yields",
            formula="output = input * (1 - loss_rate - contamination)",
            inputs={"total_input_tonnes": str(total_input)},
            output=f"{total_output} tonnes output, {recycling_efficiency} efficiency",
        ))

        # Step 4: Calculate process emissions
        process_emissions_kg = Decimal("0")
        for material in materials:
            ef = self._process_emissions.get(
                material.waste_type.value,
                self._process_emissions["default"]
            )
            emissions = material.input_tonnes * ef
            process_emissions_kg += emissions

        process_emissions_kg = process_emissions_kg.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        steps.append(CalculationStep(
            step_number=4,
            description="Calculate process emissions",
            formula="process_emissions = sum(input_tonnes * process_EF)",
            inputs={"num_materials": len(materials)},
            output=f"{process_emissions_kg} kg CO2e",
        ))

        # Step 5: Calculate energy emissions
        energy_emissions_kg = (
            total_input * input_data.electricity_kwh_per_tonne * input_data.grid_emission_factor
        ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        steps.append(CalculationStep(
            step_number=5,
            description="Calculate energy emissions",
            formula="energy_emissions = input_tonnes * kWh_per_tonne * grid_EF",
            inputs={
                "electricity_kwh_per_tonne": str(input_data.electricity_kwh_per_tonne),
                "grid_ef": str(input_data.grid_emission_factor),
            },
            output=f"{energy_emissions_kg} kg CO2e",
        ))

        # Step 6: Calculate total gross emissions
        total_gross_kg = process_emissions_kg + energy_emissions_kg

        steps.append(CalculationStep(
            step_number=6,
            description="Calculate total gross emissions",
            formula="gross = process + energy",
            inputs={
                "process_emissions_kg": str(process_emissions_kg),
                "energy_emissions_kg": str(energy_emissions_kg),
            },
            output=f"{total_gross_kg} kg CO2e",
        ))

        # Step 7: Calculate avoided emissions
        avoided_emissions_kg = Decimal("0")

        if input_data.include_avoided_emissions:
            for material in materials:
                # Get avoided emissions factor
                avoided_ef = self._avoided_emissions.get(
                    material.waste_type.value,
                    self._avoided_emissions["default"]
                )

                # Get quality factor
                quality_factor = material.quality_factor
                if quality_factor is None:
                    quality_data = self._quality_factors.get(
                        material.recycling_type.value,
                        self._quality_factors[RecyclingType.OPEN_LOOP.value]
                    )
                    quality_factor = quality_data["factor"]

                # Get output for this material
                loss_rate = material.recycling_yield
                if loss_rate is None:
                    base_loss = self._loss_rates.get(
                        material.waste_type.value,
                        self._loss_rates["default"]
                    )
                    loss_rate = Decimal("1") - base_loss - material.contamination_rate
                    if loss_rate < Decimal("0"):
                        loss_rate = Decimal("0")

                output_tonnes = material.input_tonnes * loss_rate

                # Avoided = output * virgin_EF * quality_factor
                avoided = output_tonnes * avoided_ef * quality_factor
                avoided_emissions_kg += avoided

                # Record emission factor
                ef_record = EmissionFactor(
                    factor_id=f"recycling_avoided_{material.waste_type.value}",
                    factor_value=avoided_ef,
                    factor_unit="kg CO2e/tonne",
                    source="EPA_WARM",
                    source_uri="https://www.epa.gov/warm",
                    version="2024",
                    last_updated="2024-01-01",
                    uncertainty_pct=20.0,
                    data_quality_tier=DataQualityTier.TIER_2,
                    geographic_scope="US",
                    waste_type=material.waste_type,
                    treatment_method=TreatmentMethod.RECYCLING,
                )
                emission_factors.append(ef_record)

            avoided_emissions_kg = avoided_emissions_kg.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        steps.append(CalculationStep(
            step_number=7,
            description="Calculate avoided emissions from virgin displacement",
            formula="avoided = output_tonnes * virgin_EF * quality_factor",
            inputs={"include_avoided": input_data.include_avoided_emissions},
            output=f"{avoided_emissions_kg} kg CO2e avoided",
        ))

        # Step 8: Calculate net emissions
        net_emissions_kg = (total_gross_kg - avoided_emissions_kg).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        steps.append(CalculationStep(
            step_number=8,
            description="Calculate net emissions",
            formula="net = gross - avoided",
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
            "facility_type": input_data.facility_type,
            "total_input_tonnes": str(total_input),
            "total_output_tonnes": str(total_output),
            "recycling_efficiency": str(recycling_efficiency),
        }

        # Build output
        output = RecyclingOutput(
            agent_id=self.AGENT_ID,
            agent_version=self.AGENT_VERSION,
            total_emissions_kg_co2e=total_gross_kg,
            total_emissions_mt_co2e=self._kg_to_metric_tons(total_gross_kg),
            co2_kg=total_gross_kg,  # All CO2 for recycling
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
            # Recycling-specific fields
            total_input_tonnes=total_input,
            total_recycled_output_tonnes=total_output,
            recycling_efficiency=recycling_efficiency,
            process_emissions_kg_co2e=process_emissions_kg,
            transport_emissions_kg_co2e=Decimal("0"),  # Not calculated here
            energy_emissions_kg_co2e=energy_emissions_kg,
            avoided_emissions_kg_co2e=avoided_emissions_kg,
            net_emissions_kg_co2e=net_emissions_kg,
            material_breakdown=material_breakdown,
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

    def _collect_material_data(self, input_data: RecyclingInput) -> List[RecycledMaterial]:
        """Collect and normalize recycled material data."""
        materials = list(input_data.recycled_materials)

        if not materials and input_data.total_recycled_tonnes:
            materials.append(RecycledMaterial(
                waste_type=input_data.default_material_type,
                input_tonnes=input_data.total_recycled_tonnes,
            ))

        return materials
