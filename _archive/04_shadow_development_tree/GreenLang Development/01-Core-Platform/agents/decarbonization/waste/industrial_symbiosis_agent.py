# -*- coding: utf-8 -*-
"""
GL-DECARB-WST-006: Industrial Symbiosis Agent
==============================================

Plans industrial symbiosis networks where waste/byproducts from one
process become inputs for another.

Key Features:
- Waste stream matching and exchange
- Material flow optimization
- Network design and analysis
- Economic benefit sharing
- Geographic clustering analysis

Author: GreenLang Framework Team
Version: 1.0.0
"""

from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional
from enum import Enum
import logging

from pydantic import BaseModel, Field

from greenlang.agents.decarbonization.waste.base import (
    BaseWasteDecarbAgent,
    WasteDecarbInput,
    WasteDecarbOutput,
    DecarbonizationStrategy,
    DecarbonizationIntervention,
    DecarbonizationPathway,
    ImplementationTimeline,
    CostCategory,
    ConfidenceLevel,
)

logger = logging.getLogger(__name__)


class WasteStreamCategory(str, Enum):
    """Categories of industrial waste streams."""
    ORGANIC_MATERIALS = "organic_materials"
    METALS = "metals"
    PLASTICS = "plastics"
    CHEMICALS = "chemicals"
    HEAT_ENERGY = "heat_energy"
    WATER = "water"
    GASES = "gases"
    MINERALS = "minerals"


class SymbiosisExchange(BaseModel):
    """Individual symbiosis exchange."""
    exchange_id: str
    waste_type: WasteStreamCategory
    source_facility: str
    recipient_facility: str
    annual_volume_tonnes: Decimal = Field(Decimal("0"))
    avoided_disposal_cost_usd: Decimal = Field(Decimal("0"))
    avoided_virgin_material_cost_usd: Decimal = Field(Decimal("0"))
    emission_reduction_kg_co2e: Decimal = Field(Decimal("0"))
    transport_distance_km: Decimal = Field(Decimal("0"))


class IndustrialSymbiosisInput(WasteDecarbInput):
    """Input model for Industrial Symbiosis Agent."""
    industrial_park_id: Optional[str] = Field(None, description="Industrial park identifier")
    participating_facilities: List[str] = Field(
        default_factory=list, description="Participating facility IDs"
    )
    waste_streams: Dict[str, Decimal] = Field(
        default_factory=dict, description="Available waste streams by type (tonnes/year)"
    )
    material_demands: Dict[str, Decimal] = Field(
        default_factory=dict, description="Material demands by type (tonnes/year)"
    )
    max_transport_distance_km: Decimal = Field(
        Decimal("100"), ge=0, description="Maximum transport distance"
    )
    existing_exchanges: List[SymbiosisExchange] = Field(
        default_factory=list, description="Existing symbiosis exchanges"
    )


class SymbiosisNetworkMetrics(BaseModel):
    """Symbiosis network metrics."""
    connectivity_index: Decimal = Field(Decimal("0"), description="Network connectivity (0-1)")
    material_synergy_rate: Decimal = Field(Decimal("0"), description="Waste utilized rate")
    economic_benefit_usd: Decimal = Field(Decimal("0"))
    environmental_benefit_kg_co2e: Decimal = Field(Decimal("0"))


class IndustrialSymbiosisOutput(WasteDecarbOutput):
    """Output model for Industrial Symbiosis Agent."""
    proposed_exchanges: List[SymbiosisExchange] = Field(default_factory=list)
    network_metrics: Optional[SymbiosisNetworkMetrics] = None
    total_waste_diverted_tonnes: Decimal = Field(Decimal("0"))
    total_virgin_materials_avoided_tonnes: Decimal = Field(Decimal("0"))
    annual_cost_savings_usd: Decimal = Field(Decimal("0"))
    network_visualization_data: Dict[str, Any] = Field(default_factory=dict)


class IndustrialSymbiosisAgent(BaseWasteDecarbAgent[IndustrialSymbiosisInput, IndustrialSymbiosisOutput]):
    """
    GL-DECARB-WST-006: Industrial Symbiosis Agent

    Plans industrial symbiosis networks based on Kalundborg and global
    eco-industrial park best practices.

    Symbiosis Principles:
    1. Waste = Resource: One facility's waste is another's input
    2. Proximity: Geographic clustering reduces transport
    3. Trust: Information sharing between partners
    4. Diversity: Mix of industries creates more opportunities
    5. Anchor tenant: Large facility provides stable flows

    Example:
        >>> agent = IndustrialSymbiosisAgent()
        >>> input_data = IndustrialSymbiosisInput(
        ...     organization_id="ORG001",
        ...     baseline_year=2020,
        ...     baseline_emissions_kg_co2e=Decimal("5000000"),
        ...     current_year=2024,
        ...     target_year=2030,
        ...     waste_streams={"organic_materials": Decimal("1000"), "metals": Decimal("500")},
        ...     material_demands={"organic_materials": Decimal("800"), "metals": Decimal("600")},
        ... )
        >>> result = agent.plan(input_data)
    """

    AGENT_ID = "GL-DECARB-WST-006"
    AGENT_NAME = "Industrial Symbiosis Agent"
    AGENT_VERSION = "1.0.0"
    STRATEGY = DecarbonizationStrategy.INDUSTRIAL_SYMBIOSIS

    # Emission factors for avoided virgin materials (kg CO2e/tonne)
    VIRGIN_MATERIAL_EF = {
        WasteStreamCategory.ORGANIC_MATERIALS.value: Decimal("500"),
        WasteStreamCategory.METALS.value: Decimal("2000"),
        WasteStreamCategory.PLASTICS.value: Decimal("1500"),
        WasteStreamCategory.CHEMICALS.value: Decimal("1200"),
        WasteStreamCategory.HEAT_ENERGY.value: Decimal("300"),  # per MWh equivalent
        WasteStreamCategory.WATER.value: Decimal("1"),  # per m3
        WasteStreamCategory.GASES.value: Decimal("200"),
        WasteStreamCategory.MINERALS.value: Decimal("100"),
    }

    # Disposal costs avoided (USD/tonne)
    DISPOSAL_COSTS = {
        WasteStreamCategory.ORGANIC_MATERIALS.value: Decimal("80"),
        WasteStreamCategory.METALS.value: Decimal("50"),
        WasteStreamCategory.PLASTICS.value: Decimal("100"),
        WasteStreamCategory.CHEMICALS.value: Decimal("300"),
        WasteStreamCategory.HEAT_ENERGY.value: Decimal("0"),
        WasteStreamCategory.WATER.value: Decimal("2"),
        WasteStreamCategory.GASES.value: Decimal("50"),
        WasteStreamCategory.MINERALS.value: Decimal("40"),
    }

    def plan(self, input_data: IndustrialSymbiosisInput) -> IndustrialSymbiosisOutput:
        """Generate industrial symbiosis plan."""
        start_time = datetime.now(timezone.utc)

        interventions: List[DecarbonizationIntervention] = []
        proposed_exchanges: List[SymbiosisExchange] = []

        total_waste_diverted = Decimal("0")
        total_virgin_avoided = Decimal("0")
        total_emission_reduction = Decimal("0")
        total_cost_savings = Decimal("0")

        # Match waste streams to demands
        exchange_id = 1
        for waste_type, available in input_data.waste_streams.items():
            demand = input_data.material_demands.get(waste_type, Decimal("0"))

            if demand > Decimal("0") and available > Decimal("0"):
                exchange_volume = min(available, demand)

                # Get emission factor
                ef = self.VIRGIN_MATERIAL_EF.get(waste_type, Decimal("500"))
                disposal_cost = self.DISPOSAL_COSTS.get(waste_type, Decimal("50"))

                emission_reduction = (exchange_volume * ef).quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP
                )
                cost_savings = (exchange_volume * (disposal_cost + Decimal("100"))).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )  # Disposal + virgin cost

                exchange = SymbiosisExchange(
                    exchange_id=f"SYM-{exchange_id:03d}",
                    waste_type=WasteStreamCategory(waste_type) if waste_type in [e.value for e in WasteStreamCategory] else WasteStreamCategory.ORGANIC_MATERIALS,
                    source_facility=f"Facility_Source_{exchange_id}",
                    recipient_facility=f"Facility_Recipient_{exchange_id}",
                    annual_volume_tonnes=exchange_volume,
                    avoided_disposal_cost_usd=exchange_volume * disposal_cost,
                    avoided_virgin_material_cost_usd=exchange_volume * Decimal("100"),
                    emission_reduction_kg_co2e=emission_reduction,
                    transport_distance_km=Decimal("30"),  # Assumed average
                )
                proposed_exchanges.append(exchange)

                total_waste_diverted += exchange_volume
                total_virgin_avoided += exchange_volume
                total_emission_reduction += emission_reduction
                total_cost_savings += cost_savings

                exchange_id += 1

        # Create interventions for each exchange type
        for category in WasteStreamCategory:
            category_exchanges = [e for e in proposed_exchanges if e.waste_type.value == category.value]
            if category_exchanges:
                category_reduction = sum(e.emission_reduction_kg_co2e for e in category_exchanges)
                category_volume = sum(e.annual_volume_tonnes for e in category_exchanges)

                interventions.append(DecarbonizationIntervention(
                    intervention_id=f"SYM-{category.value.upper()[:4]}-001",
                    strategy=DecarbonizationStrategy.INDUSTRIAL_SYMBIOSIS,
                    description=f"Establish {category.value} exchange network",
                    timeline=ImplementationTimeline.MEDIUM_TERM,
                    cost_category=CostCategory.LOW_COST,
                    reduction_potential_kg_co2e=category_reduction,
                    reduction_potential_pct=(category_reduction / input_data.baseline_emissions_kg_co2e * Decimal("100")).quantize(
                        Decimal("0.1"), rounding=ROUND_HALF_UP
                    ) if input_data.baseline_emissions_kg_co2e > 0 else Decimal("0"),
                    capex_usd=category_volume * Decimal("50"),  # Infrastructure
                    confidence=ConfidenceLevel.MEDIUM,
                    co_benefits=["Cost savings", "Resource efficiency", "Local economic development"],
                ))

        # Calculate network metrics
        total_available = sum(input_data.waste_streams.values())
        synergy_rate = (total_waste_diverted / total_available * Decimal("100")).quantize(
            Decimal("0.1"), rounding=ROUND_HALF_UP
        ) if total_available > 0 else Decimal("0")

        network_metrics = SymbiosisNetworkMetrics(
            connectivity_index=min(Decimal("1"), Decimal(str(len(proposed_exchanges))) / Decimal("10")),
            material_synergy_rate=synergy_rate,
            economic_benefit_usd=total_cost_savings,
            environmental_benefit_kg_co2e=total_emission_reduction,
        )

        # Create pathway
        total_capex = sum(i.capex_usd for i in interventions)

        pathway = DecarbonizationPathway(
            pathway_id="SYM-001",
            name="Industrial Symbiosis Network",
            description="Cross-industry waste-to-resource exchange network",
            interventions=interventions,
            total_reduction_kg_co2e=total_emission_reduction,
            total_capex_usd=total_capex,
            start_year=input_data.current_year,
            target_year=input_data.target_year,
            baseline_emissions_kg_co2e=input_data.baseline_emissions_kg_co2e,
            target_emissions_kg_co2e=input_data.baseline_emissions_kg_co2e - total_emission_reduction,
            reduction_pct=(total_emission_reduction / input_data.baseline_emissions_kg_co2e * Decimal("100")).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            ) if input_data.baseline_emissions_kg_co2e > 0 else Decimal("0"),
        )

        # Network visualization data
        network_data = {
            "nodes": [{"id": e.source_facility, "type": "source"} for e in proposed_exchanges] +
                     [{"id": e.recipient_facility, "type": "recipient"} for e in proposed_exchanges],
            "edges": [{"source": e.source_facility, "target": e.recipient_facility, "volume": str(e.annual_volume_tonnes)} for e in proposed_exchanges],
        }

        output = IndustrialSymbiosisOutput(
            agent_id=self.AGENT_ID,
            agent_version=self.AGENT_VERSION,
            recommended_pathway=pathway,
            baseline_emissions_kg_co2e=input_data.baseline_emissions_kg_co2e,
            achievable_reduction_kg_co2e=total_emission_reduction,
            achievable_reduction_pct=(total_emission_reduction / input_data.baseline_emissions_kg_co2e * Decimal("100")).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            ) if input_data.baseline_emissions_kg_co2e > 0 else Decimal("0"),
            total_capex_usd=total_capex,
            provenance_hash="",
            calculation_timestamp=datetime.now(timezone.utc),
            calculation_duration_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
            proposed_exchanges=proposed_exchanges,
            network_metrics=network_metrics,
            total_waste_diverted_tonnes=total_waste_diverted,
            total_virgin_materials_avoided_tonnes=total_virgin_avoided,
            annual_cost_savings_usd=total_cost_savings,
            network_visualization_data=network_data,
        )

        output.provenance_hash = self._generate_provenance_hash(
            input_data={"waste_streams": str(input_data.waste_streams)},
            output_data={"total_emission_reduction": str(total_emission_reduction)},
        )

        return output
