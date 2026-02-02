# -*- coding: utf-8 -*-
"""
GreenLang Decarbonization Transport Sector Agents
GL-DECARB-TRN-001 to GL-DECARB-TRN-013

Transport sector decarbonization planning agents for fleet electrification,
fuel switching, and logistics optimization.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import logging

from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.agents.base_agents import DeterministicAgent

logger = logging.getLogger(__name__)


class TransportDecarbBaseAgent(DeterministicAgent):
    """Base class for transport decarbonization agents."""

    def __init__(self, agent_id: str, version: str = "1.0.0"):
        super().__init__(enable_audit_trail=True)
        self.agent_id = agent_id
        self.version = version


class RoadFleetElectrificationAgent(TransportDecarbBaseAgent):
    """
    GL-DECARB-TRN-001: Road Fleet Electrification Agent

    Plans transition of road vehicle fleets to electric vehicles including
    TCO analysis, charging infrastructure, and timeline optimization.
    """

    AGENT_ID = "GL-DECARB-TRN-001"
    AGENT_NAME = "Road Fleet Electrification Agent"
    category = AgentCategory.RECOMMENDATION
    metadata = AgentMetadata(
        name="GL-DECARB-TRN-001",
        category=AgentCategory.RECOMMENDATION,
        uses_chat_session=True,
        description="Road fleet electrification planning"
    )

    EV_TCO_FACTORS = {
        "light_duty": {"ev_premium": 10000, "fuel_savings_year": 3000, "maintenance_savings_year": 800},
        "medium_duty": {"ev_premium": 40000, "fuel_savings_year": 8000, "maintenance_savings_year": 2000},
        "heavy_duty": {"ev_premium": 100000, "fuel_savings_year": 20000, "maintenance_savings_year": 4000},
    }

    def __init__(self):
        super().__init__(agent_id="GL-DECARB-TRN-001", version="1.0.0")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        fleet_size = inputs.get("fleet_size", 100)
        vehicle_type = inputs.get("vehicle_type", "light_duty")
        annual_mileage = inputs.get("annual_mileage_per_vehicle", 15000)

        factors = self.EV_TCO_FACTORS.get(vehicle_type, self.EV_TCO_FACTORS["light_duty"])
        annual_savings = factors["fuel_savings_year"] + factors["maintenance_savings_year"]
        payback_years = factors["ev_premium"] / annual_savings if annual_savings > 0 else 10

        return {
            "organization_id": inputs.get("organization_id", ""),
            "fleet_size": fleet_size,
            "recommended_ev_share_pct": min(80, fleet_size * 0.5),
            "tco_savings_per_vehicle": annual_savings * 10 - factors["ev_premium"],
            "payback_years": round(payback_years, 1),
            "emissions_reduction_pct": 70,
            "charging_stations_needed": fleet_size // 5,
        }


class AviationDecarbonizationAgent(TransportDecarbBaseAgent):
    """
    GL-DECARB-TRN-002: Aviation Decarbonization Agent

    Plans aviation emissions reduction through SAF adoption, fleet renewal,
    and operational efficiency improvements.
    """

    AGENT_ID = "GL-DECARB-TRN-002"
    AGENT_NAME = "Aviation Decarbonization Agent"
    category = AgentCategory.RECOMMENDATION
    metadata = AgentMetadata(
        name="GL-DECARB-TRN-002",
        category=AgentCategory.RECOMMENDATION,
        description="Aviation decarbonization and SAF planning"
    )

    def __init__(self):
        super().__init__(agent_id="GL-DECARB-TRN-002", version="1.0.0")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        annual_fuel_tonnes = inputs.get("annual_fuel_consumption_tonnes", 10000)
        saf_target_pct = inputs.get("saf_target_pct", 10)

        saf_volume = annual_fuel_tonnes * saf_target_pct / 100
        saf_premium_per_tonne = 1500  # SAF premium over jet fuel
        emissions_reduction = saf_volume * 2.5  # ~80% lifecycle reduction

        return {
            "organization_id": inputs.get("organization_id", ""),
            "saf_volume_tonnes": round(saf_volume, 0),
            "saf_cost_premium_million_usd": round(saf_volume * saf_premium_per_tonne / 1e6, 2),
            "emissions_reduction_tonnes_co2e": round(emissions_reduction, 0),
            "recommended_actions": [
                "SAF offtake agreements",
                "Fleet modernization to new-generation aircraft",
                "Operational efficiency improvements (CDO, taxiing)",
            ],
        }


class MaritimeDecarbonizationAgent(TransportDecarbBaseAgent):
    """
    GL-DECARB-TRN-003: Maritime Decarbonization Agent

    Plans shipping decarbonization through alternative fuels, slow steaming,
    and vessel efficiency improvements.
    """

    AGENT_ID = "GL-DECARB-TRN-003"
    AGENT_NAME = "Maritime Decarbonization Agent"
    category = AgentCategory.RECOMMENDATION
    metadata = AgentMetadata(
        name="GL-DECARB-TRN-003",
        category=AgentCategory.RECOMMENDATION,
        description="Maritime decarbonization planning"
    )

    def __init__(self):
        super().__init__(agent_id="GL-DECARB-TRN-003", version="1.0.0")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        fleet_size = inputs.get("fleet_size", 10)
        annual_fuel_tonnes = inputs.get("annual_fuel_consumption_tonnes", 50000)

        return {
            "organization_id": inputs.get("organization_id", ""),
            "slow_steaming_savings_pct": 15,
            "alternative_fuel_options": ["LNG", "Methanol", "Ammonia", "Hydrogen"],
            "eexi_compliance_investments_million": fleet_size * 2,
            "emissions_reduction_potential_pct": 40,
        }


class RailDecarbonizationAgent(TransportDecarbBaseAgent):
    """
    GL-DECARB-TRN-004: Rail Decarbonization Agent

    Plans rail electrification and alternative traction systems.
    """

    AGENT_ID = "GL-DECARB-TRN-004"
    AGENT_NAME = "Rail Decarbonization Agent"
    category = AgentCategory.RECOMMENDATION
    metadata = AgentMetadata(
        name="GL-DECARB-TRN-004",
        category=AgentCategory.RECOMMENDATION,
        description="Rail electrification and decarbonization"
    )

    def __init__(self):
        super().__init__(agent_id="GL-DECARB-TRN-004", version="1.0.0")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        route_km = inputs.get("route_km_diesel", 500)
        trains = inputs.get("locomotive_count", 20)

        return {
            "organization_id": inputs.get("organization_id", ""),
            "electrification_cost_million": route_km * 1.5,
            "battery_train_option_cost": trains * 5,
            "hydrogen_train_option_cost": trains * 8,
            "emissions_reduction_pct": 85,
        }


class LastMileDeliveryAgent(TransportDecarbBaseAgent):
    """
    GL-DECARB-TRN-005: Last Mile Delivery Agent

    Plans decarbonization of urban delivery and last-mile logistics.
    """

    AGENT_ID = "GL-DECARB-TRN-005"
    AGENT_NAME = "Last Mile Delivery Agent"
    category = AgentCategory.RECOMMENDATION
    metadata = AgentMetadata(
        name="GL-DECARB-TRN-005",
        category=AgentCategory.RECOMMENDATION,
        description="Last mile delivery decarbonization"
    )

    def __init__(self):
        super().__init__(agent_id="GL-DECARB-TRN-005", version="1.0.0")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        daily_deliveries = inputs.get("daily_deliveries", 1000)

        return {
            "organization_id": inputs.get("organization_id", ""),
            "ev_van_transition_pct": 80,
            "cargo_bike_suitable_pct": 20,
            "micro_hub_locations_needed": daily_deliveries // 500,
            "emissions_reduction_pct": 75,
        }


class EVChargingInfrastructureAgent(TransportDecarbBaseAgent):
    """
    GL-DECARB-TRN-006: EV Charging Infrastructure Agent

    Plans charging infrastructure deployment for fleet electrification.
    """

    AGENT_ID = "GL-DECARB-TRN-006"
    AGENT_NAME = "EV Charging Infrastructure Agent"
    category = AgentCategory.RECOMMENDATION
    metadata = AgentMetadata(
        name="GL-DECARB-TRN-006",
        category=AgentCategory.RECOMMENDATION,
        description="EV charging infrastructure planning"
    )

    def __init__(self):
        super().__init__(agent_id="GL-DECARB-TRN-006", version="1.0.0")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        ev_count = inputs.get("ev_fleet_size", 50)

        return {
            "organization_id": inputs.get("organization_id", ""),
            "level2_chargers_needed": ev_count,
            "dc_fast_chargers_needed": ev_count // 10,
            "grid_upgrade_cost_million": ev_count * 0.01,
            "total_infrastructure_cost_million": ev_count * 0.015,
        }


class LogisticsOptimizationAgent(TransportDecarbBaseAgent):
    """
    GL-DECARB-TRN-007: Logistics Optimization Agent

    Optimizes logistics to reduce emissions through route optimization,
    load consolidation, and network design.
    """

    AGENT_ID = "GL-DECARB-TRN-007"
    AGENT_NAME = "Logistics Optimization Agent"
    category = AgentCategory.RECOMMENDATION
    metadata = AgentMetadata(
        name="GL-DECARB-TRN-007",
        category=AgentCategory.RECOMMENDATION,
        description="Logistics optimization for emissions reduction"
    )

    def __init__(self):
        super().__init__(agent_id="GL-DECARB-TRN-007", version="1.0.0")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        annual_shipments = inputs.get("annual_shipments", 10000)

        return {
            "organization_id": inputs.get("organization_id", ""),
            "route_optimization_savings_pct": 12,
            "load_consolidation_savings_pct": 8,
            "network_redesign_savings_pct": 15,
            "total_emissions_reduction_pct": 30,
        }


class BusinessTravelDecarbonizationAgent(TransportDecarbBaseAgent):
    """
    GL-DECARB-TRN-008: Business Travel Decarbonization Agent

    Plans reduction of business travel emissions through policy changes,
    virtual meetings, and sustainable travel options.
    """

    AGENT_ID = "GL-DECARB-TRN-008"
    AGENT_NAME = "Business Travel Decarbonization Agent"
    category = AgentCategory.RECOMMENDATION
    metadata = AgentMetadata(
        name="GL-DECARB-TRN-008",
        category=AgentCategory.RECOMMENDATION,
        description="Business travel decarbonization"
    )

    def __init__(self):
        super().__init__(agent_id="GL-DECARB-TRN-008", version="1.0.0")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        annual_travel_emissions = inputs.get("annual_travel_emissions_tco2e", 500)

        return {
            "organization_id": inputs.get("organization_id", ""),
            "virtual_meeting_replacement_pct": 40,
            "rail_substitution_pct": 20,
            "saf_offset_pct": 10,
            "total_emissions_reduction_pct": 55,
            "cost_savings_pct": 30,
        }


class HydrogenMobilityAgent(TransportDecarbBaseAgent):
    """
    GL-DECARB-TRN-009: Hydrogen Mobility Agent

    Plans hydrogen fuel cell adoption for heavy-duty and long-range transport.
    """

    AGENT_ID = "GL-DECARB-TRN-009"
    AGENT_NAME = "Hydrogen Mobility Agent"
    category = AgentCategory.RECOMMENDATION
    metadata = AgentMetadata(
        name="GL-DECARB-TRN-009",
        category=AgentCategory.RECOMMENDATION,
        description="Hydrogen mobility planning"
    )

    def __init__(self):
        super().__init__(agent_id="GL-DECARB-TRN-009", version="1.0.0")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        heavy_duty_fleet = inputs.get("heavy_duty_vehicle_count", 50)

        return {
            "organization_id": inputs.get("organization_id", ""),
            "fcev_suitable_applications": ["Long-haul trucking", "Buses", "Rail"],
            "hydrogen_stations_needed": heavy_duty_fleet // 20,
            "annual_hydrogen_demand_tonnes": heavy_duty_fleet * 10,
            "emissions_reduction_pct": 90,
        }


class SustainableFuelAgent(TransportDecarbBaseAgent):
    """
    GL-DECARB-TRN-010: Sustainable Fuel Agent

    Plans adoption of sustainable fuels including biofuels, e-fuels, and SAF.
    """

    AGENT_ID = "GL-DECARB-TRN-010"
    AGENT_NAME = "Sustainable Fuel Agent"
    category = AgentCategory.RECOMMENDATION
    metadata = AgentMetadata(
        name="GL-DECARB-TRN-010",
        category=AgentCategory.RECOMMENDATION,
        description="Sustainable fuel adoption planning"
    )

    def __init__(self):
        super().__init__(agent_id="GL-DECARB-TRN-010", version="1.0.0")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        annual_fuel_litres = inputs.get("annual_fuel_consumption_litres", 1000000)

        return {
            "organization_id": inputs.get("organization_id", ""),
            "biodiesel_blend_pct": 20,
            "hvo_blend_pct": 30,
            "emissions_reduction_pct": 50,
            "fuel_cost_premium_pct": 25,
        }


class ModeShiftPlannerAgent(TransportDecarbBaseAgent):
    """
    GL-DECARB-TRN-011: Mode Shift Planner Agent

    Plans freight mode shift from road to rail and waterways.
    """

    AGENT_ID = "GL-DECARB-TRN-011"
    AGENT_NAME = "Mode Shift Planner Agent"
    category = AgentCategory.RECOMMENDATION
    metadata = AgentMetadata(
        name="GL-DECARB-TRN-011",
        category=AgentCategory.RECOMMENDATION,
        description="Freight mode shift planning"
    )

    def __init__(self):
        super().__init__(agent_id="GL-DECARB-TRN-011", version="1.0.0")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        road_freight_tonne_km = inputs.get("road_freight_tonne_km_million", 100)

        return {
            "organization_id": inputs.get("organization_id", ""),
            "rail_shift_potential_pct": 25,
            "waterway_shift_potential_pct": 15,
            "emissions_reduction_pct": 35,
            "intermodal_terminals_needed": 5,
        }


class FleetTransitionAgent(TransportDecarbBaseAgent):
    """
    GL-DECARB-TRN-012: Fleet Transition Agent

    Plans comprehensive fleet transition roadmap across all vehicle types.
    """

    AGENT_ID = "GL-DECARB-TRN-012"
    AGENT_NAME = "Fleet Transition Agent"
    category = AgentCategory.RECOMMENDATION
    metadata = AgentMetadata(
        name="GL-DECARB-TRN-012",
        category=AgentCategory.RECOMMENDATION,
        description="Comprehensive fleet transition planning"
    )

    def __init__(self):
        super().__init__(agent_id="GL-DECARB-TRN-012", version="1.0.0")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        total_fleet = inputs.get("total_fleet_size", 200)

        return {
            "organization_id": inputs.get("organization_id", ""),
            "year_1_ev_adoption_pct": 10,
            "year_3_ev_adoption_pct": 30,
            "year_5_ev_adoption_pct": 60,
            "total_transition_cost_million": total_fleet * 0.02,
            "emissions_reduction_by_2030_pct": 70,
        }


class SupplyChainDecarbonizationAgent(TransportDecarbBaseAgent):
    """
    GL-DECARB-TRN-013: Supply Chain Decarbonization Agent

    Plans decarbonization of end-to-end supply chain transport.
    """

    AGENT_ID = "GL-DECARB-TRN-013"
    AGENT_NAME = "Supply Chain Decarbonization Agent"
    category = AgentCategory.RECOMMENDATION
    metadata = AgentMetadata(
        name="GL-DECARB-TRN-013",
        category=AgentCategory.RECOMMENDATION,
        description="Supply chain transport decarbonization"
    )

    def __init__(self):
        super().__init__(agent_id="GL-DECARB-TRN-013", version="1.0.0")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        scope3_transport_emissions = inputs.get("scope3_transport_emissions_tco2e", 5000)

        return {
            "organization_id": inputs.get("organization_id", ""),
            "supplier_engagement_targets": 80,
            "low_carbon_logistics_providers_pct": 50,
            "nearshoring_potential_pct": 20,
            "emissions_reduction_pct": 40,
        }
