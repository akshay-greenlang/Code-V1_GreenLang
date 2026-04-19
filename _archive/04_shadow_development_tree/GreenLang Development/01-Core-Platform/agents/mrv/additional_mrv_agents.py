# -*- coding: utf-8 -*-
"""
GL-MRV-X-016 to GL-MRV-X-030: Additional MRV Agents
====================================================

Additional MRV agents per GHG Protocol requirements covering specialized
calculation needs for comprehensive GHG inventory management.

Agents included:
    - GL-MRV-X-016: Business Travel Calculator
    - GL-MRV-X-017: Employee Commuting Calculator
    - GL-MRV-X-018: Waste Emissions Calculator
    - GL-MRV-X-019: Upstream Transport Calculator
    - GL-MRV-X-020: Capital Goods Calculator
    - GL-MRV-X-021: Fuel & Energy Related Calculator
    - GL-MRV-X-022: Purchased Goods Calculator
    - GL-MRV-X-023: Downstream Transport Calculator
    - GL-MRV-X-024: Product Use Phase Calculator
    - GL-MRV-X-025: End-of-Life Calculator
    - GL-MRV-X-026: Leased Assets Calculator
    - GL-MRV-X-027: Franchises Calculator
    - GL-MRV-X-028: Investments Calculator
    - GL-MRV-X-029: Biogenic Carbon Tracker
    - GL-MRV-X-030: Removals & Offsets Tracker

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.agents.base_agents import DeterministicAgent
from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.utilities.determinism.clock import DeterministicClock

logger = logging.getLogger(__name__)


# =============================================================================
# SHARED UTILITIES
# =============================================================================

def compute_provenance_hash(data: Any) -> str:
    """Compute SHA-256 provenance hash."""
    json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()


# =============================================================================
# GL-MRV-X-016: Business Travel Calculator
# =============================================================================

class TravelMode(str, Enum):
    AIR_SHORT = "air_short_haul"
    AIR_MEDIUM = "air_medium_haul"
    AIR_LONG = "air_long_haul"
    RAIL = "rail"
    CAR_RENTAL = "car_rental"
    TAXI = "taxi"
    BUS = "bus"


TRAVEL_EMISSION_FACTORS = {
    TravelMode.AIR_SHORT: Decimal("0.255"),
    TravelMode.AIR_MEDIUM: Decimal("0.195"),
    TravelMode.AIR_LONG: Decimal("0.195"),
    TravelMode.RAIL: Decimal("0.041"),
    TravelMode.CAR_RENTAL: Decimal("0.171"),
    TravelMode.TAXI: Decimal("0.203"),
    TravelMode.BUS: Decimal("0.089"),
}


class BusinessTravelTrip(BaseModel):
    mode: TravelMode = Field(...)
    distance_km: float = Field(..., gt=0)
    travelers: int = Field(default=1, ge=1)
    cabin_class: str = Field(default="economy")


class BusinessTravelInput(BaseModel):
    trips: List[BusinessTravelTrip] = Field(..., min_length=1)
    organization_id: Optional[str] = Field(None)


class BusinessTravelOutput(BaseModel):
    success: bool = Field(...)
    total_tco2e: float = Field(...)
    emissions_by_mode: Dict[str, float] = Field(default_factory=dict)
    processing_time_ms: float = Field(...)
    provenance_hash: str = Field(...)
    validation_status: str = Field(...)


class BusinessTravelAgent(DeterministicAgent):
    """GL-MRV-X-016: Business Travel Calculator"""

    AGENT_ID = "GL-MRV-X-016"
    AGENT_NAME = "Business Travel Calculator"
    VERSION = "1.0.0"
    category = AgentCategory.CRITICAL

    def __init__(self, enable_audit_trail: bool = True):
        super().__init__(enable_audit_trail=enable_audit_trail)

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        start_time = DeterministicClock.now()
        try:
            travel_input = BusinessTravelInput(**inputs)
            total = Decimal("0")
            by_mode: Dict[str, Decimal] = {}

            for trip in travel_input.trips:
                ef = TRAVEL_EMISSION_FACTORS.get(trip.mode, Decimal("0.2"))
                # Apply cabin class multiplier
                multiplier = Decimal("1.0")
                if trip.cabin_class.lower() == "business":
                    multiplier = Decimal("2.0")
                elif trip.cabin_class.lower() == "first":
                    multiplier = Decimal("3.0")

                emissions = (Decimal(str(trip.distance_km)) * ef * multiplier *
                             Decimal(str(trip.travelers))) / Decimal("1000")
                total += emissions
                mode_key = trip.mode.value
                by_mode[mode_key] = by_mode.get(mode_key, Decimal("0")) + emissions

            end_time = DeterministicClock.now()
            output = BusinessTravelOutput(
                success=True,
                total_tco2e=float(total.quantize(Decimal("0.0001"))),
                emissions_by_mode={k: float(v) for k, v in by_mode.items()},
                processing_time_ms=(end_time - start_time).total_seconds() * 1000,
                provenance_hash=compute_provenance_hash({"total": float(total)}),
                validation_status="PASS"
            )
            return output.model_dump()
        except Exception as e:
            return {"success": False, "error": str(e), "validation_status": "FAIL"}


# =============================================================================
# GL-MRV-X-017: Employee Commuting Calculator
# =============================================================================

class CommuteMode(str, Enum):
    CAR_SOLO = "car_solo"
    CAR_CARPOOL = "car_carpool"
    PUBLIC_TRANSIT = "public_transit"
    BICYCLE = "bicycle"
    WALKING = "walking"
    REMOTE = "remote"


COMMUTE_FACTORS = {
    CommuteMode.CAR_SOLO: Decimal("0.171"),
    CommuteMode.CAR_CARPOOL: Decimal("0.085"),
    CommuteMode.PUBLIC_TRANSIT: Decimal("0.089"),
    CommuteMode.BICYCLE: Decimal("0"),
    CommuteMode.WALKING: Decimal("0"),
    CommuteMode.REMOTE: Decimal("0"),
}


class EmployeeCommutingInput(BaseModel):
    employees_by_mode: Dict[str, int] = Field(...)
    avg_commute_km: float = Field(default=20)
    working_days_per_year: int = Field(default=235)
    organization_id: Optional[str] = Field(None)


class EmployeeCommutingAgent(DeterministicAgent):
    """GL-MRV-X-017: Employee Commuting Calculator"""

    AGENT_ID = "GL-MRV-X-017"
    AGENT_NAME = "Employee Commuting Calculator"
    VERSION = "1.0.0"
    category = AgentCategory.CRITICAL

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        start_time = DeterministicClock.now()
        try:
            comm_input = EmployeeCommutingInput(**inputs)
            total = Decimal("0")
            by_mode: Dict[str, float] = {}

            for mode_str, count in comm_input.employees_by_mode.items():
                try:
                    mode = CommuteMode(mode_str)
                except ValueError:
                    continue

                ef = COMMUTE_FACTORS.get(mode, Decimal("0.1"))
                annual_km = Decimal(str(comm_input.avg_commute_km * 2 * comm_input.working_days_per_year))
                emissions = (annual_km * ef * Decimal(str(count))) / Decimal("1000")
                total += emissions
                by_mode[mode_str] = float(emissions)

            end_time = DeterministicClock.now()
            return {
                "success": True,
                "total_tco2e": float(total.quantize(Decimal("0.0001"))),
                "emissions_by_mode": by_mode,
                "processing_time_ms": (end_time - start_time).total_seconds() * 1000,
                "provenance_hash": compute_provenance_hash({"total": float(total)}),
                "validation_status": "PASS"
            }
        except Exception as e:
            return {"success": False, "error": str(e), "validation_status": "FAIL"}


# =============================================================================
# GL-MRV-X-018: Waste Emissions Calculator
# =============================================================================

class WasteType(str, Enum):
    LANDFILL = "landfill"
    INCINERATION = "incineration"
    RECYCLING = "recycling"
    COMPOSTING = "composting"
    WASTEWATER = "wastewater"


WASTE_FACTORS = {
    WasteType.LANDFILL: Decimal("0.587"),
    WasteType.INCINERATION: Decimal("0.021"),
    WasteType.RECYCLING: Decimal("0.021"),
    WasteType.COMPOSTING: Decimal("0.01"),
    WasteType.WASTEWATER: Decimal("0.5"),
}


class WasteRecord(BaseModel):
    waste_type: WasteType = Field(...)
    quantity_tonnes: float = Field(..., gt=0)
    material: Optional[str] = Field(None)


class WasteEmissionsAgent(DeterministicAgent):
    """GL-MRV-X-018: Waste Emissions Calculator"""

    AGENT_ID = "GL-MRV-X-018"
    AGENT_NAME = "Waste Emissions Calculator"
    VERSION = "1.0.0"
    category = AgentCategory.CRITICAL

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        start_time = DeterministicClock.now()
        try:
            records = [WasteRecord(**r) for r in inputs.get("waste_records", [])]
            total = Decimal("0")
            by_type: Dict[str, float] = {}

            for record in records:
                ef = WASTE_FACTORS.get(record.waste_type, Decimal("0.5"))
                emissions = Decimal(str(record.quantity_tonnes)) * ef
                total += emissions
                type_key = record.waste_type.value
                by_type[type_key] = by_type.get(type_key, 0) + float(emissions)

            end_time = DeterministicClock.now()
            return {
                "success": True,
                "total_tco2e": float(total.quantize(Decimal("0.0001"))),
                "emissions_by_type": by_type,
                "processing_time_ms": (end_time - start_time).total_seconds() * 1000,
                "provenance_hash": compute_provenance_hash({"total": float(total)}),
                "validation_status": "PASS"
            }
        except Exception as e:
            return {"success": False, "error": str(e), "validation_status": "FAIL"}


# =============================================================================
# GL-MRV-X-019: Upstream Transport Calculator
# =============================================================================

class TransportMode(str, Enum):
    ROAD_TRUCK = "road_truck"
    ROAD_VAN = "road_van"
    RAIL_FREIGHT = "rail_freight"
    SEA_CONTAINER = "sea_container"
    AIR_FREIGHT = "air_freight"


TRANSPORT_FACTORS = {
    TransportMode.ROAD_TRUCK: Decimal("0.062"),
    TransportMode.ROAD_VAN: Decimal("0.210"),
    TransportMode.RAIL_FREIGHT: Decimal("0.028"),
    TransportMode.SEA_CONTAINER: Decimal("0.016"),
    TransportMode.AIR_FREIGHT: Decimal("1.093"),
}


class UpstreamTransportAgent(DeterministicAgent):
    """GL-MRV-X-019: Upstream Transport Calculator"""

    AGENT_ID = "GL-MRV-X-019"
    AGENT_NAME = "Upstream Transport Calculator"
    VERSION = "1.0.0"
    category = AgentCategory.CRITICAL

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        start_time = DeterministicClock.now()
        try:
            shipments = inputs.get("shipments", [])
            total = Decimal("0")

            for shipment in shipments:
                mode = TransportMode(shipment.get("mode", "road_truck"))
                tonne_km = Decimal(str(shipment.get("weight_tonnes", 0) * shipment.get("distance_km", 0)))
                ef = TRANSPORT_FACTORS.get(mode, Decimal("0.1"))
                emissions = tonne_km * ef / Decimal("1000")
                total += emissions

            end_time = DeterministicClock.now()
            return {
                "success": True,
                "total_tco2e": float(total.quantize(Decimal("0.0001"))),
                "processing_time_ms": (end_time - start_time).total_seconds() * 1000,
                "provenance_hash": compute_provenance_hash({"total": float(total)}),
                "validation_status": "PASS"
            }
        except Exception as e:
            return {"success": False, "error": str(e), "validation_status": "FAIL"}


# =============================================================================
# GL-MRV-X-020: Capital Goods Calculator
# =============================================================================

class CapitalGoodsAgent(DeterministicAgent):
    """GL-MRV-X-020: Capital Goods Calculator"""

    AGENT_ID = "GL-MRV-X-020"
    AGENT_NAME = "Capital Goods Calculator"
    VERSION = "1.0.0"
    category = AgentCategory.CRITICAL

    SPEND_FACTORS = {
        "machinery": Decimal("0.5"),
        "vehicles": Decimal("0.4"),
        "buildings": Decimal("0.3"),
        "equipment": Decimal("0.45"),
        "it_hardware": Decimal("0.35"),
    }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        start_time = DeterministicClock.now()
        try:
            purchases = inputs.get("capital_purchases", [])
            total = Decimal("0")

            for purchase in purchases:
                category = purchase.get("category", "equipment").lower()
                spend = Decimal(str(purchase.get("spend_usd", 0)))
                ef = self.SPEND_FACTORS.get(category, Decimal("0.4"))
                emissions = spend * ef / Decimal("1000")
                total += emissions

            end_time = DeterministicClock.now()
            return {
                "success": True,
                "total_tco2e": float(total.quantize(Decimal("0.0001"))),
                "processing_time_ms": (end_time - start_time).total_seconds() * 1000,
                "provenance_hash": compute_provenance_hash({"total": float(total)}),
                "validation_status": "PASS"
            }
        except Exception as e:
            return {"success": False, "error": str(e), "validation_status": "FAIL"}


# =============================================================================
# GL-MRV-X-021: Fuel & Energy Related Calculator
# =============================================================================

class FuelEnergyRelatedAgent(DeterministicAgent):
    """GL-MRV-X-021: Fuel & Energy Related Activities Calculator"""

    AGENT_ID = "GL-MRV-X-021"
    AGENT_NAME = "Fuel & Energy Related Calculator"
    VERSION = "1.0.0"
    category = AgentCategory.CRITICAL

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        start_time = DeterministicClock.now()
        try:
            scope1_fuel_tco2e = Decimal(str(inputs.get("scope1_fuel_tco2e", 0)))
            scope2_electricity_tco2e = Decimal(str(inputs.get("scope2_electricity_tco2e", 0)))

            upstream_fuel_pct = Decimal("0.20")
            upstream_elec_pct = Decimal("0.10")
            td_loss_pct = Decimal("0.05")

            upstream_fuel = scope1_fuel_tco2e * upstream_fuel_pct
            upstream_elec = scope2_electricity_tco2e * upstream_elec_pct
            td_losses = scope2_electricity_tco2e * td_loss_pct

            total = upstream_fuel + upstream_elec + td_losses

            end_time = DeterministicClock.now()
            return {
                "success": True,
                "total_tco2e": float(total.quantize(Decimal("0.0001"))),
                "upstream_fuel_tco2e": float(upstream_fuel),
                "upstream_electricity_tco2e": float(upstream_elec),
                "td_losses_tco2e": float(td_losses),
                "processing_time_ms": (end_time - start_time).total_seconds() * 1000,
                "provenance_hash": compute_provenance_hash({"total": float(total)}),
                "validation_status": "PASS"
            }
        except Exception as e:
            return {"success": False, "error": str(e), "validation_status": "FAIL"}


# =============================================================================
# GL-MRV-X-022: Purchased Goods Calculator
# =============================================================================

class PurchasedGoodsAgent(DeterministicAgent):
    """GL-MRV-X-022: Purchased Goods & Services Calculator"""

    AGENT_ID = "GL-MRV-X-022"
    AGENT_NAME = "Purchased Goods Calculator"
    VERSION = "1.0.0"
    category = AgentCategory.CRITICAL

    SPEND_BASED_FACTORS = {
        "raw_materials": Decimal("0.8"),
        "components": Decimal("0.6"),
        "packaging": Decimal("0.4"),
        "services": Decimal("0.1"),
        "office_supplies": Decimal("0.3"),
    }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        start_time = DeterministicClock.now()
        try:
            purchases = inputs.get("purchases", [])
            total = Decimal("0")
            by_category: Dict[str, float] = {}

            for purchase in purchases:
                category = purchase.get("category", "services").lower()
                spend = Decimal(str(purchase.get("spend_usd", 0)))
                ef = self.SPEND_BASED_FACTORS.get(category, Decimal("0.3"))
                emissions = spend * ef / Decimal("1000")
                total += emissions
                by_category[category] = by_category.get(category, 0) + float(emissions)

            end_time = DeterministicClock.now()
            return {
                "success": True,
                "total_tco2e": float(total.quantize(Decimal("0.0001"))),
                "emissions_by_category": by_category,
                "processing_time_ms": (end_time - start_time).total_seconds() * 1000,
                "provenance_hash": compute_provenance_hash({"total": float(total)}),
                "validation_status": "PASS"
            }
        except Exception as e:
            return {"success": False, "error": str(e), "validation_status": "FAIL"}


# =============================================================================
# GL-MRV-X-023 to GL-MRV-X-030: Remaining Agents
# =============================================================================

class DownstreamTransportAgent(DeterministicAgent):
    """GL-MRV-X-023: Downstream Transport Calculator"""
    AGENT_ID = "GL-MRV-X-023"
    AGENT_NAME = "Downstream Transport Calculator"
    VERSION = "1.0.0"
    category = AgentCategory.CRITICAL

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Similar to upstream transport
        return UpstreamTransportAgent().execute(inputs)


class ProductUsePhaseAgent(DeterministicAgent):
    """GL-MRV-X-024: Product Use Phase Calculator"""
    AGENT_ID = "GL-MRV-X-024"
    AGENT_NAME = "Product Use Phase Calculator"
    VERSION = "1.0.0"
    category = AgentCategory.CRITICAL

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        start_time = DeterministicClock.now()
        try:
            products = inputs.get("products", [])
            total = Decimal("0")

            for product in products:
                units_sold = Decimal(str(product.get("units_sold", 0)))
                energy_per_use_kwh = Decimal(str(product.get("energy_per_use_kwh", 0)))
                uses_per_lifetime = Decimal(str(product.get("uses_per_lifetime", 1)))
                grid_factor = Decimal(str(product.get("grid_factor_kgco2e_kwh", 0.4)))

                emissions = (units_sold * energy_per_use_kwh * uses_per_lifetime * grid_factor) / Decimal("1000")
                total += emissions

            end_time = DeterministicClock.now()
            return {
                "success": True,
                "total_tco2e": float(total.quantize(Decimal("0.0001"))),
                "processing_time_ms": (end_time - start_time).total_seconds() * 1000,
                "provenance_hash": compute_provenance_hash({"total": float(total)}),
                "validation_status": "PASS"
            }
        except Exception as e:
            return {"success": False, "error": str(e), "validation_status": "FAIL"}


class EndOfLifeAgent(DeterministicAgent):
    """GL-MRV-X-025: End-of-Life Calculator"""
    AGENT_ID = "GL-MRV-X-025"
    AGENT_NAME = "End-of-Life Calculator"
    VERSION = "1.0.0"
    category = AgentCategory.CRITICAL

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Similar to waste calculator
        return WasteEmissionsAgent().execute(inputs)


class LeasedAssetsAgent(DeterministicAgent):
    """GL-MRV-X-026: Leased Assets Calculator"""
    AGENT_ID = "GL-MRV-X-026"
    AGENT_NAME = "Leased Assets Calculator"
    VERSION = "1.0.0"
    category = AgentCategory.CRITICAL

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        start_time = DeterministicClock.now()
        try:
            assets = inputs.get("leased_assets", [])
            total = Decimal("0")

            for asset in assets:
                floor_area_sqm = Decimal(str(asset.get("floor_area_sqm", 0)))
                energy_intensity = Decimal(str(asset.get("energy_intensity_kwh_sqm", 200)))
                grid_factor = Decimal(str(asset.get("grid_factor", 0.4)))

                annual_energy = floor_area_sqm * energy_intensity
                emissions = (annual_energy * grid_factor) / Decimal("1000")
                total += emissions

            end_time = DeterministicClock.now()
            return {
                "success": True,
                "total_tco2e": float(total.quantize(Decimal("0.0001"))),
                "processing_time_ms": (end_time - start_time).total_seconds() * 1000,
                "provenance_hash": compute_provenance_hash({"total": float(total)}),
                "validation_status": "PASS"
            }
        except Exception as e:
            return {"success": False, "error": str(e), "validation_status": "FAIL"}


class FranchisesAgent(DeterministicAgent):
    """GL-MRV-X-027: Franchises Calculator"""
    AGENT_ID = "GL-MRV-X-027"
    AGENT_NAME = "Franchises Calculator"
    VERSION = "1.0.0"
    category = AgentCategory.CRITICAL

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        start_time = DeterministicClock.now()
        try:
            franchises = inputs.get("franchises", [])
            total = Decimal("0")

            for franchise in franchises:
                scope1_2_tco2e = Decimal(str(franchise.get("scope1_2_tco2e", 0)))
                total += scope1_2_tco2e

            end_time = DeterministicClock.now()
            return {
                "success": True,
                "total_tco2e": float(total.quantize(Decimal("0.0001"))),
                "processing_time_ms": (end_time - start_time).total_seconds() * 1000,
                "provenance_hash": compute_provenance_hash({"total": float(total)}),
                "validation_status": "PASS"
            }
        except Exception as e:
            return {"success": False, "error": str(e), "validation_status": "FAIL"}


class InvestmentsAgent(DeterministicAgent):
    """GL-MRV-X-028: Investments Calculator"""
    AGENT_ID = "GL-MRV-X-028"
    AGENT_NAME = "Investments Calculator"
    VERSION = "1.0.0"
    category = AgentCategory.CRITICAL

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        start_time = DeterministicClock.now()
        try:
            investments = inputs.get("investments", [])
            total = Decimal("0")

            for inv in investments:
                investee_emissions = Decimal(str(inv.get("investee_emissions_tco2e", 0)))
                ownership_pct = Decimal(str(inv.get("ownership_pct", 0))) / Decimal("100")
                attributed = investee_emissions * ownership_pct
                total += attributed

            end_time = DeterministicClock.now()
            return {
                "success": True,
                "total_tco2e": float(total.quantize(Decimal("0.0001"))),
                "processing_time_ms": (end_time - start_time).total_seconds() * 1000,
                "provenance_hash": compute_provenance_hash({"total": float(total)}),
                "validation_status": "PASS"
            }
        except Exception as e:
            return {"success": False, "error": str(e), "validation_status": "FAIL"}


class BiogenicCarbonAgent(DeterministicAgent):
    """GL-MRV-X-029: Biogenic Carbon Tracker"""
    AGENT_ID = "GL-MRV-X-029"
    AGENT_NAME = "Biogenic Carbon Tracker"
    VERSION = "1.0.0"
    category = AgentCategory.CRITICAL

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        start_time = DeterministicClock.now()
        try:
            biogenic_sources = inputs.get("biogenic_sources", [])
            total_biogenic_co2 = Decimal("0")

            for source in biogenic_sources:
                quantity_tonnes = Decimal(str(source.get("quantity_tonnes", 0)))
                carbon_content_pct = Decimal(str(source.get("carbon_content_pct", 50))) / Decimal("100")
                biogenic_co2 = quantity_tonnes * carbon_content_pct * Decimal("3.67")
                total_biogenic_co2 += biogenic_co2

            end_time = DeterministicClock.now()
            return {
                "success": True,
                "biogenic_co2_tonnes": float(total_biogenic_co2.quantize(Decimal("0.0001"))),
                "note": "Reported outside scopes per GHG Protocol",
                "processing_time_ms": (end_time - start_time).total_seconds() * 1000,
                "provenance_hash": compute_provenance_hash({"total": float(total_biogenic_co2)}),
                "validation_status": "PASS"
            }
        except Exception as e:
            return {"success": False, "error": str(e), "validation_status": "FAIL"}


class RemovalsOffsetsAgent(DeterministicAgent):
    """GL-MRV-X-030: Removals & Offsets Tracker"""
    AGENT_ID = "GL-MRV-X-030"
    AGENT_NAME = "Removals & Offsets Tracker"
    VERSION = "1.0.0"
    category = AgentCategory.CRITICAL

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        start_time = DeterministicClock.now()
        try:
            removals = inputs.get("removals", [])
            offsets = inputs.get("offsets", [])

            total_removals = Decimal("0")
            total_offsets = Decimal("0")

            for removal in removals:
                total_removals += Decimal(str(removal.get("tco2e", 0)))

            for offset in offsets:
                total_offsets += Decimal(str(offset.get("tco2e", 0)))

            end_time = DeterministicClock.now()
            return {
                "success": True,
                "total_removals_tco2e": float(total_removals.quantize(Decimal("0.0001"))),
                "total_offsets_tco2e": float(total_offsets.quantize(Decimal("0.0001"))),
                "total_tco2e": float((total_removals + total_offsets).quantize(Decimal("0.0001"))),
                "note": "Removals and offsets tracked separately from gross emissions",
                "processing_time_ms": (end_time - start_time).total_seconds() * 1000,
                "provenance_hash": compute_provenance_hash({
                    "removals": float(total_removals),
                    "offsets": float(total_offsets)
                }),
                "validation_status": "PASS"
            }
        except Exception as e:
            return {"success": False, "error": str(e), "validation_status": "FAIL"}
