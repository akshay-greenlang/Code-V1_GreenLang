# -*- coding: utf-8 -*-
"""
GL-MRV-TRN-006: EV Fleet MRV Agent
==================================

This module implements the EV Fleet MRV Agent for measuring, reporting,
and verifying greenhouse gas emissions from electric vehicle fleets.

Supported Features:
- Battery electric vehicles (BEV)
- Plug-in hybrid electric vehicles (PHEV)
- Electric buses and trucks
- Grid electricity emissions (location and market-based)
- Well-to-wheel emissions
- Charging infrastructure emissions

Reference Standards:
- GHG Protocol Scope 2 Guidance
- GHG Protocol Scope 3, Categories 4, 7, 9
- DEFRA Conversion Factors 2024
- IEA CO2 Emissions from Fuel Combustion

Example:
    >>> agent = EVFleetMRVAgent()
    >>> input_data = EVFleetInput(
    ...     organization_id="ORG001",
    ...     reporting_year=2024,
    ...     vehicles=[
    ...         EVVehicleRecord(
    ...             vehicle_type=EVType.BEV_CAR,
    ...             electricity_kwh=Decimal("5000"),
    ...             distance_km=Decimal("20000"),
    ...         )
    ...     ],
    ...     grid_emission_factor_kg_per_kwh=Decimal("0.233"),
    ... )
    >>> result = agent.calculate(input_data)
"""

import logging
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any
from enum import Enum

from pydantic import BaseModel, Field

from greenlang.agents.mrv.transport.base import (
    BaseTransportMRVAgent,
    TransportMRVInput,
    TransportMRVOutput,
    TransportMode,
    EmissionScope,
    DataQualityTier,
    EmissionFactor,
    EmissionFactorSource,
    CalculationStep,
)

logger = logging.getLogger(__name__)


# =============================================================================
# EV-Specific Enums
# =============================================================================

class EVType(str, Enum):
    """Electric vehicle types."""
    BEV_CAR_SMALL = "bev_car_small"
    BEV_CAR_MEDIUM = "bev_car_medium"
    BEV_CAR_LARGE = "bev_car_large"
    BEV_CAR = "bev_car"
    BEV_VAN = "bev_van"
    BEV_TRUCK_LIGHT = "bev_truck_light"
    BEV_TRUCK_MEDIUM = "bev_truck_medium"
    BEV_TRUCK_HEAVY = "bev_truck_heavy"
    BEV_BUS = "bev_bus"
    PHEV_CAR = "phev_car"
    PHEV_VAN = "phev_van"
    E_SCOOTER = "e_scooter"
    E_BIKE = "e_bike"


class ChargingType(str, Enum):
    """Charging infrastructure types."""
    HOME_CHARGING = "home_charging"
    WORKPLACE_CHARGING = "workplace_charging"
    PUBLIC_AC = "public_ac"
    PUBLIC_DC_FAST = "public_dc_fast"
    DEPOT_CHARGING = "depot_charging"


class EmissionMethod(str, Enum):
    """Scope 2 emission calculation method."""
    LOCATION_BASED = "location_based"
    MARKET_BASED = "market_based"


# =============================================================================
# Default Grid Emission Factors (kg CO2e/kWh)
# =============================================================================

GRID_FACTORS_BY_COUNTRY: Dict[str, Decimal] = {
    "UK": Decimal("0.20707"),  # DEFRA 2024
    "US": Decimal("0.373"),
    "DE": Decimal("0.366"),
    "FR": Decimal("0.052"),
    "CN": Decimal("0.555"),
    "IN": Decimal("0.708"),
    "JP": Decimal("0.455"),
    "AU": Decimal("0.656"),
    "CA": Decimal("0.130"),
    "BR": Decimal("0.074"),
    "global": Decimal("0.436"),
}

# EV efficiency factors (kWh per km)
EV_EFFICIENCY: Dict[str, Decimal] = {
    EVType.BEV_CAR_SMALL.value: Decimal("0.145"),
    EVType.BEV_CAR_MEDIUM.value: Decimal("0.170"),
    EVType.BEV_CAR_LARGE.value: Decimal("0.215"),
    EVType.BEV_CAR.value: Decimal("0.170"),
    EVType.BEV_VAN.value: Decimal("0.280"),
    EVType.BEV_TRUCK_LIGHT.value: Decimal("0.450"),
    EVType.BEV_TRUCK_MEDIUM.value: Decimal("0.850"),
    EVType.BEV_TRUCK_HEAVY.value: Decimal("1.500"),
    EVType.BEV_BUS.value: Decimal("1.100"),
    EVType.PHEV_CAR.value: Decimal("0.120"),  # Electric mode only
    EVType.PHEV_VAN.value: Decimal("0.200"),
    EVType.E_SCOOTER.value: Decimal("0.025"),
    EVType.E_BIKE.value: Decimal("0.010"),
}

# Charging losses by type
CHARGING_LOSSES: Dict[str, Decimal] = {
    ChargingType.HOME_CHARGING.value: Decimal("0.12"),  # 12% loss
    ChargingType.WORKPLACE_CHARGING.value: Decimal("0.10"),
    ChargingType.PUBLIC_AC.value: Decimal("0.10"),
    ChargingType.PUBLIC_DC_FAST.value: Decimal("0.08"),
    ChargingType.DEPOT_CHARGING.value: Decimal("0.10"),
}


# =============================================================================
# Input Models
# =============================================================================

class EVVehicleRecord(BaseModel):
    """Individual EV record."""

    # Vehicle identification
    vehicle_id: Optional[str] = Field(None, description="Vehicle identifier")
    vehicle_type: EVType = Field(..., description="EV type")
    registration: Optional[str] = Field(None, description="Registration number")

    # Energy consumption
    electricity_kwh: Optional[Decimal] = Field(
        None, ge=0, description="Electricity consumed in kWh"
    )

    # Distance data
    distance_km: Optional[Decimal] = Field(
        None, ge=0, description="Distance traveled in km"
    )

    # PHEV specific
    fuel_liters: Optional[Decimal] = Field(
        None, ge=0, description="Fuel consumed (PHEV only) in liters"
    )
    electric_mode_pct: Optional[Decimal] = Field(
        None, ge=0, le=100, description="Percentage in electric mode"
    )

    # Charging details
    charging_type: ChargingType = Field(
        ChargingType.DEPOT_CHARGING, description="Primary charging type"
    )

    class Config:
        use_enum_values = True


class EVFleetInput(TransportMRVInput):
    """Input model for EV Fleet MRV Agent."""

    # Vehicle records
    vehicles: List[EVVehicleRecord] = Field(
        default_factory=list, description="List of EV records"
    )

    # Aggregated data
    total_electricity_kwh: Optional[Decimal] = Field(
        None, ge=0, description="Total electricity consumed"
    )
    total_distance_km: Optional[Decimal] = Field(
        None, ge=0, description="Total distance traveled"
    )
    default_vehicle_type: EVType = Field(
        EVType.BEV_CAR, description="Default vehicle type"
    )

    # Grid emission factors
    grid_emission_factor_kg_per_kwh: Optional[Decimal] = Field(
        None, ge=0, description="Custom grid emission factor"
    )
    country_code: str = Field("UK", description="Country for default grid factor")
    emission_method: EmissionMethod = Field(
        EmissionMethod.LOCATION_BASED, description="Scope 2 method"
    )

    # Renewable energy
    renewable_energy_pct: Decimal = Field(
        Decimal("0"), ge=0, le=100, description="Percentage from renewables"
    )
    has_renewable_certificates: bool = Field(
        False, description="Has renewable energy certificates"
    )

    # Charging losses
    include_charging_losses: bool = Field(
        True, description="Include charging losses"
    )
    default_charging_type: ChargingType = Field(
        ChargingType.DEPOT_CHARGING, description="Default charging type"
    )

    # Fleet ownership
    is_owned_fleet: bool = Field(
        True, description="Whether fleet is owned (affects scope)"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# Output Model
# =============================================================================

class EVFleetOutput(TransportMRVOutput):
    """Output model for EV Fleet MRV Agent."""

    # EV-specific metrics
    total_vehicles: int = Field(0, ge=0, description="Number of vehicles")
    total_electricity_kwh: Decimal = Field(
        Decimal("0"), ge=0, description="Total electricity consumed"
    )
    total_distance_km: Decimal = Field(
        Decimal("0"), ge=0, description="Total distance traveled"
    )

    # Scope breakdown
    scope_2_emissions_kg: Decimal = Field(
        Decimal("0"), ge=0, description="Scope 2 (electricity)"
    )
    scope_3_emissions_kg: Decimal = Field(
        Decimal("0"), ge=0, description="Scope 3 (upstream/if not owned)"
    )

    # Efficiency metrics
    emissions_per_km: Optional[Decimal] = Field(
        None, description="kg CO2e per km"
    )
    emissions_per_kwh: Optional[Decimal] = Field(
        None, description="kg CO2e per kWh"
    )
    efficiency_kwh_per_km: Optional[Decimal] = Field(
        None, description="Average kWh per km"
    )

    # Renewable energy
    renewable_energy_pct: Decimal = Field(
        Decimal("0"), description="Renewable energy percentage"
    )
    avoided_emissions_kg: Decimal = Field(
        Decimal("0"), ge=0, description="Emissions avoided by renewables"
    )

    # Charging losses
    charging_losses_kwh: Decimal = Field(
        Decimal("0"), ge=0, description="Energy lost in charging"
    )

    # Breakdown
    emissions_by_vehicle_type: Dict[str, Decimal] = Field(
        default_factory=dict, description="Emissions by vehicle type"
    )


# =============================================================================
# EV Fleet MRV Agent
# =============================================================================

class EVFleetMRVAgent(BaseTransportMRVAgent):
    """
    GL-MRV-TRN-006: EV Fleet MRV Agent

    Calculates greenhouse gas emissions from electric vehicle fleets.

    Key Features:
    - BEV and PHEV support
    - Location and market-based Scope 2
    - Charging loss calculations
    - Renewable energy credit handling
    - Country-specific grid factors

    Zero-Hallucination Guarantee:
    - All calculations use deterministic formulas
    - No LLM calls in the calculation path
    - Full audit trail with SHA-256 provenance
    """

    AGENT_ID = "GL-MRV-TRN-006"
    AGENT_NAME = "EV Fleet MRV Agent"
    AGENT_VERSION = "1.0.0"
    TRANSPORT_MODE = TransportMode.ROAD
    DEFAULT_SCOPE = EmissionScope.SCOPE_2

    def calculate(self, input_data: EVFleetInput) -> EVFleetOutput:
        """
        Calculate EV fleet emissions.

        Args:
            input_data: EV fleet input data

        Returns:
            Complete calculation result with audit trail
        """
        start_time = datetime.utcnow()
        steps: List[CalculationStep] = []
        emission_factors: List[EmissionFactor] = []
        warnings: List[str] = []

        # Get grid emission factor
        grid_factor = self._get_grid_factor(input_data)

        # Initialize totals
        total_emissions_kg = Decimal("0")
        total_electricity_kwh = Decimal("0")
        total_distance_km = Decimal("0")
        total_charging_losses = Decimal("0")
        scope_2_emissions = Decimal("0")
        scope_3_emissions = Decimal("0")
        emissions_by_type: Dict[str, Decimal] = {}

        steps.append(CalculationStep(
            step_number=1,
            description="Initialize EV fleet emissions calculation",
            inputs={
                "organization_id": input_data.organization_id,
                "reporting_year": input_data.reporting_year,
                "num_vehicles": len(input_data.vehicles),
                "grid_factor": str(grid_factor),
                "country": input_data.country_code,
                "emission_method": input_data.emission_method.value if hasattr(input_data.emission_method, 'value') else str(input_data.emission_method),
            },
        ))

        # Process individual vehicles
        for vehicle in input_data.vehicles:
            result = self._calculate_vehicle_emissions(
                vehicle=vehicle,
                grid_factor=grid_factor,
                include_losses=input_data.include_charging_losses,
                default_charging=input_data.default_charging_type,
                step_offset=len(steps),
            )

            steps.extend(result["steps"])
            emission_factors.extend(result["factors"])

            total_emissions_kg += result["total_kg"]
            total_electricity_kwh += result["electricity_kwh"]
            total_distance_km += result["distance_km"]
            total_charging_losses += result["charging_losses_kwh"]

            # Track by vehicle type
            vtype = vehicle.vehicle_type.value if hasattr(vehicle.vehicle_type, 'value') else str(vehicle.vehicle_type)
            emissions_by_type[vtype] = emissions_by_type.get(
                vtype, Decimal("0")
            ) + result["total_kg"]

        # Process aggregated data
        if input_data.total_electricity_kwh and not input_data.vehicles:
            agg_result = self._calculate_aggregated_emissions(
                electricity_kwh=input_data.total_electricity_kwh,
                distance_km=input_data.total_distance_km,
                grid_factor=grid_factor,
                vehicle_type=input_data.default_vehicle_type,
                include_losses=input_data.include_charging_losses,
                charging_type=input_data.default_charging_type,
                step_offset=len(steps),
            )

            steps.extend(agg_result["steps"])
            total_emissions_kg += agg_result["total_kg"]
            total_electricity_kwh += input_data.total_electricity_kwh
            if input_data.total_distance_km:
                total_distance_km += input_data.total_distance_km
            total_charging_losses += agg_result["charging_losses_kwh"]

        # Apply renewable energy reduction
        avoided_emissions = Decimal("0")
        if input_data.renewable_energy_pct > 0:
            if input_data.has_renewable_certificates and input_data.emission_method == EmissionMethod.MARKET_BASED:
                # Full reduction for market-based with certificates
                avoided_emissions = total_emissions_kg * (input_data.renewable_energy_pct / Decimal("100"))
                total_emissions_kg -= avoided_emissions

                steps.append(CalculationStep(
                    step_number=len(steps) + 1,
                    description="Apply renewable energy reduction (market-based)",
                    formula="reduction = emissions x renewable_pct",
                    inputs={
                        "renewable_pct": str(input_data.renewable_energy_pct),
                        "has_certificates": True,
                    },
                    output=str(avoided_emissions),
                ))

        # Determine scope classification
        if input_data.is_owned_fleet:
            scope_2_emissions = total_emissions_kg
        else:
            scope_3_emissions = total_emissions_kg

        # Calculate efficiency metrics
        emissions_per_km = None
        emissions_per_kwh = None
        efficiency_kwh_per_km = None

        if total_distance_km > 0:
            emissions_per_km = (total_emissions_kg / total_distance_km).quantize(
                Decimal("0.00001"), rounding=ROUND_HALF_UP
            )
        if total_electricity_kwh > 0:
            emissions_per_kwh = (total_emissions_kg / total_electricity_kwh).quantize(
                Decimal("0.00001"), rounding=ROUND_HALF_UP
            )
        if total_distance_km > 0 and total_electricity_kwh > 0:
            efficiency_kwh_per_km = (total_electricity_kwh / total_distance_km).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )

        # Final summary
        steps.append(CalculationStep(
            step_number=len(steps) + 1,
            description="Aggregate total EV fleet emissions",
            inputs={
                "total_vehicles": len(input_data.vehicles),
                "total_electricity_kwh": str(total_electricity_kwh),
                "total_distance_km": str(total_distance_km),
            },
            output=str(total_emissions_kg),
        ))

        # Determine scope
        scope = EmissionScope.SCOPE_2 if input_data.is_owned_fleet else EmissionScope.SCOPE_3

        # Build activity summary
        activity_summary = {
            "organization_id": input_data.organization_id,
            "reporting_year": input_data.reporting_year,
            "transport_mode": "ev_fleet",
            "total_vehicles": len(input_data.vehicles),
            "total_electricity_kwh": str(total_electricity_kwh),
            "total_distance_km": str(total_distance_km),
            "grid_factor": str(grid_factor),
            "country": input_data.country_code,
        }

        # Create base output
        base_output = self._create_output(
            total_emissions_kg=total_emissions_kg,
            co2_kg=total_emissions_kg,
            ch4_kg=Decimal("0"),
            n2o_kg=Decimal("0"),
            steps=steps,
            emission_factors=emission_factors,
            activity_summary=activity_summary,
            start_time=start_time,
            scope=scope,
            warnings=warnings,
        )

        return EVFleetOutput(
            **base_output.dict(),
            total_vehicles=len(input_data.vehicles),
            total_electricity_kwh=total_electricity_kwh,
            total_distance_km=total_distance_km,
            scope_2_emissions_kg=scope_2_emissions,
            scope_3_emissions_kg=scope_3_emissions,
            emissions_per_km=emissions_per_km,
            emissions_per_kwh=emissions_per_kwh,
            efficiency_kwh_per_km=efficiency_kwh_per_km,
            renewable_energy_pct=input_data.renewable_energy_pct,
            avoided_emissions_kg=avoided_emissions,
            charging_losses_kwh=total_charging_losses,
            emissions_by_vehicle_type=emissions_by_type,
        )

    def _get_grid_factor(self, input_data: EVFleetInput) -> Decimal:
        """Get appropriate grid emission factor."""
        if input_data.grid_emission_factor_kg_per_kwh:
            return input_data.grid_emission_factor_kg_per_kwh
        return GRID_FACTORS_BY_COUNTRY.get(
            input_data.country_code,
            GRID_FACTORS_BY_COUNTRY["global"]
        )

    def _calculate_vehicle_emissions(
        self,
        vehicle: EVVehicleRecord,
        grid_factor: Decimal,
        include_losses: bool,
        default_charging: ChargingType,
        step_offset: int,
    ) -> Dict[str, Any]:
        """Calculate emissions for a single EV."""
        steps: List[CalculationStep] = []
        factors: List[EmissionFactor] = []

        vtype = vehicle.vehicle_type.value if hasattr(vehicle.vehicle_type, 'value') else str(vehicle.vehicle_type)

        # Calculate electricity consumption
        if vehicle.electricity_kwh:
            electricity_kwh = vehicle.electricity_kwh
        elif vehicle.distance_km:
            efficiency = EV_EFFICIENCY.get(vtype, EV_EFFICIENCY[EVType.BEV_CAR.value])
            electricity_kwh = (vehicle.distance_km * efficiency).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            )
        else:
            electricity_kwh = Decimal("0")

        # Calculate distance
        if vehicle.distance_km:
            distance_km = vehicle.distance_km
        elif vehicle.electricity_kwh:
            efficiency = EV_EFFICIENCY.get(vtype, EV_EFFICIENCY[EVType.BEV_CAR.value])
            distance_km = (vehicle.electricity_kwh / efficiency).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            )
        else:
            distance_km = Decimal("0")

        # Calculate charging losses
        charging_losses_kwh = Decimal("0")
        if include_losses:
            ctype = vehicle.charging_type.value if hasattr(vehicle.charging_type, 'value') else str(vehicle.charging_type)
            loss_rate = CHARGING_LOSSES.get(ctype, CHARGING_LOSSES[ChargingType.DEPOT_CHARGING.value])
            charging_losses_kwh = (electricity_kwh * loss_rate).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            )
            electricity_kwh = electricity_kwh + charging_losses_kwh

        # Calculate emissions
        total_kg = (electricity_kwh * grid_factor).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        ef_record = EmissionFactor(
            factor_id="grid_electricity",
            factor_value=grid_factor,
            factor_unit="kg CO2e/kWh",
            source=EmissionFactorSource.DEFRA,
            source_uri="https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2024",
            version="2024",
            last_updated="2024-06-01",
            data_quality_tier=DataQualityTier.TIER_2,
        )
        factors.append(ef_record)

        steps.append(CalculationStep(
            step_number=step_offset + 1,
            description=f"Calculate EV emissions: {vehicle.vehicle_id or vtype}",
            formula="emissions = electricity_kwh x grid_factor",
            inputs={
                "vehicle_type": vtype,
                "electricity_kwh": str(electricity_kwh),
                "distance_km": str(distance_km),
                "charging_losses_kwh": str(charging_losses_kwh),
                "grid_factor": str(grid_factor),
            },
            output=str(total_kg),
            emission_factor=ef_record,
        ))

        return {
            "total_kg": total_kg,
            "electricity_kwh": electricity_kwh,
            "distance_km": distance_km,
            "charging_losses_kwh": charging_losses_kwh,
            "steps": steps,
            "factors": factors,
        }

    def _calculate_aggregated_emissions(
        self,
        electricity_kwh: Decimal,
        distance_km: Optional[Decimal],
        grid_factor: Decimal,
        vehicle_type: EVType,
        include_losses: bool,
        charging_type: ChargingType,
        step_offset: int,
    ) -> Dict[str, Any]:
        """Calculate emissions from aggregated data."""
        steps: List[CalculationStep] = []

        # Calculate charging losses
        charging_losses_kwh = Decimal("0")
        total_electricity = electricity_kwh
        if include_losses:
            ctype = charging_type.value if hasattr(charging_type, 'value') else str(charging_type)
            loss_rate = CHARGING_LOSSES.get(ctype, CHARGING_LOSSES[ChargingType.DEPOT_CHARGING.value])
            charging_losses_kwh = (electricity_kwh * loss_rate).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            )
            total_electricity = electricity_kwh + charging_losses_kwh

        total_kg = (total_electricity * grid_factor).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        steps.append(CalculationStep(
            step_number=step_offset + 1,
            description="Calculate aggregated EV fleet emissions",
            formula="emissions = electricity_kwh x grid_factor",
            inputs={
                "total_electricity_kwh": str(total_electricity),
                "charging_losses_kwh": str(charging_losses_kwh),
                "grid_factor": str(grid_factor),
            },
            output=str(total_kg),
        ))

        return {
            "total_kg": total_kg,
            "charging_losses_kwh": charging_losses_kwh,
            "steps": steps,
        }
