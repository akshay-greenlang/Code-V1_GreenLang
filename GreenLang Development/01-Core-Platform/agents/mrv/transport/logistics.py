# -*- coding: utf-8 -*-
"""
GL-MRV-TRN-007: Logistics MRV Agent
===================================

This module implements the Logistics MRV Agent for measuring, reporting,
and verifying greenhouse gas emissions from logistics and supply chain operations.

Supported Features:
- Multi-modal logistics emissions
- Warehousing emissions
- Cross-docking operations
- Intermodal transport
- GLEC Framework compliance
- ISO 14083 alignment

Reference Standards:
- GLEC Framework 2.0
- ISO 14083:2023 (Logistics emissions)
- GHG Protocol Scope 3, Categories 4, 9
- DEFRA Conversion Factors 2024

Example:
    >>> agent = LogisticsMRVAgent()
    >>> input_data = LogisticsInput(
    ...     organization_id="ORG001",
    ...     reporting_year=2024,
    ...     shipments=[
    ...         LogisticsShipmentRecord(
    ...             origin="Rotterdam, NL",
    ...             destination="Munich, DE",
    ...             cargo_weight_tonnes=Decimal("20"),
    ...             transport_modes=[
    ...                 TransportLeg(mode="truck", distance_km=Decimal("800"))
    ...             ],
    ...         )
    ...     ]
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
# Logistics-Specific Enums
# =============================================================================

class LogisticsMode(str, Enum):
    """Transport modes for logistics."""
    TRUCK = "truck"
    TRUCK_LTL = "truck_ltl"  # Less than truckload
    TRUCK_FTL = "truck_ftl"  # Full truckload
    RAIL = "rail"
    OCEAN_CONTAINER = "ocean_container"
    OCEAN_BULK = "ocean_bulk"
    AIR_FREIGHT = "air_freight"
    BARGE = "barge"
    COURIER = "courier"
    INTERMODAL = "intermodal"


class WarehouseType(str, Enum):
    """Warehouse types for logistics."""
    AMBIENT = "ambient"
    TEMPERATURE_CONTROLLED = "temperature_controlled"
    COLD_STORAGE = "cold_storage"
    FROZEN = "frozen"
    CROSS_DOCK = "cross_dock"
    DISTRIBUTION_CENTER = "distribution_center"


# =============================================================================
# GLEC/DEFRA Emission Factors
# =============================================================================

# Freight emission factors (kg CO2e per tonne-km)
LOGISTICS_FREIGHT_FACTORS: Dict[str, Decimal] = {
    LogisticsMode.TRUCK.value: Decimal("0.10666"),
    LogisticsMode.TRUCK_LTL.value: Decimal("0.13332"),
    LogisticsMode.TRUCK_FTL.value: Decimal("0.08000"),
    LogisticsMode.RAIL.value: Decimal("0.02781"),
    LogisticsMode.OCEAN_CONTAINER.value: Decimal("0.01609"),
    LogisticsMode.OCEAN_BULK.value: Decimal("0.00476"),
    LogisticsMode.AIR_FREIGHT.value: Decimal("0.60284"),
    LogisticsMode.BARGE.value: Decimal("0.03214"),
    LogisticsMode.COURIER.value: Decimal("0.30000"),  # Express/courier premium
}

# Warehousing factors (kg CO2e per m2 per year)
WAREHOUSE_FACTORS: Dict[str, Decimal] = {
    WarehouseType.AMBIENT.value: Decimal("28.5"),
    WarehouseType.TEMPERATURE_CONTROLLED.value: Decimal("52.3"),
    WarehouseType.COLD_STORAGE.value: Decimal("95.2"),
    WarehouseType.FROZEN.value: Decimal("142.8"),
    WarehouseType.CROSS_DOCK.value: Decimal("18.2"),
    WarehouseType.DISTRIBUTION_CENTER.value: Decimal("35.4"),
}

# Handling/transhipment factors (kg CO2e per tonne handled)
HANDLING_FACTORS: Dict[str, Decimal] = {
    "port_container": Decimal("2.5"),
    "rail_terminal": Decimal("1.8"),
    "warehouse_standard": Decimal("3.2"),
    "cross_dock": Decimal("1.5"),
    "airport_cargo": Decimal("4.2"),
}


# =============================================================================
# Input Models
# =============================================================================

class TransportLeg(BaseModel):
    """Single leg of a multi-modal shipment."""

    mode: LogisticsMode = Field(..., description="Transport mode")
    distance_km: Decimal = Field(..., ge=0, description="Distance in km")
    vehicle_type: Optional[str] = Field(None, description="Specific vehicle type")
    fuel_type: Optional[str] = Field(None, description="Fuel type if known")
    load_factor: Optional[Decimal] = Field(
        None, ge=0, le=1, description="Load factor"
    )
    carrier: Optional[str] = Field(None, description="Carrier name")

    class Config:
        use_enum_values = True


class WarehouseOperation(BaseModel):
    """Warehouse/storage operation record."""

    warehouse_type: WarehouseType = Field(..., description="Warehouse type")
    area_m2: Decimal = Field(..., ge=0, description="Area in m2")
    duration_days: Decimal = Field(..., ge=0, description="Storage duration in days")
    throughput_tonnes: Optional[Decimal] = Field(
        None, ge=0, description="Tonnes handled"
    )

    class Config:
        use_enum_values = True


class LogisticsShipmentRecord(BaseModel):
    """Complete logistics shipment record."""

    # Shipment identification
    shipment_id: Optional[str] = Field(None, description="Shipment ID")

    # Route information
    origin: str = Field(..., description="Origin location")
    destination: str = Field(..., description="Destination location")

    # Cargo details
    cargo_weight_tonnes: Decimal = Field(..., ge=0, description="Cargo weight")
    cargo_volume_m3: Optional[Decimal] = Field(None, ge=0, description="Cargo volume")
    cargo_type: Optional[str] = Field(None, description="Cargo type/category")

    # Transport legs
    transport_modes: List[TransportLeg] = Field(
        default_factory=list, description="Transport legs"
    )

    # Aggregated distance (if legs not specified)
    total_distance_km: Optional[Decimal] = Field(
        None, ge=0, description="Total distance"
    )
    primary_mode: LogisticsMode = Field(
        LogisticsMode.TRUCK, description="Primary mode if legs not specified"
    )

    # Warehousing
    warehouse_operations: List[WarehouseOperation] = Field(
        default_factory=list, description="Warehouse operations"
    )

    # Handling
    handling_events: int = Field(0, ge=0, description="Number of handling events")

    class Config:
        use_enum_values = True


class LogisticsInput(TransportMRVInput):
    """Input model for Logistics MRV Agent."""

    # Shipment records
    shipments: List[LogisticsShipmentRecord] = Field(
        default_factory=list, description="Logistics shipment records"
    )

    # Aggregated data
    total_tonne_km: Optional[Decimal] = Field(
        None, ge=0, description="Total tonne-km"
    )
    default_mode: LogisticsMode = Field(
        LogisticsMode.TRUCK, description="Default transport mode"
    )

    # Warehousing totals
    total_warehouse_m2_days: Optional[Decimal] = Field(
        None, ge=0, description="Total m2-days of warehousing"
    )
    default_warehouse_type: WarehouseType = Field(
        WarehouseType.AMBIENT, description="Default warehouse type"
    )

    # Empty running
    include_empty_running: bool = Field(
        True, description="Include empty return trips"
    )
    empty_running_factor: Decimal = Field(
        Decimal("1.30"), ge=1, description="Empty running multiplier"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# Output Model
# =============================================================================

class LogisticsOutput(TransportMRVOutput):
    """Output model for Logistics MRV Agent."""

    # Logistics-specific metrics
    total_shipments: int = Field(0, ge=0, description="Total shipments")
    total_tonne_km: Decimal = Field(Decimal("0"), ge=0, description="Total tonne-km")
    total_distance_km: Decimal = Field(
        Decimal("0"), ge=0, description="Total distance"
    )
    total_cargo_tonnes: Decimal = Field(
        Decimal("0"), ge=0, description="Total cargo weight"
    )

    # Emission breakdown
    transport_emissions_kg: Decimal = Field(
        Decimal("0"), ge=0, description="Transport emissions"
    )
    warehousing_emissions_kg: Decimal = Field(
        Decimal("0"), ge=0, description="Warehousing emissions"
    )
    handling_emissions_kg: Decimal = Field(
        Decimal("0"), ge=0, description="Handling emissions"
    )

    # Efficiency
    emissions_per_tonne_km: Optional[Decimal] = Field(
        None, description="kg CO2e per tonne-km"
    )
    emissions_per_tonne: Optional[Decimal] = Field(
        None, description="kg CO2e per tonne shipped"
    )

    # Breakdown by mode
    emissions_by_mode: Dict[str, Decimal] = Field(
        default_factory=dict, description="Emissions by transport mode"
    )
    tonne_km_by_mode: Dict[str, Decimal] = Field(
        default_factory=dict, description="Tonne-km by transport mode"
    )


# =============================================================================
# Logistics MRV Agent
# =============================================================================

class LogisticsMRVAgent(BaseTransportMRVAgent):
    """
    GL-MRV-TRN-007: Logistics MRV Agent

    Calculates greenhouse gas emissions from logistics operations
    including transport, warehousing, and handling.

    Key Features:
    - Multi-modal transport emissions
    - Warehousing and storage emissions
    - Handling/transhipment emissions
    - GLEC Framework compliance
    - Empty running calculations

    Zero-Hallucination Guarantee:
    - All calculations use deterministic formulas
    - No LLM calls in the calculation path
    - Full audit trail with SHA-256 provenance
    """

    AGENT_ID = "GL-MRV-TRN-007"
    AGENT_NAME = "Logistics MRV Agent"
    AGENT_VERSION = "1.0.0"
    TRANSPORT_MODE = TransportMode.INTERMODAL
    DEFAULT_SCOPE = EmissionScope.SCOPE_3

    def calculate(self, input_data: LogisticsInput) -> LogisticsOutput:
        """
        Calculate logistics emissions.

        Args:
            input_data: Logistics input data

        Returns:
            Complete calculation result with audit trail
        """
        start_time = datetime.utcnow()
        steps: List[CalculationStep] = []
        emission_factors: List[EmissionFactor] = []
        warnings: List[str] = []

        # Initialize totals
        total_emissions_kg = Decimal("0")
        total_tonne_km = Decimal("0")
        total_distance_km = Decimal("0")
        total_cargo_tonnes = Decimal("0")
        transport_emissions = Decimal("0")
        warehousing_emissions = Decimal("0")
        handling_emissions = Decimal("0")
        emissions_by_mode: Dict[str, Decimal] = {}
        tonne_km_by_mode: Dict[str, Decimal] = {}

        steps.append(CalculationStep(
            step_number=1,
            description="Initialize logistics emissions calculation",
            inputs={
                "organization_id": input_data.organization_id,
                "reporting_year": input_data.reporting_year,
                "num_shipments": len(input_data.shipments),
                "empty_running_factor": str(input_data.empty_running_factor),
            },
        ))

        # Process individual shipments
        for shipment in input_data.shipments:
            result = self._calculate_shipment_emissions(
                shipment=shipment,
                include_empty_running=input_data.include_empty_running,
                empty_factor=input_data.empty_running_factor,
                step_offset=len(steps),
            )

            steps.extend(result["steps"])
            emission_factors.extend(result["factors"])

            total_emissions_kg += result["total_kg"]
            total_tonne_km += result["tonne_km"]
            total_distance_km += result["distance_km"]
            total_cargo_tonnes += shipment.cargo_weight_tonnes
            transport_emissions += result["transport_kg"]
            warehousing_emissions += result["warehousing_kg"]
            handling_emissions += result["handling_kg"]

            # Track by mode
            for mode, tkm in result["tonne_km_by_mode"].items():
                tonne_km_by_mode[mode] = tonne_km_by_mode.get(mode, Decimal("0")) + tkm
            for mode, em in result["emissions_by_mode"].items():
                emissions_by_mode[mode] = emissions_by_mode.get(mode, Decimal("0")) + em

        # Process aggregated data
        if input_data.total_tonne_km and not input_data.shipments:
            mode = input_data.default_mode.value if hasattr(input_data.default_mode, 'value') else str(input_data.default_mode)
            factor = LOGISTICS_FREIGHT_FACTORS.get(
                mode, LOGISTICS_FREIGHT_FACTORS[LogisticsMode.TRUCK.value]
            )

            # Apply empty running
            adjusted_tonne_km = input_data.total_tonne_km
            if input_data.include_empty_running:
                adjusted_tonne_km = input_data.total_tonne_km * input_data.empty_running_factor

            agg_transport = (adjusted_tonne_km * factor).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )

            total_emissions_kg += agg_transport
            transport_emissions += agg_transport
            total_tonne_km += input_data.total_tonne_km

            steps.append(CalculationStep(
                step_number=len(steps) + 1,
                description="Calculate aggregated transport emissions",
                formula="emissions = tonne_km x empty_factor x EF",
                inputs={
                    "total_tonne_km": str(input_data.total_tonne_km),
                    "mode": mode,
                    "emission_factor": str(factor),
                    "empty_running_factor": str(input_data.empty_running_factor),
                },
                output=str(agg_transport),
            ))

        # Process aggregated warehousing
        if input_data.total_warehouse_m2_days:
            wtype = input_data.default_warehouse_type.value if hasattr(input_data.default_warehouse_type, 'value') else str(input_data.default_warehouse_type)
            wh_factor = WAREHOUSE_FACTORS.get(
                wtype, WAREHOUSE_FACTORS[WarehouseType.AMBIENT.value]
            )
            # Convert from per year to per day
            daily_factor = wh_factor / Decimal("365")
            agg_warehousing = (input_data.total_warehouse_m2_days * daily_factor).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )

            total_emissions_kg += agg_warehousing
            warehousing_emissions += agg_warehousing

            steps.append(CalculationStep(
                step_number=len(steps) + 1,
                description="Calculate aggregated warehousing emissions",
                inputs={
                    "total_m2_days": str(input_data.total_warehouse_m2_days),
                    "warehouse_type": wtype,
                },
                output=str(agg_warehousing),
            ))

        # Calculate efficiency metrics
        emissions_per_tonne_km = None
        emissions_per_tonne = None

        if total_tonne_km > 0:
            emissions_per_tonne_km = (transport_emissions / total_tonne_km).quantize(
                Decimal("0.00001"), rounding=ROUND_HALF_UP
            )
        if total_cargo_tonnes > 0:
            emissions_per_tonne = (total_emissions_kg / total_cargo_tonnes).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )

        # Final summary
        steps.append(CalculationStep(
            step_number=len(steps) + 1,
            description="Aggregate total logistics emissions",
            formula="total = transport + warehousing + handling",
            inputs={
                "transport_emissions_kg": str(transport_emissions),
                "warehousing_emissions_kg": str(warehousing_emissions),
                "handling_emissions_kg": str(handling_emissions),
            },
            output=str(total_emissions_kg),
        ))

        # Build activity summary
        activity_summary = {
            "organization_id": input_data.organization_id,
            "reporting_year": input_data.reporting_year,
            "transport_mode": "logistics",
            "total_shipments": len(input_data.shipments),
            "total_tonne_km": str(total_tonne_km),
            "total_cargo_tonnes": str(total_cargo_tonnes),
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
            scope=EmissionScope.SCOPE_3,
            warnings=warnings,
        )

        return LogisticsOutput(
            **base_output.dict(),
            total_shipments=len(input_data.shipments),
            total_tonne_km=total_tonne_km,
            total_distance_km=total_distance_km,
            total_cargo_tonnes=total_cargo_tonnes,
            transport_emissions_kg=transport_emissions,
            warehousing_emissions_kg=warehousing_emissions,
            handling_emissions_kg=handling_emissions,
            emissions_per_tonne_km=emissions_per_tonne_km,
            emissions_per_tonne=emissions_per_tonne,
            emissions_by_mode=emissions_by_mode,
            tonne_km_by_mode=tonne_km_by_mode,
        )

    def _calculate_shipment_emissions(
        self,
        shipment: LogisticsShipmentRecord,
        include_empty_running: bool,
        empty_factor: Decimal,
        step_offset: int,
    ) -> Dict[str, Any]:
        """Calculate emissions for a single shipment."""
        steps: List[CalculationStep] = []
        factors: List[EmissionFactor] = []

        transport_kg = Decimal("0")
        warehousing_kg = Decimal("0")
        handling_kg = Decimal("0")
        total_tonne_km = Decimal("0")
        total_distance = Decimal("0")
        emissions_by_mode: Dict[str, Decimal] = {}
        tonne_km_by_mode: Dict[str, Decimal] = {}

        # Calculate transport emissions from legs
        if shipment.transport_modes:
            for leg in shipment.transport_modes:
                mode = leg.mode.value if hasattr(leg.mode, 'value') else str(leg.mode)
                factor = LOGISTICS_FREIGHT_FACTORS.get(
                    mode, LOGISTICS_FREIGHT_FACTORS[LogisticsMode.TRUCK.value]
                )

                leg_tonne_km = leg.distance_km * shipment.cargo_weight_tonnes

                # Apply empty running
                if include_empty_running:
                    leg_tonne_km = leg_tonne_km * empty_factor

                leg_emissions = (leg_tonne_km * factor).quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP
                )

                transport_kg += leg_emissions
                total_tonne_km += leg.distance_km * shipment.cargo_weight_tonnes
                total_distance += leg.distance_km
                tonne_km_by_mode[mode] = tonne_km_by_mode.get(
                    mode, Decimal("0")
                ) + leg.distance_km * shipment.cargo_weight_tonnes
                emissions_by_mode[mode] = emissions_by_mode.get(
                    mode, Decimal("0")
                ) + leg_emissions

        # Use aggregated distance if no legs
        elif shipment.total_distance_km:
            mode = shipment.primary_mode.value if hasattr(shipment.primary_mode, 'value') else str(shipment.primary_mode)
            factor = LOGISTICS_FREIGHT_FACTORS.get(
                mode, LOGISTICS_FREIGHT_FACTORS[LogisticsMode.TRUCK.value]
            )

            leg_tonne_km = shipment.total_distance_km * shipment.cargo_weight_tonnes
            if include_empty_running:
                leg_tonne_km = leg_tonne_km * empty_factor

            transport_kg = (leg_tonne_km * factor).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
            total_tonne_km = shipment.total_distance_km * shipment.cargo_weight_tonnes
            total_distance = shipment.total_distance_km
            tonne_km_by_mode[mode] = total_tonne_km
            emissions_by_mode[mode] = transport_kg

        steps.append(CalculationStep(
            step_number=step_offset + 1,
            description=f"Calculate transport emissions: {shipment.origin} to {shipment.destination}",
            formula="emissions = SUM(leg_tonne_km x EF)",
            inputs={
                "shipment_id": shipment.shipment_id or "N/A",
                "cargo_tonnes": str(shipment.cargo_weight_tonnes),
                "total_distance_km": str(total_distance),
                "total_tonne_km": str(total_tonne_km),
            },
            output=str(transport_kg),
        ))

        # Calculate warehousing emissions
        for wh_op in shipment.warehouse_operations:
            wtype = wh_op.warehouse_type.value if hasattr(wh_op.warehouse_type, 'value') else str(wh_op.warehouse_type)
            wh_factor = WAREHOUSE_FACTORS.get(
                wtype, WAREHOUSE_FACTORS[WarehouseType.AMBIENT.value]
            )
            # Convert from per year to per day
            daily_factor = wh_factor / Decimal("365")
            m2_days = wh_op.area_m2 * wh_op.duration_days
            wh_emissions = (m2_days * daily_factor).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
            warehousing_kg += wh_emissions

        if warehousing_kg > 0:
            steps.append(CalculationStep(
                step_number=step_offset + 2,
                description="Calculate warehousing emissions",
                output=str(warehousing_kg),
            ))

        # Calculate handling emissions
        if shipment.handling_events > 0:
            handling_factor = HANDLING_FACTORS["warehouse_standard"]
            handling_kg = (
                Decimal(str(shipment.handling_events)) *
                shipment.cargo_weight_tonnes *
                handling_factor
            ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

            steps.append(CalculationStep(
                step_number=step_offset + 3,
                description="Calculate handling emissions",
                inputs={
                    "handling_events": shipment.handling_events,
                    "cargo_tonnes": str(shipment.cargo_weight_tonnes),
                },
                output=str(handling_kg),
            ))

        total_kg = transport_kg + warehousing_kg + handling_kg

        return {
            "total_kg": total_kg,
            "transport_kg": transport_kg,
            "warehousing_kg": warehousing_kg,
            "handling_kg": handling_kg,
            "tonne_km": total_tonne_km,
            "distance_km": total_distance,
            "emissions_by_mode": emissions_by_mode,
            "tonne_km_by_mode": tonne_km_by_mode,
            "steps": steps,
            "factors": factors,
        }
