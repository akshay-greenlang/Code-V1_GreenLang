# -*- coding: utf-8 -*-
"""
GL-MRV-TRN-003: Maritime MRV Agent
==================================

This module implements the Maritime MRV Agent for measuring, reporting,
and verifying greenhouse gas emissions from maritime shipping activities.

Supported Features:
- Container ship emissions
- Bulk carrier emissions
- Tanker emissions
- Ro-Ro vessel emissions
- Fuel-based method (direct consumption)
- Distance-based method (tonne-km)
- EU MRV and IMO DCS compliance
- Well-to-wake emissions

Reference Standards:
- IMO Data Collection System (DCS)
- EU MRV Regulation 2015/757
- GHG Protocol Scope 3, Categories 4, 9
- DEFRA Conversion Factors 2024
- IMO Fourth GHG Study (2020)

Example:
    >>> agent = MaritimeMRVAgent()
    >>> input_data = MaritimeInput(
    ...     organization_id="ORG001",
    ...     reporting_year=2024,
    ...     voyages=[
    ...         VoyageRecord(
    ...             vessel_type=VesselType.CONTAINER_SHIP,
    ...             origin_port="NLRTM",
    ...             destination_port="CNSHA",
    ...             cargo_tonnes=Decimal("10000"),
    ...             fuel_consumed_tonnes=Decimal("500"),
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

from pydantic import BaseModel, Field, model_validator

from greenlang.agents.mrv.transport.base import (
    BaseTransportMRVAgent,
    TransportMRVInput,
    TransportMRVOutput,
    TransportMode,
    FuelType,
    EmissionScope,
    CalculationMethod,
    DataQualityTier,
    EmissionFactor,
    EmissionFactorSource,
    CalculationStep,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Maritime-Specific Enums
# =============================================================================

class VesselType(str, Enum):
    """Types of maritime vessels."""
    CONTAINER_SHIP = "container_ship"
    CONTAINER_SMALL = "container_small"  # <1000 TEU
    CONTAINER_MEDIUM = "container_medium"  # 1000-5000 TEU
    CONTAINER_LARGE = "container_large"  # >5000 TEU
    BULK_CARRIER = "bulk_carrier"
    BULK_SMALL = "bulk_small"  # <10000 DWT
    BULK_MEDIUM = "bulk_medium"  # 10000-60000 DWT
    BULK_LARGE = "bulk_large"  # >60000 DWT
    TANKER = "tanker"
    TANKER_CRUDE = "tanker_crude"
    TANKER_PRODUCT = "tanker_product"
    TANKER_CHEMICAL = "tanker_chemical"
    TANKER_LNG = "tanker_lng"
    RORO = "roro"
    RORO_VEHICLE = "roro_vehicle"
    RORO_PASSEN = "roro_passenger"
    GENERAL_CARGO = "general_cargo"
    REFRIGERATED = "refrigerated"
    FERRY = "ferry"
    CRUISE = "cruise"


class MarineFuelType(str, Enum):
    """Marine fuel types."""
    HFO = "hfo"  # Heavy Fuel Oil
    VLSFO = "vlsfo"  # Very Low Sulphur Fuel Oil
    MGO = "mgo"  # Marine Gas Oil
    MDO = "mdo"  # Marine Diesel Oil
    LNG = "lng"  # Liquefied Natural Gas
    METHANOL = "methanol"
    AMMONIA = "ammonia"
    BIOFUEL = "biofuel"


# =============================================================================
# DEFRA 2024 Maritime Emission Factors
# =============================================================================

# Fuel-based factors (kg CO2e per tonne of fuel)
MARINE_FUEL_FACTORS: Dict[str, Dict[str, Decimal]] = {
    MarineFuelType.HFO.value: {
        "co2_per_tonne": Decimal("3114"),
        "ch4_per_tonne": Decimal("0.12"),
        "n2o_per_tonne": Decimal("0.58"),
        "total_per_tonne": Decimal("3114.70"),
    },
    MarineFuelType.VLSFO.value: {
        "co2_per_tonne": Decimal("3151"),
        "ch4_per_tonne": Decimal("0.12"),
        "n2o_per_tonne": Decimal("0.58"),
        "total_per_tonne": Decimal("3151.70"),
    },
    MarineFuelType.MGO.value: {
        "co2_per_tonne": Decimal("3206"),
        "ch4_per_tonne": Decimal("0.12"),
        "n2o_per_tonne": Decimal("0.58"),
        "total_per_tonne": Decimal("3206.70"),
    },
    MarineFuelType.MDO.value: {
        "co2_per_tonne": Decimal("3206"),
        "ch4_per_tonne": Decimal("0.12"),
        "n2o_per_tonne": Decimal("0.58"),
        "total_per_tonne": Decimal("3206.70"),
    },
    MarineFuelType.LNG.value: {
        "co2_per_tonne": Decimal("2750"),
        "ch4_per_tonne": Decimal("50"),  # Methane slip
        "n2o_per_tonne": Decimal("0.11"),
        "total_per_tonne": Decimal("2800.11"),
    },
}

# Distance-based factors (kg CO2e per tonne-km)
MARINE_TONNE_KM_FACTORS: Dict[str, Decimal] = {
    VesselType.CONTAINER_SHIP.value: Decimal("0.01609"),
    VesselType.CONTAINER_SMALL.value: Decimal("0.02891"),
    VesselType.CONTAINER_MEDIUM.value: Decimal("0.01609"),
    VesselType.CONTAINER_LARGE.value: Decimal("0.00823"),
    VesselType.BULK_CARRIER.value: Decimal("0.00476"),
    VesselType.BULK_SMALL.value: Decimal("0.01011"),
    VesselType.BULK_MEDIUM.value: Decimal("0.00519"),
    VesselType.BULK_LARGE.value: Decimal("0.00325"),
    VesselType.TANKER.value: Decimal("0.00761"),
    VesselType.TANKER_CRUDE.value: Decimal("0.00432"),
    VesselType.TANKER_PRODUCT.value: Decimal("0.00854"),
    VesselType.TANKER_CHEMICAL.value: Decimal("0.01087"),
    VesselType.TANKER_LNG.value: Decimal("0.01219"),
    VesselType.RORO.value: Decimal("0.05095"),
    VesselType.GENERAL_CARGO.value: Decimal("0.01842"),
    VesselType.REFRIGERATED.value: Decimal("0.01285"),
    VesselType.FERRY.value: Decimal("0.11559"),
}

# Port-to-port distances (sample, in nautical miles)
PORT_DISTANCES: Dict[str, Dict[str, int]] = {
    "NLRTM": {  # Rotterdam
        "CNSHA": 10509,  # Shanghai
        "SGSIN": 8261,  # Singapore
        "USNYC": 3458,  # New York
        "GBFXT": 168,  # Felixstowe
        "DEHAM": 254,  # Hamburg
    },
    "SGSIN": {  # Singapore
        "CNSHA": 2242,
        "JPTYO": 2890,
        "AESYD": 5325,
    },
}


# =============================================================================
# Input Models
# =============================================================================

class VoyageRecord(BaseModel):
    """Individual voyage record for maritime transport."""

    # Voyage identification
    voyage_id: Optional[str] = Field(None, description="Unique voyage identifier")

    # Vessel details
    vessel_type: VesselType = Field(..., description="Type of vessel")
    vessel_name: Optional[str] = Field(None, description="Vessel name")
    imo_number: Optional[str] = Field(None, description="IMO number")
    vessel_dwt: Optional[Decimal] = Field(
        None, ge=0, description="Deadweight tonnage"
    )
    vessel_teu: Optional[int] = Field(
        None, ge=0, description="TEU capacity (container ships)"
    )

    # Route information
    origin_port: str = Field(..., description="Origin port (UN/LOCODE)")
    destination_port: str = Field(..., description="Destination port (UN/LOCODE)")
    distance_nm: Optional[Decimal] = Field(
        None, ge=0, description="Distance in nautical miles"
    )
    distance_km: Optional[Decimal] = Field(
        None, ge=0, description="Distance in kilometers"
    )

    # Cargo details
    cargo_tonnes: Optional[Decimal] = Field(
        None, ge=0, description="Cargo weight in tonnes"
    )
    cargo_teu: Optional[int] = Field(
        None, ge=0, description="Cargo in TEU (containers)"
    )
    load_factor: Optional[Decimal] = Field(
        None, ge=0, le=1, description="Load factor (0-1)"
    )

    # Fuel consumption
    fuel_type: MarineFuelType = Field(
        MarineFuelType.VLSFO, description="Primary fuel type"
    )
    fuel_consumed_tonnes: Optional[Decimal] = Field(
        None, ge=0, description="Fuel consumed in tonnes"
    )
    fuel_consumed_liters: Optional[Decimal] = Field(
        None, ge=0, description="Fuel consumed in liters"
    )

    # Voyage details
    voyage_duration_days: Optional[Decimal] = Field(
        None, ge=0, description="Voyage duration in days"
    )
    average_speed_knots: Optional[Decimal] = Field(
        None, ge=0, description="Average speed in knots"
    )

    # Port operations
    port_time_hours: Optional[Decimal] = Field(
        None, ge=0, description="Time at port in hours"
    )
    port_fuel_tonnes: Optional[Decimal] = Field(
        None, ge=0, description="Fuel consumed at port"
    )

    class Config:
        use_enum_values = True

    @model_validator(mode="before")
    @classmethod
    def convert_nm_to_km(cls, data: Any) -> Any:
        """Convert nautical miles to kilometers if not provided."""
        if isinstance(data, dict):
            if data.get("distance_km") is None and data.get("distance_nm"):
                # 1 nautical mile = 1.852 km
                data["distance_km"] = Decimal(str(data["distance_nm"])) * Decimal("1.852")
        return data


class MaritimeInput(TransportMRVInput):
    """Input model for Maritime MRV Agent."""

    # Voyage records
    voyages: List[VoyageRecord] = Field(
        default_factory=list, description="List of voyage records"
    )

    # Aggregated data (alternative to individual voyages)
    total_tonne_km: Optional[Decimal] = Field(
        None, ge=0, description="Total tonne-km transported"
    )
    default_vessel_type: VesselType = Field(
        VesselType.CONTAINER_SHIP, description="Default vessel type"
    )

    # Fleet summary
    owned_vessels: bool = Field(
        False, description="Whether vessels are owned (Scope 1)"
    )

    # EU MRV specific
    eu_mrv_reporting: bool = Field(
        False, description="EU MRV regulation reporting"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# Output Model
# =============================================================================

class MaritimeOutput(TransportMRVOutput):
    """Output model for Maritime MRV Agent."""

    # Maritime-specific metrics
    total_voyages: int = Field(0, ge=0, description="Total number of voyages")
    total_distance_nm: Decimal = Field(
        Decimal("0"), ge=0, description="Total distance in nautical miles"
    )
    total_distance_km: Decimal = Field(
        Decimal("0"), ge=0, description="Total distance in kilometers"
    )
    total_cargo_tonnes: Decimal = Field(
        Decimal("0"), ge=0, description="Total cargo transported"
    )
    total_tonne_km: Decimal = Field(
        Decimal("0"), ge=0, description="Total tonne-km"
    )
    total_fuel_tonnes: Decimal = Field(
        Decimal("0"), ge=0, description="Total fuel consumed"
    )

    # Efficiency metrics
    emissions_per_tonne_km: Optional[Decimal] = Field(
        None, description="kg CO2e per tonne-km"
    )
    fuel_per_tonne_km: Optional[Decimal] = Field(
        None, description="kg fuel per tonne-km"
    )
    eeoi: Optional[Decimal] = Field(
        None, description="Energy Efficiency Operational Indicator"
    )

    # Breakdown by vessel type
    emissions_by_vessel_type: Dict[str, Decimal] = Field(
        default_factory=dict, description="Emissions by vessel type"
    )

    # Breakdown by fuel type
    emissions_by_fuel_type: Dict[str, Decimal] = Field(
        default_factory=dict, description="Emissions by fuel type"
    )


# =============================================================================
# Maritime MRV Agent
# =============================================================================

class MaritimeMRVAgent(BaseTransportMRVAgent):
    """
    GL-MRV-TRN-003: Maritime MRV Agent

    Calculates greenhouse gas emissions from maritime shipping activities
    including container ships, bulk carriers, tankers, and other vessels.

    Key Features:
    - Fuel-based calculation (IMO DCS method)
    - Distance-based calculation (tonne-km method)
    - Multiple vessel types with specific factors
    - EU MRV and IMO DCS compliance
    - EEOI calculation

    Zero-Hallucination Guarantee:
    - All calculations use deterministic formulas
    - No LLM calls in the calculation path
    - Full audit trail with SHA-256 provenance
    """

    AGENT_ID = "GL-MRV-TRN-003"
    AGENT_NAME = "Maritime MRV Agent"
    AGENT_VERSION = "1.0.0"
    TRANSPORT_MODE = TransportMode.MARITIME
    DEFAULT_SCOPE = EmissionScope.SCOPE_3

    def __init__(self):
        """Initialize Maritime MRV Agent."""
        super().__init__()
        self._fuel_factors = MARINE_FUEL_FACTORS
        self._tonne_km_factors = MARINE_TONNE_KM_FACTORS

    def calculate(self, input_data: MaritimeInput) -> MaritimeOutput:
        """
        Calculate maritime emissions.

        Args:
            input_data: Maritime input data

        Returns:
            Complete calculation result with audit trail
        """
        start_time = datetime.utcnow()
        steps: List[CalculationStep] = []
        emission_factors: List[EmissionFactor] = []
        warnings: List[str] = []

        # Initialize totals
        total_emissions_kg = Decimal("0")
        total_co2_kg = Decimal("0")
        total_ch4_kg = Decimal("0")
        total_n2o_kg = Decimal("0")
        total_distance_nm = Decimal("0")
        total_distance_km = Decimal("0")
        total_cargo_tonnes = Decimal("0")
        total_tonne_km = Decimal("0")
        total_fuel_tonnes = Decimal("0")
        emissions_by_vessel: Dict[str, Decimal] = {}
        emissions_by_fuel: Dict[str, Decimal] = {}

        steps.append(CalculationStep(
            step_number=1,
            description="Initialize maritime emissions calculation",
            inputs={
                "organization_id": input_data.organization_id,
                "reporting_year": input_data.reporting_year,
                "num_voyages": len(input_data.voyages),
                "calculation_method": input_data.calculation_method,
            },
        ))

        # Process individual voyages
        for idx, voyage in enumerate(input_data.voyages):
            voyage_result = self._calculate_voyage_emissions(
                voyage=voyage,
                method=input_data.calculation_method,
                step_offset=len(steps),
            )

            steps.extend(voyage_result["steps"])
            emission_factors.extend(voyage_result["factors"])

            # Accumulate totals
            total_emissions_kg += voyage_result["total_kg"]
            total_co2_kg += voyage_result["co2_kg"]
            total_ch4_kg += voyage_result["ch4_kg"]
            total_n2o_kg += voyage_result["n2o_kg"]

            if voyage.distance_nm:
                total_distance_nm += voyage.distance_nm
            if voyage_result["distance_km"] > 0:
                total_distance_km += voyage_result["distance_km"]
            if voyage.cargo_tonnes:
                total_cargo_tonnes += voyage.cargo_tonnes
            total_tonne_km += voyage_result["tonne_km"]
            if voyage.fuel_consumed_tonnes:
                total_fuel_tonnes += voyage.fuel_consumed_tonnes

            # Track by vessel type
            vtype = voyage.vessel_type.value if hasattr(voyage.vessel_type, 'value') else str(voyage.vessel_type)
            emissions_by_vessel[vtype] = emissions_by_vessel.get(
                vtype, Decimal("0")
            ) + voyage_result["total_kg"]

            # Track by fuel type
            ftype = voyage.fuel_type.value if hasattr(voyage.fuel_type, 'value') else str(voyage.fuel_type)
            emissions_by_fuel[ftype] = emissions_by_fuel.get(
                ftype, Decimal("0")
            ) + voyage_result["total_kg"]

        # Process aggregated data
        if input_data.total_tonne_km and not input_data.voyages:
            agg_result = self._calculate_aggregated_emissions(
                tonne_km=input_data.total_tonne_km,
                vessel_type=input_data.default_vessel_type,
                step_offset=len(steps),
            )
            steps.extend(agg_result["steps"])
            total_emissions_kg += agg_result["total_kg"]
            total_co2_kg += agg_result["total_kg"]
            total_tonne_km += input_data.total_tonne_km

        # Calculate efficiency metrics
        emissions_per_tonne_km = None
        fuel_per_tonne_km = None
        eeoi = None

        if total_tonne_km > 0:
            emissions_per_tonne_km = (total_emissions_kg / total_tonne_km).quantize(
                Decimal("0.00001"), rounding=ROUND_HALF_UP
            )
            if total_fuel_tonnes > 0:
                fuel_per_tonne_km = (total_fuel_tonnes / total_tonne_km * Decimal("1000000")).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )  # g fuel per tonne-km
                # EEOI = (Fuel x Carbon Factor) / (Cargo x Distance)
                eeoi = (total_co2_kg / total_tonne_km).quantize(
                    Decimal("0.00001"), rounding=ROUND_HALF_UP
                )

        # Final summary step
        steps.append(CalculationStep(
            step_number=len(steps) + 1,
            description="Aggregate total maritime emissions",
            inputs={
                "total_voyages": len(input_data.voyages),
                "total_distance_nm": str(total_distance_nm),
                "total_cargo_tonnes": str(total_cargo_tonnes),
                "total_fuel_tonnes": str(total_fuel_tonnes),
            },
            output=str(total_emissions_kg),
        ))

        # Determine scope
        scope = EmissionScope.SCOPE_1 if input_data.owned_vessels else EmissionScope.SCOPE_3

        # Build activity summary
        activity_summary = {
            "organization_id": input_data.organization_id,
            "reporting_year": input_data.reporting_year,
            "transport_mode": "maritime",
            "total_voyages": len(input_data.voyages),
            "total_distance_nm": str(total_distance_nm),
            "total_cargo_tonnes": str(total_cargo_tonnes),
            "total_fuel_tonnes": str(total_fuel_tonnes),
            "owned_vessels": input_data.owned_vessels,
        }

        # Create base output
        base_output = self._create_output(
            total_emissions_kg=total_emissions_kg,
            co2_kg=total_co2_kg,
            ch4_kg=total_ch4_kg,
            n2o_kg=total_n2o_kg,
            steps=steps,
            emission_factors=emission_factors,
            activity_summary=activity_summary,
            start_time=start_time,
            scope=scope,
            warnings=warnings,
        )

        return MaritimeOutput(
            **base_output.dict(),
            total_voyages=len(input_data.voyages),
            total_distance_nm=total_distance_nm,
            total_distance_km=total_distance_km,
            total_cargo_tonnes=total_cargo_tonnes,
            total_tonne_km=total_tonne_km,
            total_fuel_tonnes=total_fuel_tonnes,
            emissions_per_tonne_km=emissions_per_tonne_km,
            fuel_per_tonne_km=fuel_per_tonne_km,
            eeoi=eeoi,
            emissions_by_vessel_type=emissions_by_vessel,
            emissions_by_fuel_type=emissions_by_fuel,
        )

    def _calculate_voyage_emissions(
        self,
        voyage: VoyageRecord,
        method: CalculationMethod,
        step_offset: int,
    ) -> Dict[str, Any]:
        """
        Calculate emissions for a single voyage.

        Args:
            voyage: Voyage record
            method: Calculation method
            step_offset: Step number offset

        Returns:
            Dictionary with emissions and calculation details
        """
        steps: List[CalculationStep] = []
        factors: List[EmissionFactor] = []

        total_kg = Decimal("0")
        co2_kg = Decimal("0")
        ch4_kg = Decimal("0")
        n2o_kg = Decimal("0")
        tonne_km = Decimal("0")
        distance_km = Decimal("0")

        # Prefer fuel-based if available
        if voyage.fuel_consumed_tonnes and (
            method == CalculationMethod.FUEL_BASED or
            method == CalculationMethod.ACTIVITY_BASED
        ):
            fuel_data = self._fuel_factors.get(
                voyage.fuel_type.value if hasattr(voyage.fuel_type, 'value') else str(voyage.fuel_type),
                self._fuel_factors[MarineFuelType.VLSFO.value]
            )

            total_kg = (voyage.fuel_consumed_tonnes * fuel_data["total_per_tonne"]).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
            co2_kg = (voyage.fuel_consumed_tonnes * fuel_data["co2_per_tonne"]).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
            ch4_kg = (voyage.fuel_consumed_tonnes * fuel_data["ch4_per_tonne"]).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
            n2o_kg = (voyage.fuel_consumed_tonnes * fuel_data["n2o_per_tonne"]).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )

            # Create emission factor record
            ftype = voyage.fuel_type.value if hasattr(voyage.fuel_type, 'value') else str(voyage.fuel_type)
            ef_record = EmissionFactor(
                factor_id=f"defra_2024_marine_{ftype}",
                factor_value=fuel_data["total_per_tonne"],
                factor_unit="kg CO2e/tonne fuel",
                source=EmissionFactorSource.DEFRA,
                source_uri="https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2024",
                version="2024",
                last_updated="2024-06-01",
                uncertainty_pct=5.0,
                data_quality_tier=DataQualityTier.TIER_1,
            )
            factors.append(ef_record)

            steps.append(CalculationStep(
                step_number=step_offset + 1,
                description=f"Calculate fuel-based emissions: {voyage.origin_port} to {voyage.destination_port}",
                formula="emissions = fuel_tonnes x EF",
                inputs={
                    "voyage_id": voyage.voyage_id or "N/A",
                    "fuel_type": ftype,
                    "fuel_tonnes": str(voyage.fuel_consumed_tonnes),
                    "emission_factor": str(fuel_data["total_per_tonne"]),
                },
                output=str(total_kg),
                emission_factor=ef_record,
            ))

            # Calculate tonne-km for metrics
            if voyage.distance_km and voyage.cargo_tonnes:
                distance_km = voyage.distance_km
                tonne_km = voyage.distance_km * voyage.cargo_tonnes

        # Fall back to distance-based
        elif voyage.distance_km and voyage.cargo_tonnes:
            distance_km = voyage.distance_km
            tonne_km = voyage.distance_km * voyage.cargo_tonnes

            vtype = voyage.vessel_type.value if hasattr(voyage.vessel_type, 'value') else str(voyage.vessel_type)
            factor = self._tonne_km_factors.get(
                vtype,
                self._tonne_km_factors[VesselType.CONTAINER_SHIP.value]
            )

            total_kg = (tonne_km * factor).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
            co2_kg = total_kg

            ef_record = EmissionFactor(
                factor_id=f"defra_2024_marine_{vtype}",
                factor_value=factor,
                factor_unit="kg CO2e/tonne-km",
                source=EmissionFactorSource.DEFRA,
                source_uri="https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2024",
                version="2024",
                last_updated="2024-06-01",
                uncertainty_pct=15.0,
                data_quality_tier=DataQualityTier.TIER_2,
            )
            factors.append(ef_record)

            steps.append(CalculationStep(
                step_number=step_offset + 1,
                description=f"Calculate distance-based emissions: {voyage.origin_port} to {voyage.destination_port}",
                formula="emissions = tonne_km x EF",
                inputs={
                    "voyage_id": voyage.voyage_id or "N/A",
                    "vessel_type": vtype,
                    "distance_km": str(voyage.distance_km),
                    "cargo_tonnes": str(voyage.cargo_tonnes),
                    "tonne_km": str(tonne_km),
                    "emission_factor": str(factor),
                },
                output=str(total_kg),
                emission_factor=ef_record,
            ))

        return {
            "total_kg": total_kg,
            "co2_kg": co2_kg,
            "ch4_kg": ch4_kg,
            "n2o_kg": n2o_kg,
            "tonne_km": tonne_km,
            "distance_km": distance_km,
            "steps": steps,
            "factors": factors,
        }

    def _calculate_aggregated_emissions(
        self,
        tonne_km: Decimal,
        vessel_type: VesselType,
        step_offset: int,
    ) -> Dict[str, Any]:
        """
        Calculate emissions from aggregated tonne-km data.

        Args:
            tonne_km: Total tonne-km
            vessel_type: Default vessel type
            step_offset: Step number offset

        Returns:
            Dictionary with emissions and calculation details
        """
        steps: List[CalculationStep] = []

        vtype = vessel_type.value if hasattr(vessel_type, 'value') else str(vessel_type)
        factor = self._tonne_km_factors.get(
            vtype,
            self._tonne_km_factors[VesselType.CONTAINER_SHIP.value]
        )

        total_kg = (tonne_km * factor).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        steps.append(CalculationStep(
            step_number=step_offset + 1,
            description="Calculate emissions from aggregated tonne-km",
            formula="emissions = tonne_km x EF",
            inputs={
                "total_tonne_km": str(tonne_km),
                "vessel_type": vtype,
                "emission_factor": str(factor),
            },
            output=str(total_kg),
        ))

        return {
            "total_kg": total_kg,
            "steps": steps,
        }
