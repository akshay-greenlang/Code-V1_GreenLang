# -*- coding: utf-8 -*-
"""
Transport MRV Base Module
=========================

This module provides base classes and common functionality for all
transport MRV (Monitoring, Reporting, Verification) agents.

Design Principles:
- Zero-hallucination guarantee (no LLM in calculation path)
- Full audit trail with SHA-256 provenance hashing
- GHG Protocol Scope 1 and Scope 3 Category 4, 6, 7, 9 compliant
- DEFRA, EPA, and ICAO emission factor support
- Pydantic models for type safety

Reference Standards:
- GHG Protocol Corporate Standard (2015)
- GHG Protocol Scope 3 Standard (2011)
- ICAO Carbon Emissions Calculator Methodology
- IMO DCS and EU MRV Regulation
- EPA SmartWay Transport Partnership
"""

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class TransportMode(str, Enum):
    """Transport modes for emissions calculation."""
    ROAD = "road"
    RAIL = "rail"
    AVIATION = "aviation"
    MARITIME = "maritime"
    INLAND_WATERWAY = "inland_waterway"
    PIPELINE = "pipeline"
    INTERMODAL = "intermodal"


class FuelType(str, Enum):
    """Fuel types for transport emissions."""
    # Liquid fuels
    DIESEL = "diesel"
    PETROL = "petrol"
    BIODIESEL = "biodiesel"
    BIOETHANOL = "bioethanol"
    LPG = "lpg"
    CNG = "cng"
    LNG = "lng"
    JET_A = "jet_a"
    JET_A1 = "jet_a1"
    AVGAS = "avgas"
    MARINE_FUEL_OIL = "marine_fuel_oil"
    MARINE_GAS_OIL = "marine_gas_oil"
    MARINE_DIESEL = "marine_diesel"
    HFO = "hfo"  # Heavy Fuel Oil

    # Electricity
    ELECTRICITY = "electricity"

    # Hydrogen
    HYDROGEN_GREEN = "hydrogen_green"
    HYDROGEN_GREY = "hydrogen_grey"
    HYDROGEN_BLUE = "hydrogen_blue"

    # Biofuels
    B20 = "b20"
    B100 = "b100"
    E10 = "e10"
    E85 = "e85"
    HVO = "hvo"  # Hydrotreated Vegetable Oil


class VehicleType(str, Enum):
    """Vehicle types for road transport."""
    # Cars
    CAR_SMALL = "car_small"
    CAR_MEDIUM = "car_medium"
    CAR_LARGE = "car_large"
    CAR_SUV = "car_suv"
    CAR_ELECTRIC = "car_electric"
    CAR_HYBRID = "car_hybrid"
    CAR_PHEV = "car_phev"

    # Vans/LCV
    VAN_SMALL = "van_small"
    VAN_MEDIUM = "van_medium"
    VAN_LARGE = "van_large"

    # Trucks/HGV
    TRUCK_RIGID_SMALL = "truck_rigid_small"  # <7.5t
    TRUCK_RIGID_MEDIUM = "truck_rigid_medium"  # 7.5-17t
    TRUCK_RIGID_LARGE = "truck_rigid_large"  # >17t
    TRUCK_ARTICULATED = "truck_articulated"
    TRUCK_ARTICULATED_SMALL = "truck_articulated_small"  # <33t
    TRUCK_ARTICULATED_LARGE = "truck_articulated_large"  # >33t

    # Buses/Coaches
    BUS_LOCAL = "bus_local"
    BUS_COACH = "bus_coach"
    BUS_ELECTRIC = "bus_electric"

    # Motorcycles
    MOTORCYCLE_SMALL = "motorcycle_small"
    MOTORCYCLE_MEDIUM = "motorcycle_medium"
    MOTORCYCLE_LARGE = "motorcycle_large"


class EmissionScope(str, Enum):
    """GHG Protocol emission scopes."""
    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"


class CalculationMethod(str, Enum):
    """Calculation methods for transport emissions."""
    FUEL_BASED = "fuel_based"
    DISTANCE_BASED = "distance_based"
    SPEND_BASED = "spend_based"
    ACTIVITY_BASED = "activity_based"


class DataQualityTier(str, Enum):
    """Data quality tiers per GHG Protocol."""
    TIER_1 = "tier_1"  # Primary data from direct measurement
    TIER_2 = "tier_2"  # Secondary data from published sources
    TIER_3 = "tier_3"  # Estimated or modeled data


class EmissionFactorSource(str, Enum):
    """Sources for emission factors."""
    DEFRA = "defra"
    EPA = "epa"
    ICAO = "icao"
    IMO = "imo"
    IPCC = "ipcc"
    GLEC = "glec"  # Global Logistics Emissions Council
    CUSTOM = "custom"


# =============================================================================
# Emission Factor Models
# =============================================================================

class EmissionFactor(BaseModel):
    """Emission factor record with provenance."""

    factor_id: str = Field(..., description="Unique factor identifier")
    factor_value: Decimal = Field(..., ge=0, description="Emission factor value")
    factor_unit: str = Field(..., description="Factor unit")
    source: EmissionFactorSource = Field(..., description="Factor source")
    source_uri: str = Field("", description="URI to source documentation")
    version: str = Field(..., description="Factor version/year")
    last_updated: str = Field(..., description="Last update date")
    uncertainty_pct: Optional[float] = Field(
        None, ge=0, le=100, description="Uncertainty percentage"
    )
    data_quality_tier: DataQualityTier = Field(
        DataQualityTier.TIER_2, description="Data quality tier"
    )
    geographic_scope: str = Field("global", description="Geographic applicability")
    fuel_type: Optional[FuelType] = Field(None, description="Applicable fuel type")
    vehicle_type: Optional[VehicleType] = Field(None, description="Applicable vehicle type")

    class Config:
        use_enum_values = True


# =============================================================================
# Calculation Step Model
# =============================================================================

class CalculationStep(BaseModel):
    """Single step in the calculation audit trail."""

    step_number: int = Field(..., ge=1, description="Step sequence number")
    description: str = Field(..., description="Step description")
    formula: Optional[str] = Field(None, description="Formula applied")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Input values")
    output: Optional[str] = Field(None, description="Output value")
    emission_factor: Optional[EmissionFactor] = Field(
        None, description="Emission factor used"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Step timestamp"
    )


# =============================================================================
# Base Input/Output Models
# =============================================================================

class TransportMRVInput(BaseModel):
    """Base input model for all transport MRV agents."""

    # Identification
    organization_id: str = Field(..., description="Organization identifier")
    facility_id: Optional[str] = Field(None, description="Facility identifier")
    request_id: Optional[str] = Field(None, description="Unique request ID")

    # Reporting period
    reporting_year: int = Field(..., ge=2020, le=2100, description="Reporting year")
    reporting_period_start: Optional[datetime] = Field(
        None, description="Period start date"
    )
    reporting_period_end: Optional[datetime] = Field(
        None, description="Period end date"
    )

    # Calculation parameters
    calculation_method: CalculationMethod = Field(
        CalculationMethod.FUEL_BASED, description="Calculation method"
    )

    # Geographic context
    region: str = Field("global", description="Geographic region")
    country: Optional[str] = Field(None, description="Country code (ISO 3166-1)")

    class Config:
        use_enum_values = True

    @field_validator("reporting_year")
    @classmethod
    def validate_reporting_year(cls, v: int) -> int:
        """Validate reporting year is reasonable."""
        current_year = datetime.now().year
        if v > current_year + 1:
            raise ValueError(f"Reporting year {v} is too far in the future")
        return v


class TransportMRVOutput(BaseModel):
    """Base output model for all transport MRV agents."""

    # Agent identification
    agent_id: str = Field(..., description="Agent identifier")
    agent_version: str = Field("1.0.0", description="Agent version")

    # Emission results
    total_emissions_kg_co2e: Decimal = Field(
        ..., ge=0, description="Total emissions in kg CO2e"
    )
    total_emissions_mt_co2e: Decimal = Field(
        ..., ge=0, description="Total emissions in metric tons CO2e"
    )

    # Gas breakdown
    co2_kg: Decimal = Field(Decimal("0"), ge=0, description="CO2 in kg")
    ch4_kg: Decimal = Field(Decimal("0"), ge=0, description="CH4 in kg")
    n2o_kg: Decimal = Field(Decimal("0"), ge=0, description="N2O in kg")

    # Scope classification
    scope: EmissionScope = Field(..., description="GHG Protocol scope")

    # Audit trail
    calculation_steps: List[CalculationStep] = Field(
        default_factory=list, description="Calculation audit trail"
    )
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")

    # Data quality
    data_quality_tier: DataQualityTier = Field(
        DataQualityTier.TIER_2, description="Overall data quality"
    )
    uncertainty_pct: Optional[float] = Field(
        None, ge=0, le=100, description="Uncertainty percentage"
    )

    # Timestamps
    calculation_timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Calculation timestamp"
    )
    calculation_duration_ms: float = Field(
        0.0, ge=0, description="Calculation duration in ms"
    )

    # Status
    status: str = Field("success", description="Calculation status")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    errors: List[str] = Field(default_factory=list, description="Error messages")

    # Metadata
    emission_factors_used: List[EmissionFactor] = Field(
        default_factory=list, description="Emission factors applied"
    )
    activity_data_summary: Dict[str, Any] = Field(
        default_factory=dict, description="Summary of activity data"
    )

    class Config:
        use_enum_values = True
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat(),
        }


# =============================================================================
# Emission Factor Database
# =============================================================================

# DEFRA 2024 Fuel Emission Factors (kg CO2e per liter)
DEFRA_FUEL_FACTORS: Dict[str, Dict[str, Any]] = {
    FuelType.DIESEL.value: {
        "co2_per_liter": Decimal("2.70496"),
        "ch4_per_liter": Decimal("0.00007"),
        "n2o_per_liter": Decimal("0.02636"),
        "total_per_liter": Decimal("2.73139"),
        "density_kg_per_liter": Decimal("0.8379"),
    },
    FuelType.PETROL.value: {
        "co2_per_liter": Decimal("2.31463"),
        "ch4_per_liter": Decimal("0.00201"),
        "n2o_per_liter": Decimal("0.00702"),
        "total_per_liter": Decimal("2.32366"),
        "density_kg_per_liter": Decimal("0.7489"),
    },
    FuelType.LPG.value: {
        "co2_per_liter": Decimal("1.55377"),
        "ch4_per_liter": Decimal("0.00066"),
        "n2o_per_liter": Decimal("0.00027"),
        "total_per_liter": Decimal("1.5547"),
        "density_kg_per_liter": Decimal("0.51"),
    },
    FuelType.CNG.value: {
        "co2_per_kg": Decimal("2.53973"),
        "ch4_per_kg": Decimal("0.01018"),
        "n2o_per_kg": Decimal("0.00027"),
        "total_per_kg": Decimal("2.55018"),
    },
    FuelType.JET_A1.value: {
        "co2_per_liter": Decimal("2.54052"),
        "ch4_per_liter": Decimal("0.00001"),
        "n2o_per_liter": Decimal("0.00264"),
        "total_per_liter": Decimal("2.54317"),
        "density_kg_per_liter": Decimal("0.8"),
    },
    FuelType.MARINE_FUEL_OIL.value: {
        "co2_per_tonne": Decimal("3114"),
        "ch4_per_tonne": Decimal("0.12"),
        "n2o_per_tonne": Decimal("0.58"),
        "total_per_tonne": Decimal("3114.7"),
    },
    FuelType.MARINE_GAS_OIL.value: {
        "co2_per_tonne": Decimal("3206"),
        "ch4_per_tonne": Decimal("0.12"),
        "n2o_per_tonne": Decimal("0.58"),
        "total_per_tonne": Decimal("3206.7"),
    },
}

# DEFRA 2024 Distance-based factors (kg CO2e per km)
DEFRA_VEHICLE_FACTORS: Dict[str, Dict[str, Any]] = {
    VehicleType.CAR_SMALL.value: {
        "co2e_per_km": Decimal("0.14901"),
        "co2e_per_passenger_km": Decimal("0.09934"),
    },
    VehicleType.CAR_MEDIUM.value: {
        "co2e_per_km": Decimal("0.17049"),
        "co2e_per_passenger_km": Decimal("0.11366"),
    },
    VehicleType.CAR_LARGE.value: {
        "co2e_per_km": Decimal("0.21016"),
        "co2e_per_passenger_km": Decimal("0.14011"),
    },
    VehicleType.CAR_ELECTRIC.value: {
        "co2e_per_km": Decimal("0.05297"),  # UK grid average
        "co2e_per_passenger_km": Decimal("0.03531"),
    },
    VehicleType.VAN_SMALL.value: {
        "co2e_per_km": Decimal("0.20624"),
    },
    VehicleType.VAN_MEDIUM.value: {
        "co2e_per_km": Decimal("0.24102"),
    },
    VehicleType.VAN_LARGE.value: {
        "co2e_per_km": Decimal("0.30963"),
    },
    VehicleType.TRUCK_RIGID_SMALL.value: {
        "co2e_per_km": Decimal("0.48618"),
        "co2e_per_tonne_km": Decimal("0.29541"),
    },
    VehicleType.TRUCK_RIGID_MEDIUM.value: {
        "co2e_per_km": Decimal("0.59627"),
        "co2e_per_tonne_km": Decimal("0.17835"),
    },
    VehicleType.TRUCK_RIGID_LARGE.value: {
        "co2e_per_km": Decimal("0.85285"),
        "co2e_per_tonne_km": Decimal("0.10668"),
    },
    VehicleType.TRUCK_ARTICULATED.value: {
        "co2e_per_km": Decimal("0.92907"),
        "co2e_per_tonne_km": Decimal("0.10666"),
    },
}

# DEFRA 2024 Tonne-km factors for freight
DEFRA_FREIGHT_FACTORS: Dict[str, Decimal] = {
    "truck_average": Decimal("0.10666"),
    "rail_freight": Decimal("0.02781"),
    "rail_freight_electric": Decimal("0.01763"),
    "ship_container_average": Decimal("0.01609"),
    "ship_bulk_carrier": Decimal("0.00476"),
    "ship_tanker": Decimal("0.00761"),
    "air_freight_short_haul": Decimal("1.12764"),
    "air_freight_long_haul": Decimal("0.45940"),
    "barge_inland": Decimal("0.03214"),
}


# =============================================================================
# Base Transport MRV Agent
# =============================================================================

class BaseTransportMRVAgent(ABC):
    """
    Abstract base class for transport MRV agents.

    All transport MRV agents inherit from this class and implement
    the calculate() method with transport-mode-specific logic.

    Key Guarantees:
    - ZERO HALLUCINATION: No LLM calls in calculation path
    - DETERMINISTIC: Same input always produces same output
    - AUDITABLE: Complete SHA-256 provenance tracking
    - GHG PROTOCOL COMPLIANT: Scope 1, 2, and 3 aligned
    """

    # Class attributes to be overridden by subclasses
    AGENT_ID: str = "GL-MRV-TRN-000"
    AGENT_NAME: str = "Base Transport MRV Agent"
    AGENT_VERSION: str = "1.0.0"
    TRANSPORT_MODE: TransportMode = TransportMode.ROAD
    DEFAULT_SCOPE: EmissionScope = EmissionScope.SCOPE_1

    def __init__(self):
        """Initialize the transport MRV agent."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._fuel_factors = DEFRA_FUEL_FACTORS
        self._vehicle_factors = DEFRA_VEHICLE_FACTORS
        self._freight_factors = DEFRA_FREIGHT_FACTORS
        self.logger.info(f"Initialized {self.AGENT_ID} v{self.AGENT_VERSION}")

    @abstractmethod
    def calculate(self, input_data: TransportMRVInput) -> TransportMRVOutput:
        """
        Execute the emission calculation.

        Args:
            input_data: Transport-specific input data

        Returns:
            Complete calculation result with audit trail

        Raises:
            ValueError: If input validation fails
            CalculationError: If calculation fails
        """
        pass

    def _get_fuel_factor(self, fuel_type: FuelType) -> EmissionFactor:
        """
        Get emission factor for a fuel type.

        Args:
            fuel_type: Type of fuel

        Returns:
            EmissionFactor record
        """
        fuel_data = self._fuel_factors.get(fuel_type.value)
        if not fuel_data:
            self.logger.warning(f"No factor for fuel type {fuel_type}, using diesel")
            fuel_data = self._fuel_factors[FuelType.DIESEL.value]

        # Determine unit based on available data
        if "total_per_liter" in fuel_data:
            factor_value = fuel_data["total_per_liter"]
            factor_unit = "kg CO2e/liter"
        elif "total_per_kg" in fuel_data:
            factor_value = fuel_data["total_per_kg"]
            factor_unit = "kg CO2e/kg"
        elif "total_per_tonne" in fuel_data:
            factor_value = fuel_data["total_per_tonne"]
            factor_unit = "kg CO2e/tonne"
        else:
            raise ValueError(f"Invalid fuel factor data for {fuel_type}")

        return EmissionFactor(
            factor_id=f"defra_2024_{fuel_type.value}",
            factor_value=factor_value,
            factor_unit=factor_unit,
            source=EmissionFactorSource.DEFRA,
            source_uri="https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2024",
            version="2024",
            last_updated="2024-06-01",
            uncertainty_pct=5.0,
            data_quality_tier=DataQualityTier.TIER_2,
            geographic_scope="UK",
            fuel_type=fuel_type,
        )

    def _get_vehicle_factor(self, vehicle_type: VehicleType) -> EmissionFactor:
        """
        Get emission factor for a vehicle type.

        Args:
            vehicle_type: Type of vehicle

        Returns:
            EmissionFactor record
        """
        vehicle_data = self._vehicle_factors.get(vehicle_type.value)
        if not vehicle_data:
            self.logger.warning(
                f"No factor for vehicle type {vehicle_type}, using medium car"
            )
            vehicle_data = self._vehicle_factors[VehicleType.CAR_MEDIUM.value]

        return EmissionFactor(
            factor_id=f"defra_2024_{vehicle_type.value}",
            factor_value=vehicle_data["co2e_per_km"],
            factor_unit="kg CO2e/km",
            source=EmissionFactorSource.DEFRA,
            source_uri="https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2024",
            version="2024",
            last_updated="2024-06-01",
            uncertainty_pct=10.0,
            data_quality_tier=DataQualityTier.TIER_2,
            geographic_scope="UK",
            vehicle_type=vehicle_type,
        )

    def _calculate_fuel_based(
        self,
        fuel_liters: Decimal,
        fuel_type: FuelType,
    ) -> tuple[Decimal, Decimal, Decimal, Decimal]:
        """
        Calculate emissions from fuel consumption.

        Formula: Emissions = Fuel (liters) x EF (kg CO2e/liter)

        Args:
            fuel_liters: Fuel consumption in liters
            fuel_type: Type of fuel

        Returns:
            Tuple of (total_kg, co2_kg, ch4_kg, n2o_kg)
        """
        fuel_data = self._fuel_factors.get(
            fuel_type.value,
            self._fuel_factors[FuelType.DIESEL.value]
        )

        if "total_per_liter" in fuel_data:
            total_kg = (fuel_liters * fuel_data["total_per_liter"]).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
            co2_kg = (fuel_liters * fuel_data["co2_per_liter"]).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
            ch4_kg = (fuel_liters * fuel_data["ch4_per_liter"]).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
            n2o_kg = (fuel_liters * fuel_data["n2o_per_liter"]).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
        else:
            total_kg = Decimal("0")
            co2_kg = Decimal("0")
            ch4_kg = Decimal("0")
            n2o_kg = Decimal("0")

        return total_kg, co2_kg, ch4_kg, n2o_kg

    def _calculate_distance_based(
        self,
        distance_km: Decimal,
        vehicle_type: VehicleType,
    ) -> Decimal:
        """
        Calculate emissions from distance traveled.

        Formula: Emissions = Distance (km) x EF (kg CO2e/km)

        Args:
            distance_km: Distance in kilometers
            vehicle_type: Type of vehicle

        Returns:
            Emissions in kg CO2e
        """
        vehicle_data = self._vehicle_factors.get(
            vehicle_type.value,
            self._vehicle_factors[VehicleType.CAR_MEDIUM.value]
        )

        emissions_kg = (distance_km * vehicle_data["co2e_per_km"]).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        return emissions_kg

    def _kg_to_metric_tons(self, kg: Decimal) -> Decimal:
        """Convert kg to metric tons."""
        return (kg / Decimal("1000")).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

    def _generate_provenance_hash(
        self,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        steps: List[CalculationStep],
    ) -> str:
        """
        Generate SHA-256 provenance hash for audit trail.

        Args:
            input_data: Input data dictionary
            output_data: Output data dictionary
            steps: Calculation steps

        Returns:
            SHA-256 hex digest
        """
        provenance_data = {
            "agent_id": self.AGENT_ID,
            "input": input_data,
            "output": {
                k: str(v) if isinstance(v, Decimal) else v
                for k, v in output_data.items()
                if k not in ["provenance_hash", "calculation_timestamp"]
            },
            "steps": [
                {
                    "step_number": s.step_number,
                    "description": s.description,
                    "formula": s.formula,
                    "inputs": s.inputs,
                    "output": s.output,
                }
                for s in steps
            ],
        }
        data_str = json.dumps(provenance_data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def _create_output(
        self,
        total_emissions_kg: Decimal,
        co2_kg: Decimal,
        ch4_kg: Decimal,
        n2o_kg: Decimal,
        steps: List[CalculationStep],
        emission_factors: List[EmissionFactor],
        activity_summary: Dict[str, Any],
        start_time: datetime,
        scope: Optional[EmissionScope] = None,
        warnings: Optional[List[str]] = None,
    ) -> TransportMRVOutput:
        """
        Create standardized output with provenance.

        Args:
            total_emissions_kg: Total emissions in kg
            co2_kg: CO2 component
            ch4_kg: CH4 component
            n2o_kg: N2O component
            steps: Calculation steps
            emission_factors: Factors used
            activity_summary: Summary of activity data
            start_time: Calculation start time
            scope: Emission scope (optional, uses default)
            warnings: Warning messages

        Returns:
            Complete TransportMRVOutput
        """
        end_time = datetime.utcnow()
        duration_ms = (end_time - start_time).total_seconds() * 1000

        output_data = {
            "total_emissions_kg_co2e": total_emissions_kg,
            "co2_kg": co2_kg,
            "ch4_kg": ch4_kg,
            "n2o_kg": n2o_kg,
        }

        provenance_hash = self._generate_provenance_hash(
            input_data=activity_summary,
            output_data=output_data,
            steps=steps,
        )

        # Determine overall data quality
        if emission_factors:
            quality_tiers = [ef.data_quality_tier for ef in emission_factors]
            if DataQualityTier.TIER_3 in quality_tiers:
                overall_quality = DataQualityTier.TIER_3
            elif DataQualityTier.TIER_2 in quality_tiers:
                overall_quality = DataQualityTier.TIER_2
            else:
                overall_quality = DataQualityTier.TIER_1
        else:
            overall_quality = DataQualityTier.TIER_3

        return TransportMRVOutput(
            agent_id=self.AGENT_ID,
            agent_version=self.AGENT_VERSION,
            total_emissions_kg_co2e=total_emissions_kg,
            total_emissions_mt_co2e=self._kg_to_metric_tons(total_emissions_kg),
            co2_kg=co2_kg,
            ch4_kg=ch4_kg,
            n2o_kg=n2o_kg,
            scope=scope or self.DEFAULT_SCOPE,
            calculation_steps=steps,
            provenance_hash=provenance_hash,
            data_quality_tier=overall_quality,
            calculation_timestamp=end_time,
            calculation_duration_ms=duration_ms,
            emission_factors_used=emission_factors,
            activity_data_summary=activity_summary,
            warnings=warnings or [],
        )
