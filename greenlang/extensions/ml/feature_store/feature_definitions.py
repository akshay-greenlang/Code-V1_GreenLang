# -*- coding: utf-8 -*-
"""
Feature Definitions for GreenLang Process Heat Agents

This module defines all feature types and entity definitions used by
Process Heat agents (GL-001 through GL-020), including:
- Entity definitions: equipment_id, timestamp, facility_id
- BoilerFeatures: steam_flow, fuel_rate, efficiency, excess_air
- CombustionFeatures: O2, CO, NOx, stack_temp, air_fuel_ratio
- SteamFeatures: pressure, temperature, quality, enthalpy
- EmissionsFeatures: CO2_rate, CH4_rate, N2O_rate, intensity
- PredictiveFeatures: fouling_index, failure_probability, remaining_life

All feature definitions include validation, documentation, and provenance
tracking with SHA-256 hashes for regulatory compliance.

Example:
    >>> from greenlang.ml.feature_store.feature_definitions import (
    ...     BoilerFeatures, CombustionFeatures, EquipmentEntity
    ... )
    >>>
    >>> entity = EquipmentEntity(
    ...     equipment_id="boiler-001",
    ...     facility_id="plant-north",
    ...     equipment_type="fire_tube_boiler"
    ... )
    >>>
    >>> features = BoilerFeatures(
    ...     steam_flow=1000.0,
    ...     fuel_rate=50.0,
    ...     efficiency=0.85,
    ...     excess_air=15.0
    ... )
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator, root_validator
from enum import Enum
from datetime import datetime, timezone
from dataclasses import dataclass
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class EquipmentType(str, Enum):
    """Types of process heat equipment."""
    FIRE_TUBE_BOILER = "fire_tube_boiler"
    WATER_TUBE_BOILER = "water_tube_boiler"
    PACKAGE_BOILER = "package_boiler"
    WASTE_HEAT_BOILER = "waste_heat_boiler"
    STEAM_GENERATOR = "steam_generator"
    THERMAL_FLUID_HEATER = "thermal_fluid_heater"
    FURNACE = "furnace"
    KILN = "kiln"
    DRYER = "dryer"
    HEAT_EXCHANGER = "heat_exchanger"
    HEAT_PUMP = "heat_pump"
    CHP_UNIT = "chp_unit"


class FuelType(str, Enum):
    """Types of fuels used in process heat."""
    NATURAL_GAS = "natural_gas"
    LPG = "lpg"
    DIESEL = "diesel"
    FUEL_OIL = "fuel_oil"
    COAL = "coal"
    BIOMASS = "biomass"
    BIOGAS = "biogas"
    HYDROGEN = "hydrogen"
    ELECTRICITY = "electricity"
    WASTE_HEAT = "waste_heat"


class FeatureDataType(str, Enum):
    """Data types for features."""
    FLOAT = "float"
    INT = "int"
    STRING = "string"
    BOOL = "bool"
    TIMESTAMP = "timestamp"
    ARRAY = "array"


class FeatureCategory(str, Enum):
    """Categories of features."""
    OPERATIONAL = "operational"
    PERFORMANCE = "performance"
    ENVIRONMENTAL = "environmental"
    PREDICTIVE = "predictive"
    DERIVED = "derived"
    RAW = "raw"


# =============================================================================
# ENTITY DEFINITIONS
# =============================================================================

class EquipmentEntity(BaseModel):
    """
    Entity definition for process heat equipment.

    This represents a unique piece of equipment that features are attached to.
    All feature lookups require an equipment entity.

    Attributes:
        equipment_id: Unique identifier for the equipment
        facility_id: ID of the facility containing the equipment
        equipment_type: Type of equipment
        manufacturer: Equipment manufacturer (optional)
        model: Equipment model (optional)
        installation_date: Date equipment was installed (optional)
        metadata: Additional equipment metadata

    Example:
        >>> entity = EquipmentEntity(
        ...     equipment_id="boiler-001",
        ...     facility_id="plant-north",
        ...     equipment_type="fire_tube_boiler"
        ... )
    """

    equipment_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Unique equipment identifier"
    )
    facility_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Facility identifier"
    )
    equipment_type: EquipmentType = Field(
        ...,
        description="Type of equipment"
    )
    manufacturer: Optional[str] = Field(
        None,
        max_length=256,
        description="Equipment manufacturer"
    )
    model: Optional[str] = Field(
        None,
        max_length=256,
        description="Equipment model"
    )
    installation_date: Optional[datetime] = Field(
        None,
        description="Installation date"
    )
    capacity_kw: Optional[float] = Field(
        None,
        ge=0,
        description="Equipment capacity in kW"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    @validator('equipment_id', 'facility_id')
    def validate_ids(cls, v):
        """Validate ID format."""
        if not v or not v.strip():
            raise ValueError("ID cannot be empty or whitespace")
        return v.strip()

    class Config:
        """Pydantic configuration."""
        use_enum_values = True

    def to_feast_entity(self) -> Dict[str, str]:
        """Convert to Feast entity format."""
        return {
            "equipment_id": self.equipment_id,
            "facility_id": self.facility_id
        }


class FacilityEntity(BaseModel):
    """
    Entity definition for facilities.

    Represents a facility that contains multiple pieces of equipment.
    Used for facility-level feature aggregations.

    Attributes:
        facility_id: Unique facility identifier
        facility_name: Human-readable facility name
        location: Geographic location
        industry: Industry sector
        metadata: Additional facility metadata
    """

    facility_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Unique facility identifier"
    )
    facility_name: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Facility name"
    )
    location: Optional[str] = Field(
        None,
        description="Geographic location"
    )
    latitude: Optional[float] = Field(
        None,
        ge=-90,
        le=90,
        description="Latitude"
    )
    longitude: Optional[float] = Field(
        None,
        ge=-180,
        le=180,
        description="Longitude"
    )
    industry: Optional[str] = Field(
        None,
        description="Industry sector (e.g., chemical, food, paper)"
    )
    country_code: Optional[str] = Field(
        None,
        min_length=2,
        max_length=3,
        description="ISO country code"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    class Config:
        """Pydantic configuration."""
        use_enum_values = True


# =============================================================================
# FEATURE PROVENANCE
# =============================================================================

class FeatureProvenance(BaseModel):
    """
    Provenance tracking for features.

    Implements SHA-256 hashing for complete audit trail of feature
    computations, supporting GreenLang's zero-hallucination principles.

    Attributes:
        feature_hash: SHA-256 hash of feature values
        computation_timestamp: When features were computed
        source_data_hash: Hash of source data used
        pipeline_version: Version of feature pipeline
        agent_id: ID of agent that computed features
    """

    feature_hash: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 hash of feature values"
    )
    computation_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of computation"
    )
    source_data_hash: Optional[str] = Field(
        None,
        min_length=64,
        max_length=64,
        description="SHA-256 hash of source data"
    )
    pipeline_version: str = Field(
        default="1.0.0",
        description="Feature pipeline version"
    )
    agent_id: Optional[str] = Field(
        None,
        description="Agent that computed features"
    )
    transformation_chain: List[str] = Field(
        default_factory=list,
        description="List of transformations applied"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional provenance metadata"
    )

    @staticmethod
    def compute_hash(data: Dict[str, Any]) -> str:
        """
        Compute SHA-256 hash for feature data.

        Args:
            data: Feature data to hash

        Returns:
            SHA-256 hex digest
        """
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()


# =============================================================================
# BASE FEATURE CLASS
# =============================================================================

class BaseFeatures(BaseModel):
    """
    Base class for all feature definitions.

    Provides common functionality for feature validation, serialization,
    and provenance tracking.
    """

    event_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of the feature values"
    )
    created_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When features were created/computed"
    )
    provenance: Optional[FeatureProvenance] = Field(
        None,
        description="Provenance tracking for audit trail"
    )

    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True

    def compute_provenance(self, agent_id: Optional[str] = None) -> FeatureProvenance:
        """
        Compute provenance for this feature set.

        Args:
            agent_id: Optional agent identifier

        Returns:
            FeatureProvenance with computed hash
        """
        # Get feature values (exclude provenance and timestamps for hashing)
        feature_data = self.dict(
            exclude={'provenance', 'event_timestamp', 'created_timestamp'}
        )

        feature_hash = FeatureProvenance.compute_hash(feature_data)

        self.provenance = FeatureProvenance(
            feature_hash=feature_hash,
            agent_id=agent_id
        )

        return self.provenance

    def to_feast_row(self) -> Dict[str, Any]:
        """Convert to Feast row format."""
        return self.dict(exclude={'provenance'})

    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        exclude_fields = {'event_timestamp', 'created_timestamp', 'provenance'}
        return [f for f in self.__fields__.keys() if f not in exclude_fields]


# =============================================================================
# BOILER FEATURES (GL-002)
# =============================================================================

class BoilerFeatures(BaseFeatures):
    """
    Features for boiler performance analysis (GL-002 Boiler Efficiency Agent).

    These features capture the operational state and efficiency metrics
    of steam boilers used in process heat applications.

    Attributes:
        steam_flow: Steam output flow rate (kg/h)
        fuel_rate: Fuel consumption rate (kg/h or m3/h)
        efficiency: Boiler thermal efficiency (0-1 fraction)
        excess_air: Excess air percentage (%)
        blowdown_rate: Blowdown rate (% of steam flow)
        feedwater_temp: Feedwater temperature (C)
        steam_pressure: Steam header pressure (bar)
        steam_temp: Steam temperature (C)

    Example:
        >>> features = BoilerFeatures(
        ...     steam_flow=1000.0,
        ...     fuel_rate=50.0,
        ...     efficiency=0.85,
        ...     excess_air=15.0
        ... )
        >>> print(f"Efficiency: {features.efficiency:.1%}")
        Efficiency: 85.0%
    """

    steam_flow: float = Field(
        ...,
        ge=0,
        le=1000000,
        description="Steam flow rate (kg/h)"
    )
    fuel_rate: float = Field(
        ...,
        ge=0,
        le=100000,
        description="Fuel consumption rate (kg/h or m3/h)"
    )
    efficiency: float = Field(
        ...,
        ge=0,
        le=1.0,
        description="Thermal efficiency (0-1 fraction)"
    )
    excess_air: float = Field(
        ...,
        ge=0,
        le=200,
        description="Excess air percentage (%)"
    )
    blowdown_rate: Optional[float] = Field(
        None,
        ge=0,
        le=20,
        description="Blowdown rate (% of steam flow)"
    )
    feedwater_temp: Optional[float] = Field(
        None,
        ge=0,
        le=200,
        description="Feedwater temperature (C)"
    )
    steam_pressure: Optional[float] = Field(
        None,
        ge=0,
        le=200,
        description="Steam pressure (bar)"
    )
    steam_temp: Optional[float] = Field(
        None,
        ge=0,
        le=600,
        description="Steam temperature (C)"
    )
    heat_input: Optional[float] = Field(
        None,
        ge=0,
        description="Heat input (kW)"
    )
    heat_output: Optional[float] = Field(
        None,
        ge=0,
        description="Heat output (kW)"
    )

    @validator('efficiency')
    def validate_efficiency(cls, v):
        """Validate efficiency is a reasonable value."""
        if v > 0.98:
            logger.warning(f"Efficiency {v} exceeds typical maximum of 98%")
        return v

    @root_validator
    def validate_heat_balance(cls, values):
        """Validate heat input/output consistency."""
        heat_input = values.get('heat_input')
        heat_output = values.get('heat_output')
        efficiency = values.get('efficiency')

        if heat_input and heat_output and efficiency:
            calculated_efficiency = heat_output / heat_input
            if abs(calculated_efficiency - efficiency) > 0.05:
                logger.warning(
                    f"Efficiency {efficiency} inconsistent with heat balance "
                    f"(calculated: {calculated_efficiency:.3f})"
                )

        return values


# =============================================================================
# COMBUSTION FEATURES (GL-003)
# =============================================================================

class CombustionFeatures(BaseFeatures):
    """
    Features for combustion analysis (GL-003 Combustion Analyzer Agent).

    These features capture flue gas composition and combustion quality
    metrics for optimizing burner performance.

    Attributes:
        o2_percentage: Oxygen in flue gas (%)
        co_ppm: Carbon monoxide in flue gas (ppm)
        nox_ppm: NOx in flue gas (ppm)
        stack_temp: Stack/flue gas temperature (C)
        air_fuel_ratio: Air-to-fuel ratio (mass basis)
        combustion_efficiency: Combustion efficiency (0-1)
        flame_temp: Flame temperature (C)
        excess_air_ratio: Excess air ratio (lambda)

    Example:
        >>> features = CombustionFeatures(
        ...     o2_percentage=3.0,
        ...     co_ppm=50,
        ...     nox_ppm=100,
        ...     stack_temp=180,
        ...     air_fuel_ratio=15.0
        ... )
    """

    o2_percentage: float = Field(
        ...,
        ge=0,
        le=21,
        description="Oxygen in flue gas (%)"
    )
    co_ppm: float = Field(
        ...,
        ge=0,
        le=50000,
        description="Carbon monoxide (ppm)"
    )
    nox_ppm: float = Field(
        ...,
        ge=0,
        le=5000,
        description="NOx concentration (ppm)"
    )
    stack_temp: float = Field(
        ...,
        ge=0,
        le=1000,
        description="Stack temperature (C)"
    )
    air_fuel_ratio: float = Field(
        ...,
        ge=1,
        le=100,
        description="Air-to-fuel ratio (mass basis)"
    )
    combustion_efficiency: Optional[float] = Field(
        None,
        ge=0,
        le=1.0,
        description="Combustion efficiency (0-1)"
    )
    flame_temp: Optional[float] = Field(
        None,
        ge=500,
        le=2500,
        description="Flame temperature (C)"
    )
    excess_air_ratio: Optional[float] = Field(
        None,
        ge=1.0,
        le=3.0,
        description="Excess air ratio (lambda)"
    )
    co2_percentage: Optional[float] = Field(
        None,
        ge=0,
        le=20,
        description="CO2 in flue gas (%)"
    )
    so2_ppm: Optional[float] = Field(
        None,
        ge=0,
        le=5000,
        description="SO2 concentration (ppm)"
    )
    unburned_carbon: Optional[float] = Field(
        None,
        ge=0,
        le=10,
        description="Unburned carbon in ash (%)"
    )

    @validator('o2_percentage')
    def validate_o2(cls, v):
        """Validate O2 is in typical combustion range."""
        if v < 1:
            logger.warning(f"O2 {v}% is dangerously low - incomplete combustion risk")
        elif v > 10:
            logger.warning(f"O2 {v}% indicates excessive air - efficiency loss")
        return v

    @validator('co_ppm')
    def validate_co(cls, v):
        """Validate CO is within acceptable limits."""
        if v > 400:
            logger.warning(f"CO {v} ppm exceeds typical limit of 400 ppm")
        return v


# =============================================================================
# STEAM FEATURES (GL-004)
# =============================================================================

class SteamFeatures(BaseFeatures):
    """
    Features for steam quality analysis (GL-004 Steam Quality Agent).

    These features capture thermodynamic properties of steam at various
    points in the steam system.

    Attributes:
        pressure: Steam pressure (bar absolute)
        temperature: Steam temperature (C)
        quality: Steam quality/dryness fraction (0-1)
        enthalpy: Specific enthalpy (kJ/kg)
        entropy: Specific entropy (kJ/kg-K)
        specific_volume: Specific volume (m3/kg)
        saturation_temp: Saturation temperature (C)
        superheat_deg: Degrees of superheat (C)

    Example:
        >>> features = SteamFeatures(
        ...     pressure=10.0,
        ...     temperature=200.0,
        ...     quality=0.98,
        ...     enthalpy=2800.0
        ... )
    """

    pressure: float = Field(
        ...,
        ge=0,
        le=300,
        description="Steam pressure (bar absolute)"
    )
    temperature: float = Field(
        ...,
        ge=0,
        le=650,
        description="Steam temperature (C)"
    )
    quality: float = Field(
        ...,
        ge=0,
        le=1.0,
        description="Steam quality/dryness fraction (0-1)"
    )
    enthalpy: float = Field(
        ...,
        ge=0,
        le=4000,
        description="Specific enthalpy (kJ/kg)"
    )
    entropy: Optional[float] = Field(
        None,
        ge=0,
        le=10,
        description="Specific entropy (kJ/kg-K)"
    )
    specific_volume: Optional[float] = Field(
        None,
        ge=0,
        description="Specific volume (m3/kg)"
    )
    saturation_temp: Optional[float] = Field(
        None,
        ge=0,
        le=400,
        description="Saturation temperature at pressure (C)"
    )
    superheat_deg: Optional[float] = Field(
        None,
        ge=0,
        le=300,
        description="Degrees of superheat (C)"
    )
    flow_rate: Optional[float] = Field(
        None,
        ge=0,
        description="Steam flow rate (kg/h)"
    )
    velocity: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Steam velocity (m/s)"
    )

    @root_validator
    def validate_thermodynamic_consistency(cls, values):
        """Validate thermodynamic consistency."""
        pressure = values.get('pressure')
        temperature = values.get('temperature')
        saturation_temp = values.get('saturation_temp')
        quality = values.get('quality')

        # For saturated steam, temperature should equal saturation temp
        if saturation_temp and quality and quality < 1.0:
            if abs(temperature - saturation_temp) > 5:
                logger.warning(
                    f"Temperature {temperature}C inconsistent with saturation temp "
                    f"{saturation_temp}C for quality {quality}"
                )

        # For superheated steam, temperature must exceed saturation temp
        if saturation_temp and quality == 1.0:
            if temperature < saturation_temp:
                logger.warning(
                    f"Superheated steam temperature {temperature}C below "
                    f"saturation temp {saturation_temp}C"
                )

        return values


# =============================================================================
# EMISSIONS FEATURES (GL-005)
# =============================================================================

class EmissionsFeatures(BaseFeatures):
    """
    Features for emissions calculation (GL-005 Emissions Calculator Agent).

    These features capture greenhouse gas emissions and emission intensity
    metrics for regulatory compliance (e.g., CBAM, EPA).

    Attributes:
        co2_rate: CO2 emission rate (kg/h)
        ch4_rate: CH4 emission rate (kg/h)
        n2o_rate: N2O emission rate (kg/h)
        intensity: Emission intensity (kgCO2e/unit)
        total_ghg: Total GHG emissions (kgCO2e/h)
        scope1_emissions: Scope 1 emissions (kgCO2e)
        scope2_emissions: Scope 2 emissions (kgCO2e)
        emission_factor: Emission factor used (kgCO2e/unit)

    Example:
        >>> features = EmissionsFeatures(
        ...     co2_rate=500.0,
        ...     ch4_rate=0.1,
        ...     n2o_rate=0.01,
        ...     intensity=0.25,
        ...     total_ghg=510.0
        ... )
    """

    co2_rate: float = Field(
        ...,
        ge=0,
        description="CO2 emission rate (kg/h)"
    )
    ch4_rate: float = Field(
        ...,
        ge=0,
        description="CH4 emission rate (kg/h)"
    )
    n2o_rate: float = Field(
        ...,
        ge=0,
        description="N2O emission rate (kg/h)"
    )
    intensity: float = Field(
        ...,
        ge=0,
        description="Emission intensity (kgCO2e/unit output)"
    )
    total_ghg: float = Field(
        ...,
        ge=0,
        description="Total GHG emissions (kgCO2e/h)"
    )
    scope1_emissions: Optional[float] = Field(
        None,
        ge=0,
        description="Scope 1 emissions (kgCO2e)"
    )
    scope2_emissions: Optional[float] = Field(
        None,
        ge=0,
        description="Scope 2 emissions (kgCO2e)"
    )
    scope3_emissions: Optional[float] = Field(
        None,
        ge=0,
        description="Scope 3 emissions (kgCO2e)"
    )
    emission_factor: Optional[float] = Field(
        None,
        ge=0,
        description="Emission factor used (kgCO2e/unit fuel)"
    )
    fuel_type: Optional[FuelType] = Field(
        None,
        description="Fuel type for emission factor"
    )
    gwp_co2: float = Field(
        default=1.0,
        ge=0,
        description="Global warming potential for CO2"
    )
    gwp_ch4: float = Field(
        default=28.0,
        ge=0,
        description="Global warming potential for CH4 (AR5)"
    )
    gwp_n2o: float = Field(
        default=265.0,
        ge=0,
        description="Global warming potential for N2O (AR5)"
    )

    @root_validator
    def validate_ghg_calculation(cls, values):
        """Validate total GHG is consistent with component emissions."""
        co2_rate = values.get('co2_rate', 0)
        ch4_rate = values.get('ch4_rate', 0)
        n2o_rate = values.get('n2o_rate', 0)
        total_ghg = values.get('total_ghg', 0)
        gwp_co2 = values.get('gwp_co2', 1.0)
        gwp_ch4 = values.get('gwp_ch4', 28.0)
        gwp_n2o = values.get('gwp_n2o', 265.0)

        calculated_ghg = (
            co2_rate * gwp_co2 +
            ch4_rate * gwp_ch4 +
            n2o_rate * gwp_n2o
        )

        if total_ghg > 0 and abs(calculated_ghg - total_ghg) / total_ghg > 0.01:
            logger.warning(
                f"Total GHG {total_ghg} kgCO2e/h differs from calculated "
                f"{calculated_ghg:.2f} kgCO2e/h by more than 1%"
            )

        return values


# =============================================================================
# PREDICTIVE FEATURES (GL-009)
# =============================================================================

class PredictiveFeatures(BaseFeatures):
    """
    Features for predictive maintenance (GL-009 Predictive Maintenance Agent).

    These features capture equipment health indicators and failure
    predictions for proactive maintenance scheduling.

    Attributes:
        fouling_index: Fouling severity index (0-100)
        failure_probability: Probability of failure (0-1)
        remaining_life: Estimated remaining useful life (hours)
        maintenance_score: Maintenance urgency score (0-100)
        anomaly_score: Anomaly detection score (0-1)
        health_index: Overall equipment health (0-100)
        vibration_trend: Vibration trend indicator (-1 to 1)
        thermal_stress: Thermal stress factor (0-100)

    Example:
        >>> features = PredictiveFeatures(
        ...     fouling_index=25.0,
        ...     failure_probability=0.05,
        ...     remaining_life=8760.0,
        ...     maintenance_score=30.0
        ... )
    """

    fouling_index: float = Field(
        ...,
        ge=0,
        le=100,
        description="Fouling severity index (0-100, 100=severe)"
    )
    failure_probability: float = Field(
        ...,
        ge=0,
        le=1.0,
        description="Probability of failure in next period (0-1)"
    )
    remaining_life: float = Field(
        ...,
        ge=0,
        description="Estimated remaining useful life (hours)"
    )
    maintenance_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Maintenance urgency score (0-100, 100=urgent)"
    )
    anomaly_score: Optional[float] = Field(
        None,
        ge=0,
        le=1.0,
        description="Anomaly detection score (0-1, 1=anomalous)"
    )
    health_index: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Overall equipment health (0-100, 100=perfect)"
    )
    vibration_trend: Optional[float] = Field(
        None,
        ge=-1,
        le=1,
        description="Vibration trend indicator (-1=improving, 1=degrading)"
    )
    thermal_stress: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Thermal stress factor (0-100)"
    )
    corrosion_rate: Optional[float] = Field(
        None,
        ge=0,
        description="Corrosion rate (mm/year)"
    )
    tube_thickness: Optional[float] = Field(
        None,
        ge=0,
        description="Remaining tube wall thickness (mm)"
    )
    efficiency_degradation: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Efficiency degradation from baseline (%)"
    )
    operating_hours: Optional[float] = Field(
        None,
        ge=0,
        description="Total operating hours since maintenance"
    )
    cycle_count: Optional[int] = Field(
        None,
        ge=0,
        description="Number of start/stop cycles"
    )

    @validator('failure_probability')
    def validate_failure_prob(cls, v):
        """Validate and warn on high failure probability."""
        if v > 0.5:
            logger.warning(f"High failure probability: {v:.1%} - maintenance recommended")
        return v

    @validator('remaining_life')
    def validate_remaining_life(cls, v):
        """Validate remaining life is reasonable."""
        if v < 168:  # Less than 1 week
            logger.warning(f"Critical: Only {v:.0f} hours remaining life estimated")
        return v


# =============================================================================
# FUEL FEATURES (GL-001)
# =============================================================================

class FuelFeatures(BaseFeatures):
    """
    Features for fuel analysis (GL-001 Fuel Analyzer Agent).

    These features capture fuel properties and composition for
    accurate combustion and emissions calculations.

    Attributes:
        fuel_type: Type of fuel
        fuel_flow_rate: Fuel consumption rate (kg/h or m3/h)
        energy_content: Energy content/heating value (MJ/kg)
        carbon_content: Carbon content (% by mass)
        sulfur_content: Sulfur content (% by mass)
        ash_content: Ash content (% by mass)
        moisture_content: Moisture content (% by mass)
        heating_value: Net calorific value (MJ/kg)
    """

    fuel_type: FuelType = Field(
        ...,
        description="Type of fuel"
    )
    fuel_flow_rate: float = Field(
        ...,
        ge=0,
        description="Fuel consumption rate (kg/h or m3/h)"
    )
    energy_content: float = Field(
        ...,
        ge=0,
        le=150,
        description="Gross calorific value (MJ/kg)"
    )
    carbon_content: float = Field(
        ...,
        ge=0,
        le=100,
        description="Carbon content (% by mass)"
    )
    sulfur_content: Optional[float] = Field(
        None,
        ge=0,
        le=10,
        description="Sulfur content (% by mass)"
    )
    ash_content: Optional[float] = Field(
        None,
        ge=0,
        le=50,
        description="Ash content (% by mass)"
    )
    moisture_content: Optional[float] = Field(
        None,
        ge=0,
        le=70,
        description="Moisture content (% by mass)"
    )
    heating_value: Optional[float] = Field(
        None,
        ge=0,
        le=140,
        description="Net calorific value (MJ/kg)"
    )
    hydrogen_content: Optional[float] = Field(
        None,
        ge=0,
        le=25,
        description="Hydrogen content (% by mass)"
    )
    nitrogen_content: Optional[float] = Field(
        None,
        ge=0,
        le=5,
        description="Nitrogen content (% by mass)"
    )
    oxygen_content: Optional[float] = Field(
        None,
        ge=0,
        le=50,
        description="Oxygen content (% by mass)"
    )
    volatile_matter: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Volatile matter (% by mass)"
    )
    fixed_carbon: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Fixed carbon (% by mass)"
    )


# =============================================================================
# HEAT RECOVERY FEATURES (GL-006)
# =============================================================================

class HeatRecoveryFeatures(BaseFeatures):
    """
    Features for waste heat recovery analysis (GL-006, GL-019 agents).

    Attributes:
        waste_heat_available: Available waste heat (kW)
        recovery_potential: Recovery potential (kW)
        recovery_efficiency: Heat recovery efficiency (0-1)
        source_temperature: Heat source temperature (C)
        sink_temperature: Heat sink temperature (C)
        temperature_approach: Minimum temperature approach (C)
        heat_exchanger_area: Required heat exchanger area (m2)
        payback_years: Simple payback period (years)
    """

    waste_heat_available: float = Field(
        ...,
        ge=0,
        description="Available waste heat (kW)"
    )
    recovery_potential: float = Field(
        ...,
        ge=0,
        description="Recoverable heat (kW)"
    )
    recovery_efficiency: float = Field(
        ...,
        ge=0,
        le=1.0,
        description="Heat recovery efficiency (0-1)"
    )
    source_temperature: float = Field(
        ...,
        ge=0,
        le=1500,
        description="Heat source temperature (C)"
    )
    sink_temperature: float = Field(
        ...,
        ge=0,
        le=500,
        description="Heat sink temperature (C)"
    )
    temperature_approach: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Minimum temperature approach (C)"
    )
    heat_exchanger_area: Optional[float] = Field(
        None,
        ge=0,
        description="Heat exchanger area (m2)"
    )
    payback_years: Optional[float] = Field(
        None,
        ge=0,
        description="Simple payback period (years)"
    )
    annual_savings_kwh: Optional[float] = Field(
        None,
        ge=0,
        description="Annual energy savings (kWh)"
    )
    annual_co2_avoided: Optional[float] = Field(
        None,
        ge=0,
        description="Annual CO2 emissions avoided (tonnes)"
    )


# =============================================================================
# THERMAL STORAGE FEATURES (GL-007)
# =============================================================================

class ThermalStorageFeatures(BaseFeatures):
    """
    Features for thermal storage analysis (GL-007 Thermal Storage Agent).

    Attributes:
        storage_capacity: Storage capacity (kWh)
        state_of_charge: Current state of charge (0-1)
        charge_rate: Charge rate (kW)
        discharge_rate: Discharge rate (kW)
        round_trip_efficiency: Round-trip efficiency (0-1)
        storage_temperature: Storage medium temperature (C)
        standby_losses: Standby thermal losses (kW)
        cycles_remaining: Estimated remaining charge cycles
    """

    storage_capacity: float = Field(
        ...,
        ge=0,
        description="Thermal storage capacity (kWh)"
    )
    state_of_charge: float = Field(
        ...,
        ge=0,
        le=1.0,
        description="State of charge (0-1)"
    )
    charge_rate: float = Field(
        ...,
        ge=0,
        description="Current charge rate (kW)"
    )
    discharge_rate: float = Field(
        ...,
        ge=0,
        description="Current discharge rate (kW)"
    )
    round_trip_efficiency: float = Field(
        ...,
        ge=0,
        le=1.0,
        description="Round-trip efficiency (0-1)"
    )
    storage_temperature: Optional[float] = Field(
        None,
        ge=-50,
        le=1000,
        description="Storage medium temperature (C)"
    )
    standby_losses: Optional[float] = Field(
        None,
        ge=0,
        description="Standby thermal losses (kW)"
    )
    cycles_remaining: Optional[int] = Field(
        None,
        ge=0,
        description="Estimated remaining charge cycles"
    )
    max_charge_rate: Optional[float] = Field(
        None,
        ge=0,
        description="Maximum charge rate (kW)"
    )
    max_discharge_rate: Optional[float] = Field(
        None,
        ge=0,
        description="Maximum discharge rate (kW)"
    )


# =============================================================================
# FEATURE COLLECTION
# =============================================================================

class ProcessHeatFeatureSet(BaseModel):
    """
    Complete feature set for a Process Heat equipment entity.

    Combines all feature types for comprehensive analysis.
    """

    entity: EquipmentEntity = Field(
        ...,
        description="Equipment entity"
    )
    boiler_features: Optional[BoilerFeatures] = Field(
        None,
        description="Boiler performance features"
    )
    combustion_features: Optional[CombustionFeatures] = Field(
        None,
        description="Combustion analysis features"
    )
    steam_features: Optional[SteamFeatures] = Field(
        None,
        description="Steam quality features"
    )
    emissions_features: Optional[EmissionsFeatures] = Field(
        None,
        description="Emissions calculation features"
    )
    predictive_features: Optional[PredictiveFeatures] = Field(
        None,
        description="Predictive maintenance features"
    )
    fuel_features: Optional[FuelFeatures] = Field(
        None,
        description="Fuel analysis features"
    )
    heat_recovery_features: Optional[HeatRecoveryFeatures] = Field(
        None,
        description="Heat recovery features"
    )
    thermal_storage_features: Optional[ThermalStorageFeatures] = Field(
        None,
        description="Thermal storage features"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Feature set timestamp"
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash of complete feature set"
    )

    def compute_provenance(self) -> str:
        """Compute SHA-256 hash for complete feature set."""
        feature_data = self.dict(exclude={'provenance_hash'})
        json_str = json.dumps(feature_data, sort_keys=True, default=str)
        self.provenance_hash = hashlib.sha256(json_str.encode('utf-8')).hexdigest()
        return self.provenance_hash

    class Config:
        """Pydantic configuration."""
        use_enum_values = True
