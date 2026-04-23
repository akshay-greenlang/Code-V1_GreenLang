"""
Combustion Schemas - Core data models for combustion monitoring and control.

This module defines the fundamental Pydantic models for representing combustion
data, fuel properties, flue gas composition, burner state, and operating envelopes.
All models follow GreenLang's zero-hallucination principles with strict validation.

Example:
    >>> from combustion_schemas import CombustionData, FuelProperties
    >>> combustion = CombustionData(
    ...     fuel_flow_kg_per_s=2.5,
    ...     air_flow_kg_per_s=35.0,
    ...     o2_percent=3.2,
    ...     co_ppm=25.0,
    ...     nox_ppm=45.0
    ... )
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator, computed_field
import hashlib


class FuelType(str, Enum):
    """Enumeration of supported fuel types."""
    NATURAL_GAS = "natural_gas"
    PROPANE = "propane"
    BUTANE = "butane"
    FUEL_OIL_NO2 = "fuel_oil_no2"
    FUEL_OIL_NO6 = "fuel_oil_no6"
    HYDROGEN = "hydrogen"
    BIOGAS = "biogas"
    SYNGAS = "syngas"
    COAL = "coal"
    BIOMASS = "biomass"
    MIXED = "mixed"


class CombustionPhase(str, Enum):
    """Enumeration of combustion phases."""
    STARTUP = "startup"
    WARMUP = "warmup"
    STEADY_STATE = "steady_state"
    LOAD_CHANGE = "load_change"
    SHUTDOWN = "shutdown"
    EMERGENCY_STOP = "emergency_stop"


class TemperatureReading(BaseModel):
    """Temperature reading from a sensor location."""
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    location: str = Field(..., min_length=1, max_length=100, description="Location identifier for the temperature sensor")
    value_celsius: float = Field(..., ge=-273.15, le=2500.0, description="Temperature value in degrees Celsius")
    sensor_id: str = Field(..., min_length=1, max_length=50, description="Unique identifier for the temperature sensor")
    quality: str = Field(default="good", description="Data quality indicator (good, bad, uncertain, stale)")
    timestamp: Optional[datetime] = Field(default=None, description="Timestamp of the reading")


class PressureReading(BaseModel):
    """Pressure reading from a sensor location."""
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    location: str = Field(..., min_length=1, max_length=100, description="Location identifier for the pressure sensor")
    value_kpa: float = Field(..., ge=0.0, le=50000.0, description="Pressure value in kilopascals (kPa)")
    sensor_id: str = Field(..., min_length=1, max_length=50, description="Unique identifier for the pressure sensor")
    quality: str = Field(default="good", description="Data quality indicator (good, bad, uncertain, stale)")
    timestamp: Optional[datetime] = Field(default=None, description="Timestamp of the reading")


class CombustionData(BaseModel):
    """
    Core combustion measurement data from sensors.

    This model captures the essential real-time measurements from a combustion
    system including flows, emissions, temperatures, and pressures. All values
    are validated to ensure physical plausibility.
    """
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    fuel_flow_kg_per_s: float = Field(..., ge=0.0, le=1000.0, description="Fuel mass flow rate in kilograms per second")
    air_flow_kg_per_s: float = Field(..., ge=0.0, le=10000.0, description="Combustion air mass flow rate in kilograms per second")
    o2_percent: float = Field(..., ge=0.0, le=21.0, description="Oxygen concentration in flue gas (dry basis) as percentage")
    co_ppm: float = Field(..., ge=0.0, le=100000.0, description="Carbon monoxide concentration in parts per million")
    nox_ppm: float = Field(..., ge=0.0, le=10000.0, description="Nitrogen oxides (as NO2) concentration in parts per million")
    so2_ppm: Optional[float] = Field(default=None, ge=0.0, le=50000.0, description="Sulfur dioxide concentration in parts per million")
    co2_percent: Optional[float] = Field(default=None, ge=0.0, le=25.0, description="Carbon dioxide concentration (dry basis) as percentage")
    temperatures: List[TemperatureReading] = Field(default_factory=list, description="List of temperature readings from various sensor locations")
    pressures: List[PressureReading] = Field(default_factory=list, description="List of pressure readings from various sensor locations")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp when measurements were recorded")
    combustion_phase: CombustionPhase = Field(default=CombustionPhase.STEADY_STATE, description="Current combustion operating phase")
    data_quality_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Overall data quality score (0.0 = poor, 1.0 = excellent)")

    @computed_field
    @property
    def air_fuel_ratio(self) -> float:
        """Calculate the actual air-to-fuel ratio."""
        if self.fuel_flow_kg_per_s > 0:
            return self.air_flow_kg_per_s / self.fuel_flow_kg_per_s
        return 0.0

    @computed_field
    @property
    def excess_air_percent(self) -> float:
        """Estimate excess air percentage from O2 measurement."""
        if self.o2_percent < 21.0:
            return (self.o2_percent / (21.0 - self.o2_percent)) * 100.0
        return 0.0

    @field_validator('temperatures')
    @classmethod
    def validate_temperature_locations(cls, v: List[TemperatureReading]) -> List[TemperatureReading]:
        """Ensure no duplicate sensor IDs in temperature readings."""
        sensor_ids = [t.sensor_id for t in v]
        if len(sensor_ids) != len(set(sensor_ids)):
            raise ValueError("Duplicate temperature sensor IDs detected")
        return v

    @field_validator('pressures')
    @classmethod
    def validate_pressure_locations(cls, v: List[PressureReading]) -> List[PressureReading]:
        """Ensure no duplicate sensor IDs in pressure readings."""
        sensor_ids = [p.sensor_id for p in v]
        if len(sensor_ids) != len(set(sensor_ids)):
            raise ValueError("Duplicate pressure sensor IDs detected")
        return v

    def get_temperature(self, location: str) -> Optional[float]:
        """Get temperature value for a specific location."""
        for temp in self.temperatures:
            if temp.location == location:
                return temp.value_celsius
        return None

    def get_pressure(self, location: str) -> Optional[float]:
        """Get pressure value for a specific location."""
        for pressure in self.pressures:
            if pressure.location == location:
                return pressure.value_kpa
        return None

    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash for provenance tracking."""
        data_str = self.model_dump_json(exclude={'timestamp'})
        return hashlib.sha256(data_str.encode()).hexdigest()


class FuelComposition(BaseModel):
    """Chemical composition of fuel for combustion calculations."""
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    methane_ch4: float = Field(default=0.0, ge=0.0, le=100.0, description="Methane (CH4) content in volume percent")
    ethane_c2h6: float = Field(default=0.0, ge=0.0, le=100.0, description="Ethane (C2H6) content in volume percent")
    propane_c3h8: float = Field(default=0.0, ge=0.0, le=100.0, description="Propane (C3H8) content in volume percent")
    butane_c4h10: float = Field(default=0.0, ge=0.0, le=100.0, description="Butane (C4H10) content in volume percent")
    hydrogen_h2: float = Field(default=0.0, ge=0.0, le=100.0, description="Hydrogen (H2) content in volume percent")
    carbon_monoxide_co: float = Field(default=0.0, ge=0.0, le=100.0, description="Carbon monoxide (CO) content in volume percent")
    carbon_dioxide_co2: float = Field(default=0.0, ge=0.0, le=100.0, description="Carbon dioxide (CO2) content in volume percent")
    nitrogen_n2: float = Field(default=0.0, ge=0.0, le=100.0, description="Nitrogen (N2) content in volume percent")
    oxygen_o2: float = Field(default=0.0, ge=0.0, le=100.0, description="Oxygen (O2) content in volume percent")
    carbon_c: float = Field(default=0.0, ge=0.0, le=100.0, description="Carbon (C) content in mass percent")
    hydrogen_h: float = Field(default=0.0, ge=0.0, le=100.0, description="Hydrogen (H) content in mass percent")
    sulfur_s: float = Field(default=0.0, ge=0.0, le=100.0, description="Sulfur (S) content in mass percent")
    oxygen_o: float = Field(default=0.0, ge=0.0, le=100.0, description="Oxygen (O) content in mass percent")
    nitrogen_n: float = Field(default=0.0, ge=0.0, le=100.0, description="Nitrogen (N) content in mass percent")
    ash: float = Field(default=0.0, ge=0.0, le=100.0, description="Ash content in mass percent")
    moisture: float = Field(default=0.0, ge=0.0, le=100.0, description="Moisture content in mass percent")

    @computed_field
    @property
    def total_gaseous_percent(self) -> float:
        """Calculate total of gaseous components."""
        return (self.methane_ch4 + self.ethane_c2h6 + self.propane_c3h8 +
                self.butane_c4h10 + self.hydrogen_h2 + self.carbon_monoxide_co +
                self.carbon_dioxide_co2 + self.nitrogen_n2 + self.oxygen_o2)

    @computed_field
    @property
    def total_solid_liquid_percent(self) -> float:
        """Calculate total of solid/liquid components."""
        return (self.carbon_c + self.hydrogen_h + self.sulfur_s +
                self.oxygen_o + self.nitrogen_n + self.ash + self.moisture)


class FuelProperties(BaseModel):
    """Complete fuel properties for combustion calculations."""
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    fuel_type: FuelType = Field(..., description="Type of fuel being used")
    fuel_name: Optional[str] = Field(default=None, max_length=200, description="Custom name or description for the fuel")
    hhv_mj_per_kg: float = Field(..., gt=0.0, le=150.0, description="Higher heating value (gross) in megajoules per kilogram")
    lhv_mj_per_kg: float = Field(..., gt=0.0, le=150.0, description="Lower heating value (net) in megajoules per kilogram")
    density_kg_per_m3: float = Field(..., gt=0.0, le=2000.0, description="Fuel density in kilograms per cubic meter")
    specific_gravity: Optional[float] = Field(default=None, gt=0.0, le=5.0, description="Specific gravity relative to air/water")
    viscosity_cp: Optional[float] = Field(default=None, ge=0.0, le=100000.0, description="Dynamic viscosity in centipoise")
    wobbe_index_mj_per_m3: Optional[float] = Field(default=None, gt=0.0, le=100.0, description="Wobbe index for gas interchangeability")
    composition: Optional[FuelComposition] = Field(default=None, description="Detailed chemical composition of the fuel")
    stoich_air_fuel_ratio: Optional[float] = Field(default=None, gt=0.0, le=50.0, description="Stoichiometric air-to-fuel ratio")
    carbon_content_percent: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Carbon content of fuel in mass percent")
    source: Optional[str] = Field(default=None, max_length=500, description="Source of fuel property data")
    analysis_date: Optional[datetime] = Field(default=None, description="Date when fuel was analyzed")

    @model_validator(mode='after')
    def validate_heating_values(self) -> 'FuelProperties':
        """Ensure LHV is less than or equal to HHV."""
        if self.lhv_mj_per_kg > self.hhv_mj_per_kg:
            raise ValueError("LHV cannot exceed HHV")
        return self

    @computed_field
    @property
    def hhv_lhv_ratio(self) -> float:
        """Calculate the ratio of HHV to LHV."""
        return self.hhv_mj_per_kg / self.lhv_mj_per_kg

    @computed_field
    @property
    def co2_emission_factor_kg_per_mj(self) -> Optional[float]:
        """Calculate CO2 emission factor in kg CO2 per MJ."""
        if self.carbon_content_percent is not None:
            carbon_kg_per_kg_fuel = self.carbon_content_percent / 100.0
            co2_kg_per_kg_fuel = carbon_kg_per_kg_fuel * 3.667
            return co2_kg_per_kg_fuel / self.lhv_mj_per_kg
        return None


class FlueGasComposition(BaseModel):
    """Flue gas (exhaust gas) composition from combustion."""
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    co2_percent: float = Field(..., ge=0.0, le=25.0, description="Carbon dioxide concentration (dry basis) in volume percent")
    h2o_percent: float = Field(..., ge=0.0, le=30.0, description="Water vapor concentration (wet basis) in volume percent")
    n2_percent: float = Field(..., ge=0.0, le=90.0, description="Nitrogen concentration (dry basis) in volume percent")
    o2_percent: float = Field(..., ge=0.0, le=21.0, description="Oxygen concentration (dry basis) in volume percent")
    ar_percent: Optional[float] = Field(default=None, ge=0.0, le=5.0, description="Argon concentration (dry basis) in volume percent")
    co_ppm: float = Field(..., ge=0.0, le=100000.0, description="Carbon monoxide concentration in parts per million")
    nox_ppm: float = Field(..., ge=0.0, le=10000.0, description="Nitrogen oxides (as NO2) concentration in ppm")
    so2_ppm: Optional[float] = Field(default=None, ge=0.0, le=50000.0, description="Sulfur dioxide concentration in ppm")
    so3_ppm: Optional[float] = Field(default=None, ge=0.0, le=10000.0, description="Sulfur trioxide concentration in ppm")
    uhc_ppm: Optional[float] = Field(default=None, ge=0.0, le=50000.0, description="Unburned hydrocarbons concentration in ppm")
    temperature_celsius: Optional[float] = Field(default=None, ge=0.0, le=2000.0, description="Flue gas temperature in degrees Celsius")
    flow_rate_kg_per_s: Optional[float] = Field(default=None, ge=0.0, le=10000.0, description="Flue gas mass flow rate in kg/s")
    measurement_basis: str = Field(default="dry", description="Measurement basis: 'dry' or 'wet'")
    reference_o2_percent: Optional[float] = Field(default=None, ge=0.0, le=21.0, description="Reference O2 for normalized emissions")

    @computed_field
    @property
    def dry_basis_total(self) -> float:
        """Calculate sum of dry basis components."""
        return self.co2_percent + self.n2_percent + self.o2_percent + (self.ar_percent or 0.0)

    def correct_to_reference_o2(self, target_o2: float = 3.0) -> Dict[str, float]:
        """Correct emissions to a reference O2 level."""
        if self.o2_percent >= 21.0:
            return {"co_ppm_corrected": self.co_ppm, "nox_ppm_corrected": self.nox_ppm}
        correction_factor = (21.0 - target_o2) / (21.0 - self.o2_percent)
        return {
            "co_ppm_corrected": self.co_ppm * correction_factor,
            "nox_ppm_corrected": self.nox_ppm * correction_factor,
            "correction_factor": correction_factor,
            "reference_o2_percent": target_o2
        }


class StabilityIndicator(str, Enum):
    """Combustion stability classification."""
    STABLE = "stable"
    MARGINAL = "marginal"
    UNSTABLE = "unstable"
    CRITICAL = "critical"


class BurnerState(BaseModel):
    """Complete burner operating state at a point in time."""
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    burner_id: str = Field(..., min_length=1, max_length=50, description="Unique identifier for the burner")
    operating_point_percent: float = Field(..., ge=0.0, le=120.0, description="Current operating load as percentage of rated capacity")
    firing_rate_mw: Optional[float] = Field(default=None, ge=0.0, le=1000.0, description="Current firing rate in megawatts (thermal)")
    stability_score: float = Field(..., ge=0.0, le=1.0, description="Stability metric where 0.0 = unstable, 1.0 = perfectly stable")
    stability_indicator: StabilityIndicator = Field(default=StabilityIndicator.STABLE, description="Categorical stability classification")
    flame_detected: bool = Field(..., description="Whether flame is currently detected")
    efficiency_percent: float = Field(..., ge=0.0, le=100.0, description="Combustion/thermal efficiency as percentage")
    emissions: Optional[FlueGasComposition] = Field(default=None, description="Current flue gas composition and emissions")
    air_fuel_ratio: Optional[float] = Field(default=None, gt=0.0, le=100.0, description="Current air-to-fuel mass ratio")
    excess_air_percent: Optional[float] = Field(default=None, ge=-50.0, le=500.0, description="Current excess air percentage")
    fuel_valve_position_percent: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Fuel control valve position")
    air_damper_position_percent: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Combustion air damper position")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of this state snapshot")
    data_quality_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Overall data quality score for this state")

    @computed_field
    @property
    def is_operating(self) -> bool:
        """Determine if burner is currently operating."""
        return self.flame_detected and self.operating_point_percent > 0.0

    @computed_field
    @property
    def needs_attention(self) -> bool:
        """Determine if burner needs operator attention."""
        return (self.stability_score < 0.7 or
                self.stability_indicator in [StabilityIndicator.UNSTABLE, StabilityIndicator.CRITICAL] or
                not self.flame_detected and self.operating_point_percent > 0.0)


class ParameterLimit(BaseModel):
    """Minimum and maximum limits for a single parameter."""
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    parameter_name: str = Field(..., min_length=1, max_length=100, description="Name of the parameter")
    min_value: float = Field(..., description="Minimum allowable value")
    max_value: float = Field(..., description="Maximum allowable value")
    unit: str = Field(..., min_length=1, max_length=50, description="Engineering unit for the parameter")
    warning_margin_percent: float = Field(default=10.0, ge=0.0, le=50.0, description="Warning margin as percentage of range")

    @model_validator(mode='after')
    def validate_min_max(self) -> 'ParameterLimit':
        """Ensure min is less than max."""
        if self.min_value >= self.max_value:
            raise ValueError("min_value must be less than max_value")
        return self

    @computed_field
    @property
    def range(self) -> float:
        """Calculate the parameter range."""
        return self.max_value - self.min_value

    def is_within_limits(self, value: float) -> bool:
        """Check if a value is within the limits."""
        return self.min_value <= value <= self.max_value

    def is_in_warning_zone(self, value: float) -> bool:
        """Check if a value is in the warning zone."""
        margin = self.range * (self.warning_margin_percent / 100.0)
        return ((self.min_value <= value < self.min_value + margin) or
                (self.max_value - margin < value <= self.max_value))


class OperatingEnvelope(BaseModel):
    """Complete operating envelope defining safe/valid operating region."""
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    envelope_id: str = Field(..., min_length=1, max_length=50, description="Unique identifier for this operating envelope")
    description: str = Field(..., min_length=1, max_length=500, description="Human-readable description of this envelope")
    limits: List[ParameterLimit] = Field(..., min_length=1, description="List of parameter limits defining the envelope")
    min_turndown_percent: float = Field(default=10.0, ge=0.0, le=100.0, description="Minimum turndown as percentage of rated capacity")
    max_capacity_percent: float = Field(default=100.0, ge=0.0, le=150.0, description="Maximum capacity as percentage of rated capacity")
    ambient_temp_min_celsius: Optional[float] = Field(default=None, ge=-50.0, le=60.0, description="Minimum ambient temperature for operation")
    ambient_temp_max_celsius: Optional[float] = Field(default=None, ge=-50.0, le=60.0, description="Maximum ambient temperature for operation")
    version: str = Field(default="1.0", description="Version of this envelope definition")
    effective_date: Optional[datetime] = Field(default=None, description="Date when this envelope became effective")
    approved_by: Optional[str] = Field(default=None, max_length=100, description="Person who approved this envelope")

    @field_validator('limits')
    @classmethod
    def validate_unique_parameters(cls, v: List[ParameterLimit]) -> List[ParameterLimit]:
        """Ensure no duplicate parameter names."""
        names = [limit.parameter_name for limit in v]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate parameter names in limits")
        return v

    def get_limit(self, parameter_name: str) -> Optional[ParameterLimit]:
        """Get the limit for a specific parameter."""
        for limit in self.limits:
            if limit.parameter_name == parameter_name:
                return limit
        return None

    def check_point(self, values: Dict[str, float]) -> Dict[str, bool]:
        """Check if a set of values is within the envelope."""
        results = {}
        for limit in self.limits:
            if limit.parameter_name in values:
                results[limit.parameter_name] = limit.is_within_limits(values[limit.parameter_name])
        return results

    def is_fully_within_envelope(self, values: Dict[str, float]) -> bool:
        """Check if all values are within the envelope."""
        check_results = self.check_point(values)
        return all(check_results.values())


__all__ = [
    "FuelType",
    "CombustionPhase",
    "TemperatureReading",
    "PressureReading",
    "CombustionData",
    "FuelComposition",
    "FuelProperties",
    "FlueGasComposition",
    "StabilityIndicator",
    "BurnerState",
    "ParameterLimit",
    "OperatingEnvelope",
]
