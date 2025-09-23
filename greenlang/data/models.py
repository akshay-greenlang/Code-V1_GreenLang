from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Literal
from enum import Enum
from datetime import datetime


class BuildingType(str, Enum):
    COMMERCIAL_OFFICE = "commercial_office"
    RETAIL = "retail"
    HOSPITAL = "hospital"
    DATA_CENTER = "data_center"
    WAREHOUSE = "warehouse"
    HOTEL = "hotel"
    EDUCATION = "education"
    RESTAURANT = "restaurant"
    MULTIFAMILY = "multifamily"
    INDUSTRIAL = "industrial"
    OTHER = "other"


class ClimateZone(str, Enum):
    TROPICAL = "tropical"
    DRY = "dry"
    TEMPERATE = "temperate"
    CONTINENTAL = "continental"
    POLAR = "polar"
    ZONE_1A = "1A"  # Very Hot-Humid
    ZONE_2A = "2A"  # Hot-Humid
    ZONE_3A = "3A"  # Warm-Humid
    ZONE_4A = "4A"  # Mixed-Humid
    ZONE_5A = "5A"  # Cool-Humid
    ZONE_6A = "6A"  # Cold-Humid
    ZONE_7 = "7"  # Very Cold
    ZONE_8 = "8"  # Subarctic


class HVACType(str, Enum):
    CENTRAL_AC = "central_ac"
    VRF = "vrf"
    SPLIT = "split"
    CHILLER = "chiller"
    HEAT_PUMP = "heat_pump"
    PACKAGED_ROOFTOP = "packaged_rooftop"
    GEOTHERMAL = "geothermal"
    OTHER = "other"


class GlazingType(str, Enum):
    SINGLE = "single"
    DOUBLE = "double"
    TRIPLE = "triple"
    LOW_E = "low_e"
    TINTED = "tinted"


class Country(str, Enum):
    USA = "US"
    INDIA = "IN"
    EU = "EU"
    CHINA = "CN"
    JAPAN = "JP"
    SOUTH_KOREA = "KR"
    BRAZIL = "BR"
    UK = "UK"
    GERMANY = "DE"
    FRANCE = "FR"
    CANADA = "CA"
    AUSTRALIA = "AU"
    SINGAPORE = "SG"
    UAE = "AE"


class BuildingMetadata(BaseModel):
    building_type: BuildingType
    area: float = Field(
        ..., description="Building area in square feet or square meters"
    )
    area_unit: Literal["sqft", "sqm"] = "sqft"
    location: Dict[str, str] = Field(..., description="Country, region, city")
    occupancy: Optional[int] = Field(None, description="Average daily occupancy")
    floor_count: Optional[int] = Field(None, description="Number of floors")
    building_age: Optional[int] = Field(None, description="Building age in years")
    climate_zone: Optional[ClimateZone] = None
    construction_year: Optional[int] = None
    last_retrofit_year: Optional[int] = None
    operating_hours: Optional[float] = Field(
        None, description="Operating hours per day"
    )

    @validator("location")
    def validate_location(cls, v):
        if "country" not in v:
            raise ValueError("Country is required in location")
        return v

    @validator("area")
    def validate_area(cls, v):
        if v <= 0:
            raise ValueError("Area must be positive")
        return v


class EnergyConsumption(BaseModel):
    electricity: Optional[Dict[str, float]] = Field(
        None, description="Electricity consumption"
    )
    natural_gas: Optional[Dict[str, float]] = Field(
        None, description="Natural gas consumption"
    )
    diesel: Optional[Dict[str, float]] = Field(None, description="Diesel consumption")
    lpg_propane: Optional[Dict[str, float]] = Field(
        None, description="LPG/Propane consumption"
    )
    heating_oil: Optional[Dict[str, float]] = Field(
        None, description="Heating oil consumption"
    )
    district_heating: Optional[Dict[str, float]] = Field(
        None, description="District heating"
    )
    solar_pv_generation: Optional[Dict[str, float]] = Field(
        None, description="Solar PV generation"
    )
    renewable_energy: Optional[Dict[str, float]] = Field(
        None, description="Renewable energy"
    )
    coal: Optional[Dict[str, float]] = Field(None, description="Coal consumption")
    biomass: Optional[Dict[str, float]] = Field(None, description="Biomass consumption")

    period: Literal["annual", "monthly", "daily"] = "annual"
    period_details: Optional[Dict[str, Any]] = None  # Year, month, date ranges


class HVACSystem(BaseModel):
    hvac_type: Optional[HVACType] = None
    hvac_efficiency_cop: Optional[float] = Field(None, ge=1.0, le=10.0)
    hvac_efficiency_eer: Optional[float] = Field(None, ge=5.0, le=20.0)
    hvac_efficiency_seer: Optional[float] = Field(None, ge=8.0, le=30.0)
    cooling_capacity: Optional[float] = Field(
        None, description="Cooling capacity in tons"
    )
    heating_capacity: Optional[float] = Field(
        None, description="Heating capacity in kW"
    )
    age: Optional[int] = Field(None, description="System age in years")


class BuildingEnvelope(BaseModel):
    window_to_wall_ratio: Optional[float] = Field(None, ge=0.0, le=1.0)
    insulation_r_value: Optional[float] = Field(None, description="R-value or U-value")
    glazing_type: Optional[GlazingType] = None
    roof_type: Optional[str] = None
    roof_reflectance: Optional[float] = Field(None, ge=0.0, le=1.0)
    air_leakage_rate: Optional[float] = Field(None, description="ACH50 value")


class ApplianceLoad(BaseModel):
    lighting_load: Optional[float] = Field(
        None, description="Annual lighting load in kWh"
    )
    it_load: Optional[float] = Field(None, description="IT equipment load in kWh")
    water_heating_load: Optional[float] = Field(
        None, description="Water heating load in kWh"
    )
    plug_load: Optional[float] = Field(None, description="Plug load in kWh")
    elevator_load: Optional[float] = Field(None, description="Elevator load in kWh")
    kitchen_load: Optional[float] = Field(
        None, description="Kitchen equipment load in kWh"
    )


class BuildingInput(BaseModel):
    metadata: BuildingMetadata
    energy_consumption: EnergyConsumption
    hvac_system: Optional[HVACSystem] = None
    building_envelope: Optional[BuildingEnvelope] = None
    appliance_loads: Optional[ApplianceLoad] = None
    renewable_systems: Optional[Dict[str, Any]] = None
    water_consumption: Optional[Dict[str, float]] = None
    waste_generation: Optional[Dict[str, float]] = None
    transportation: Optional[Dict[str, Any]] = None

    class Config:
        schema_extra = {
            "example": {
                "metadata": {
                    "building_type": "commercial_office",
                    "area": 50000,
                    "area_unit": "sqft",
                    "location": {
                        "country": "US",
                        "region": "California",
                        "city": "San Francisco",
                    },
                    "occupancy": 200,
                    "floor_count": 10,
                    "building_age": 15,
                    "climate_zone": "3A",
                },
                "energy_consumption": {
                    "electricity": {"value": 500000, "unit": "kWh"},
                    "natural_gas": {"value": 10000, "unit": "therms"},
                    "period": "annual",
                },
            }
        }


class EmissionResult(BaseModel):
    total_emissions_kgco2e: float
    emissions_by_source: Dict[str, float]
    emissions_intensity: Optional[Dict[str, float]] = None
    benchmark_rating: Optional[str] = None
    recommendations: Optional[List[str]] = None
    period: str
    calculation_date: datetime = Field(default_factory=datetime.now)
    methodology: str = "IPCC 2021"
    uncertainty_range: Optional[Dict[str, float]] = None


class BenchmarkData(BaseModel):
    building_type: BuildingType
    country: Country
    excellent_threshold: float  # kgCO2e/sqft/year
    good_threshold: float
    average_threshold: float
    poor_threshold: float
    source: str
    year: int
    notes: Optional[str] = None
