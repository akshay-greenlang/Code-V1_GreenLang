# -*- coding: utf-8 -*-
"""
GreenLang Data Contracts

Pydantic models for enterprise-grade data validation across CBAM, emissions,
energy, and activity data. Ensures data integrity at the contract level.
"""

from datetime import datetime, date
from decimal import Decimal
from typing import Optional, List, Dict, Any, Literal, Annotated
from enum import Enum
from pydantic import BaseModel, Field, field_validator, condecimal, StringConstraints
from uuid import UUID, uuid4


class GHGScope(str, Enum):
    """GHG Protocol Scopes"""
    SCOPE_1 = "scope_1"  # Direct emissions
    SCOPE_2 = "scope_2"  # Indirect emissions from purchased energy
    SCOPE_3 = "scope_3"  # Other indirect emissions


class EmissionFactorSource(str, Enum):
    """Emission factor data sources"""
    DEFRA_2024 = "defra_2024"
    EPA_EGRID_2023 = "epa_egrid_2023"
    IPCC_2021 = "ipcc_2021"
    IEA_2023 = "iea_2023"
    CUSTOM = "custom"


class DataQualityLevel(str, Enum):
    """Data quality rating"""
    EXCELLENT = "excellent"  # 90-100%
    GOOD = "good"            # 70-89%
    FAIR = "fair"            # 50-69%
    POOR = "poor"            # <50%


class CBAMProductCategory(str, Enum):
    """CBAM covered product categories"""
    CEMENT = "cement"
    ELECTRICITY = "electricity"
    FERTILIZERS = "fertilizers"
    IRON_STEEL = "iron_steel"
    ALUMINIUM = "aluminium"
    HYDROGEN = "hydrogen"


class EnergyType(str, Enum):
    """Energy types"""
    ELECTRICITY = "electricity"
    NATURAL_GAS = "natural_gas"
    DIESEL = "diesel"
    GASOLINE = "gasoline"
    FUEL_OIL = "fuel_oil"
    COAL = "coal"
    BIOMASS = "biomass"
    SOLAR = "solar"
    WIND = "wind"
    HYDRO = "hydro"


class ActivityType(str, Enum):
    """Activity data types"""
    FUEL_COMBUSTION = "fuel_combustion"
    ELECTRICITY_CONSUMPTION = "electricity_consumption"
    MATERIAL_PROCESSING = "material_processing"
    TRANSPORT = "transport"
    REFRIGERANT_LEAKAGE = "refrigerant_leakage"
    WASTE_DISPOSAL = "waste_disposal"


# ============================================================================
# CBAM DATA CONTRACT
# ============================================================================

class CBAMDataContract(BaseModel):
    """
    Carbon Border Adjustment Mechanism (CBAM) Data Contract

    EU CBAM requires importers to report embedded emissions in imported goods.
    This contract validates CBAM declarations for goods imported into the EU.

    Reference: EU Regulation 2023/956 (CBAM Regulation)
    """

    id: UUID = Field(default_factory=uuid4, description="Unique record identifier")

    # Import Details
    importer_id: Annotated[str, StringConstraints(min_length=1, max_length=50)] = Field(
        ...,
        description="EU EORI number or tax ID of importer"
    )

    import_date: date = Field(..., description="Date goods entered EU customs")

    declaration_period: Annotated[str, StringConstraints(pattern=r"^\d{4}-Q[1-4]$")] = Field(
        ...,
        description="Reporting quarter (e.g., '2024-Q1')",
        examples=["2024-Q1"]
    )

    # Product Information
    product_category: CBAMProductCategory = Field(
        ...,
        description="CBAM product category"
    )

    cn_code: Annotated[str, StringConstraints(pattern=r"^\d{8}$")] = Field(
        ...,
        description="8-digit Combined Nomenclature code",
        example="72071100"
    )

    product_description: str = Field(
        ...,
        max_length=500,
        description="Description of imported goods"
    )

    quantity: condecimal(gt=Decimal(0), max_digits=12, decimal_places=3) = Field(
        ...,
        description="Quantity of goods imported"
    )

    quantity_unit: str = Field(
        ...,
        description="Unit of measurement (tonnes, MWh, etc.)",
        example="tonnes"
    )

    # Origin Information
    country_of_origin: Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}$")] = Field(
        ...,
        description="ISO 3166-1 alpha-2 country code",
        example="CN"
    )

    installation_id: Optional[str] = Field(
        None,
        description="ID of production installation if known"
    )

    # Embedded Emissions
    direct_emissions_co2e: condecimal(ge=Decimal(0), max_digits=12, decimal_places=3) = Field(
        ...,
        description="Direct (Scope 1) emissions in tonnes CO2e"
    )

    indirect_emissions_co2e: condecimal(ge=Decimal(0), max_digits=12, decimal_places=3) = Field(
        ...,
        description="Indirect (Scope 2) emissions in tonnes CO2e"
    )

    total_embedded_emissions: condecimal(ge=Decimal(0), max_digits=12, decimal_places=3) = Field(
        ...,
        description="Total embedded emissions (tonnes CO2e)"
    )

    specific_emissions: condecimal(ge=Decimal(0), max_digits=10, decimal_places=4) = Field(
        ...,
        description="Emissions per unit of product (tCO2e/unit)"
    )

    # Emission Factor Source
    emission_factor_source: EmissionFactorSource = Field(
        ...,
        description="Source of emission factors used"
    )

    methodology: str = Field(
        ...,
        description="Calculation methodology (e.g., 'ISO 14064-1', 'GHG Protocol')"
    )

    # Verification
    is_verified: bool = Field(
        default=False,
        description="Whether emissions data has been third-party verified"
    )

    verifier_name: Optional[str] = Field(
        None,
        description="Name of accredited verifier"
    )

    verification_date: Optional[date] = Field(
        None,
        description="Date of verification"
    )

    # Data Quality
    data_quality_level: DataQualityLevel = Field(
        ...,
        description="Overall data quality assessment"
    )

    uncertainty_percentage: Optional[condecimal(ge=Decimal(0), le=Decimal(100))] = Field(
        None,
        description="Uncertainty in emissions estimate (%)"
    )

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    @field_validator('total_embedded_emissions')
    def validate_total_emissions(cls, v, info):
        """Ensure total equals direct + indirect emissions"""
        if 'direct_emissions_co2e' in info.data and 'indirect_emissions_co2e' in info.data:
            expected_total = info.data['direct_emissions_co2e'] + info.data['indirect_emissions_co2e']
            if abs(v - expected_total) > Decimal('0.001'):  # Allow small rounding
                raise ValueError(
                    f"Total emissions {v} must equal direct {info.data['direct_emissions_co2e']} "
                    f"+ indirect {info.data['indirect_emissions_co2e']}"
                )
        return v

    @field_validator('specific_emissions')
    def validate_specific_emissions(cls, v, info):
        """Ensure specific emissions calculation is correct"""
        if 'total_embedded_emissions' in info.data and 'quantity' in info.data:
            if info.data['quantity'] > 0:
                expected = info.data['total_embedded_emissions'] / info.data['quantity']
                if abs(v - expected) > Decimal('0.0001'):
                    raise ValueError(
                        f"Specific emissions {v} must equal total {info.data['total_embedded_emissions']} "
                        f"/ quantity {info.data['quantity']}"
                    )
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "importer_id": "GB123456789000",
                "import_date": "2024-03-15",
                "declaration_period": "2024-Q1",
                "product_category": "iron_steel",
                "cn_code": "72071100",
                "product_description": "Semi-finished products of iron or non-alloy steel",
                "quantity": "1000.000",
                "quantity_unit": "tonnes",
                "country_of_origin": "CN",
                "direct_emissions_co2e": "1800.000",
                "indirect_emissions_co2e": "200.000",
                "total_embedded_emissions": "2000.000",
                "specific_emissions": "2.0000",
                "emission_factor_source": "defra_2024",
                "methodology": "ISO 14064-1",
                "is_verified": True,
                "verifier_name": "Bureau Veritas",
                "verification_date": "2024-04-01",
                "data_quality_level": "excellent"
            }
        }


# ============================================================================
# EMISSIONS DATA CONTRACT
# ============================================================================

class EmissionsDataContract(BaseModel):
    """
    GHG Emissions Data Contract

    Validates greenhouse gas emissions data across Scope 1, 2, and 3.
    Compliant with GHG Protocol and ISO 14064-1.
    """

    id: UUID = Field(default_factory=uuid4, description="Unique record identifier")

    # Organization & Period
    organization_id: Annotated[str, StringConstraints(min_length=1, max_length=100)] = Field(
        ...,
        description="Organization identifier"
    )

    facility_id: Optional[str] = Field(
        None,
        description="Facility or site identifier"
    )

    reporting_period_start: date = Field(
        ...,
        description="Start date of reporting period"
    )

    reporting_period_end: date = Field(
        ...,
        description="End date of reporting period"
    )

    # Emission Details
    ghg_scope: GHGScope = Field(
        ...,
        description="GHG Protocol scope (1, 2, or 3)"
    )

    emission_source: str = Field(
        ...,
        description="Source of emissions (e.g., 'Natural gas combustion', 'Purchased electricity')"
    )

    activity_type: ActivityType = Field(
        ...,
        description="Type of activity causing emissions"
    )

    # GHG Breakdown by Gas
    co2_tonnes: condecimal(ge=Decimal(0), max_digits=12, decimal_places=3) = Field(
        default=Decimal(0),
        description="Carbon dioxide emissions (tonnes)"
    )

    ch4_tonnes: condecimal(ge=Decimal(0), max_digits=12, decimal_places=6) = Field(
        default=Decimal(0),
        description="Methane emissions (tonnes)"
    )

    n2o_tonnes: condecimal(ge=Decimal(0), max_digits=12, decimal_places=6) = Field(
        default=Decimal(0),
        description="Nitrous oxide emissions (tonnes)"
    )

    hfcs_tonnes: condecimal(ge=Decimal(0), max_digits=12, decimal_places=6) = Field(
        default=Decimal(0),
        description="Hydrofluorocarbon emissions (tonnes)"
    )

    pfcs_tonnes: condecimal(ge=Decimal(0), max_digits=12, decimal_places=6) = Field(
        default=Decimal(0),
        description="Perfluorocarbon emissions (tonnes)"
    )

    sf6_tonnes: condecimal(ge=Decimal(0), max_digits=12, decimal_places=6) = Field(
        default=Decimal(0),
        description="Sulfur hexafluoride emissions (tonnes)"
    )

    nf3_tonnes: condecimal(ge=Decimal(0), max_digits=12, decimal_places=6) = Field(
        default=Decimal(0),
        description="Nitrogen trifluoride emissions (tonnes)"
    )

    # CO2e Total
    total_co2e_tonnes: condecimal(ge=Decimal(0), max_digits=12, decimal_places=3) = Field(
        ...,
        description="Total emissions in CO2 equivalent (tonnes)"
    )

    # Activity Data
    activity_amount: condecimal(gt=Decimal(0), max_digits=12, decimal_places=3) = Field(
        ...,
        description="Amount of activity (e.g., fuel consumed, electricity used)"
    )

    activity_unit: str = Field(
        ...,
        description="Unit of activity (kWh, liters, km, etc.)"
    )

    # Emission Factor
    emission_factor_value: condecimal(gt=Decimal(0), max_digits=12, decimal_places=6) = Field(
        ...,
        description="Emission factor used"
    )

    emission_factor_unit: str = Field(
        ...,
        description="Unit of emission factor (kgCO2e/kWh, etc.)"
    )

    emission_factor_source: EmissionFactorSource = Field(
        ...,
        description="Source of emission factor"
    )

    # Geographic & Temporal Context
    location_country: Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}$")] = Field(
        ...,
        description="ISO 3166-1 alpha-2 country code"
    )

    location_region: Optional[str] = Field(
        None,
        description="State/province/region"
    )

    # Data Quality
    data_quality_level: DataQualityLevel = Field(
        ...,
        description="Data quality assessment"
    )

    uncertainty_percentage: Optional[condecimal(ge=Decimal(0), le=Decimal(100))] = Field(
        None,
        description="Uncertainty in calculation (%)"
    )

    # Verification
    is_assured: bool = Field(
        default=False,
        description="Whether data has been externally assured"
    )

    assurance_level: Optional[Literal["limited", "reasonable"]] = Field(
        None,
        description="Level of external assurance"
    )

    # Metadata
    calculation_method: str = Field(
        ...,
        description="Calculation methodology used"
    )

    notes: Optional[str] = Field(
        None,
        max_length=1000,
        description="Additional notes"
    )

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('reporting_period_end')
    def validate_period(cls, v, info):
        """Ensure end date is after start date"""
        if 'reporting_period_start' in info.data:
            if v < info.data['reporting_period_start']:
                raise ValueError("Period end must be after period start")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "organization_id": "ORG-12345",
                "facility_id": "PLANT-A",
                "reporting_period_start": "2024-01-01",
                "reporting_period_end": "2024-12-31",
                "ghg_scope": "scope_1",
                "emission_source": "Natural gas combustion for heating",
                "activity_type": "fuel_combustion",
                "co2_tonnes": "1500.000",
                "ch4_tonnes": "0.150",
                "n2o_tonnes": "0.030",
                "total_co2e_tonnes": "1520.000",
                "activity_amount": "800000.000",
                "activity_unit": "kWh",
                "emission_factor_value": "0.001900",
                "emission_factor_unit": "tCO2e/kWh",
                "emission_factor_source": "defra_2024",
                "location_country": "GB",
                "data_quality_level": "good",
                "calculation_method": "GHG Protocol - Stationary Combustion"
            }
        }


# ============================================================================
# ENERGY DATA CONTRACT
# ============================================================================

class EnergyDataContract(BaseModel):
    """
    Energy Consumption Data Contract

    Tracks energy consumption across different sources and types.
    Used for Scope 2 emissions and energy efficiency analysis.
    """

    id: UUID = Field(default_factory=uuid4, description="Unique record identifier")

    # Organization & Period
    organization_id: Annotated[str, StringConstraints(min_length=1, max_length=100)] = Field(
        ...,
        description="Organization identifier"
    )

    facility_id: Optional[str] = Field(
        None,
        description="Facility or site identifier"
    )

    meter_id: Optional[str] = Field(
        None,
        description="Utility meter identifier"
    )

    consumption_period_start: datetime = Field(
        ...,
        description="Start of consumption period"
    )

    consumption_period_end: datetime = Field(
        ...,
        description="End of consumption period"
    )

    # Energy Details
    energy_type: EnergyType = Field(
        ...,
        description="Type of energy consumed"
    )

    consumption_amount: condecimal(gt=Decimal(0), max_digits=12, decimal_places=3) = Field(
        ...,
        description="Amount of energy consumed"
    )

    consumption_unit: str = Field(
        ...,
        description="Unit of measurement (kWh, therms, liters, etc.)"
    )

    # Cost Information
    energy_cost: Optional[condecimal(ge=Decimal(0), max_digits=12, decimal_places=2)] = Field(
        None,
        description="Cost of energy consumed"
    )

    currency: Optional[Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")]] = Field(
        None,
        description="ISO 4217 currency code",
        example="USD"
    )

    # Renewable Energy
    is_renewable: bool = Field(
        default=False,
        description="Whether energy is from renewable sources"
    )

    renewable_percentage: Optional[condecimal(ge=Decimal(0), le=Decimal(100))] = Field(
        None,
        description="Percentage of renewable energy in mix"
    )

    has_green_certificate: bool = Field(
        default=False,
        description="Whether backed by renewable energy certificate (REC/GO)"
    )

    # Grid Information (for electricity)
    grid_region: Optional[str] = Field(
        None,
        description="Electricity grid region (for location-based Scope 2)"
    )

    supplier_name: Optional[str] = Field(
        None,
        description="Energy supplier name"
    )

    supplier_emission_factor: Optional[condecimal(ge=Decimal(0), max_digits=10, decimal_places=6)] = Field(
        None,
        description="Supplier-specific emission factor (for market-based Scope 2)"
    )

    # Associated Emissions
    scope_2_location_based_co2e: Optional[condecimal(ge=Decimal(0), max_digits=12, decimal_places=3)] = Field(
        None,
        description="Location-based Scope 2 emissions (tonnes CO2e)"
    )

    scope_2_market_based_co2e: Optional[condecimal(ge=Decimal(0), max_digits=12, decimal_places=3)] = Field(
        None,
        description="Market-based Scope 2 emissions (tonnes CO2e)"
    )

    # Data Source
    data_source: Literal["utility_bill", "meter_reading", "estimate", "sub_meter"] = Field(
        ...,
        description="Source of consumption data"
    )

    data_quality_level: DataQualityLevel = Field(
        ...,
        description="Data quality assessment"
    )

    # Geographic Context
    location_country: Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}$")] = Field(
        ...,
        description="ISO 3166-1 alpha-2 country code"
    )

    location_region: Optional[str] = Field(
        None,
        description="State/province/region"
    )

    # Metadata
    notes: Optional[str] = Field(
        None,
        max_length=1000,
        description="Additional notes"
    )

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('consumption_period_end')
    def validate_period(cls, v, info):
        """Ensure end is after start"""
        if 'consumption_period_start' in info.data:
            if v <= info.data['consumption_period_start']:
                raise ValueError("Period end must be after period start")
        return v

    @field_validator('renewable_percentage')
    def validate_renewable(cls, v, info):
        """If renewable, percentage should be set"""
        if info.data.get('is_renewable') and v is None:
            raise ValueError("Renewable percentage required when is_renewable=True")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "organization_id": "ORG-12345",
                "facility_id": "PLANT-A",
                "meter_id": "METER-001",
                "consumption_period_start": "2024-01-01T00:00:00Z",
                "consumption_period_end": "2024-01-31T23:59:59Z",
                "energy_type": "electricity",
                "consumption_amount": "50000.000",
                "consumption_unit": "kWh",
                "energy_cost": "8500.00",
                "currency": "USD",
                "is_renewable": False,
                "grid_region": "WECC",
                "supplier_name": "City Power Company",
                "scope_2_location_based_co2e": "22.500",
                "scope_2_market_based_co2e": "25.000",
                "data_source": "utility_bill",
                "data_quality_level": "excellent",
                "location_country": "US",
                "location_region": "CA"
            }
        }


# ============================================================================
# ACTIVITY DATA CONTRACT
# ============================================================================

class ActivityDataContract(BaseModel):
    """
    Activity Data Contract

    Raw activity data that drives emissions calculations.
    Includes fuel consumption, electricity use, materials, transport, etc.
    """

    id: UUID = Field(default_factory=uuid4, description="Unique record identifier")

    # Organization & Period
    organization_id: Annotated[str, StringConstraints(min_length=1, max_length=100)] = Field(
        ...,
        description="Organization identifier"
    )

    facility_id: Optional[str] = Field(
        None,
        description="Facility or site identifier"
    )

    activity_date: date = Field(
        ...,
        description="Date of activity"
    )

    # Activity Details
    activity_type: ActivityType = Field(
        ...,
        description="Type of activity"
    )

    activity_description: str = Field(
        ...,
        max_length=500,
        description="Description of activity"
    )

    activity_amount: condecimal(gt=Decimal(0), max_digits=12, decimal_places=3) = Field(
        ...,
        description="Quantity of activity"
    )

    activity_unit: str = Field(
        ...,
        description="Unit of measurement"
    )

    # Context
    asset_id: Optional[str] = Field(
        None,
        description="Asset or equipment identifier"
    )

    process_name: Optional[str] = Field(
        None,
        description="Process or operation name"
    )

    # For Transport Activities
    transport_mode: Optional[Literal["road", "rail", "air", "sea"]] = Field(
        None,
        description="Mode of transport"
    )

    vehicle_type: Optional[str] = Field(
        None,
        description="Type of vehicle (truck, car, ship, etc.)"
    )

    distance_km: Optional[condecimal(ge=Decimal(0), max_digits=12, decimal_places=2)] = Field(
        None,
        description="Distance traveled (km)"
    )

    load_weight_tonnes: Optional[condecimal(ge=Decimal(0), max_digits=12, decimal_places=3)] = Field(
        None,
        description="Weight of load transported (tonnes)"
    )

    # For Fuel Activities
    fuel_type: Optional[str] = Field(
        None,
        description="Type of fuel (diesel, natural gas, etc.)"
    )

    fuel_density: Optional[condecimal(gt=Decimal(0), max_digits=8, decimal_places=4)] = Field(
        None,
        description="Fuel density for volume to mass conversion"
    )

    # For Material Processing
    material_type: Optional[str] = Field(
        None,
        description="Type of material processed"
    )

    material_weight_tonnes: Optional[condecimal(ge=Decimal(0), max_digits=12, decimal_places=3)] = Field(
        None,
        description="Weight of material (tonnes)"
    )

    # Data Quality
    data_source: Literal[
        "direct_measurement",
        "utility_bill",
        "invoice",
        "estimate",
        "calculation"
    ] = Field(
        ...,
        description="Source of activity data"
    )

    data_quality_level: DataQualityLevel = Field(
        ...,
        description="Data quality assessment"
    )

    is_estimated: bool = Field(
        default=False,
        description="Whether data is estimated vs measured"
    )

    # Geographic Context
    location_country: Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}$")] = Field(
        ...,
        description="ISO 3166-1 alpha-2 country code"
    )

    location_latitude: Optional[condecimal(ge=Decimal(-90), le=Decimal(90))] = Field(
        None,
        description="Latitude of activity location"
    )

    location_longitude: Optional[condecimal(ge=Decimal(-180), le=Decimal(180))] = Field(
        None,
        description="Longitude of activity location"
    )

    # Emission Linkage
    linked_emission_id: Optional[UUID] = Field(
        None,
        description="ID of associated emission record"
    )

    # Metadata
    notes: Optional[str] = Field(
        None,
        max_length=1000,
        description="Additional notes"
    )

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "organization_id": "ORG-12345",
                "facility_id": "PLANT-A",
                "activity_date": "2024-01-15",
                "activity_type": "fuel_combustion",
                "activity_description": "Natural gas combustion in boiler for process heat",
                "activity_amount": "15000.000",
                "activity_unit": "m3",
                "asset_id": "BOILER-001",
                "process_name": "Steam generation",
                "fuel_type": "natural_gas",
                "data_source": "direct_measurement",
                "data_quality_level": "excellent",
                "is_estimated": False,
                "location_country": "US"
            }
        }
