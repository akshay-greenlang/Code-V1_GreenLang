"""
PCF Exchange Data Models
PACT Pathfinder v2.0 Compatible Models

Models for Product Carbon Footprint exchange following:
- PACT Technical Specifications v2.0
- ISO 14067:2018
- GHG Protocol Product Standard

Version: 1.0.0
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, validator


class DataQualityRating(str, Enum):
    """Data Quality Rating per PACT Pathfinder v2.0."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


class CrossSectoralStandard(str, Enum):
    """Cross-sectoral standards for PCF calculation."""
    GHG_PROTOCOL = "GHG Protocol Product standard"
    ISO_14067 = "ISO 14067"
    ISO_14044 = "ISO 14044"


class BiogenicAccountingMethodology(str, Enum):
    """Biogenic carbon accounting methodology."""
    PEF = "PEF"
    ISO_14067 = "ISO 14067"
    GHGP = "GHG Protocol"
    QUANTIS = "Quantis"


class ProductOrSectorSpecificRule(BaseModel):
    """Product or sector-specific rule."""
    operator: str = Field(description="Operator of the PCR")
    rule_names: List[str] = Field(description="Names of the rules")
    other_operator_name: Optional[str] = Field(default=None)


class CarbonFootprint(BaseModel):
    """
    Carbon Footprint per PACT Pathfinder v2.0.

    Represents the carbon footprint of a product according to the
    Pathfinder Framework.
    """
    declared_unit: str = Field(
        description="Unit of measurement (e.g., '1 kilogram')"
    )

    unitary_product_amount: float = Field(
        gt=0,
        description="Amount of product for this footprint"
    )

    p_cf_excluding_biogenic: float = Field(
        description="PCF excluding biogenic carbon (kgCO2e)"
    )

    p_cf_including_biogenic: Optional[float] = Field(
        default=None,
        description="PCF including biogenic carbon (kgCO2e)"
    )

    fossil_ghg_emissions: float = Field(
        description="Fossil GHG emissions (kgCO2e)"
    )

    fossil_carbon_content: Optional[float] = Field(
        default=None,
        description="Fossil carbon content (kgC)"
    )

    biogenic_carbon_content: Optional[float] = Field(
        default=None,
        description="Biogenic carbon content (kgC)"
    )

    d_luc_ghg_emissions: Optional[float] = Field(
        default=None,
        description="Direct land use change emissions (kgCO2e)"
    )

    land_management_ghg_emissions: Optional[float] = Field(
        default=None,
        description="Land management emissions (kgCO2e)"
    )

    other_biogenic_ghg_emissions: Optional[float] = Field(
        default=None,
        description="Other biogenic emissions (kgCO2e)"
    )

    i_luc_ghg_emissions: Optional[float] = Field(
        default=None,
        description="Indirect land use change emissions (kgCO2e)"
    )

    biogenic_carbon_withdrawal: Optional[float] = Field(
        default=None,
        description="Biogenic carbon withdrawal (kgCO2)"
    )

    aircraft_ghg_emissions: Optional[float] = Field(
        default=None,
        description="Aircraft GHG emissions (kgCO2e)"
    )

    characterization_factors: str = Field(
        default="AR6",
        description="IPCC Assessment Report (AR5, AR6)"
    )

    cross_sectoral_standards_used: List[CrossSectoralStandard] = Field(
        description="Cross-sectoral standards used"
    )

    product_or_sector_specific_rules: Optional[List[ProductOrSectorSpecificRule]] = Field(
        default=None,
        description="Product or sector-specific rules"
    )

    biogenic_accounting_methodology: Optional[BiogenicAccountingMethodology] = Field(
        default=None,
        description="Biogenic accounting methodology"
    )

    boundary_processes_description: str = Field(
        description="Description of system boundary"
    )

    reference_period_start: datetime = Field(
        description="Start of reference period"
    )

    reference_period_end: datetime = Field(
        description="End of reference period"
    )

    geography_country_subdivision: Optional[str] = Field(
        default=None,
        description="ISO 3166-2 subdivision code"
    )

    geography_country: Optional[str] = Field(
        default=None,
        description="ISO 3166-1 alpha-2 country code"
    )

    geography_region_or_subregion: Optional[str] = Field(
        default=None,
        description="UN M.49 region code"
    )

    secondary_emission_factor_sources: Optional[List[str]] = Field(
        default=None,
        description="Secondary emission factor sources"
    )

    exempted_emissions_percent: float = Field(
        default=0.0,
        ge=0,
        le=5,
        description="Percentage of exempted emissions (max 5%)"
    )

    exempted_emissions_description: Optional[str] = Field(
        default=None,
        description="Description of exempted emissions"
    )

    packaging_emissions_included: bool = Field(
        description="Whether packaging emissions are included"
    )

    packaging_ghg_emissions: Optional[float] = Field(
        default=None,
        description="Packaging GHG emissions if separately reported"
    )

    allocation_rules_description: Optional[str] = Field(
        default=None,
        description="Description of allocation rules"
    )

    uncertainty_assessment_description: Optional[str] = Field(
        default=None,
        description="Description of uncertainty assessment"
    )

    primary_data_share: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Share of primary data (%)"
    )

    dqi: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Data Quality Indicators"
    )

    assurance: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Assurance information"
    )


class PCFDataModel(BaseModel):
    """
    Product Carbon Footprint Data Model.

    Complete PCF data model following PACT Pathfinder v2.0 specification.
    """
    id: str = Field(
        description="Unique PCF identifier (UUID)"
    )

    spec_version: str = Field(
        default="2.0.0",
        description="PACT specification version"
    )

    version: int = Field(
        default=1,
        ge=1,
        description="Version of this PCF"
    )

    created: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation timestamp"
    )

    updated: Optional[datetime] = Field(
        default=None,
        description="Last update timestamp"
    )

    status: str = Field(
        default="Active",
        description="PCF status (Active, Deprecated)"
    )

    company_name: str = Field(
        description="Company name"
    )

    company_ids: List[str] = Field(
        description="Company identifiers (e.g., DUNS, LEI)"
    )

    product_description: str = Field(
        description="Product description"
    )

    product_ids: List[str] = Field(
        description="Product identifiers (e.g., GTIN, CAS)"
    )

    product_category_cpc: str = Field(
        description="CPC (Central Product Classification) code"
    )

    product_name_company: str = Field(
        description="Company-specific product name"
    )

    comment: Optional[str] = Field(
        default=None,
        description="Additional comments"
    )

    pcf: CarbonFootprint = Field(
        description="Carbon footprint data"
    )

    preceding_pf_ids: Optional[List[str]] = Field(
        default=None,
        description="IDs of preceding PCFs"
    )

    extensions: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Extension data"
    )


class PCFExchangeRequest(BaseModel):
    """Request for PCF exchange."""

    pcf_id: Optional[str] = Field(
        default=None,
        description="PCF ID to fetch (for import)"
    )

    pcf_data: Optional[PCFDataModel] = Field(
        default=None,
        description="PCF data (for export)"
    )

    operation: str = Field(
        description="Operation type (import, export, validate)"
    )

    target_system: str = Field(
        description="Target system (pact, catenax, sap_sdx)"
    )

    validate_only: bool = Field(
        default=False,
        description="Only validate, don't exchange"
    )


class PCFExchangeResponse(BaseModel):
    """Response from PCF exchange."""

    success: bool = Field(
        description="Whether operation succeeded"
    )

    pcf_data: Optional[PCFDataModel] = Field(
        default=None,
        description="PCF data (for import)"
    )

    validation_errors: Optional[List[str]] = Field(
        default=None,
        description="Validation errors if any"
    )

    warnings: Optional[List[str]] = Field(
        default=None,
        description="Warnings"
    )

    exchange_id: Optional[str] = Field(
        default=None,
        description="Exchange transaction ID"
    )

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp"
    )
