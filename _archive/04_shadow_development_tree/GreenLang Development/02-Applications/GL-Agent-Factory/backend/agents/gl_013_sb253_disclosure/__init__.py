"""
GL-013: California SB 253 Climate Disclosure Agent

This package provides the SB253DisclosureAgent for calculating and reporting
GHG emissions in compliance with California's Climate Corporate Data
Accountability Act (SB 253).

Key Features:
- Scope 1 direct emissions (stationary, mobile, process, fugitive)
- Scope 2 indirect emissions (location-based and market-based)
- Scope 3 value chain emissions (all 15 GHG Protocol categories)
- CARB portal filing format generation
- Third-party assurance package preparation
- Complete SHA-256 provenance tracking

Regulatory Timeline:
- Revenue threshold: >$1B USD
- First Scope 1&2 reports: June 30, 2026
- First Scope 3 reports: June 30, 2027
- Enforcement: California Air Resources Board (CARB)

Example Usage:
    >>> from backend.agents.gl_013_sb253_disclosure import (
    ...     SB253DisclosureAgent,
    ...     SB253ReportInput,
    ...     CompanyInfo,
    ...     Scope1Source,
    ...     Scope2Source,
    ...     FuelType,
    ...     FuelUnit,
    ...     SourceCategory,
    ... )
    >>>
    >>> agent = SB253DisclosureAgent()
    >>> input_data = SB253ReportInput(
    ...     company_info=CompanyInfo(
    ...         company_name="California Manufacturing Inc.",
    ...         ein="12-3456789",
    ...         total_revenue_usd=2_500_000_000,
    ...         naics_code="331110",
    ...     ),
    ...     fiscal_year=2025,
    ...     scope1_sources=[
    ...         Scope1Source(
    ...             facility_id="FAC001",
    ...             source_category=SourceCategory.STATIONARY_COMBUSTION,
    ...             fuel_type=FuelType.NATURAL_GAS,
    ...             quantity=10000.0,
    ...             unit=FuelUnit.THERMS,
    ...         )
    ...     ],
    ...     scope2_sources=[
    ...         Scope2Source(
    ...             facility_id="FAC001",
    ...             kwh=2_000_000.0,
    ...             egrid_subregion="CAMX",
    ...         )
    ...     ],
    ... )
    >>> result = agent.run(input_data)
    >>> print(f"Total emissions: {result.total_emissions_mtco2e} MTCO2e")
"""

from .agent import (
    # Main Agent
    SB253DisclosureAgent,

    # Input Models
    SB253ReportInput,
    CompanyInfo,
    FacilityInfo,
    Scope1Source,
    Scope2Source,
    Scope3Data,
    Scope3CategoryData,
    ReportingPeriod,

    # Output Models
    SB253ReportOutput,
    EmissionBreakdown,
    Scope1Result,
    Scope2Result,
    Scope3Result,
    Scope3CategoryResult,
    AssurancePackage,
    CARBFilingData,
    ProvenanceRecord,

    # Emission Factor Models
    Scope1EmissionFactor,
    EGridFactor,
    EEIOFactor,

    # Enumerations
    OrganizationalBoundary,
    FuelType,
    FuelUnit,
    SourceCategory,
    Scope2Method,
    Scope3Category,
    CalculationMethod,
    DataQualityScore,
    GWPSet,
    AssuranceLevel,
    RefrigerantType,

    # Pack Specification
    PACK_SPEC,
)

__all__ = [
    # Main Agent
    "SB253DisclosureAgent",

    # Input Models
    "SB253ReportInput",
    "CompanyInfo",
    "FacilityInfo",
    "Scope1Source",
    "Scope2Source",
    "Scope3Data",
    "Scope3CategoryData",
    "ReportingPeriod",

    # Output Models
    "SB253ReportOutput",
    "EmissionBreakdown",
    "Scope1Result",
    "Scope2Result",
    "Scope3Result",
    "Scope3CategoryResult",
    "AssurancePackage",
    "CARBFilingData",
    "ProvenanceRecord",

    # Emission Factor Models
    "Scope1EmissionFactor",
    "EGridFactor",
    "EEIOFactor",

    # Enumerations
    "OrganizationalBoundary",
    "FuelType",
    "FuelUnit",
    "SourceCategory",
    "Scope2Method",
    "Scope3Category",
    "CalculationMethod",
    "DataQualityScore",
    "GWPSet",
    "AssuranceLevel",
    "RefrigerantType",

    # Pack Specification
    "PACK_SPEC",
]

# Agent metadata
__version__ = "1.0.0"
__agent_id__ = "regulatory/sb253_disclosure_v1"
