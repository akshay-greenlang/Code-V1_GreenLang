# -*- coding: utf-8 -*-
"""
GreenLang Data Engineering Module

Provides data contracts, ETL pipelines, quality checks, and emission factor management.
"""

# EUDR Commodities Database
from .eudr_commodities import (
    CommodityType,
    RiskCategory,
    TraceabilityLevel,
    EUDR_CUTOFF_DATE,
    CNCode,
    Commodity,
    EUDR_COMMODITIES,
    CN_CODE_DATABASE,
    get_commodity_by_cn_code,
    is_eudr_regulated,
    get_commodity_type,
    get_traceability_requirements,
    get_risk_category,
    get_all_regulated_cn_codes,
    get_cn_codes_by_commodity,
    get_commodity_info,
    classify_cn_code,
)

# EUDR Country Risk Database
from .eudr_country_risk import (
    RiskLevel,
    DueDiligenceLevel,
    ForestType,
    ForestData,
    CountryRisk,
    RegionRisk,
    COUNTRY_RISK_DATABASE,
    REGION_RISK_DATABASE,
    get_country_risk,
    get_risk_level,
    get_risk_score,
    get_commodity_risk,
    get_due_diligence_level,
    requires_satellite_verification,
    get_regions_of_concern,
    get_region_risks,
    is_deforestation_hotspot,
    get_forest_data,
    assess_country_risk,
    get_high_risk_countries,
    get_low_risk_countries,
)

# Comment out imports that have missing dependencies for now
# from .contracts import (
#     CBAMDataContract,
#     EmissionsDataContract,
#     EnergyDataContract,
#     ActivityDataContract,
#     GHGScope,
#     EmissionFactorSource,
#     DataQualityLevel
# )
#
# from .emission_factors import (
#     EmissionFactorLoader,
#     EmissionFactor,
#     load_defra_factors,
#     load_epa_egrid_factors
# )
#
# from .quality import (
#     DataQualityChecker,
#     DataQualityReport,
#     QualityCheck,
#     check_data_quality
# )
#
# from .sample_data import (
#     SampleDataGenerator,
#     generate_cbam_sample,
#     generate_emissions_sample,
#     generate_energy_sample
# )

__all__ = [
    # EUDR Commodities
    "CommodityType",
    "RiskCategory",
    "TraceabilityLevel",
    "EUDR_CUTOFF_DATE",
    "CNCode",
    "Commodity",
    "EUDR_COMMODITIES",
    "CN_CODE_DATABASE",
    "get_commodity_by_cn_code",
    "is_eudr_regulated",
    "get_commodity_type",
    "get_traceability_requirements",
    "get_risk_category",
    "get_all_regulated_cn_codes",
    "get_cn_codes_by_commodity",
    "get_commodity_info",
    "classify_cn_code",
    # EUDR Country Risk
    "RiskLevel",
    "DueDiligenceLevel",
    "ForestType",
    "ForestData",
    "CountryRisk",
    "RegionRisk",
    "COUNTRY_RISK_DATABASE",
    "REGION_RISK_DATABASE",
    "get_country_risk",
    "get_risk_level",
    "get_risk_score",
    "get_commodity_risk",
    "get_due_diligence_level",
    "requires_satellite_verification",
    "get_regions_of_concern",
    "get_region_risks",
    "is_deforestation_hotspot",
    "get_forest_data",
    "assess_country_risk",
    "get_high_risk_countries",
    "get_low_risk_countries",
]

__version__ = "1.0.0"
