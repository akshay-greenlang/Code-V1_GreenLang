"""
PACK-014 CSRD Retail & Consumer Goods Pack - Configuration Module

This module exports all configuration classes, enums, constants, and utility
functions for the CSRD Retail Pack. Import from this module to access the
full configuration API.

Usage:
    >>> from packs.eu_compliance.PACK_014_csrd_retail.config import (
    ...     PackConfig,
    ...     CSRDRetailConfig,
    ...     RetailSubSector,
    ...     RetailTier,
    ...     get_subsector_info,
    ...     get_default_config,
    ... )
    >>> config = PackConfig.from_preset("grocery_retail")
    >>> print(config.pack.sub_sectors)
    [RetailSubSector.GROCERY]
"""

from .pack_config import (
    # Enums
    ComplianceStatus,
    DisclosureFormat,
    DueDiligenceRisk,
    EPRScheme,
    ESRSTopic,
    EUDRCommodity,
    FoodWasteCategory,
    GreenClaimType,
    PackagingMaterial,
    RefrigerantType,
    ReportingFrequency,
    RetailSubSector,
    RetailTier,
    Scope3Priority,
    StoreType,
    SupplierTier,
    # Sub-config models
    AuditTrailConfig,
    BenchmarkConfig,
    CircularEconomyConfig,
    DisclosureConfig,
    FoodWasteConfig,
    OmnibusConfig,
    PackagingConfig,
    ProductSustainabilityConfig,
    RetailScope3Config,
    StoreConfig,
    StoreEmissionsConfig,
    SupplyChainDDConfig,
    # Main config models
    CSRDRetailConfig,
    PackConfig,
    # Constants
    AVAILABLE_PRESETS,
    EPR_RECYCLING_TARGETS,
    FGAS_GWP_VALUES,
    PPWR_RECYCLED_CONTENT_TARGETS,
    PRIORITY_SCOPE3_BY_SUBSECTOR,
    RETAIL_SCOPE3_PRIORITY,
    SUBSECTOR_INFO,
    # Directories
    CONFIG_DIR,
    PACK_BASE_DIR,
    # Utility functions
    get_default_config,
    get_fgas_gwp,
    get_ppwr_target,
    get_subsector_info,
    list_available_presets,
    load_preset,
    validate_config,
)

__all__ = [
    # Enums
    "ComplianceStatus",
    "DisclosureFormat",
    "DueDiligenceRisk",
    "EPRScheme",
    "ESRSTopic",
    "EUDRCommodity",
    "FoodWasteCategory",
    "GreenClaimType",
    "PackagingMaterial",
    "RefrigerantType",
    "ReportingFrequency",
    "RetailSubSector",
    "RetailTier",
    "Scope3Priority",
    "StoreType",
    "SupplierTier",
    # Sub-config models
    "AuditTrailConfig",
    "BenchmarkConfig",
    "CircularEconomyConfig",
    "DisclosureConfig",
    "FoodWasteConfig",
    "OmnibusConfig",
    "PackagingConfig",
    "ProductSustainabilityConfig",
    "RetailScope3Config",
    "StoreConfig",
    "StoreEmissionsConfig",
    "SupplyChainDDConfig",
    # Main config models
    "CSRDRetailConfig",
    "PackConfig",
    # Constants
    "AVAILABLE_PRESETS",
    "EPR_RECYCLING_TARGETS",
    "FGAS_GWP_VALUES",
    "PPWR_RECYCLED_CONTENT_TARGETS",
    "PRIORITY_SCOPE3_BY_SUBSECTOR",
    "RETAIL_SCOPE3_PRIORITY",
    "SUBSECTOR_INFO",
    # Directories
    "CONFIG_DIR",
    "PACK_BASE_DIR",
    # Utility functions
    "get_default_config",
    "get_fgas_gwp",
    "get_ppwr_target",
    "get_subsector_info",
    "list_available_presets",
    "load_preset",
    "validate_config",
]
