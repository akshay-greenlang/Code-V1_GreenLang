"""
PACK-014 CSRD Retail & Consumer Goods Pack - Configuration Manager

This module implements the CSRDRetailConfig and PackConfig classes that load,
merge, and validate all configuration for the CSRD Retail Pack. It provides
comprehensive Pydantic v2 models for every aspect of retail sector CSRD
compliance: store-level emissions, Scope 3 supply chain mapping, PPWR packaging
compliance, product sustainability (DPP, PEF, ECGT), food waste management,
CSDDD/EUDR supply chain due diligence, retail circular economy (EPR), and
sector benchmarking.

Retail Sub-Sectors:
    - GROCERY: Supermarkets, hypermarkets, convenience stores
    - APPAREL: Fashion, textile, footwear, luxury retail
    - ELECTRONICS: Consumer electronics, IT, appliances
    - HOME_FURNISHING: Furniture, home decor, DIY
    - DEPARTMENT_STORE: Multi-category department stores
    - CONVENIENCE: Small-format convenience and neighborhood stores
    - HYPERMARKET: Large-format hypermarkets and superstores
    - SPECIALTY: Specialty retail (sport, beauty, pet, garden)
    - E_COMMERCE: Online-only, marketplace, direct-to-consumer
    - WHOLESALE: Cash & carry, wholesale distribution
    - PHARMACY: Pharmacy and health retail
    - DIY_HARDWARE: DIY, hardware, building materials
    - LUXURY: Luxury goods, jewelry, watches
    - DISCOUNT: Discount and value retailers
    - FOOD_SERVICE: Quick-service, cafeterias, catering
    - OTHER: Other retail sub-sectors

Retail Tiers:
    - ENTERPRISE: Large listed retailers (>10,000 employees, >EUR 1B revenue)
    - MID_MARKET: Mid-sized retailers (1,000-10,000 employees)
    - SME: Small and medium retailers (<1,000 employees)
    - FRANCHISE: Franchise-based retail networks

Configuration Merge Order (later overrides earlier):
    1. Base pack.yaml manifest
    2. Preset YAML (grocery_retail / apparel_retail / electronics_retail /
       general_retail / online_retail / sme_retailer)
    3. Environment overrides (CSRD_RETAIL_PACK_* environment variables)
    4. Explicit runtime overrides

Regulatory Context:
    - CSRD: Directive (EU) 2022/2464
    - ESRS: Delegated Regulation (EU) 2023/2772 (Set 1)
    - PPWR: Regulation (EU) 2025/40
    - EUDR: Regulation (EU) 2023/1115
    - CSDDD: Directive (EU) 2024/1760
    - ESPR: Regulation (EU) 2024/1781
    - ECGT: Directive (EU) 2024/825
    - EED: Directive (EU) 2023/1791
    - F-Gas: Regulation (EU) 2024/573

Example:
    >>> config = PackConfig.from_preset("grocery_retail")
    >>> print(config.pack.sub_sectors)
    [RetailSubSector.GROCERY]
    >>> print(config.pack.food_waste.enabled)
    True
    >>> print(config.pack.store_emissions.refrigerant_tracking)
    True
"""

import hashlib
import json
import logging
import os
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# Base directory for all pack configuration files
PACK_BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = Path(__file__).parent


# =============================================================================
# Enums - Retail-specific enumeration types (16 enums)
# =============================================================================


class RetailSubSector(str, Enum):
    """Retail sub-sector classification."""

    GROCERY = "GROCERY"
    APPAREL = "APPAREL"
    ELECTRONICS = "ELECTRONICS"
    HOME_FURNISHING = "HOME_FURNISHING"
    DEPARTMENT_STORE = "DEPARTMENT_STORE"
    CONVENIENCE = "CONVENIENCE"
    HYPERMARKET = "HYPERMARKET"
    SPECIALTY = "SPECIALTY"
    E_COMMERCE = "E_COMMERCE"
    WHOLESALE = "WHOLESALE"
    PHARMACY = "PHARMACY"
    DIY_HARDWARE = "DIY_HARDWARE"
    LUXURY = "LUXURY"
    DISCOUNT = "DISCOUNT"
    FOOD_SERVICE = "FOOD_SERVICE"
    OTHER = "OTHER"


class RetailTier(str, Enum):
    """Retail company tier classification."""

    ENTERPRISE = "ENTERPRISE"
    MID_MARKET = "MID_MARKET"
    SME = "SME"
    FRANCHISE = "FRANCHISE"


class PackagingMaterial(str, Enum):
    """Packaging material type classification for PPWR tracking."""

    PET = "PET"
    HDPE = "HDPE"
    PP = "PP"
    PS = "PS"
    PVC = "PVC"
    GLASS = "GLASS"
    ALUMINIUM = "ALUMINIUM"
    STEEL = "STEEL"
    PAPER_BOARD = "PAPER_BOARD"
    WOOD = "WOOD"
    COMPOSITE = "COMPOSITE"
    BIOPLASTIC = "BIOPLASTIC"


class EPRScheme(str, Enum):
    """Extended Producer Responsibility scheme types."""

    PACKAGING = "PACKAGING"
    WEEE = "WEEE"
    BATTERIES = "BATTERIES"
    TEXTILES = "TEXTILES"
    VEHICLES = "VEHICLES"
    FURNITURE = "FURNITURE"


class EUDRCommodity(str, Enum):
    """EUDR regulated commodity types."""

    PALM_OIL = "PALM_OIL"
    SOY = "SOY"
    COCOA = "COCOA"
    COFFEE = "COFFEE"
    RUBBER = "RUBBER"
    TIMBER = "TIMBER"
    CATTLE = "CATTLE"


class FoodWasteCategory(str, Enum):
    """Food waste category classification."""

    BAKERY = "BAKERY"
    PRODUCE = "PRODUCE"
    DAIRY = "DAIRY"
    MEAT = "MEAT"
    PREPARED_FOOD = "PREPARED_FOOD"
    PACKAGED = "PACKAGED"
    BEVERAGES = "BEVERAGES"


class StoreType(str, Enum):
    """Store format/type classification."""

    FLAGSHIP = "FLAGSHIP"
    STANDARD = "STANDARD"
    EXPRESS = "EXPRESS"
    OUTLET = "OUTLET"
    WAREHOUSE = "WAREHOUSE"
    DARK_STORE = "DARK_STORE"
    POP_UP = "POP_UP"


class RefrigerantType(str, Enum):
    """Refrigerant type classification for F-gas tracking."""

    R404A = "R404A"
    R134A = "R134A"
    R410A = "R410A"
    R32 = "R32"
    R290 = "R290"
    R744 = "R744"
    R1234YF = "R1234YF"
    R1234ZE = "R1234ZE"
    AMMONIA = "AMMONIA"


class Scope3Priority(str, Enum):
    """Scope 3 category priority level for retail."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class SupplierTier(str, Enum):
    """Supplier tier classification for due diligence depth."""

    TIER_1 = "TIER_1"
    TIER_2 = "TIER_2"
    TIER_3 = "TIER_3"
    TIER_4_PLUS = "TIER_4_PLUS"


class DueDiligenceRisk(str, Enum):
    """Due diligence risk level classification."""

    VERY_HIGH = "VERY_HIGH"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NEGLIGIBLE = "NEGLIGIBLE"


class GreenClaimType(str, Enum):
    """Green claim type classification for ECGT audit."""

    CARBON_NEUTRAL = "CARBON_NEUTRAL"
    CLIMATE_POSITIVE = "CLIMATE_POSITIVE"
    ECO_FRIENDLY = "ECO_FRIENDLY"
    SUSTAINABLE = "SUSTAINABLE"
    RECYCLABLE = "RECYCLABLE"
    BIODEGRADABLE = "BIODEGRADABLE"
    ORGANIC = "ORGANIC"


class ESRSTopic(str, Enum):
    """ESRS topical standards for materiality assessment."""

    E1 = "E1"  # Climate change
    E2 = "E2"  # Pollution
    E3 = "E3"  # Water and marine resources
    E4 = "E4"  # Biodiversity and ecosystems
    E5 = "E5"  # Resource use and circular economy
    S1 = "S1"  # Own workforce
    S2 = "S2"  # Workers in the value chain
    S3 = "S3"  # Affected communities
    S4 = "S4"  # Consumers and end-users
    G1 = "G1"  # Business conduct


class ReportingFrequency(str, Enum):
    """Reporting and disclosure frequency."""

    ANNUAL = "ANNUAL"
    SEMI_ANNUAL = "SEMI_ANNUAL"
    QUARTERLY = "QUARTERLY"
    MONTHLY = "MONTHLY"


class ComplianceStatus(str, Enum):
    """Overall compliance status."""

    COMPLIANT = "COMPLIANT"
    PARTIALLY_COMPLIANT = "PARTIALLY_COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    NOT_ASSESSED = "NOT_ASSESSED"
    EXEMPT = "EXEMPT"


class DisclosureFormat(str, Enum):
    """Output format for disclosure documents."""

    XBRL = "XBRL"
    PDF = "PDF"
    HTML = "HTML"
    JSON = "JSON"


# =============================================================================
# Reference Data Constants
# =============================================================================

# Retail sub-sector display names and NACE codes
SUBSECTOR_INFO: Dict[str, Dict[str, Any]] = {
    "GROCERY": {
        "name": "Grocery & Food Retail",
        "nace": "G47.11",
        "typical_emissions_profile": "Scope 1 dominated by F-gas refrigerants; Scope 3 Cat 1 at 60-80%",
        "key_regulations": ["CSRD", "PPWR", "EUDR", "F-Gas", "CSDDD"],
    },
    "APPAREL": {
        "name": "Apparel & Fashion Retail",
        "nace": "G47.71",
        "typical_emissions_profile": "Scope 3 Cat 1 dominant (raw materials/textiles); Cat 12 material",
        "key_regulations": ["CSRD", "ESPR", "ECGT", "CSDDD", "EUDR", "Textiles EPR"],
    },
    "ELECTRONICS": {
        "name": "Electronics & IT Retail",
        "nace": "G47.41",
        "typical_emissions_profile": "Scope 3 Cat 1 and Cat 11 (use-phase) dominant; WEEE critical",
        "key_regulations": ["CSRD", "ESPR", "WEEE", "Batteries", "ECGT"],
    },
    "HOME_FURNISHING": {
        "name": "Home Furnishing & Decor Retail",
        "nace": "G47.59",
        "typical_emissions_profile": "Scope 3 Cat 1 and Cat 4 material; furniture EPR applicable",
        "key_regulations": ["CSRD", "ESPR", "EUDR", "Furniture EPR"],
    },
    "DEPARTMENT_STORE": {
        "name": "Department & Variety Stores",
        "nace": "G47.19",
        "typical_emissions_profile": "Balanced profile; multi-category requiring broad coverage",
        "key_regulations": ["CSRD", "PPWR", "ESPR", "ECGT", "CSDDD"],
    },
    "CONVENIENCE": {
        "name": "Convenience & Neighborhood Stores",
        "nace": "G47.11",
        "typical_emissions_profile": "Small format; refrigerant per sqm higher; limited Scope 3 data",
        "key_regulations": ["CSRD", "PPWR", "F-Gas"],
    },
    "HYPERMARKET": {
        "name": "Hypermarkets & Superstores",
        "nace": "G47.11",
        "typical_emissions_profile": "Similar to grocery but larger footprint; significant energy use",
        "key_regulations": ["CSRD", "PPWR", "EUDR", "F-Gas", "EED", "CSDDD"],
    },
    "SPECIALTY": {
        "name": "Specialty Retail",
        "nace": "G47.7",
        "typical_emissions_profile": "Category-dependent; focused Scope 3",
        "key_regulations": ["CSRD", "PPWR", "ESPR"],
    },
    "E_COMMERCE": {
        "name": "E-Commerce & Online Retail",
        "nace": "G47.91",
        "typical_emissions_profile": "Cat 4/9 (transport) critical; packaging high; no store Scope 1",
        "key_regulations": ["CSRD", "PPWR", "ESPR", "ECGT", "CSDDD"],
    },
    "WHOLESALE": {
        "name": "Wholesale & Cash & Carry",
        "nace": "G46",
        "typical_emissions_profile": "Warehouse-centric; Cat 4/9 material; large packaging volumes",
        "key_regulations": ["CSRD", "PPWR", "EUDR"],
    },
    "PHARMACY": {
        "name": "Pharmacy & Health Retail",
        "nace": "G47.73",
        "typical_emissions_profile": "Smaller footprint; cold chain for pharmaceuticals",
        "key_regulations": ["CSRD", "PPWR"],
    },
    "DIY_HARDWARE": {
        "name": "DIY & Hardware Retail",
        "nace": "G47.52",
        "typical_emissions_profile": "Cat 1 building materials; Cat 11 use-phase for power tools",
        "key_regulations": ["CSRD", "PPWR", "ESPR", "EUDR"],
    },
    "LUXURY": {
        "name": "Luxury Goods Retail",
        "nace": "G47.77",
        "typical_emissions_profile": "Supply chain DD critical; EUDR leather/exotic materials",
        "key_regulations": ["CSRD", "CSDDD", "EUDR", "ECGT"],
    },
    "DISCOUNT": {
        "name": "Discount & Value Retail",
        "nace": "G47.11",
        "typical_emissions_profile": "High volume, low margin; packaging efficiency critical",
        "key_regulations": ["CSRD", "PPWR", "EUDR", "F-Gas"],
    },
    "FOOD_SERVICE": {
        "name": "Food Service & Catering Retail",
        "nace": "I56",
        "typical_emissions_profile": "Food waste dominant concern; energy for cooking/heating",
        "key_regulations": ["CSRD", "PPWR", "EUDR"],
    },
    "OTHER": {
        "name": "Other Retail",
        "nace": "G47",
        "typical_emissions_profile": "Generic retail profile",
        "key_regulations": ["CSRD", "PPWR"],
    },
}

# Scope 3 priority by sub-sector
RETAIL_SCOPE3_PRIORITY: Dict[str, Dict[int, str]] = {
    "GROCERY": {
        1: "CRITICAL", 2: "LOW", 3: "MEDIUM", 4: "HIGH", 5: "HIGH",
        6: "LOW", 7: "LOW", 8: "MEDIUM", 9: "HIGH", 10: "MEDIUM",
        11: "LOW", 12: "MEDIUM", 13: "LOW", 14: "HIGH", 15: "LOW",
    },
    "APPAREL": {
        1: "CRITICAL", 2: "LOW", 3: "MEDIUM", 4: "HIGH", 5: "MEDIUM",
        6: "LOW", 7: "LOW", 8: "MEDIUM", 9: "HIGH", 10: "LOW",
        11: "LOW", 12: "CRITICAL", 13: "LOW", 14: "MEDIUM", 15: "LOW",
    },
    "ELECTRONICS": {
        1: "CRITICAL", 2: "MEDIUM", 3: "MEDIUM", 4: "HIGH", 5: "MEDIUM",
        6: "LOW", 7: "LOW", 8: "MEDIUM", 9: "HIGH", 10: "LOW",
        11: "CRITICAL", 12: "HIGH", 13: "LOW", 14: "MEDIUM", 15: "LOW",
    },
    "E_COMMERCE": {
        1: "CRITICAL", 2: "MEDIUM", 3: "MEDIUM", 4: "CRITICAL", 5: "MEDIUM",
        6: "LOW", 7: "LOW", 8: "HIGH", 9: "CRITICAL", 10: "LOW",
        11: "MEDIUM", 12: "HIGH", 13: "LOW", 14: "LOW", 15: "LOW",
    },
    "DEPARTMENT_STORE": {
        1: "CRITICAL", 2: "MEDIUM", 3: "MEDIUM", 4: "HIGH", 5: "MEDIUM",
        6: "LOW", 7: "LOW", 8: "MEDIUM", 9: "HIGH", 10: "LOW",
        11: "MEDIUM", 12: "MEDIUM", 13: "LOW", 14: "MEDIUM", 15: "LOW",
    },
}

# PPWR recycled content targets by material and year
PPWR_RECYCLED_CONTENT_TARGETS: Dict[str, Dict[int, float]] = {
    "PET": {2030: 30.0, 2035: 40.0, 2040: 50.0},
    "HDPE": {2030: 10.0, 2035: 25.0, 2040: 35.0},
    "PP": {2030: 10.0, 2035: 25.0, 2040: 35.0},
    "PS": {2030: 10.0, 2035: 25.0, 2040: 35.0},
    "GLASS": {2030: 0.0, 2035: 0.0, 2040: 0.0},  # Exempt (high baseline)
    "ALUMINIUM": {2030: 0.0, 2035: 0.0, 2040: 0.0},  # Exempt
    "STEEL": {2030: 0.0, 2035: 0.0, 2040: 0.0},  # Exempt
    "PAPER_BOARD": {2030: 0.0, 2035: 0.0, 2040: 0.0},  # Separate rules
}

# EPR recycling targets by scheme
EPR_RECYCLING_TARGETS: Dict[str, float] = {
    "PACKAGING": 70.0,
    "WEEE": 65.0,
    "BATTERIES": 70.0,
    "TEXTILES": 50.0,
    "FURNITURE": 45.0,
    "VEHICLES": 85.0,
}

# F-gas GWP values (IPCC AR6)
FGAS_GWP_VALUES: Dict[str, int] = {
    "R404A": 3922,
    "R134A": 1430,
    "R410A": 2088,
    "R32": 675,
    "R290": 3,
    "R744": 1,
    "R1234YF": 4,
    "R1234ZE": 7,
    "AMMONIA": 0,
}

# Available presets
AVAILABLE_PRESETS: Dict[str, str] = {
    "grocery_retail": "Grocery/food retail with refrigeration, food waste, and EUDR commodities",
    "apparel_retail": "Fashion/textile retail with supply chain DD, textile EPR, and DPP",
    "electronics_retail": "Electronics retail with WEEE, use-phase emissions, and batteries EPR",
    "general_retail": "Balanced department store profile with broad regulatory coverage",
    "online_retail": "E-commerce with last-mile delivery, packaging, and returns logistics",
    "sme_retailer": "Simplified SME retailer with Omnibus threshold assessment",
}

# Priority Scope 3 categories by retail sub-sector
PRIORITY_SCOPE3_BY_SUBSECTOR: Dict[str, List[int]] = {
    "GROCERY": [1, 4, 5, 9, 14],
    "APPAREL": [1, 4, 9, 12],
    "ELECTRONICS": [1, 4, 9, 11, 12],
    "E_COMMERCE": [1, 4, 9, 12],
    "DEPARTMENT_STORE": [1, 4, 9, 11, 12],
    "CONVENIENCE": [1, 4, 9],
    "WHOLESALE": [1, 4, 9],
}


# =============================================================================
# Pydantic Sub-Config Models (12 models)
# =============================================================================


class StoreConfig(BaseModel):
    """Configuration for a single retail store or location."""

    store_id: str = Field(
        "",
        description="Unique identifier for the store",
    )
    store_name: str = Field(
        "",
        description="Human-readable store name",
    )
    country: str = Field(
        "DE",
        description="ISO 3166-1 alpha-2 country code",
    )
    sub_sector: RetailSubSector = Field(
        RetailSubSector.GROCERY,
        description="Retail sub-sector of this store",
    )
    store_type: StoreType = Field(
        StoreType.STANDARD,
        description="Store format/type",
    )
    floor_area_sqm: Optional[float] = Field(
        None,
        ge=0,
        description="Total floor area in square meters",
    )
    sales_area_sqm: Optional[float] = Field(
        None,
        ge=0,
        description="Sales floor area in square meters",
    )
    employees: Optional[int] = Field(
        None,
        ge=0,
        description="Number of employees at the store",
    )
    has_refrigeration: bool = Field(
        False,
        description="Whether the store has commercial refrigeration systems",
    )
    refrigerant_types: List[RefrigerantType] = Field(
        default_factory=list,
        description="Types of refrigerants used in the store",
    )
    has_food_department: bool = Field(
        False,
        description="Whether the store has a food/fresh department",
    )
    annual_revenue_eur: Optional[float] = Field(
        None,
        ge=0,
        description="Annual revenue in EUR",
    )


class StoreEmissionsConfig(BaseModel):
    """Configuration for store-level emissions engine.

    Handles Scope 1 (heating, F-gas, fleet) and Scope 2 (electricity,
    district energy) for retail store operations.
    """

    enabled: bool = Field(
        True,
        description="Enable store emissions calculation",
    )
    store_types: List[StoreType] = Field(
        default_factory=lambda: [StoreType.STANDARD],
        description="Store format types in the portfolio",
    )
    refrigerant_tracking: bool = Field(
        True,
        description="Enable F-gas refrigerant leakage tracking",
    )
    f_gas_phase_down: bool = Field(
        True,
        description="Track F-gas phase-down compliance per Regulation (EU) 2024/573",
    )
    fleet_tracking: bool = Field(
        True,
        description="Track company-owned delivery fleet emissions",
    )
    renewable_on_site: bool = Field(
        False,
        description="Track on-site renewable energy generation (solar PV, etc.)",
    )
    heating_sources: List[str] = Field(
        default_factory=lambda: ["natural_gas", "electricity"],
        description="Heating energy sources across stores",
    )
    eed_energy_audit: bool = Field(
        False,
        description="Enable EED energy audit compliance for large retail buildings",
    )
    per_sqm_normalization: bool = Field(
        True,
        description="Calculate emissions per sqm intensity metric",
    )
    per_revenue_normalization: bool = Field(
        True,
        description="Calculate emissions per EUR revenue intensity metric",
    )
    grid_factor_source: str = Field(
        "AIB_RESIDUAL_MIX",
        description="Grid emission factor source: AIB_RESIDUAL_MIX, IEA, NATIONAL",
    )
    year_over_year_tracking: bool = Field(
        True,
        description="Enable year-over-year store emissions trend analysis",
    )


class RetailScope3Config(BaseModel):
    """Configuration for retail Scope 3 emissions engine.

    Covers all 15 Scope 3 categories with retail-specific prioritization
    and calculation methodologies.
    """

    enabled: bool = Field(
        True,
        description="Enable Scope 3 emissions calculation",
    )
    priority_categories: List[int] = Field(
        default_factory=lambda: [1, 4, 5, 9, 11, 12, 14],
        description="Priority Scope 3 categories (1-15) for detailed calculation",
    )
    calculation_methods: Dict[str, str] = Field(
        default_factory=lambda: {
            "cat_1": "hybrid",
            "cat_4": "distance_based",
            "cat_9": "distance_based",
            "default": "spend_based",
        },
        description="Calculation method by category: hybrid, spend_based, supplier_specific, average_data, distance_based",
    )
    supplier_data_collection: bool = Field(
        True,
        description="Enable supplier-specific emissions data collection",
    )
    cat1_methodology: str = Field(
        "hybrid",
        description="Cat 1 calculation method: hybrid, spend_based, supplier_specific, average_data",
    )
    spend_based_screening: bool = Field(
        True,
        description="Use spend-based screening for all uncovered categories",
    )
    hotspot_analysis: bool = Field(
        True,
        description="Identify emission hotspots by category, supplier, and product group",
    )
    supplier_engagement_scoring: bool = Field(
        True,
        description="Score suppliers on emissions data quality and reduction targets",
    )
    flag_pathway_enabled: bool = Field(
        False,
        description="Enable SBTi FLAG pathway for food/agriculture Scope 3",
    )
    last_mile_tracking: bool = Field(
        False,
        description="Track last-mile delivery emissions (Cat 9) in detail",
    )

    @field_validator("priority_categories")
    @classmethod
    def validate_categories(cls, v: List[int]) -> List[int]:
        """Validate Scope 3 category numbers."""
        invalid = [c for c in v if c < 1 or c > 15]
        if invalid:
            raise ValueError(
                f"Invalid Scope 3 categories: {invalid}. Must be 1-15."
            )
        return sorted(set(v))


class PackagingConfig(BaseModel):
    """Configuration for PPWR packaging compliance engine.

    Manages packaging compliance per Regulation (EU) 2025/40.
    """

    enabled: bool = Field(
        True,
        description="Enable packaging compliance tracking",
    )
    ppwr_compliance: bool = Field(
        True,
        description="Enable PPWR compliance assessment",
    )
    recycled_content_tracking: bool = Field(
        True,
        description="Track recycled content by material against PPWR targets",
    )
    materials_tracked: List[PackagingMaterial] = Field(
        default_factory=lambda: [
            PackagingMaterial.PET,
            PackagingMaterial.HDPE,
            PackagingMaterial.PP,
            PackagingMaterial.PAPER_BOARD,
            PackagingMaterial.GLASS,
        ],
        description="Packaging materials to track",
    )
    epr_scheme: bool = Field(
        True,
        description="Enable EPR eco-modulation fee calculation",
    )
    labeling_requirements: bool = Field(
        True,
        description="Track harmonized labeling compliance (material ID, sorting, QR)",
    )
    reuse_targets: bool = Field(
        True,
        description="Track PPWR reuse targets for transport packaging",
    )
    ecommerce_packaging_reuse: bool = Field(
        False,
        description="Track e-commerce packaging reuse targets (PPWR 2030)",
    )
    over_packaging_monitoring: bool = Field(
        True,
        description="Monitor packaging weight minimization and over-packaging",
    )
    target_year: int = Field(
        2030,
        ge=2025,
        le=2040,
        description="Target year for compliance milestone tracking",
    )


class ProductSustainabilityConfig(BaseModel):
    """Configuration for product sustainability engine.

    Manages DPP, PEF, ECGT green claims, and microplastics requirements.
    """

    enabled: bool = Field(
        True,
        description="Enable product sustainability assessment",
    )
    dpp_enabled: bool = Field(
        False,
        description="Enable Digital Product Passport (DPP) data generation per ESPR",
    )
    dpp_product_categories: List[str] = Field(
        default_factory=list,
        description="Product categories requiring DPP (textiles, electronics, etc.)",
    )
    pef_methodology: bool = Field(
        True,
        description="Use EU Product Environmental Footprint (PEF) methodology",
    )
    green_claims_audit: bool = Field(
        True,
        description="Audit green claims per ECGT substantiation requirements",
    )
    banned_claims: List[GreenClaimType] = Field(
        default_factory=lambda: [
            GreenClaimType.CARBON_NEUTRAL,
            GreenClaimType.ECO_FRIENDLY,
        ],
        description="Green claim types that are banned/require substantiation under ECGT",
    )
    textile_microplastics: bool = Field(
        False,
        description="Track textile microplastics release per ESPR",
    )
    product_recyclability_scoring: bool = Field(
        True,
        description="Calculate product recyclability scores",
    )
    right_to_repair: bool = Field(
        False,
        description="Track right-to-repair compliance for electronics",
    )
    emission_factor_database: str = Field(
        "ECOINVENT",
        description="LCA emission factor database: ECOINVENT, EF_3_1, GABI",
    )


class FoodWasteConfig(BaseModel):
    """Configuration for food waste management engine.

    Tracks food waste per EU 30% reduction target (retail level by 2030).
    """

    enabled: bool = Field(
        False,
        description="Enable food waste tracking (primarily for grocery/food retail)",
    )
    measurement_method: str = Field(
        "direct_weighing",
        description="Waste measurement: direct_weighing, scanning, extrapolation",
    )
    reduction_target_pct: float = Field(
        30.0,
        ge=0.0,
        le=100.0,
        description="Food waste reduction target (%) vs. baseline year",
    )
    baseline_year: int = Field(
        2020,
        ge=2015,
        le=2025,
        description="Baseline year for food waste reduction measurement",
    )
    redistribution_tracking: bool = Field(
        True,
        description="Track food redistribution to charities and food banks",
    )
    waste_categories: List[FoodWasteCategory] = Field(
        default_factory=lambda: [
            FoodWasteCategory.BAKERY,
            FoodWasteCategory.PRODUCE,
            FoodWasteCategory.DAIRY,
            FoodWasteCategory.MEAT,
        ],
        description="Food waste categories to track",
    )
    date_marking_optimization: bool = Field(
        True,
        description="Track best-before vs use-by optimization opportunities",
    )
    markdown_effectiveness: bool = Field(
        True,
        description="Track markdown/clearance program effectiveness",
    )
    animal_feed_diversion: bool = Field(
        False,
        description="Track diversion to animal feed",
    )
    composting_tracking: bool = Field(
        True,
        description="Track composting and anaerobic digestion volumes",
    )
    waste_intensity_metric: str = Field(
        "kg_per_eur_revenue",
        description="Waste intensity: kg_per_eur_revenue, kg_per_sqm, waste_to_sales_ratio",
    )


class SupplyChainDDConfig(BaseModel):
    """Configuration for supply chain due diligence engine.

    Manages CSDDD and EUDR compliance for retail supply chains.
    """

    enabled: bool = Field(
        True,
        description="Enable supply chain due diligence",
    )
    csddd_compliance: bool = Field(
        True,
        description="Enable CSDDD due diligence process",
    )
    eudr_commodities: List[EUDRCommodity] = Field(
        default_factory=list,
        description="EUDR commodities to trace in supply chain",
    )
    forced_labour_screening: bool = Field(
        True,
        description="Screen suppliers against forced labour databases",
    )
    supplier_tier_depth: int = Field(
        2,
        ge=1,
        le=5,
        description="Depth of supply chain tiers to assess (1=Tier 1 only)",
    )
    country_risk_assessment: bool = Field(
        True,
        description="Assess country-level human rights and environmental risk",
    )
    living_wage_gap_analysis: bool = Field(
        False,
        description="Analyse living wage gaps in supply chain",
    )
    sanctions_screening: bool = Field(
        True,
        description="Screen suppliers against international sanctions lists",
    )
    stakeholder_engagement: bool = Field(
        True,
        description="Track stakeholder engagement activities per CSDDD",
    )
    remediation_tracking: bool = Field(
        True,
        description="Track remediation actions for identified adverse impacts",
    )
    geolocation_verification: bool = Field(
        False,
        description="Verify EUDR geolocation data using satellite imagery",
    )
    risk_assessment_platform: str = Field(
        "INTERNAL",
        description="Risk assessment platform: INTERNAL, MAPLECROFT, REFINITIV, SEDEX",
    )

    @field_validator("eudr_commodities")
    @classmethod
    def validate_commodities(cls, v: List[EUDRCommodity]) -> List[EUDRCommodity]:
        """Deduplicate EUDR commodities."""
        seen = set()
        result = []
        for c in v:
            if c not in seen:
                seen.add(c)
                result.append(c)
        return result


class CircularEconomyConfig(BaseModel):
    """Configuration for retail circular economy engine.

    Manages EPR obligations, take-back programs, and circularity metrics.
    """

    enabled: bool = Field(
        True,
        description="Enable circular economy metrics tracking",
    )
    epr_schemes: List[EPRScheme] = Field(
        default_factory=lambda: [EPRScheme.PACKAGING],
        description="EPR schemes applicable to the retailer",
    )
    take_back_programs: List[str] = Field(
        default_factory=list,
        description="Active take-back programs (e.g., electronics, textiles, furniture)",
    )
    mci_tracking: bool = Field(
        True,
        description="Calculate Material Circularity Indicator (MCI)",
    )
    product_recyclability: bool = Field(
        True,
        description="Calculate product recyclability scores",
    )
    second_hand_tracking: bool = Field(
        False,
        description="Track second-hand and refurbished product sales",
    )
    return_logistics_emissions: bool = Field(
        False,
        description="Track emissions from returns and reverse logistics",
    )
    repair_services: bool = Field(
        False,
        description="Track repair service volumes and impact",
    )
    multi_country_epr: bool = Field(
        True,
        description="Handle EPR obligations across multiple EU member states",
    )


class BenchmarkConfig(BaseModel):
    """Configuration for retail benchmark engine.

    Benchmarks against sector peers, SBTi pathways, and leading practices.
    """

    enabled: bool = Field(
        True,
        description="Enable retail benchmarking",
    )
    kpi_set: List[str] = Field(
        default_factory=lambda: [
            "emissions_per_sqm",
            "emissions_per_revenue",
            "energy_intensity_kwh_sqm",
            "refrigerant_leakage_rate",
            "scope3_supplier_coverage",
            "packaging_recycled_content",
            "food_waste_ratio",
            "sbti_alignment",
        ],
        description="KPI set for benchmarking",
    )
    peer_group: str = Field(
        "NACE_SECTOR",
        description="Peer group: NACE_SECTOR, STORE_FORMAT, GEOGRAPHY, REVENUE_BAND",
    )
    sbti_pathway: str = Field(
        "1.5C",
        description="SBTi pathway for alignment: 1.5C, WELL_BELOW_2C, NET_ZERO",
    )
    sbti_flag_alignment: bool = Field(
        False,
        description="Track SBTi FLAG pathway alignment for food/agriculture",
    )
    percentile_tracking: bool = Field(
        True,
        description="Track percentile ranking within peer group",
    )
    gap_analysis_enabled: bool = Field(
        True,
        description="Generate gap analysis vs. SBTi pathway and leading practice",
    )
    target_years: List[int] = Field(
        default_factory=lambda: [2025, 2030, 2035, 2040, 2050],
        description="Milestone years for trajectory tracking",
    )


class AuditTrailConfig(BaseModel):
    """Configuration for audit trail and provenance tracking."""

    enabled: bool = Field(
        True,
        description="Enable audit trail for all calculations",
    )
    sha256_provenance: bool = Field(
        True,
        description="Generate SHA-256 provenance hashes for all outputs",
    )
    calculation_logging: bool = Field(
        True,
        description="Log all intermediate calculation steps",
    )
    assumption_tracking: bool = Field(
        True,
        description="Track all assumptions used in calculations",
    )
    data_lineage_enabled: bool = Field(
        True,
        description="Track full data lineage from source to output",
    )
    retention_years: int = Field(
        10,
        ge=1,
        le=15,
        description="Audit trail retention period in years",
    )
    external_audit_export: bool = Field(
        True,
        description="Enable export format for external auditors",
    )


class DisclosureConfig(BaseModel):
    """Configuration for disclosure document generation."""

    esrs_chapter_enabled: bool = Field(
        True,
        description="Generate retail ESRS chapter for management report",
    )
    store_emissions_report_enabled: bool = Field(
        True,
        description="Generate store emissions report",
    )
    supply_chain_report_enabled: bool = Field(
        True,
        description="Generate supply chain due diligence report",
    )
    packaging_report_enabled: bool = Field(
        True,
        description="Generate packaging compliance report",
    )
    product_sustainability_report_enabled: bool = Field(
        True,
        description="Generate product sustainability report",
    )
    food_waste_report_enabled: bool = Field(
        False,
        description="Generate food waste report",
    )
    circular_economy_report_enabled: bool = Field(
        True,
        description="Generate circular economy report",
    )
    scorecard_enabled: bool = Field(
        True,
        description="Generate retail ESG scorecard",
    )
    output_formats: List[DisclosureFormat] = Field(
        default_factory=lambda: [DisclosureFormat.PDF, DisclosureFormat.XBRL],
        description="Output formats for disclosure documents",
    )
    multi_language_support: bool = Field(
        False,
        description="Enable multi-language disclosure generation",
    )
    supported_languages: List[str] = Field(
        default_factory=lambda: ["en"],
        description="Supported languages for disclosures",
    )
    review_workflow_enabled: bool = Field(
        True,
        description="Enable review and approval workflow",
    )
    watermark_draft: bool = Field(
        True,
        description="Apply DRAFT watermark to unapproved documents",
    )


class OmnibusConfig(BaseModel):
    """Configuration for CSRD Omnibus Directive threshold assessment."""

    enabled: bool = Field(
        True,
        description="Enable Omnibus Directive threshold assessment",
    )
    total_assets_eur: Optional[float] = Field(
        None,
        ge=0,
        description="Total assets in EUR for threshold assessment",
    )
    net_turnover_eur: Optional[float] = Field(
        None,
        ge=0,
        description="Net turnover in EUR for threshold assessment",
    )
    average_employees: Optional[int] = Field(
        None,
        ge=0,
        description="Average number of employees for threshold assessment",
    )
    listed_entity: bool = Field(
        False,
        description="Whether the entity is listed on a regulated market",
    )


# =============================================================================
# Main Configuration Model
# =============================================================================


class CSRDRetailConfig(BaseModel):
    """Main configuration for PACK-014 CSRD Retail & Consumer Goods Pack.

    This is the root configuration model that contains all sub-configurations
    for retail sector CSRD compliance. The sub_sectors field drives which
    engines are prioritized and which regulatory requirements are most critical.
    """

    # Company identification
    company_name: str = Field(
        "",
        description="Legal entity name of the retail company",
    )
    reporting_year: int = Field(
        2025,
        ge=2024,
        le=2035,
        description="Reporting year for CSRD disclosure",
    )
    tier: RetailTier = Field(
        RetailTier.ENTERPRISE,
        description="Retail company tier (drives complexity and engine set)",
    )
    sub_sectors: List[RetailSubSector] = Field(
        default_factory=lambda: [RetailSubSector.GROCERY],
        description="Retail sub-sectors of the company",
    )

    # Stores
    stores: List[StoreConfig] = Field(
        default_factory=list,
        description="List of store/location configurations",
    )

    # Omnibus threshold
    omnibus_threshold: OmnibusConfig = Field(
        default_factory=OmnibusConfig,
        description="CSRD Omnibus Directive threshold assessment",
    )

    # Sub-configurations for each engine
    store_emissions: StoreEmissionsConfig = Field(
        default_factory=StoreEmissionsConfig,
        description="Store emissions engine configuration",
    )
    scope3: RetailScope3Config = Field(
        default_factory=RetailScope3Config,
        description="Retail Scope 3 engine configuration",
    )
    packaging: PackagingConfig = Field(
        default_factory=PackagingConfig,
        description="Packaging compliance engine configuration",
    )
    product_sustainability: ProductSustainabilityConfig = Field(
        default_factory=ProductSustainabilityConfig,
        description="Product sustainability engine configuration",
    )
    food_waste: FoodWasteConfig = Field(
        default_factory=FoodWasteConfig,
        description="Food waste engine configuration",
    )
    supply_chain_dd: SupplyChainDDConfig = Field(
        default_factory=SupplyChainDDConfig,
        description="Supply chain due diligence engine configuration",
    )
    circular_economy: CircularEconomyConfig = Field(
        default_factory=CircularEconomyConfig,
        description="Retail circular economy engine configuration",
    )
    benchmark: BenchmarkConfig = Field(
        default_factory=BenchmarkConfig,
        description="Retail benchmark engine configuration",
    )

    # Supporting configurations
    audit_trail: AuditTrailConfig = Field(
        default_factory=AuditTrailConfig,
        description="Audit trail and provenance configuration",
    )
    disclosure: DisclosureConfig = Field(
        default_factory=DisclosureConfig,
        description="Disclosure document generation configuration",
    )

    @model_validator(mode="after")
    def validate_grocery_requires_food_waste_and_refrigerant(self) -> "CSRDRetailConfig":
        """Ensure grocery sub-sector has food waste and refrigerant tracking."""
        has_grocery = any(
            s in (RetailSubSector.GROCERY, RetailSubSector.HYPERMARKET,
                  RetailSubSector.CONVENIENCE)
            for s in self.sub_sectors
        )
        if has_grocery:
            if not self.food_waste.enabled:
                logger.warning(
                    "Food waste tracking is critical for grocery/food retail. "
                    "Enabling food_waste."
                )
                object.__setattr__(self.food_waste, "enabled", True)
            if not self.store_emissions.refrigerant_tracking:
                logger.warning(
                    "Refrigerant tracking is critical for grocery retail "
                    "(commercial refrigeration). Enabling refrigerant_tracking."
                )
                object.__setattr__(self.store_emissions, "refrigerant_tracking", True)
        return self

    @model_validator(mode="after")
    def validate_apparel_requires_textile_epr(self) -> "CSRDRetailConfig":
        """Ensure apparel sub-sector has textile EPR enabled."""
        has_apparel = RetailSubSector.APPAREL in self.sub_sectors
        if has_apparel:
            if EPRScheme.TEXTILES not in self.circular_economy.epr_schemes:
                logger.warning(
                    "Textile EPR is critical for apparel retail. "
                    "Adding TEXTILES to EPR schemes."
                )
                self.circular_economy.epr_schemes.append(EPRScheme.TEXTILES)
        return self

    @model_validator(mode="after")
    def validate_electronics_requires_weee(self) -> "CSRDRetailConfig":
        """Ensure electronics sub-sector has WEEE EPR enabled."""
        has_electronics = RetailSubSector.ELECTRONICS in self.sub_sectors
        if has_electronics:
            if EPRScheme.WEEE not in self.circular_economy.epr_schemes:
                logger.warning(
                    "WEEE EPR is mandatory for electronics retail. "
                    "Adding WEEE to EPR schemes."
                )
                self.circular_economy.epr_schemes.append(EPRScheme.WEEE)
        return self


# =============================================================================
# Pack Configuration Wrapper
# =============================================================================


class PackConfig(BaseModel):
    """Top-level pack configuration wrapper.

    Handles preset loading, environment variable overrides, and
    configuration merging.
    """

    pack: CSRDRetailConfig = Field(
        default_factory=CSRDRetailConfig,
        description="Main CSRD Retail configuration",
    )
    preset_name: Optional[str] = Field(
        None,
        description="Name of the loaded preset",
    )
    config_version: str = Field(
        "1.0.0",
        description="Configuration schema version",
    )
    pack_id: str = Field(
        "PACK-014-csrd-retail",
        description="Pack identifier",
    )

    @classmethod
    def from_preset(cls, preset_name: str, overrides: Optional[Dict[str, Any]] = None) -> "PackConfig":
        """Load configuration from a named preset.

        Args:
            preset_name: Name of the preset (grocery_retail, apparel_retail, etc.)
            overrides: Optional dictionary of configuration overrides.

        Returns:
            PackConfig instance with preset values applied.

        Raises:
            FileNotFoundError: If preset YAML file does not exist.
            ValueError: If preset_name is not in AVAILABLE_PRESETS.
        """
        if preset_name not in AVAILABLE_PRESETS:
            raise ValueError(
                f"Unknown preset: {preset_name}. "
                f"Available presets: {sorted(AVAILABLE_PRESETS.keys())}"
            )

        preset_path = CONFIG_DIR / "presets" / f"{preset_name}.yaml"
        if not preset_path.exists():
            raise FileNotFoundError(
                f"Preset file not found: {preset_path}. "
                f"Run setup wizard to generate presets."
            )

        with open(preset_path, "r", encoding="utf-8") as f:
            preset_data = yaml.safe_load(f) or {}

        # Apply environment variable overrides
        env_overrides = cls._load_env_overrides()
        if env_overrides:
            preset_data = cls._deep_merge(preset_data, env_overrides)

        # Apply explicit overrides
        if overrides:
            preset_data = cls._deep_merge(preset_data, overrides)

        pack_config = CSRDRetailConfig(**preset_data)
        return cls(pack=pack_config, preset_name=preset_name)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "PackConfig":
        """Load configuration from a YAML file.

        Args:
            yaml_path: Path to YAML configuration file.

        Returns:
            PackConfig instance with YAML values applied.
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

        pack_config = CSRDRetailConfig(**config_data)
        return cls(pack=pack_config)

    @staticmethod
    def _load_env_overrides() -> Dict[str, Any]:
        """Load configuration overrides from environment variables.

        Environment variables prefixed with CSRD_RETAIL_PACK_ are loaded
        and mapped to configuration keys. Nested keys use double underscore.

        Example: CSRD_RETAIL_PACK_FOOD_WASTE__ENABLED=true
        """
        overrides: Dict[str, Any] = {}
        prefix = "CSRD_RETAIL_PACK_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                parts = config_key.split("__")
                current = overrides
                for part in parts[:-1]:
                    current = current.setdefault(part, {})
                # Parse value
                if value.lower() in ("true", "yes", "1"):
                    current[parts[-1]] = True
                elif value.lower() in ("false", "no", "0"):
                    current[parts[-1]] = False
                else:
                    try:
                        current[parts[-1]] = int(value)
                    except ValueError:
                        try:
                            current[parts[-1]] = float(value)
                        except ValueError:
                            current[parts[-1]] = value
        return overrides

    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries, with override taking precedence."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = PackConfig._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def get_config_hash(self) -> str:
        """Generate SHA-256 hash of the current configuration for provenance."""
        config_json = self.model_dump_json(indent=None)
        return hashlib.sha256(config_json.encode("utf-8")).hexdigest()


# =============================================================================
# Utility Functions
# =============================================================================


def load_preset(preset_name: str, overrides: Optional[Dict[str, Any]] = None) -> PackConfig:
    """Load a named preset configuration.

    Convenience wrapper around PackConfig.from_preset().

    Args:
        preset_name: Name of the preset to load.
        overrides: Optional configuration overrides.

    Returns:
        PackConfig instance with preset applied.
    """
    return PackConfig.from_preset(preset_name, overrides)


def validate_config(config: CSRDRetailConfig) -> List[str]:
    """Validate a retail configuration and return any warnings.

    Args:
        config: CSRDRetailConfig instance to validate.

    Returns:
        List of warning messages (empty if fully valid).
    """
    warnings: List[str] = []

    # Check for missing store data
    if not config.stores:
        warnings.append(
            "No stores configured. Add at least one store for meaningful results."
        )

    # Check grocery requires food waste
    grocery_sectors = {RetailSubSector.GROCERY, RetailSubSector.HYPERMARKET,
                       RetailSubSector.CONVENIENCE}
    has_grocery = any(s in grocery_sectors for s in config.sub_sectors)
    if has_grocery and not config.food_waste.enabled:
        warnings.append(
            "Food waste tracking is strongly recommended for grocery/food retail."
        )

    # Check apparel requires textile EPR
    if RetailSubSector.APPAREL in config.sub_sectors:
        if EPRScheme.TEXTILES not in config.circular_economy.epr_schemes:
            warnings.append(
                "Textile EPR is mandatory for apparel retail. Add TEXTILES to EPR schemes."
            )

    # Check electronics requires WEEE
    if RetailSubSector.ELECTRONICS in config.sub_sectors:
        if EPRScheme.WEEE not in config.circular_economy.epr_schemes:
            warnings.append(
                "WEEE EPR is mandatory for electronics retail. Add WEEE to EPR schemes."
            )

    # Check Scope 3 coverage
    if config.scope3.enabled and len(config.scope3.priority_categories) < 3:
        warnings.append(
            "Fewer than 3 Scope 3 categories configured. Retail companies "
            "typically need at least categories 1, 4, and 9."
        )

    # Check EUDR commodity relevance
    eudr_grocery = {EUDRCommodity.PALM_OIL, EUDRCommodity.SOY, EUDRCommodity.COCOA,
                    EUDRCommodity.COFFEE}
    if has_grocery and config.supply_chain_dd.enabled:
        if not any(c in eudr_grocery for c in config.supply_chain_dd.eudr_commodities):
            warnings.append(
                "Grocery retailers typically need EUDR tracing for palm oil, soy, "
                "cocoa, and coffee. Consider adding relevant commodities."
            )

    return warnings


def get_default_config(sub_sector: RetailSubSector = RetailSubSector.GROCERY) -> CSRDRetailConfig:
    """Get default configuration for a given retail sub-sector.

    Args:
        sub_sector: Retail sub-sector to configure for.

    Returns:
        CSRDRetailConfig instance with sub-sector-appropriate defaults.
    """
    return CSRDRetailConfig(
        sub_sectors=[sub_sector],
    )


def get_subsector_info(sub_sector: Union[str, RetailSubSector]) -> Dict[str, Any]:
    """Get detailed information about a retail sub-sector.

    Args:
        sub_sector: Sub-sector enum or string value.

    Returns:
        Dictionary with name, NACE code, emissions profile, and regulations.
    """
    key = sub_sector.value if isinstance(sub_sector, RetailSubSector) else sub_sector
    return SUBSECTOR_INFO.get(key, {
        "name": key,
        "nace": "G47",
        "typical_emissions_profile": "Generic retail",
        "key_regulations": ["CSRD"],
    })


def get_ppwr_target(material: Union[str, PackagingMaterial], year: int) -> float:
    """Get PPWR recycled content target for a material and year.

    Args:
        material: Packaging material enum or string value.
        year: Target year (2030, 2035, or 2040).

    Returns:
        Required recycled content percentage.
    """
    key = material.value if isinstance(material, PackagingMaterial) else material
    targets = PPWR_RECYCLED_CONTENT_TARGETS.get(key, {})
    return targets.get(year, 0.0)


def get_fgas_gwp(refrigerant: Union[str, RefrigerantType]) -> int:
    """Get GWP value for a refrigerant type.

    Args:
        refrigerant: Refrigerant type enum or string value.

    Returns:
        Global Warming Potential (100-year, IPCC AR6).
    """
    key = refrigerant.value if isinstance(refrigerant, RefrigerantType) else refrigerant
    return FGAS_GWP_VALUES.get(key, 0)


def list_available_presets() -> Dict[str, str]:
    """List all available configuration presets.

    Returns:
        Dictionary mapping preset names to descriptions.
    """
    return AVAILABLE_PRESETS.copy()
