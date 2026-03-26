"""
PACK-042 Scope 3 Starter Pack - Configuration Manager

This module implements the Scope3StarterConfig and PackConfig classes that
load, merge, and validate all configuration for the Scope 3 Starter Pack.
It provides comprehensive Pydantic v2 models for a production-ready Scope 3
value chain emissions screening and quantification solution covering all 15
GHG Protocol Scope 3 categories.

Methodology Tiers:
    - SPEND_BASED: Spend-based method using EEIO emission factors (Tier 1)
    - AVERAGE_DATA: Average-data method using industry-average EFs (Tier 2)
    - SUPPLIER_SPECIFIC: Supplier-specific primary data (Tier 3)
    - HYBRID: Combination of spend and physical data (mixed tier)

EEIO Models:
    - EXIOBASE_3: Multi-regional input-output model (EU-focused)
    - USEEIO_2: US Environmentally-Extended Input-Output model
    - GTAP: Global Trade Analysis Project database

Classification Systems:
    - NAICS: North American Industry Classification System
    - ISIC: International Standard Industrial Classification
    - UNSPSC: United Nations Standard Products and Services Code
    - HS: Harmonized System commodity classification
    - GL_ACCOUNT: General Ledger account mapping

Data Quality Levels:
    - LEVEL_1: Audited primary data from suppliers (highest quality)
    - LEVEL_2: Non-audited primary data from suppliers
    - LEVEL_3: Industry-average data, peer-reviewed
    - LEVEL_4: Regional or sectoral proxy data
    - LEVEL_5: Estimated or spend-based proxy data (lowest quality)

Sector Presets:
    manufacturing / retail / technology / financial_services /
    food_agriculture / energy_utility / transport_logistics / sme_simplified

Configuration Merge Order (later overrides earlier):
    1. Base pack.yaml manifest
    2. Preset YAML (sector-specific defaults)
    3. Environment overrides (SCOPE3_PACK_* environment variables)
    4. Explicit runtime overrides

Regulatory Context:
    - GHG Protocol Corporate Value Chain (Scope 3) Standard (2011)
    - GHG Protocol Technical Guidance for Calculating Scope 3 (2013)
    - ISO 14064-1:2018 (Categories 3-6)
    - EU CSRD / ESRS E1 (Scope 3 phase-in)
    - CDP Climate Change 2026
    - SBTi Corporate Net-Zero Standard v1.1
    - US SEC Climate Disclosure Rules (Scope 3 safe harbour)
    - California SB 253 (Scope 3 from 2027)
    - PCAF Global GHG Accounting Standard (Category 15)

Example:
    >>> config = PackConfig.from_preset("manufacturing")
    >>> print(config.pack.sector_type)
    SectorType.MANUFACTURING
    >>> print(config.pack.screening.eeio_model)
    EEIOModel.EXIOBASE_3
    >>> print(config.pack.categories.cat_1.enabled)
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
# Enums - Scope 3 enumeration types
# =============================================================================


class Scope3Category(str, Enum):
    """GHG Protocol Scope 3 category classification (15 categories)."""

    CAT_1 = "CAT_1"    # Purchased Goods & Services
    CAT_2 = "CAT_2"    # Capital Goods
    CAT_3 = "CAT_3"    # Fuel- & Energy-Related Activities
    CAT_4 = "CAT_4"    # Upstream Transportation & Distribution
    CAT_5 = "CAT_5"    # Waste Generated in Operations
    CAT_6 = "CAT_6"    # Business Travel
    CAT_7 = "CAT_7"    # Employee Commuting
    CAT_8 = "CAT_8"    # Upstream Leased Assets
    CAT_9 = "CAT_9"    # Downstream Transportation & Distribution
    CAT_10 = "CAT_10"  # Processing of Sold Products
    CAT_11 = "CAT_11"  # Use of Sold Products
    CAT_12 = "CAT_12"  # End-of-Life Treatment of Sold Products
    CAT_13 = "CAT_13"  # Downstream Leased Assets
    CAT_14 = "CAT_14"  # Franchises
    CAT_15 = "CAT_15"  # Investments


class MethodologyTier(str, Enum):
    """Scope 3 methodology tier for emission quantification."""

    SPEND_BASED = "SPEND_BASED"
    AVERAGE_DATA = "AVERAGE_DATA"
    SUPPLIER_SPECIFIC = "SUPPLIER_SPECIFIC"
    HYBRID = "HYBRID"


class EEIOModel(str, Enum):
    """Environmentally-Extended Input-Output model for spend-based calculations."""

    EXIOBASE_3 = "EXIOBASE_3"
    USEEIO_2 = "USEEIO_2"
    GTAP = "GTAP"


class ClassificationCode(str, Enum):
    """Industry/product classification code system for spend mapping."""

    NAICS = "NAICS"
    ISIC = "ISIC"
    UNSPSC = "UNSPSC"
    HS = "HS"
    GL_ACCOUNT = "GL_ACCOUNT"


class DataQualityLevel(str, Enum):
    """Data quality level following GHG Protocol Scope 3 data quality guidance."""

    LEVEL_1 = "LEVEL_1"  # Audited primary data from suppliers
    LEVEL_2 = "LEVEL_2"  # Non-audited primary data from suppliers
    LEVEL_3 = "LEVEL_3"  # Industry-average data, peer-reviewed
    LEVEL_4 = "LEVEL_4"  # Regional or sectoral proxy data
    LEVEL_5 = "LEVEL_5"  # Estimated or spend-based proxy data


class EngagementStatus(str, Enum):
    """Supplier engagement status for primary data collection."""

    NOT_STARTED = "NOT_STARTED"
    DATA_REQUESTED = "DATA_REQUESTED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    OVERDUE = "OVERDUE"


class FrameworkType(str, Enum):
    """Regulatory and reporting framework identifiers for Scope 3."""

    GHG_PROTOCOL = "GHG_PROTOCOL"
    ESRS_E1 = "ESRS_E1"
    CDP = "CDP"
    SBTI = "SBTI"
    SEC = "SEC"
    SB_253 = "SB_253"
    ISO_14064 = "ISO_14064"
    PCAF = "PCAF"


class OutputFormat(str, Enum):
    """Output format for Scope 3 reports."""

    MARKDOWN = "MARKDOWN"
    HTML = "HTML"
    JSON = "JSON"
    CSV = "CSV"


class ReportingFrequency(str, Enum):
    """Reporting and monitoring frequency for Scope 3 inventory."""

    ANNUAL = "ANNUAL"
    SEMI_ANNUAL = "SEMI_ANNUAL"
    QUARTERLY = "QUARTERLY"


class SectorType(str, Enum):
    """Sector classification for preset selection."""

    MANUFACTURING = "MANUFACTURING"
    RETAIL = "RETAIL"
    TECHNOLOGY = "TECHNOLOGY"
    FINANCIAL = "FINANCIAL"
    FOOD_AGRICULTURE = "FOOD_AGRICULTURE"
    ENERGY = "ENERGY"
    TRANSPORT = "TRANSPORT"
    HEALTHCARE = "HEALTHCARE"
    REAL_ESTATE = "REAL_ESTATE"
    SERVICES = "SERVICES"
    SME = "SME"


class AllocationMethod(str, Enum):
    """Allocation method for avoiding double counting across categories."""

    PHYSICAL = "PHYSICAL"
    ECONOMIC = "ECONOMIC"
    MASS = "MASS"
    ENERGY = "ENERGY"


class UncertaintyMethod(str, Enum):
    """Uncertainty quantification method for Scope 3 estimates."""

    MONTE_CARLO = "MONTE_CARLO"
    ANALYTICAL = "ANALYTICAL"
    QUALITATIVE = "QUALITATIVE"


# =============================================================================
# Reference Data Constants
# =============================================================================


# Scope 3 category information for guidance and presets
CATEGORY_INFO: Dict[str, Dict[str, Any]] = {
    "CAT_1": {
        "name": "Purchased Goods & Services",
        "ghg_protocol_ref": "Chapter 1",
        "description": "Extraction, production, and transportation of goods and services purchased",
        "typical_share_pct": "30-80%",
        "common_methods": ["Spend-based", "Average-data", "Supplier-specific", "Hybrid"],
        "key_data_sources": ["Procurement spend", "Supplier LCA data", "EEIO models"],
        "upstream": True,
    },
    "CAT_2": {
        "name": "Capital Goods",
        "ghg_protocol_ref": "Chapter 2",
        "description": "Extraction, production, and transportation of capital goods purchased",
        "typical_share_pct": "2-15%",
        "common_methods": ["Spend-based", "Average-data", "Supplier-specific"],
        "key_data_sources": ["CAPEX records", "Asset register", "Supplier EPDs"],
        "upstream": True,
    },
    "CAT_3": {
        "name": "Fuel- & Energy-Related Activities",
        "ghg_protocol_ref": "Chapter 3",
        "description": "Upstream emissions from fuel and energy purchased (not in Scope 1/2)",
        "typical_share_pct": "3-10%",
        "common_methods": ["Average-data", "Supplier-specific"],
        "key_data_sources": ["Scope 1/2 fuel data", "WTT factors", "T&D loss factors"],
        "upstream": True,
    },
    "CAT_4": {
        "name": "Upstream Transportation & Distribution",
        "ghg_protocol_ref": "Chapter 4",
        "description": "Transportation and distribution of purchased products in company-contracted vehicles",
        "typical_share_pct": "2-15%",
        "common_methods": ["Spend-based", "Distance-based", "Fuel-based"],
        "key_data_sources": ["Freight invoices", "Shipping manifests", "Logistics providers"],
        "upstream": True,
    },
    "CAT_5": {
        "name": "Waste Generated in Operations",
        "ghg_protocol_ref": "Chapter 5",
        "description": "Third-party disposal and treatment of waste generated in operations",
        "typical_share_pct": "0.5-3%",
        "common_methods": ["Waste-type-specific", "Average-data", "Spend-based"],
        "key_data_sources": ["Waste manifests", "Waste contractor reports", "Waste audits"],
        "upstream": True,
    },
    "CAT_6": {
        "name": "Business Travel",
        "ghg_protocol_ref": "Chapter 6",
        "description": "Transportation of employees for business-related activities",
        "typical_share_pct": "0.5-5%",
        "common_methods": ["Distance-based", "Spend-based", "Fuel-based"],
        "key_data_sources": ["Travel management system", "Expense reports", "Travel agency"],
        "upstream": True,
    },
    "CAT_7": {
        "name": "Employee Commuting",
        "ghg_protocol_ref": "Chapter 7",
        "description": "Transportation of employees between homes and worksites",
        "typical_share_pct": "1-5%",
        "common_methods": ["Distance-based", "Average-data", "Survey-based"],
        "key_data_sources": ["Employee surveys", "HR records", "National averages"],
        "upstream": True,
    },
    "CAT_8": {
        "name": "Upstream Leased Assets",
        "ghg_protocol_ref": "Chapter 8",
        "description": "Operation of assets leased by the reporting company",
        "typical_share_pct": "0-5%",
        "common_methods": ["Asset-specific", "Average-data"],
        "key_data_sources": ["Lease contracts", "Energy bills", "Landlord data"],
        "upstream": True,
    },
    "CAT_9": {
        "name": "Downstream Transportation & Distribution",
        "ghg_protocol_ref": "Chapter 9",
        "description": "Transportation and distribution of sold products to end consumers",
        "typical_share_pct": "2-15%",
        "common_methods": ["Distance-based", "Average-data", "Spend-based"],
        "key_data_sources": ["Distribution partner data", "Sales volume", "Customer locations"],
        "upstream": False,
    },
    "CAT_10": {
        "name": "Processing of Sold Products",
        "ghg_protocol_ref": "Chapter 10",
        "description": "Processing of sold intermediate products by downstream companies",
        "typical_share_pct": "0-20%",
        "common_methods": ["Average-data", "Site-specific"],
        "key_data_sources": ["Customer processing data", "Industry studies", "LCA databases"],
        "upstream": False,
    },
    "CAT_11": {
        "name": "Use of Sold Products",
        "ghg_protocol_ref": "Chapter 11",
        "description": "End use of sold products by consumers and customers",
        "typical_share_pct": "0-60%",
        "common_methods": ["Product-specific", "Average-data"],
        "key_data_sources": ["Product specifications", "Energy ratings", "Usage profiles"],
        "upstream": False,
    },
    "CAT_12": {
        "name": "End-of-Life Treatment of Sold Products",
        "ghg_protocol_ref": "Chapter 12",
        "description": "Waste disposal and treatment of sold products at end of life",
        "typical_share_pct": "0.5-5%",
        "common_methods": ["Waste-type-specific", "Average-data"],
        "key_data_sources": ["Product composition", "Waste treatment profiles", "Country disposal mix"],
        "upstream": False,
    },
    "CAT_13": {
        "name": "Downstream Leased Assets",
        "ghg_protocol_ref": "Chapter 13",
        "description": "Operation of assets owned by the reporting company and leased to others",
        "typical_share_pct": "0-10%",
        "common_methods": ["Asset-specific", "Average-data"],
        "key_data_sources": ["Tenant energy data", "Building benchmarks", "Lease contracts"],
        "upstream": False,
    },
    "CAT_14": {
        "name": "Franchises",
        "ghg_protocol_ref": "Chapter 14",
        "description": "Operation of franchises not included in Scope 1/2",
        "typical_share_pct": "0-20%",
        "common_methods": ["Franchise-specific", "Average-data"],
        "key_data_sources": ["Franchise energy data", "Franchise reports", "Industry benchmarks"],
        "upstream": False,
    },
    "CAT_15": {
        "name": "Investments",
        "ghg_protocol_ref": "Chapter 15",
        "description": "Operation of investments not included in Scope 1/2",
        "typical_share_pct": "0-90%",
        "common_methods": ["Investment-specific", "PCAF", "Average-data"],
        "key_data_sources": ["Portfolio data", "PCAF database", "Investee reports"],
        "upstream": False,
    },
}


# Sector information for preset guidance
SECTOR_INFO: Dict[str, Dict[str, Any]] = {
    "MANUFACTURING": {
        "name": "Manufacturing",
        "dominant_categories": ["CAT_1", "CAT_4", "CAT_11", "CAT_12"],
        "cat_1_typical_pct": "40-70%",
        "description": "Cat 1 (purchased raw materials) dominant; Cat 4 (inbound logistics) and Cat 11 (product use) material",
        "typical_total_scope3_vs_scope12": "3-10x Scope 1+2",
        "key_engagement_targets": "Top 50 material suppliers",
    },
    "RETAIL": {
        "name": "Retail & Distribution",
        "dominant_categories": ["CAT_1", "CAT_4", "CAT_9"],
        "cat_1_typical_pct": "60-85%",
        "description": "Cat 1 (purchased products for resale) dominant; Cat 4 (inbound) and Cat 9 (outbound) logistics material",
        "typical_total_scope3_vs_scope12": "10-50x Scope 1+2",
        "key_engagement_targets": "Top 100 product suppliers",
    },
    "TECHNOLOGY": {
        "name": "Technology & Services",
        "dominant_categories": ["CAT_1", "CAT_6", "CAT_11"],
        "cat_1_typical_pct": "30-60%",
        "description": "Cat 1 (hardware, cloud services) and Cat 11 (product energy use) material; Cat 6 (business travel) significant for services",
        "typical_total_scope3_vs_scope12": "5-20x Scope 1+2",
        "key_engagement_targets": "Top 30 IT and cloud suppliers",
    },
    "FINANCIAL": {
        "name": "Financial Services",
        "dominant_categories": ["CAT_15", "CAT_1", "CAT_6"],
        "cat_1_typical_pct": "5-15%",
        "description": "Cat 15 (financed emissions) dominant per PCAF; Cat 1 (purchased services) and Cat 6 (travel) secondary",
        "typical_total_scope3_vs_scope12": "100-700x Scope 1+2",
        "key_engagement_targets": "Top 20 portfolio companies and fund managers",
    },
    "FOOD_AGRICULTURE": {
        "name": "Food & Agriculture",
        "dominant_categories": ["CAT_1", "CAT_10", "CAT_12"],
        "cat_1_typical_pct": "50-80%",
        "description": "Cat 1 (agricultural inputs, ingredients) dominant; Cat 10 (processing) and Cat 12 (food waste) material",
        "typical_total_scope3_vs_scope12": "5-20x Scope 1+2",
        "key_engagement_targets": "Top 50 agricultural and ingredient suppliers",
    },
    "ENERGY": {
        "name": "Energy & Utilities",
        "dominant_categories": ["CAT_3", "CAT_11", "CAT_1"],
        "cat_1_typical_pct": "10-30%",
        "description": "Cat 3 (upstream fuel) and Cat 11 (use of sold energy) dominant; Cat 1 (equipment, services) secondary",
        "typical_total_scope3_vs_scope12": "0.5-3x Scope 1+2",
        "key_engagement_targets": "Top 20 fuel and equipment suppliers",
    },
    "TRANSPORT": {
        "name": "Transport & Logistics",
        "dominant_categories": ["CAT_4", "CAT_9", "CAT_11"],
        "cat_1_typical_pct": "10-25%",
        "description": "Cat 4 (subcontracted transport) and Cat 9 (client-contracted outbound) dominant; Cat 11 (vehicle use) for manufacturers",
        "typical_total_scope3_vs_scope12": "2-5x Scope 1+2",
        "key_engagement_targets": "Top 30 subcontractors and fuel suppliers",
    },
    "SME": {
        "name": "Small-Medium Enterprise",
        "dominant_categories": ["CAT_1", "CAT_6", "CAT_7"],
        "cat_1_typical_pct": "40-70%",
        "description": "Cat 1 (purchased goods) dominant; Cat 6 (travel) and Cat 7 (commuting) secondary; simplified top-5 approach",
        "typical_total_scope3_vs_scope12": "3-10x Scope 1+2",
        "key_engagement_targets": "Top 10 suppliers",
    },
}


# Default EEIO emission factors by sector (kgCO2e per EUR, illustrative)
DEFAULT_EEIO_FACTORS: Dict[str, float] = {
    "agriculture_forestry_fishing": 1.85,
    "mining_quarrying": 0.92,
    "food_beverages_tobacco": 0.89,
    "textiles_apparel": 0.74,
    "wood_paper_printing": 0.68,
    "chemicals_pharmaceuticals": 0.61,
    "rubber_plastics": 0.72,
    "metals_fabricated_products": 0.83,
    "electronics_electrical": 0.34,
    "machinery_equipment": 0.42,
    "motor_vehicles": 0.48,
    "furniture_other_manufacturing": 0.55,
    "construction": 0.67,
    "wholesale_retail": 0.28,
    "transportation_storage": 0.94,
    "accommodation_food_services": 0.45,
    "information_communication": 0.18,
    "financial_insurance": 0.09,
    "real_estate": 0.15,
    "professional_scientific_technical": 0.14,
    "administrative_support": 0.19,
    "education": 0.21,
    "health_social_work": 0.26,
    "arts_entertainment_recreation": 0.32,
    "other_services": 0.25,
}


# Available presets
AVAILABLE_PRESETS: Dict[str, str] = {
    "manufacturing": "Manufacturing sector with dominant upstream supply chain (Cat 1, Cat 4, Cat 11)",
    "retail": "Retail and distribution with large purchased goods portfolio (Cat 1, Cat 4, Cat 9)",
    "technology": "Technology and services with hardware, cloud, and travel (Cat 1, Cat 6, Cat 11)",
    "financial_services": "Financial sector with financed emissions via PCAF (Cat 15 dominant)",
    "food_agriculture": "Food and agriculture with agricultural supply chain (Cat 1, Cat 10, Cat 12)",
    "energy_utility": "Energy sector with upstream fuel and downstream energy use (Cat 3, Cat 11)",
    "transport_logistics": "Transport and logistics with subcontracted services (Cat 4, Cat 9, Cat 11)",
    "sme_simplified": "SME simplified approach covering top 5 material categories only",
}


# =============================================================================
# Pydantic Sub-Config Models
# =============================================================================


class ScreeningConfig(BaseModel):
    """Configuration for Scope 3 initial screening using EEIO models.

    Defines the environmentally-extended input-output model, currency,
    base year, and thresholds for the spend-based screening phase
    that identifies material Scope 3 categories before detailed
    quantification begins.
    """

    eeio_model: EEIOModel = Field(
        EEIOModel.EXIOBASE_3,
        description="EEIO model for spend-based screening (EXIOBASE_3, USEEIO_2, GTAP)",
    )
    currency: str = Field(
        "EUR",
        description="Currency for spend data (ISO 4217 code)",
    )
    base_year: int = Field(
        2024,
        ge=2015,
        le=2030,
        description="Base year for EEIO factors (inflation adjustment reference)",
    )
    revenue_threshold_eur: Decimal = Field(
        Decimal("0"),
        ge=0,
        description="Minimum annual revenue (EUR) for Scope 3 reporting applicability",
    )
    inflation_adjustment: bool = Field(
        True,
        description="Apply inflation adjustment to spend data for EEIO factor year alignment",
    )
    include_second_order: bool = Field(
        False,
        description="Include second-order (indirect) EEIO effects in screening estimates",
    )


class SpendClassificationConfig(BaseModel):
    """Configuration for spend data classification and mapping.

    Controls how procurement and financial spend data is classified
    into industry sectors for EEIO emission factor application.
    """

    code_system: ClassificationCode = Field(
        ClassificationCode.NAICS,
        description="Primary classification code system for spend mapping",
    )
    secondary_code_system: Optional[ClassificationCode] = Field(
        None,
        description="Secondary classification system for cross-referencing",
    )
    confidence_threshold: float = Field(
        0.80,
        ge=0.50,
        le=1.00,
        description="Minimum confidence threshold for automated spend classification",
    )
    manual_review_threshold: float = Field(
        0.60,
        ge=0.30,
        le=0.90,
        description="Below this confidence, require manual classification review",
    )
    currency: str = Field(
        "EUR",
        description="Currency for spend data (ISO 4217 code)",
    )
    inflation_year: int = Field(
        2024,
        ge=2015,
        le=2030,
        description="Reference year for inflation-adjusted spend figures",
    )

    @field_validator("manual_review_threshold")
    @classmethod
    def validate_manual_below_auto(cls, v: float, info: Any) -> float:
        """Manual review threshold must be below auto-classification threshold."""
        # Note: cross-field validation handled in model_validator
        return v


class SingleCategoryConfig(BaseModel):
    """Configuration for a single Scope 3 category.

    Controls enablement, methodology tier, data sources, and
    materiality threshold for each of the 15 Scope 3 categories.
    """

    enabled: bool = Field(
        True,
        description="Whether this category is enabled for quantification",
    )
    tier: MethodologyTier = Field(
        MethodologyTier.SPEND_BASED,
        description="Default methodology tier for this category",
    )
    data_sources: List[str] = Field(
        default_factory=lambda: ["spend_data"],
        description="Available data sources for this category",
    )
    materiality_threshold_pct: float = Field(
        1.0,
        ge=0.0,
        le=25.0,
        description="Materiality threshold (% of total Scope 3) below which category may be excluded",
    )
    notes: str = Field(
        "",
        description="Category-specific implementation notes or assumptions",
    )


class CategoryConfig(BaseModel):
    """Configuration for all 15 Scope 3 categories.

    Each category is independently configurable with methodology tier,
    data sources, and materiality thresholds. Categories are grouped
    into upstream (1-8) and downstream (9-15) per GHG Protocol.
    """

    # Upstream categories (1-8)
    cat_1: SingleCategoryConfig = Field(
        default_factory=lambda: SingleCategoryConfig(
            enabled=True,
            tier=MethodologyTier.SPEND_BASED,
            data_sources=["spend_data", "supplier_data", "lca_databases"],
            materiality_threshold_pct=0.5,
        ),
        description="Category 1: Purchased Goods & Services",
    )
    cat_2: SingleCategoryConfig = Field(
        default_factory=lambda: SingleCategoryConfig(
            enabled=True,
            tier=MethodologyTier.SPEND_BASED,
            data_sources=["capex_records", "asset_register"],
            materiality_threshold_pct=1.0,
        ),
        description="Category 2: Capital Goods",
    )
    cat_3: SingleCategoryConfig = Field(
        default_factory=lambda: SingleCategoryConfig(
            enabled=True,
            tier=MethodologyTier.AVERAGE_DATA,
            data_sources=["scope1_fuel_data", "scope2_electricity_data"],
            materiality_threshold_pct=1.0,
        ),
        description="Category 3: Fuel- & Energy-Related Activities",
    )
    cat_4: SingleCategoryConfig = Field(
        default_factory=lambda: SingleCategoryConfig(
            enabled=True,
            tier=MethodologyTier.SPEND_BASED,
            data_sources=["freight_invoices", "logistics_data"],
            materiality_threshold_pct=1.0,
        ),
        description="Category 4: Upstream Transportation & Distribution",
    )
    cat_5: SingleCategoryConfig = Field(
        default_factory=lambda: SingleCategoryConfig(
            enabled=True,
            tier=MethodologyTier.AVERAGE_DATA,
            data_sources=["waste_manifests", "waste_contractor_reports"],
            materiality_threshold_pct=1.0,
        ),
        description="Category 5: Waste Generated in Operations",
    )
    cat_6: SingleCategoryConfig = Field(
        default_factory=lambda: SingleCategoryConfig(
            enabled=True,
            tier=MethodologyTier.SPEND_BASED,
            data_sources=["travel_management_system", "expense_reports"],
            materiality_threshold_pct=1.0,
        ),
        description="Category 6: Business Travel",
    )
    cat_7: SingleCategoryConfig = Field(
        default_factory=lambda: SingleCategoryConfig(
            enabled=True,
            tier=MethodologyTier.AVERAGE_DATA,
            data_sources=["employee_surveys", "hr_records", "national_averages"],
            materiality_threshold_pct=1.0,
        ),
        description="Category 7: Employee Commuting",
    )
    cat_8: SingleCategoryConfig = Field(
        default_factory=lambda: SingleCategoryConfig(
            enabled=False,
            tier=MethodologyTier.AVERAGE_DATA,
            data_sources=["lease_contracts", "energy_bills"],
            materiality_threshold_pct=2.0,
        ),
        description="Category 8: Upstream Leased Assets",
    )
    # Downstream categories (9-15)
    cat_9: SingleCategoryConfig = Field(
        default_factory=lambda: SingleCategoryConfig(
            enabled=False,
            tier=MethodologyTier.AVERAGE_DATA,
            data_sources=["distribution_data", "sales_volumes"],
            materiality_threshold_pct=2.0,
        ),
        description="Category 9: Downstream Transportation & Distribution",
    )
    cat_10: SingleCategoryConfig = Field(
        default_factory=lambda: SingleCategoryConfig(
            enabled=False,
            tier=MethodologyTier.AVERAGE_DATA,
            data_sources=["customer_processing_data", "industry_studies"],
            materiality_threshold_pct=2.0,
        ),
        description="Category 10: Processing of Sold Products",
    )
    cat_11: SingleCategoryConfig = Field(
        default_factory=lambda: SingleCategoryConfig(
            enabled=False,
            tier=MethodologyTier.AVERAGE_DATA,
            data_sources=["product_specs", "energy_ratings", "usage_profiles"],
            materiality_threshold_pct=2.0,
        ),
        description="Category 11: Use of Sold Products",
    )
    cat_12: SingleCategoryConfig = Field(
        default_factory=lambda: SingleCategoryConfig(
            enabled=False,
            tier=MethodologyTier.AVERAGE_DATA,
            data_sources=["product_composition", "waste_treatment_profiles"],
            materiality_threshold_pct=2.0,
        ),
        description="Category 12: End-of-Life Treatment of Sold Products",
    )
    cat_13: SingleCategoryConfig = Field(
        default_factory=lambda: SingleCategoryConfig(
            enabled=False,
            tier=MethodologyTier.AVERAGE_DATA,
            data_sources=["tenant_data", "building_benchmarks"],
            materiality_threshold_pct=2.0,
        ),
        description="Category 13: Downstream Leased Assets",
    )
    cat_14: SingleCategoryConfig = Field(
        default_factory=lambda: SingleCategoryConfig(
            enabled=False,
            tier=MethodologyTier.AVERAGE_DATA,
            data_sources=["franchise_reports", "industry_benchmarks"],
            materiality_threshold_pct=2.0,
        ),
        description="Category 14: Franchises",
    )
    cat_15: SingleCategoryConfig = Field(
        default_factory=lambda: SingleCategoryConfig(
            enabled=False,
            tier=MethodologyTier.SPEND_BASED,
            data_sources=["portfolio_data", "pcaf_database", "investee_reports"],
            materiality_threshold_pct=2.0,
        ),
        description="Category 15: Investments",
    )

    def get_enabled_categories(self) -> List[str]:
        """Return list of enabled category identifiers."""
        enabled = []
        for i in range(1, 16):
            cat_config = getattr(self, f"cat_{i}")
            if cat_config.enabled:
                enabled.append(f"CAT_{i}")
        return enabled

    def get_category_config(self, category: str) -> SingleCategoryConfig:
        """Get configuration for a specific category by identifier.

        Args:
            category: Category identifier (e.g., 'CAT_1' or 'cat_1').

        Returns:
            SingleCategoryConfig for the specified category.

        Raises:
            ValueError: If category identifier is not valid.
        """
        cat_key = category.lower().replace("cat_", "cat_")
        if not hasattr(self, cat_key):
            raise ValueError(
                f"Invalid category identifier: {category}. "
                f"Use CAT_1 through CAT_15."
            )
        return getattr(self, cat_key)


class DoubleCountingConfig(BaseModel):
    """Configuration for double-counting prevention across Scope 3 categories.

    Implements rules to prevent emissions from being counted in multiple
    categories (e.g., transport in Cat 1 vs Cat 4) and across scopes
    (e.g., Cat 3 vs Scope 1/2).
    """

    rules_enabled: List[str] = Field(
        default_factory=lambda: [
            "cat1_vs_cat4_transport",
            "cat1_vs_cat2_capex",
            "cat3_vs_scope12",
            "cat4_vs_cat9_transport",
            "cat8_vs_scope12_leases",
            "cat13_vs_scope12_leases",
            "cat14_vs_scope12_franchises",
        ],
        description="List of double-counting prevention rules to apply",
    )
    allocation_method: AllocationMethod = Field(
        AllocationMethod.ECONOMIC,
        description="Default allocation method for resolving double-counting (PHYSICAL, ECONOMIC, MASS, ENERGY)",
    )
    conservative_mode: bool = Field(
        True,
        description="Use conservative allocation (avoid understatement) when ambiguous",
    )
    document_exclusions: bool = Field(
        True,
        description="Document all double-counting exclusions with rationale",
    )
    scope12_reference_enabled: bool = Field(
        True,
        description="Cross-reference Scope 1/2 data to prevent Cat 3 double-counting",
    )


class HotspotConfig(BaseModel):
    """Configuration for Scope 3 hotspot identification and prioritisation.

    Implements Pareto analysis to identify the categories, suppliers,
    and product groups that contribute most to total Scope 3 emissions.
    """

    pareto_threshold_pct: float = Field(
        80.0,
        ge=50.0,
        le=95.0,
        description="Pareto threshold (%) for identifying material categories and suppliers",
    )
    min_categories: int = Field(
        3,
        ge=1,
        le=15,
        description="Minimum number of categories to include in hotspot analysis",
    )
    benchmark_source: str = Field(
        "ghg_protocol_sector_guidance",
        description="Benchmark data source for sector comparison",
    )
    top_n_suppliers: int = Field(
        50,
        ge=5,
        le=500,
        description="Number of top suppliers to include in hotspot drill-down",
    )
    include_product_group_analysis: bool = Field(
        True,
        description="Break down Cat 1 emissions by product group for targeted reduction",
    )
    trend_comparison_years: int = Field(
        3,
        ge=1,
        le=10,
        description="Number of prior years to include in hotspot trend comparison",
    )


class SupplierEngagementConfig(BaseModel):
    """Configuration for supplier engagement and primary data collection.

    Defines the supplier engagement programme for transitioning from
    spend-based to supplier-specific methodology.
    """

    top_percent: float = Field(
        80.0,
        ge=10.0,
        le=100.0,
        description="Target percentage of Scope 3 emissions covered by supplier engagement",
    )
    top_n_suppliers: int = Field(
        50,
        ge=5,
        le=500,
        description="Number of top suppliers to engage for primary data",
    )
    reminder_frequency_days: int = Field(
        30,
        ge=7,
        le=180,
        description="Frequency (days) for sending data request reminders to suppliers",
    )
    quality_target: DataQualityLevel = Field(
        DataQualityLevel.LEVEL_2,
        description="Target data quality level for supplier-provided data",
    )
    deadline_days: int = Field(
        90,
        ge=30,
        le=365,
        description="Deadline (days from request) for supplier data submission",
    )
    escalation_after_days: int = Field(
        60,
        ge=14,
        le=180,
        description="Escalate to procurement team after this many days without response",
    )
    use_cdp_supply_chain: bool = Field(
        False,
        description="Integrate with CDP Supply Chain programme for supplier data collection",
    )
    questionnaire_template: str = Field(
        "ghg_protocol_scope3_supplier",
        description="Template to use for supplier data request questionnaire",
    )


class DataQualityConfig(BaseModel):
    """Configuration for Scope 3 data quality assessment.

    Implements GHG Protocol Scope 3 data quality indicators (DQIs):
    technological representativeness, temporal representativeness,
    geographical representativeness, completeness, and reliability.
    """

    min_dqr: float = Field(
        1.0,
        ge=1.0,
        le=5.0,
        description="Minimum acceptable data quality rating (1=best, 5=worst)",
    )
    target_dqr: float = Field(
        3.0,
        ge=1.0,
        le=5.0,
        description="Target data quality rating for material categories",
    )
    quality_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "technological_representativeness": 0.30,
            "temporal_representativeness": 0.20,
            "geographical_representativeness": 0.20,
            "completeness": 0.15,
            "reliability": 0.15,
        },
        description="Weights for each data quality indicator (must sum to 1.0)",
    )
    improvement_plan_required: bool = Field(
        True,
        description="Require data quality improvement plan for categories below target DQR",
    )
    track_dqr_trend: bool = Field(
        True,
        description="Track DQR improvement over reporting years",
    )

    @field_validator("quality_weights")
    @classmethod
    def validate_weights_sum_to_one(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Quality indicator weights must sum to approximately 1.0."""
        total = sum(v.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Data quality indicator weights must sum to 1.0 (got {total:.3f})."
            )
        return v


class UncertaintyConfig(BaseModel):
    """Configuration for Scope 3 uncertainty quantification.

    Controls uncertainty analysis method and parameters. Scope 3
    uncertainty is typically much larger than Scope 1/2 due to the
    use of secondary data, EEIO models, and proxy data.
    """

    method: UncertaintyMethod = Field(
        UncertaintyMethod.QUALITATIVE,
        description="Uncertainty method: MONTE_CARLO, ANALYTICAL, or QUALITATIVE",
    )
    monte_carlo_iterations: int = Field(
        5000,
        ge=1000,
        le=100000,
        description="Number of Monte Carlo iterations (if method is MONTE_CARLO)",
    )
    confidence_level: float = Field(
        0.95,
        ge=0.80,
        le=0.99,
        description="Confidence level for uncertainty intervals (0.95 = 95%)",
    )
    seed: Optional[int] = Field(
        None,
        description="Random seed for reproducible Monte Carlo results (None = random)",
    )
    default_spend_uncertainty_pct: float = Field(
        50.0,
        ge=10.0,
        le=200.0,
        description="Default uncertainty range (%) for spend-based estimates",
    )
    default_average_data_uncertainty_pct: float = Field(
        30.0,
        ge=5.0,
        le=100.0,
        description="Default uncertainty range (%) for average-data estimates",
    )
    default_supplier_uncertainty_pct: float = Field(
        10.0,
        ge=2.0,
        le=50.0,
        description="Default uncertainty range (%) for supplier-specific data",
    )


class ComplianceConfig(BaseModel):
    """Configuration for regulatory compliance framework mapping for Scope 3."""

    target_frameworks: List[FrameworkType] = Field(
        default_factory=lambda: [
            FrameworkType.GHG_PROTOCOL,
            FrameworkType.ESRS_E1,
            FrameworkType.CDP,
        ],
        description="Regulatory frameworks to map Scope 3 output to",
    )
    esrs_phase_in_year: int = Field(
        2025,
        ge=2024,
        le=2030,
        description="ESRS E1 Scope 3 phase-in year (first reporting year for Scope 3)",
    )
    sbti_target_year: Optional[int] = Field(
        None,
        ge=2025,
        le=2050,
        description="SBTi Scope 3 target year (if SBTi-validated target exists)",
    )
    sbti_reduction_target_pct: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="SBTi Scope 3 reduction target (% from base year, typically 97% coverage)",
    )
    sec_safe_harbour: bool = Field(
        True,
        description="Apply SEC Scope 3 safe harbour provisions (good faith estimates)",
    )
    sb253_scope3_from: int = Field(
        2027,
        ge=2027,
        le=2030,
        description="SB 253 Scope 3 reporting start year (2027 per legislation)",
    )
    pcaf_enabled: bool = Field(
        False,
        description="Enable PCAF methodology for Category 15 financed emissions",
    )
    pcaf_asset_classes: List[str] = Field(
        default_factory=list,
        description="PCAF asset classes to cover (e.g., 'listed_equity', 'corporate_bonds', 'project_finance')",
    )
    cdp_supply_chain_module: bool = Field(
        False,
        description="Enable CDP Supply Chain module integration for supplier data",
    )

    @field_validator("target_frameworks")
    @classmethod
    def validate_frameworks_not_empty(cls, v: List[FrameworkType]) -> List[FrameworkType]:
        """At least one compliance framework must be configured."""
        if not v:
            raise ValueError("At least one compliance framework must be selected.")
        return v


class ReportingConfig(BaseModel):
    """Configuration for Scope 3 report generation."""

    formats: List[OutputFormat] = Field(
        default_factory=lambda: [OutputFormat.HTML, OutputFormat.JSON],
        description="Output formats for Scope 3 reports",
    )
    frequency: ReportingFrequency = Field(
        ReportingFrequency.ANNUAL,
        description="Reporting frequency for Scope 3 inventory updates",
    )
    include_provenance: bool = Field(
        True,
        description="Include SHA-256 provenance hashes in reports",
    )
    include_appendices: bool = Field(
        True,
        description="Include methodology appendices with emission factor sources",
    )
    include_data_quality_matrix: bool = Field(
        True,
        description="Include data quality matrix showing DQR per category",
    )
    include_supplier_engagement_status: bool = Field(
        True,
        description="Include supplier engagement progress in reports",
    )
    include_hotspot_analysis: bool = Field(
        True,
        description="Include hotspot analysis with Pareto charts in reports",
    )
    include_double_counting_log: bool = Field(
        True,
        description="Include double-counting prevention log in reports",
    )
    output_language: str = Field(
        "en",
        description="Report language (ISO 639-1)",
    )
    watermark_draft: bool = Field(
        True,
        description="Apply DRAFT watermark to unapproved reports",
    )


class IntegrationConfig(BaseModel):
    """Configuration for integration with other GreenLang packs and external systems."""

    pack041_enabled: bool = Field(
        True,
        description="Enable integration with PACK-041 (Scope 1-2 Complete) for Cat 3 cross-reference",
    )
    pack041_pack_id: str = Field(
        "PACK-041-scope-1-2-complete",
        description="PACK-041 identifier for cross-pack data exchange",
    )
    erp_type: Optional[str] = Field(
        None,
        description="ERP system type (SAP, Oracle, Dynamics) for spend data extraction",
    )
    erp_spend_module: Optional[str] = Field(
        None,
        description="ERP module for procurement spend data (e.g., SAP MM, SAP SRM)",
    )
    supplier_portal_enabled: bool = Field(
        False,
        description="Enable supplier portal for primary data collection",
    )
    supplier_portal_url: Optional[str] = Field(
        None,
        description="URL of the supplier data collection portal",
    )
    lca_database: str = Field(
        "ecoinvent_3.10",
        description="LCA database for average-data emission factors (ecoinvent, GaBi, ELCD)",
    )
    travel_management_system: Optional[str] = Field(
        None,
        description="Travel management system integration (e.g., Concur, Egencia, TripActions)",
    )
    waste_management_system: Optional[str] = Field(
        None,
        description="Waste management contractor data integration",
    )
    logistics_platform: Optional[str] = Field(
        None,
        description="Logistics platform integration for Cat 4/9 (e.g., EcoTransIT, GLEC)",
    )


class SecurityConfig(BaseModel):
    """Security and access control configuration for Scope 3 data."""

    roles: List[str] = Field(
        default_factory=lambda: [
            "scope3_manager",
            "sustainability_officer",
            "procurement_manager",
            "supply_chain_analyst",
            "data_analyst",
            "supplier_contact",
            "verifier",
            "viewer",
            "admin",
        ],
        description="Available RBAC roles for the Scope 3 pack",
    )
    data_classification: str = Field(
        "CONFIDENTIAL",
        description="Default data classification: PUBLIC, INTERNAL, CONFIDENTIAL, RESTRICTED",
    )
    audit_logging: bool = Field(
        True,
        description="Enable security audit logging for all data access",
    )
    pii_redaction: bool = Field(
        True,
        description="Enable PII redaction in exported reports (supplier contacts)",
    )
    encryption_at_rest: bool = Field(
        True,
        description="Require encryption at rest for stored Scope 3 data",
    )
    supplier_data_isolation: bool = Field(
        True,
        description="Isolate individual supplier data from other suppliers in multi-tenant mode",
    )


class PerformanceConfig(BaseModel):
    """Performance and resource limits for Scope 3 pack execution."""

    parallel_categories: int = Field(
        5,
        ge=1,
        le=15,
        description="Maximum number of Scope 3 categories to calculate in parallel",
    )
    batch_size: int = Field(
        1000,
        ge=100,
        le=10000,
        description="Batch size for processing spend line items and supplier records",
    )
    cache_ttl_seconds: int = Field(
        3600,
        ge=60,
        le=86400,
        description="Cache TTL for EEIO factors and classification lookups (seconds)",
    )
    spend_classification_timeout_seconds: int = Field(
        300,
        ge=30,
        le=1800,
        description="Timeout for spend classification engine (seconds)",
    )
    max_suppliers: int = Field(
        5000,
        ge=100,
        le=50000,
        description="Maximum number of suppliers per category calculation",
    )
    max_line_items: int = Field(
        500000,
        ge=10000,
        le=5000000,
        description="Maximum number of spend line items per calculation run",
    )
    memory_ceiling_mb: int = Field(
        8192,
        ge=1024,
        le=65536,
        description="Memory ceiling for Scope 3 calculation (MB)",
    )


class AuditTrailConfig(BaseModel):
    """Configuration for Scope 3 calculation audit trail and provenance."""

    enabled: bool = Field(
        True,
        description="Enable audit trail for all Scope 3 calculations",
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
        description="Track all assumptions (EEIO factors, proxies, allocations) used",
    )
    data_lineage_enabled: bool = Field(
        True,
        description="Track full data lineage from spend record to category total",
    )
    retention_years: int = Field(
        7,
        ge=1,
        le=15,
        description="Audit trail retention period in years",
    )
    emission_factor_citation: bool = Field(
        True,
        description="Cite EEIO/EF source and vintage for every calculation",
    )
    supplier_data_versioning: bool = Field(
        True,
        description="Version all supplier-provided data with submission timestamps",
    )


# =============================================================================
# Main Configuration Model
# =============================================================================


class Scope3StarterConfig(BaseModel):
    """Main configuration for PACK-042 Scope 3 Starter Pack.

    This is the root configuration model that contains all sub-configurations
    for Scope 3 value chain emissions screening, quantification, hotspot
    analysis, supplier engagement, and multi-framework compliance mapping.
    The sector_type field drives which Scope 3 categories are prioritised,
    which EEIO model is used, and which supplier engagement targets are set.
    """

    # Organisation identification
    company_name: str = Field(
        "",
        description="Legal entity name of the reporting company",
    )
    sector_type: SectorType = Field(
        SectorType.MANUFACTURING,
        description="Primary sector classification for preset selection",
    )
    country: str = Field(
        "DE",
        description="Primary country of operations (ISO 3166-1 alpha-2)",
    )
    reporting_year: int = Field(
        2026,
        ge=2020,
        le=2035,
        description="Reporting year for the Scope 3 inventory",
    )

    # Organisation characteristics
    revenue_meur: Optional[Decimal] = Field(
        None,
        ge=0,
        description="Annual revenue in million EUR for intensity metrics and SB 253 applicability",
    )
    employees_fte: Optional[int] = Field(
        None,
        ge=0,
        description="Full-time equivalent employees for Cat 7 estimation and intensity metrics",
    )
    total_procurement_spend_eur: Optional[Decimal] = Field(
        None,
        ge=0,
        description="Total annual procurement spend (EUR) for Cat 1 spend-based screening",
    )
    number_of_suppliers: Optional[int] = Field(
        None,
        ge=0,
        description="Total number of active suppliers for engagement planning",
    )

    # Sub-configurations
    screening: ScreeningConfig = Field(
        default_factory=ScreeningConfig,
        description="EEIO screening configuration for initial category assessment",
    )
    spend_classification: SpendClassificationConfig = Field(
        default_factory=SpendClassificationConfig,
        description="Spend data classification and mapping configuration",
    )
    categories: CategoryConfig = Field(
        default_factory=CategoryConfig,
        description="Per-category configuration for all 15 Scope 3 categories",
    )
    double_counting: DoubleCountingConfig = Field(
        default_factory=DoubleCountingConfig,
        description="Double-counting prevention configuration",
    )
    hotspot: HotspotConfig = Field(
        default_factory=HotspotConfig,
        description="Hotspot identification and prioritisation configuration",
    )
    supplier_engagement: SupplierEngagementConfig = Field(
        default_factory=SupplierEngagementConfig,
        description="Supplier engagement programme configuration",
    )
    data_quality: DataQualityConfig = Field(
        default_factory=DataQualityConfig,
        description="Data quality assessment configuration (5 DQIs)",
    )
    uncertainty: UncertaintyConfig = Field(
        default_factory=UncertaintyConfig,
        description="Uncertainty quantification configuration",
    )
    compliance: ComplianceConfig = Field(
        default_factory=ComplianceConfig,
        description="Regulatory compliance framework configuration",
    )
    reporting: ReportingConfig = Field(
        default_factory=ReportingConfig,
        description="Report generation configuration",
    )
    integration: IntegrationConfig = Field(
        default_factory=IntegrationConfig,
        description="Integration with other packs and external systems",
    )
    security: SecurityConfig = Field(
        default_factory=SecurityConfig,
        description="Security and access control",
    )
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig,
        description="Performance and resource limits",
    )
    audit_trail: AuditTrailConfig = Field(
        default_factory=AuditTrailConfig,
        description="Audit trail and provenance configuration",
    )

    @model_validator(mode="after")
    def validate_financial_sector_pcaf(self) -> "Scope3StarterConfig":
        """Financial sector should enable PCAF for Category 15."""
        if self.sector_type == SectorType.FINANCIAL:
            if not self.categories.cat_15.enabled:
                logger.info(
                    "Financial sector: enabling Category 15 (Investments) automatically."
                )
                self.categories.cat_15.enabled = True
            if not self.compliance.pcaf_enabled:
                logger.info(
                    "Financial sector: enabling PCAF methodology for Category 15."
                )
                self.compliance.pcaf_enabled = True
        return self

    @model_validator(mode="after")
    def validate_sme_simplified(self) -> "Scope3StarterConfig":
        """SME organisations use simplified configuration with fewer categories."""
        if self.sector_type == SectorType.SME:
            if self.uncertainty.monte_carlo_iterations > 3000:
                logger.info(
                    "SME sector: reducing Monte Carlo iterations to 3000 for efficiency."
                )
                self.uncertainty.monte_carlo_iterations = 3000
        return self

    @model_validator(mode="after")
    def validate_sbti_requires_scope3(self) -> "Scope3StarterConfig":
        """SBTi compliance requires Scope 3 coverage of at least 67% (near-term)."""
        if FrameworkType.SBTI in self.compliance.target_frameworks:
            enabled_cats = self.categories.get_enabled_categories()
            if len(enabled_cats) < 3:
                logger.warning(
                    "SBTi compliance requires covering at least 67% of Scope 3. "
                    "Only %d categories enabled. Consider enabling more.",
                    len(enabled_cats),
                )
        return self

    @model_validator(mode="after")
    def validate_spend_classification_thresholds(self) -> "Scope3StarterConfig":
        """Manual review threshold must be below auto-classification threshold."""
        if self.spend_classification.manual_review_threshold >= self.spend_classification.confidence_threshold:
            logger.warning(
                "Manual review threshold (%.2f) >= confidence threshold (%.2f). "
                "Setting manual review to confidence - 0.20.",
                self.spend_classification.manual_review_threshold,
                self.spend_classification.confidence_threshold,
            )
            self.spend_classification.manual_review_threshold = max(
                0.30,
                self.spend_classification.confidence_threshold - 0.20,
            )
        return self

    @model_validator(mode="after")
    def validate_cat3_requires_pack041(self) -> "Scope3StarterConfig":
        """Category 3 (FERA) calculations require Scope 1/2 fuel data from PACK-041."""
        if self.categories.cat_3.enabled and not self.integration.pack041_enabled:
            logger.warning(
                "Category 3 (Fuel- & Energy-Related Activities) requires "
                "Scope 1/2 data. Enabling PACK-041 integration automatically."
            )
            self.integration.pack041_enabled = True
        return self


# =============================================================================
# Pack Configuration Wrapper
# =============================================================================


class PackConfig(BaseModel):
    """Top-level pack configuration wrapper for PACK-042.

    Handles preset loading, environment variable overrides, and
    configuration merging. Follows the standard GreenLang pack config
    pattern with from_preset(), from_yaml(), and merge() support.
    """

    pack: Scope3StarterConfig = Field(
        default_factory=Scope3StarterConfig,
        description="Main Scope 3 Starter Pack configuration",
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
        "PACK-042-scope-3-starter",
        description="Pack identifier",
    )

    @classmethod
    def from_preset(
        cls, preset_name: str, overrides: Optional[Dict[str, Any]] = None
    ) -> "PackConfig":
        """Load configuration from a named preset.

        Args:
            preset_name: Name of the preset (manufacturing, retail, etc.)
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

        pack_config = Scope3StarterConfig(**preset_data)
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

        pack_config = Scope3StarterConfig(**config_data)
        return cls(pack=pack_config)

    @classmethod
    def merge(
        cls,
        base: "PackConfig",
        overrides: Dict[str, Any],
    ) -> "PackConfig":
        """Create a new PackConfig by merging overrides into an existing config.

        Args:
            base: Base PackConfig instance.
            overrides: Dictionary of configuration overrides.

        Returns:
            New PackConfig with merged values.
        """
        base_dict = base.pack.model_dump()
        merged = cls._deep_merge(base_dict, overrides)
        pack_config = Scope3StarterConfig(**merged)
        return cls(
            pack=pack_config,
            preset_name=base.preset_name,
            config_version=base.config_version,
        )

    @staticmethod
    def _load_env_overrides() -> Dict[str, Any]:
        """Load configuration overrides from environment variables.

        Environment variables prefixed with SCOPE3_PACK_ are loaded
        and mapped to configuration keys. Nested keys use double underscore.

        Example: SCOPE3_PACK_SCREENING__EEIO_MODEL=USEEIO_2
                 SCOPE3_PACK_UNCERTAINTY__METHOD=MONTE_CARLO
        """
        overrides: Dict[str, Any] = {}
        prefix = "SCOPE3_PACK_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                parts = config_key.split("__")
                current = overrides
                for part in parts[:-1]:
                    current = current.setdefault(part, {})
                # Parse value type
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
        """Deep merge two dictionaries, with override taking precedence.

        Args:
            base: Base dictionary.
            override: Override dictionary (values take precedence).

        Returns:
            Merged dictionary.
        """
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = PackConfig._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def get_config_hash(self) -> str:
        """Generate SHA-256 hash of the current configuration for provenance.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        config_json = self.model_dump_json(indent=None)
        return hashlib.sha256(config_json.encode("utf-8")).hexdigest()

    def validate_completeness(self) -> List[str]:
        """Validate configuration completeness and return warnings.

        Returns:
            List of warning messages (empty if fully valid).
        """
        return validate_config(self.pack)


# =============================================================================
# Utility Functions
# =============================================================================


def load_preset(
    preset_name: str, overrides: Optional[Dict[str, Any]] = None
) -> PackConfig:
    """Load a named preset configuration.

    Convenience wrapper around PackConfig.from_preset().

    Args:
        preset_name: Name of the preset to load.
        overrides: Optional configuration overrides.

    Returns:
        PackConfig instance with preset applied.
    """
    return PackConfig.from_preset(preset_name, overrides)


def validate_config(config: Scope3StarterConfig) -> List[str]:
    """Validate a Scope 3 Starter configuration and return any warnings.

    Args:
        config: Scope3StarterConfig instance to validate.

    Returns:
        List of warning messages (empty if fully valid).
    """
    warnings: List[str] = []

    # Check company identification
    if not config.company_name:
        warnings.append(
            "No company_name configured. Add a company name for report identification."
        )

    # Check at least one category is enabled
    enabled_cats = config.categories.get_enabled_categories()
    if not enabled_cats:
        warnings.append(
            "No Scope 3 categories enabled. At least one category must be configured."
        )

    # Check spend data availability for spend-based categories
    if config.total_procurement_spend_eur is None:
        spend_cats = []
        for cat_id in enabled_cats:
            cat_config = config.categories.get_category_config(cat_id)
            if cat_config.tier == MethodologyTier.SPEND_BASED:
                spend_cats.append(cat_id)
        if spend_cats:
            warnings.append(
                f"Spend-based methodology used for {spend_cats} but "
                f"total_procurement_spend_eur not provided."
            )

    # Check employee data for Cat 7
    if config.categories.cat_7.enabled and config.employees_fte is None:
        warnings.append(
            "Category 7 (Employee Commuting) enabled but employees_fte not provided."
        )

    # Check PCAF configuration for Cat 15
    if config.categories.cat_15.enabled:
        if config.compliance.pcaf_enabled and not config.compliance.pcaf_asset_classes:
            warnings.append(
                "PCAF enabled for Category 15 but no asset classes configured."
            )

    # Check SBTi coverage requirements
    if FrameworkType.SBTI in config.compliance.target_frameworks:
        if len(enabled_cats) < 5:
            warnings.append(
                "SBTi requires covering at least 67% of total Scope 3. "
                "Consider enabling more categories for comprehensive coverage."
            )
        if config.compliance.sbti_target_year is None:
            warnings.append(
                "SBTi compliance selected but no sbti_target_year configured."
            )

    # Check ESRS phase-in
    if FrameworkType.ESRS_E1 in config.compliance.target_frameworks:
        if config.reporting_year < config.compliance.esrs_phase_in_year:
            warnings.append(
                f"Reporting year ({config.reporting_year}) is before ESRS Scope 3 "
                f"phase-in year ({config.compliance.esrs_phase_in_year}). "
                f"Scope 3 disclosure may not yet be required."
            )

    # Check SB 253 applicability
    if FrameworkType.SB_253 in config.compliance.target_frameworks:
        if config.reporting_year < config.compliance.sb253_scope3_from:
            warnings.append(
                f"SB 253 Scope 3 reporting starts from {config.compliance.sb253_scope3_from}. "
                f"Reporting year {config.reporting_year} is before the start date."
            )
        if config.revenue_meur is not None and config.revenue_meur < Decimal("900"):
            warnings.append(
                "SB 253 applies to entities with revenue > USD 1 billion. "
                "Current revenue may be below threshold (check USD equivalent)."
            )

    # Check Cat 3 requires Scope 1/2 data
    if config.categories.cat_3.enabled and not config.integration.pack041_enabled:
        warnings.append(
            "Category 3 (Fuel- & Energy-Related Activities) requires Scope 1/2 "
            "fuel and energy data. Enable PACK-041 integration or provide manually."
        )

    # Check double-counting prevention
    if not config.double_counting.rules_enabled:
        warnings.append(
            "No double-counting prevention rules enabled. Consider enabling "
            "at least cat1_vs_cat4_transport and cat3_vs_scope12."
        )

    # Check data quality configuration
    if config.data_quality.target_dqr > 4.0:
        warnings.append(
            f"Target DQR of {config.data_quality.target_dqr} is very permissive. "
            f"Consider targeting DQR 3.0 or better for material categories."
        )

    # Check supplier engagement for material categories
    if config.supplier_engagement.top_n_suppliers < 10:
        warnings.append(
            "Engaging fewer than 10 suppliers may not achieve sufficient "
            "coverage for supplier-specific data improvement."
        )

    # Check sector-specific recommendations
    if config.sector_type == SectorType.FINANCIAL and not config.categories.cat_15.enabled:
        warnings.append(
            "Financial sector should enable Category 15 (Investments) with PCAF methodology."
        )

    if config.sector_type == SectorType.MANUFACTURING and not config.categories.cat_1.enabled:
        warnings.append(
            "Manufacturing sector should enable Category 1 (Purchased Goods & Services)."
        )

    if config.sector_type == SectorType.RETAIL and not config.categories.cat_9.enabled:
        warnings.append(
            "Retail sector should consider enabling Category 9 (Downstream Transport)."
        )

    return warnings


def get_default_config(
    sector_type: SectorType = SectorType.MANUFACTURING,
) -> Scope3StarterConfig:
    """Get default configuration for a given sector type.

    Args:
        sector_type: Sector type to configure for.

    Returns:
        Scope3StarterConfig instance with sector-appropriate defaults.
    """
    return Scope3StarterConfig(sector_type=sector_type)


def get_sector_info(sector_type: Union[str, SectorType]) -> Dict[str, Any]:
    """Get detailed information about a sector type and its Scope 3 profile.

    Args:
        sector_type: Sector type enum or string value.

    Returns:
        Dictionary with name, dominant categories, typical shares, and engagement targets.
    """
    key = sector_type.value if isinstance(sector_type, SectorType) else sector_type
    return SECTOR_INFO.get(
        key,
        {
            "name": key,
            "dominant_categories": ["CAT_1"],
            "cat_1_typical_pct": "30-60%",
            "description": "Varies by operation",
            "typical_total_scope3_vs_scope12": "Varies",
            "key_engagement_targets": "Top suppliers",
        },
    )


def get_category_info(category: Union[str, Scope3Category]) -> Dict[str, Any]:
    """Get detailed information about a Scope 3 category.

    Args:
        category: Category enum or string value (e.g., 'CAT_1').

    Returns:
        Dictionary with name, GHG Protocol reference, description,
        typical share, common methods, and key data sources.
    """
    key = category.value if isinstance(category, Scope3Category) else category
    return CATEGORY_INFO.get(
        key,
        {
            "name": f"Category {key}",
            "ghg_protocol_ref": "Unknown",
            "description": "Unknown category",
            "typical_share_pct": "N/A",
            "common_methods": [],
            "key_data_sources": [],
        },
    )


def list_available_presets() -> Dict[str, str]:
    """List all available configuration presets.

    Returns:
        Dictionary mapping preset names to descriptions.
    """
    return AVAILABLE_PRESETS.copy()
