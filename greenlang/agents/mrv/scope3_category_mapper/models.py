# -*- coding: utf-8 -*-
"""
Scope 3 Category Mapper Agent Models (AGENT-MRV-029)

This module provides comprehensive data models for the GHG Protocol Scope 3
Category Mapper agent, which classifies organizational data into the correct
Scope 3 category (1-15) and routes records to the appropriate category-specific
calculation agent.

Supports:
- 15 GHG Protocol Scope 3 categories (upstream + downstream)
- 13 input data source types (spend, PO, BOM, travel, fleet, waste, lease, etc.)
- 6 classification methods (NAICS, ISIC, UNSPSC, HS code, GL account, keyword)
- 4 calculation approaches (supplier-specific, hybrid, average-data, spend-based)
- 10 double-counting prevention rules (DC-SCM-001 through DC-SCM-010)
- 8 compliance frameworks (GHG Protocol, ISO 14064, CSRD, CDP, SBTi, SB 253,
  SEC Climate, EU Taxonomy)
- Multi-category split routing for records spanning multiple categories
- Completeness screening per GHG Protocol Technical Guidance
- Boundary determination with Incoterms and lease classification
- SHA-256 provenance chain with 10-stage pipeline
- Batch classification up to 50,000 records

All numeric fields use Decimal for precision in regulatory calculations.
All models are frozen (immutable) for audit trail integrity.

Example:
    >>> from greenlang.agents.mrv.scope3_category_mapper.models import (
    ...     ClassificationInput, DataSourceType, Scope3Category,
    ... )
    >>> inp = ClassificationInput(
    ...     record={"amount": "1500.00", "description": "Office supplies"},
    ...     source_type=DataSourceType.SPEND_DATA,
    ...     organization_id="org-001",
    ...     reporting_year=2025,
    ... )
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any
from datetime import datetime, date
from enum import Enum
from pydantic import ConfigDict, Field, validator
from greenlang.schemas import GreenLangBase, utcnow, new_uuid

import hashlib
import json

# ==============================================================================
# AGENT METADATA
# ==============================================================================

AGENT_ID: str = "GL-MRV-X-040"
AGENT_COMPONENT: str = "AGENT-MRV-029"
VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_scm_"

# Decimal quantization constants
_QUANT_2DP = Decimal("0.01")
_QUANT_4DP = Decimal("0.0001")
_QUANT_8DP = Decimal("0.00000001")

# ==============================================================================
# ENUMERATIONS (25)
# ==============================================================================


class Scope3Category(str, Enum):
    """GHG Protocol Scope 3 categories (1-15).

    Covers all 15 categories per the GHG Protocol Corporate Value Chain
    (Scope 3) Accounting and Reporting Standard. Categories 1-8 are upstream;
    categories 9-15 are downstream.
    """

    CAT_1_PURCHASED_GOODS_SERVICES = "1_purchased_goods_services"
    CAT_2_CAPITAL_GOODS = "2_capital_goods"
    CAT_3_FUEL_ENERGY_ACTIVITIES = "3_fuel_energy_activities"
    CAT_4_UPSTREAM_TRANSPORTATION = "4_upstream_transportation"
    CAT_5_WASTE_GENERATED = "5_waste_generated"
    CAT_6_BUSINESS_TRAVEL = "6_business_travel"
    CAT_7_EMPLOYEE_COMMUTING = "7_employee_commuting"
    CAT_8_UPSTREAM_LEASED_ASSETS = "8_upstream_leased_assets"
    CAT_9_DOWNSTREAM_TRANSPORTATION = "9_downstream_transportation"
    CAT_10_PROCESSING_SOLD_PRODUCTS = "10_processing_sold_products"
    CAT_11_USE_OF_SOLD_PRODUCTS = "11_use_of_sold_products"
    CAT_12_END_OF_LIFE_TREATMENT = "12_end_of_life_treatment"
    CAT_13_DOWNSTREAM_LEASED_ASSETS = "13_downstream_leased_assets"
    CAT_14_FRANCHISES = "14_franchises"
    CAT_15_INVESTMENTS = "15_investments"


class DataSourceType(str, Enum):
    """Input data source types for classification.

    Each source type carries different signals for category determination.
    The mapper uses source type as the first-pass classifier.
    """

    SPEND_DATA = "spend_data"
    PURCHASE_ORDER = "purchase_order"
    BOM = "bom"
    TRAVEL = "travel"
    FLEET = "fleet"
    WASTE = "waste"
    LEASE = "lease"
    LOGISTICS = "logistics"
    PRODUCT_SALES = "product_sales"
    INVESTMENT = "investment"
    FRANCHISE = "franchise"
    ENERGY = "energy"
    SUPPLIER = "supplier"


class ClassificationMethod(str, Enum):
    """Classification method used to determine category mapping.

    Each method carries a different confidence level. NAICS codes provide
    the highest confidence; keyword matching provides the lowest.
    """

    NAICS = "naics"
    ISIC = "isic"
    UNSPSC = "unspsc"
    HS_CODE = "hs_code"
    GL_ACCOUNT = "gl_account"
    KEYWORD = "keyword"


class CalculationApproach(str, Enum):
    """GHG Protocol Scope 3 calculation approaches.

    Per GHG Protocol Technical Guidance, supplier-specific is preferred,
    followed by hybrid, average-data, and spend-based (lowest quality).
    """

    SUPPLIER_SPECIFIC = "supplier_specific"
    HYBRID = "hybrid"
    AVERAGE_DATA = "average_data"
    SPEND_BASED = "spend_based"


class ConfidenceLevel(str, Enum):
    """Classification confidence level thresholds.

    very_high: >= 0.90 (deterministic code match)
    high:      >= 0.75 (strong signal match)
    medium:    >= 0.50 (moderate signal match)
    low:       >= 0.25 (weak signal, review recommended)
    very_low:  <  0.25 (uncertain, manual review required)
    """

    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


class ConsolidationApproach(str, Enum):
    """GHG Protocol organizational boundary consolidation approach."""

    OPERATIONAL_CONTROL = "operational_control"
    FINANCIAL_CONTROL = "financial_control"
    EQUITY_SHARE = "equity_share"


class ValueChainPosition(str, Enum):
    """Position in the value chain for boundary determination."""

    UPSTREAM = "upstream"
    DOWNSTREAM = "downstream"


class CategoryRelevance(str, Enum):
    """Relevance assessment for a Scope 3 category.

    Per GHG Protocol guidance, companies must identify which categories are
    relevant based on size, influence, risk, stakeholder expectations, and
    outsourcing/sector-specific criteria.
    """

    MATERIAL = "material"
    RELEVANT = "relevant"
    NOT_RELEVANT = "not_relevant"
    UNKNOWN = "unknown"


class CompanyType(str, Enum):
    """Company type for industry-specific category relevance screening."""

    MANUFACTURER = "manufacturer"
    SERVICES = "services"
    FINANCIAL = "financial"
    RETAILER = "retailer"
    ENERGY = "energy"
    MINING = "mining"
    AGRICULTURE = "agriculture"
    TRANSPORT = "transport"


class IncotermsRule(str, Enum):
    """Incoterms 2020 rules for transportation boundary determination.

    Determines whether transportation emissions fall under Category 4
    (upstream) or Category 9 (downstream) based on point of transfer.
    """

    EXW = "EXW"
    FCA = "FCA"
    CPT = "CPT"
    CIP = "CIP"
    DAP = "DAP"
    DPU = "DPU"
    DDP = "DDP"
    FAS = "FAS"
    FOB = "FOB"
    CFR = "CFR"
    CIF = "CIF"


class LeaseClassification(str, Enum):
    """Lease classification for Category 8 / Category 13 boundary.

    Operating leases and finance leases may differ in how emissions are
    reported. Short-term and low-value leases may be excluded per IFRS 16.
    """

    OPERATING_LEASE = "operating_lease"
    FINANCE_LEASE = "finance_lease"
    SHORT_TERM = "short_term"
    LOW_VALUE = "low_value"


class CapitalizationPolicy(str, Enum):
    """Asset capitalization policy for Category 1 vs Category 2 boundary.

    Determines whether a purchased asset is categorized as Cat 1
    (purchased goods -- expensed) or Cat 2 (capital goods -- capitalized).
    """

    CAPITALIZE = "capitalize"
    EXPENSE = "expense"
    THRESHOLD_BASED = "threshold_based"


class CurrencyCode(str, Enum):
    """ISO 4217 currency codes for financial record classification."""

    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CHF = "CHF"
    CAD = "CAD"
    AUD = "AUD"
    CNY = "CNY"
    KRW = "KRW"
    SGD = "SGD"
    HKD = "HKD"
    SEK = "SEK"
    NOK = "NOK"
    DKK = "DKK"
    BRL = "BRL"
    INR = "INR"
    MXN = "MXN"
    ZAR = "ZAR"
    TRY = "TRY"
    PLN = "PLN"
    THB = "THB"
    MYR = "MYR"
    IDR = "IDR"
    PHP = "PHP"
    VND = "VND"
    CLP = "CLP"
    COP = "COP"
    PEN = "PEN"
    ARS = "ARS"
    EGP = "EGP"
    NGN = "NGN"
    PKR = "PKR"
    BDT = "BDT"
    TWD = "TWD"
    SAR = "SAR"
    AED = "AED"
    QAR = "QAR"
    KWD = "KWD"
    ILS = "ILS"
    NZD = "NZD"
    CZK = "CZK"
    HUF = "HUF"
    RON = "RON"


class GWPVersion(str, Enum):
    """IPCC Global Warming Potential assessment report version."""

    SAR = "SAR"
    AR4 = "AR4"
    AR5 = "AR5"
    AR6 = "AR6"


class ComplianceFramework(str, Enum):
    """Regulatory/reporting frameworks for completeness and compliance checks."""

    GHG_PROTOCOL = "ghg_protocol"
    ISO_14064 = "iso_14064"
    CSRD_ESRS = "csrd_esrs"
    CDP = "cdp"
    SBTI = "sbti"
    SB_253 = "sb_253"
    SEC_CLIMATE = "sec_climate"
    EU_TAXONOMY = "eu_taxonomy"


class DataQualityTier(str, Enum):
    """Data quality tier (1 = best, 5 = worst).

    Tier 1: Supplier-specific, verified primary data.
    Tier 2: Supplier-specific, unverified primary data.
    Tier 3: Average data with good proxy match.
    Tier 4: Spend-based or estimated data.
    Tier 5: Extrapolated or default values.
    """

    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"
    TIER_4 = "tier_4"
    TIER_5 = "tier_5"


class MappingStatus(str, Enum):
    """Status of a record's category mapping."""

    MAPPED = "mapped"
    SPLIT = "split"
    UNMAPPED = "unmapped"
    REVIEW_REQUIRED = "review_required"
    EXCLUDED = "excluded"


class DoubleCountingRule(str, Enum):
    """Double-counting prevention rules (DC-SCM-001 through DC-SCM-010).

    Each rule addresses a specific overlap risk between Scope 3 categories
    or between Scope 1/2 and Scope 3.
    """

    DC_SCM_001 = "DC-SCM-001"
    DC_SCM_002 = "DC-SCM-002"
    DC_SCM_003 = "DC-SCM-003"
    DC_SCM_004 = "DC-SCM-004"
    DC_SCM_005 = "DC-SCM-005"
    DC_SCM_006 = "DC-SCM-006"
    DC_SCM_007 = "DC-SCM-007"
    DC_SCM_008 = "DC-SCM-008"
    DC_SCM_009 = "DC-SCM-009"
    DC_SCM_010 = "DC-SCM-010"


class MaterialityThreshold(str, Enum):
    """Materiality threshold types for category screening."""

    QUANTITATIVE_1PCT = "quantitative_1pct"
    QUALITATIVE_HIGH = "qualitative_high"
    DE_MINIMIS = "de_minimis"


class NAICSLevel(str, Enum):
    """NAICS code hierarchy levels."""

    SECTOR_2 = "sector_2"
    SUBSECTOR_3 = "subsector_3"
    INDUSTRY_GROUP_4 = "industry_group_4"
    INDUSTRY_6 = "industry_6"


class ISICLevel(str, Enum):
    """ISIC code hierarchy levels."""

    SECTION_1 = "section_1"
    DIVISION_2 = "division_2"
    GROUP_3 = "group_3"
    CLASS_4 = "class_4"


class UNSPSCLevel(str, Enum):
    """UNSPSC code hierarchy levels."""

    SEGMENT_2 = "segment_2"
    FAMILY_4 = "family_4"
    CLASS_6 = "class_6"
    COMMODITY_8 = "commodity_8"


class RoutingAction(str, Enum):
    """Action to take when routing a classified record."""

    ROUTE = "route"
    SPLIT_ROUTE = "split_route"
    QUEUE_REVIEW = "queue_review"
    EXCLUDE = "exclude"


class ScreeningResult(str, Enum):
    """Category screening data availability result."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    MISSING = "missing"


class PipelineStage(str, Enum):
    """Processing pipeline stages for provenance tracking."""

    INPUT_VALIDATION = "input_validation"
    SOURCE_IDENTIFICATION = "source_identification"
    CODE_CLASSIFICATION = "code_classification"
    BOUNDARY_DETERMINATION = "boundary_determination"
    CATEGORY_ASSIGNMENT = "category_assignment"
    DOUBLE_COUNTING_CHECK = "double_counting_check"
    COMPLETENESS_SCREENING = "completeness_screening"
    ROUTING_PLAN = "routing_plan"
    COMPLIANCE_CHECK = "compliance_check"
    OUTPUT_ASSEMBLY = "output_assembly"


# ==============================================================================
# CONSTANT TABLES
# ==============================================================================

# Scope 3 Category Metadata -- 15 categories with name, position, and agent ID
SCOPE3_CATEGORY_METADATA: Dict[str, Dict[str, str]] = {
    Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES.value: {
        "number": "1",
        "name": "Purchased Goods and Services",
        "position": "upstream",
        "agent_id": "GL-MRV-S3-001",
        "agent_component": "AGENT-MRV-014",
        "api_endpoint": "/api/v1/purchased-goods-services",
    },
    Scope3Category.CAT_2_CAPITAL_GOODS.value: {
        "number": "2",
        "name": "Capital Goods",
        "position": "upstream",
        "agent_id": "GL-MRV-S3-002",
        "agent_component": "AGENT-MRV-015",
        "api_endpoint": "/api/v1/capital-goods",
    },
    Scope3Category.CAT_3_FUEL_ENERGY_ACTIVITIES.value: {
        "number": "3",
        "name": "Fuel- and Energy-Related Activities",
        "position": "upstream",
        "agent_id": "GL-MRV-S3-003",
        "agent_component": "AGENT-MRV-016",
        "api_endpoint": "/api/v1/fuel-energy-activities",
    },
    Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION.value: {
        "number": "4",
        "name": "Upstream Transportation and Distribution",
        "position": "upstream",
        "agent_id": "GL-MRV-S3-004",
        "agent_component": "AGENT-MRV-017",
        "api_endpoint": "/api/v1/upstream-transportation",
    },
    Scope3Category.CAT_5_WASTE_GENERATED.value: {
        "number": "5",
        "name": "Waste Generated in Operations",
        "position": "upstream",
        "agent_id": "GL-MRV-S3-005",
        "agent_component": "AGENT-MRV-018",
        "api_endpoint": "/api/v1/waste-generated",
    },
    Scope3Category.CAT_6_BUSINESS_TRAVEL.value: {
        "number": "6",
        "name": "Business Travel",
        "position": "upstream",
        "agent_id": "GL-MRV-S3-006",
        "agent_component": "AGENT-MRV-019",
        "api_endpoint": "/api/v1/business-travel",
    },
    Scope3Category.CAT_7_EMPLOYEE_COMMUTING.value: {
        "number": "7",
        "name": "Employee Commuting",
        "position": "upstream",
        "agent_id": "GL-MRV-S3-007",
        "agent_component": "AGENT-MRV-020",
        "api_endpoint": "/api/v1/employee-commuting",
    },
    Scope3Category.CAT_8_UPSTREAM_LEASED_ASSETS.value: {
        "number": "8",
        "name": "Upstream Leased Assets",
        "position": "upstream",
        "agent_id": "GL-MRV-S3-008",
        "agent_component": "AGENT-MRV-021",
        "api_endpoint": "/api/v1/upstream-leased-assets",
    },
    Scope3Category.CAT_9_DOWNSTREAM_TRANSPORTATION.value: {
        "number": "9",
        "name": "Downstream Transportation and Distribution",
        "position": "downstream",
        "agent_id": "GL-MRV-S3-009",
        "agent_component": "AGENT-MRV-022",
        "api_endpoint": "/api/v1/downstream-transportation",
    },
    Scope3Category.CAT_10_PROCESSING_SOLD_PRODUCTS.value: {
        "number": "10",
        "name": "Processing of Sold Products",
        "position": "downstream",
        "agent_id": "GL-MRV-S3-010",
        "agent_component": "AGENT-MRV-023",
        "api_endpoint": "/api/v1/processing-sold-products",
    },
    Scope3Category.CAT_11_USE_OF_SOLD_PRODUCTS.value: {
        "number": "11",
        "name": "Use of Sold Products",
        "position": "downstream",
        "agent_id": "GL-MRV-S3-011",
        "agent_component": "AGENT-MRV-024",
        "api_endpoint": "/api/v1/use-of-sold-products",
    },
    Scope3Category.CAT_12_END_OF_LIFE_TREATMENT.value: {
        "number": "12",
        "name": "End-of-Life Treatment of Sold Products",
        "position": "downstream",
        "agent_id": "GL-MRV-S3-012",
        "agent_component": "AGENT-MRV-025",
        "api_endpoint": "/api/v1/end-of-life-treatment",
    },
    Scope3Category.CAT_13_DOWNSTREAM_LEASED_ASSETS.value: {
        "number": "13",
        "name": "Downstream Leased Assets",
        "position": "downstream",
        "agent_id": "GL-MRV-S3-013",
        "agent_component": "AGENT-MRV-026",
        "api_endpoint": "/api/v1/downstream-leased-assets",
    },
    Scope3Category.CAT_14_FRANCHISES.value: {
        "number": "14",
        "name": "Franchises",
        "position": "downstream",
        "agent_id": "GL-MRV-S3-014",
        "agent_component": "AGENT-MRV-027",
        "api_endpoint": "/api/v1/franchises",
    },
    Scope3Category.CAT_15_INVESTMENTS.value: {
        "number": "15",
        "name": "Investments",
        "position": "downstream",
        "agent_id": "GL-MRV-S3-015",
        "agent_component": "AGENT-MRV-028",
        "api_endpoint": "/api/v1/investments",
    },
}

# Source-type to default category mapping -- first-pass heuristic
SOURCE_TYPE_DEFAULT_CATEGORY: Dict[str, str] = {
    DataSourceType.SPEND_DATA.value: Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES.value,
    DataSourceType.PURCHASE_ORDER.value: Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES.value,
    DataSourceType.BOM.value: Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES.value,
    DataSourceType.TRAVEL.value: Scope3Category.CAT_6_BUSINESS_TRAVEL.value,
    DataSourceType.FLEET.value: Scope3Category.CAT_7_EMPLOYEE_COMMUTING.value,
    DataSourceType.WASTE.value: Scope3Category.CAT_5_WASTE_GENERATED.value,
    DataSourceType.LEASE.value: Scope3Category.CAT_8_UPSTREAM_LEASED_ASSETS.value,
    DataSourceType.LOGISTICS.value: Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION.value,
    DataSourceType.PRODUCT_SALES.value: Scope3Category.CAT_11_USE_OF_SOLD_PRODUCTS.value,
    DataSourceType.INVESTMENT.value: Scope3Category.CAT_15_INVESTMENTS.value,
    DataSourceType.FRANCHISE.value: Scope3Category.CAT_14_FRANCHISES.value,
    DataSourceType.ENERGY.value: Scope3Category.CAT_3_FUEL_ENERGY_ACTIVITIES.value,
    DataSourceType.SUPPLIER.value: Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES.value,
}

# Classification method confidence defaults
CLASSIFICATION_METHOD_CONFIDENCE: Dict[str, Decimal] = {
    ClassificationMethod.NAICS.value: Decimal("0.95"),
    ClassificationMethod.ISIC.value: Decimal("0.90"),
    ClassificationMethod.UNSPSC.value: Decimal("0.90"),
    ClassificationMethod.HS_CODE.value: Decimal("0.85"),
    ClassificationMethod.GL_ACCOUNT.value: Decimal("0.85"),
    ClassificationMethod.KEYWORD.value: Decimal("0.40"),
}

# Incoterm upstream/downstream boundary rules
INCOTERM_TRANSPORT_BOUNDARY: Dict[str, Dict[str, str]] = {
    IncotermsRule.EXW.value: {
        "buyer_freight_responsibility": "full",
        "buyer_category": Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION.value,
        "seller_category": Scope3Category.CAT_9_DOWNSTREAM_TRANSPORTATION.value,
    },
    IncotermsRule.FCA.value: {
        "buyer_freight_responsibility": "from_carrier",
        "buyer_category": Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION.value,
        "seller_category": Scope3Category.CAT_9_DOWNSTREAM_TRANSPORTATION.value,
    },
    IncotermsRule.CPT.value: {
        "buyer_freight_responsibility": "from_destination",
        "buyer_category": Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION.value,
        "seller_category": Scope3Category.CAT_9_DOWNSTREAM_TRANSPORTATION.value,
    },
    IncotermsRule.CIP.value: {
        "buyer_freight_responsibility": "from_destination",
        "buyer_category": Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION.value,
        "seller_category": Scope3Category.CAT_9_DOWNSTREAM_TRANSPORTATION.value,
    },
    IncotermsRule.DAP.value: {
        "buyer_freight_responsibility": "none",
        "buyer_category": Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES.value,
        "seller_category": Scope3Category.CAT_9_DOWNSTREAM_TRANSPORTATION.value,
    },
    IncotermsRule.DPU.value: {
        "buyer_freight_responsibility": "none",
        "buyer_category": Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES.value,
        "seller_category": Scope3Category.CAT_9_DOWNSTREAM_TRANSPORTATION.value,
    },
    IncotermsRule.DDP.value: {
        "buyer_freight_responsibility": "none",
        "buyer_category": Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES.value,
        "seller_category": Scope3Category.CAT_9_DOWNSTREAM_TRANSPORTATION.value,
    },
    IncotermsRule.FAS.value: {
        "buyer_freight_responsibility": "from_port",
        "buyer_category": Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION.value,
        "seller_category": Scope3Category.CAT_9_DOWNSTREAM_TRANSPORTATION.value,
    },
    IncotermsRule.FOB.value: {
        "buyer_freight_responsibility": "from_port",
        "buyer_category": Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION.value,
        "seller_category": Scope3Category.CAT_9_DOWNSTREAM_TRANSPORTATION.value,
    },
    IncotermsRule.CFR.value: {
        "buyer_freight_responsibility": "from_port_destination",
        "buyer_category": Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION.value,
        "seller_category": Scope3Category.CAT_9_DOWNSTREAM_TRANSPORTATION.value,
    },
    IncotermsRule.CIF.value: {
        "buyer_freight_responsibility": "from_port_destination",
        "buyer_category": Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION.value,
        "seller_category": Scope3Category.CAT_9_DOWNSTREAM_TRANSPORTATION.value,
    },
}

# Double-counting rules (DC-SCM-001 through DC-SCM-010)
DC_RULES: Dict[str, Dict[str, str]] = {
    "DC-SCM-001": {
        "rule": "Cat 1 vs Cat 2 boundary: expensed vs capitalized goods",
        "description": (
            "Purchased goods that are capitalized per the company's accounting "
            "policy should be reported under Category 2 (capital goods), not "
            "Category 1. Ensure each line item is classified under exactly one."
        ),
        "category_a": Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES.value,
        "category_b": Scope3Category.CAT_2_CAPITAL_GOODS.value,
        "action": "RECLASSIFY",
    },
    "DC-SCM-002": {
        "rule": "Cat 1 vs Cat 4 boundary: purchased goods including freight",
        "description": (
            "If purchased goods spend includes freight/shipping costs, the "
            "transportation component should be separated into Category 4 "
            "(upstream transportation) to avoid double-counting."
        ),
        "category_a": Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES.value,
        "category_b": Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION.value,
        "action": "SPLIT",
    },
    "DC-SCM-003": {
        "rule": "Cat 3 vs Scope 2: upstream energy vs purchased electricity",
        "description": (
            "Category 3 covers well-to-tank emissions of fuels, upstream "
            "emissions of purchased electricity, and T&D losses. Ensure no "
            "overlap with Scope 2 location-based or market-based reporting."
        ),
        "category_a": Scope3Category.CAT_3_FUEL_ENERGY_ACTIVITIES.value,
        "category_b": "scope_2",
        "action": "EXCLUDE",
    },
    "DC-SCM-004": {
        "rule": "Cat 4 vs Cat 9 boundary: Incoterm-based transport direction",
        "description": (
            "Transportation paid for by the reporting company for inbound goods "
            "is Category 4; outbound transportation to customers is Category 9. "
            "The Incoterm determines the boundary."
        ),
        "category_a": Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION.value,
        "category_b": Scope3Category.CAT_9_DOWNSTREAM_TRANSPORTATION.value,
        "action": "RECLASSIFY",
    },
    "DC-SCM-005": {
        "rule": "Cat 6 vs Cat 7 boundary: business travel vs commuting",
        "description": (
            "Business travel (Cat 6) covers trips for work purposes. Employee "
            "commuting (Cat 7) covers home-to-work transportation. Ensure "
            "no overlap for trips that serve dual purposes."
        ),
        "category_a": Scope3Category.CAT_6_BUSINESS_TRAVEL.value,
        "category_b": Scope3Category.CAT_7_EMPLOYEE_COMMUTING.value,
        "action": "RECLASSIFY",
    },
    "DC-SCM-006": {
        "rule": "Cat 8 vs Cat 13 boundary: upstream vs downstream leases",
        "description": (
            "Assets leased by the reporting company (lessee) are Category 8. "
            "Assets owned by the reporting company and leased to others "
            "(lessor) are Category 13. Ensure correct direction."
        ),
        "category_a": Scope3Category.CAT_8_UPSTREAM_LEASED_ASSETS.value,
        "category_b": Scope3Category.CAT_13_DOWNSTREAM_LEASED_ASSETS.value,
        "action": "RECLASSIFY",
    },
    "DC-SCM-007": {
        "rule": "Cat 8/13 vs Scope 1/2 boundary: leased asset consolidation",
        "description": (
            "If the reporting company uses the operational control approach "
            "and operates a leased asset, emissions belong in Scope 1/2, "
            "not in Category 8. Similarly for financial control."
        ),
        "category_a": Scope3Category.CAT_8_UPSTREAM_LEASED_ASSETS.value,
        "category_b": "scope_1_2",
        "action": "EXCLUDE",
    },
    "DC-SCM-008": {
        "rule": "Cat 10 vs Cat 11 boundary: processing vs use of sold products",
        "description": (
            "Category 10 covers intermediate processing of sold products by "
            "downstream entities. Category 11 covers end-use of products by "
            "consumers. Ensure each emission is counted once."
        ),
        "category_a": Scope3Category.CAT_10_PROCESSING_SOLD_PRODUCTS.value,
        "category_b": Scope3Category.CAT_11_USE_OF_SOLD_PRODUCTS.value,
        "action": "RECLASSIFY",
    },
    "DC-SCM-009": {
        "rule": "Cat 11 vs Cat 12 boundary: use-phase vs end-of-life",
        "description": (
            "Category 11 covers emissions during product use phase. Category 12 "
            "covers end-of-life treatment. For products that emit during both "
            "phases, ensure no lifecycle stage overlap."
        ),
        "category_a": Scope3Category.CAT_11_USE_OF_SOLD_PRODUCTS.value,
        "category_b": Scope3Category.CAT_12_END_OF_LIFE_TREATMENT.value,
        "action": "RECLASSIFY",
    },
    "DC-SCM-010": {
        "rule": "Cat 14 vs Cat 13 boundary: franchises vs downstream leased assets",
        "description": (
            "Franchise operations (Cat 14) and downstream leased assets (Cat 13) "
            "may overlap if franchise agreements include property leases. Ensure "
            "franchise-specific emissions are not double-counted."
        ),
        "category_a": Scope3Category.CAT_14_FRANCHISES.value,
        "category_b": Scope3Category.CAT_13_DOWNSTREAM_LEASED_ASSETS.value,
        "action": "RECLASSIFY",
    },
}

# Industry-specific category relevance matrix
INDUSTRY_CATEGORY_RELEVANCE: Dict[str, Dict[str, str]] = {
    CompanyType.MANUFACTURER.value: {
        Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES.value: CategoryRelevance.MATERIAL.value,
        Scope3Category.CAT_2_CAPITAL_GOODS.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_3_FUEL_ENERGY_ACTIVITIES.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION.value: CategoryRelevance.MATERIAL.value,
        Scope3Category.CAT_5_WASTE_GENERATED.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_6_BUSINESS_TRAVEL.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_7_EMPLOYEE_COMMUTING.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_8_UPSTREAM_LEASED_ASSETS.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_9_DOWNSTREAM_TRANSPORTATION.value: CategoryRelevance.MATERIAL.value,
        Scope3Category.CAT_10_PROCESSING_SOLD_PRODUCTS.value: CategoryRelevance.MATERIAL.value,
        Scope3Category.CAT_11_USE_OF_SOLD_PRODUCTS.value: CategoryRelevance.MATERIAL.value,
        Scope3Category.CAT_12_END_OF_LIFE_TREATMENT.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_13_DOWNSTREAM_LEASED_ASSETS.value: CategoryRelevance.NOT_RELEVANT.value,
        Scope3Category.CAT_14_FRANCHISES.value: CategoryRelevance.NOT_RELEVANT.value,
        Scope3Category.CAT_15_INVESTMENTS.value: CategoryRelevance.NOT_RELEVANT.value,
    },
    CompanyType.SERVICES.value: {
        Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES.value: CategoryRelevance.MATERIAL.value,
        Scope3Category.CAT_2_CAPITAL_GOODS.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_3_FUEL_ENERGY_ACTIVITIES.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION.value: CategoryRelevance.NOT_RELEVANT.value,
        Scope3Category.CAT_5_WASTE_GENERATED.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_6_BUSINESS_TRAVEL.value: CategoryRelevance.MATERIAL.value,
        Scope3Category.CAT_7_EMPLOYEE_COMMUTING.value: CategoryRelevance.MATERIAL.value,
        Scope3Category.CAT_8_UPSTREAM_LEASED_ASSETS.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_9_DOWNSTREAM_TRANSPORTATION.value: CategoryRelevance.NOT_RELEVANT.value,
        Scope3Category.CAT_10_PROCESSING_SOLD_PRODUCTS.value: CategoryRelevance.NOT_RELEVANT.value,
        Scope3Category.CAT_11_USE_OF_SOLD_PRODUCTS.value: CategoryRelevance.NOT_RELEVANT.value,
        Scope3Category.CAT_12_END_OF_LIFE_TREATMENT.value: CategoryRelevance.NOT_RELEVANT.value,
        Scope3Category.CAT_13_DOWNSTREAM_LEASED_ASSETS.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_14_FRANCHISES.value: CategoryRelevance.NOT_RELEVANT.value,
        Scope3Category.CAT_15_INVESTMENTS.value: CategoryRelevance.NOT_RELEVANT.value,
    },
    CompanyType.FINANCIAL.value: {
        Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_2_CAPITAL_GOODS.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_3_FUEL_ENERGY_ACTIVITIES.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION.value: CategoryRelevance.NOT_RELEVANT.value,
        Scope3Category.CAT_5_WASTE_GENERATED.value: CategoryRelevance.NOT_RELEVANT.value,
        Scope3Category.CAT_6_BUSINESS_TRAVEL.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_7_EMPLOYEE_COMMUTING.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_8_UPSTREAM_LEASED_ASSETS.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_9_DOWNSTREAM_TRANSPORTATION.value: CategoryRelevance.NOT_RELEVANT.value,
        Scope3Category.CAT_10_PROCESSING_SOLD_PRODUCTS.value: CategoryRelevance.NOT_RELEVANT.value,
        Scope3Category.CAT_11_USE_OF_SOLD_PRODUCTS.value: CategoryRelevance.NOT_RELEVANT.value,
        Scope3Category.CAT_12_END_OF_LIFE_TREATMENT.value: CategoryRelevance.NOT_RELEVANT.value,
        Scope3Category.CAT_13_DOWNSTREAM_LEASED_ASSETS.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_14_FRANCHISES.value: CategoryRelevance.NOT_RELEVANT.value,
        Scope3Category.CAT_15_INVESTMENTS.value: CategoryRelevance.MATERIAL.value,
    },
    CompanyType.RETAILER.value: {
        Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES.value: CategoryRelevance.MATERIAL.value,
        Scope3Category.CAT_2_CAPITAL_GOODS.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_3_FUEL_ENERGY_ACTIVITIES.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION.value: CategoryRelevance.MATERIAL.value,
        Scope3Category.CAT_5_WASTE_GENERATED.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_6_BUSINESS_TRAVEL.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_7_EMPLOYEE_COMMUTING.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_8_UPSTREAM_LEASED_ASSETS.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_9_DOWNSTREAM_TRANSPORTATION.value: CategoryRelevance.MATERIAL.value,
        Scope3Category.CAT_10_PROCESSING_SOLD_PRODUCTS.value: CategoryRelevance.NOT_RELEVANT.value,
        Scope3Category.CAT_11_USE_OF_SOLD_PRODUCTS.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_12_END_OF_LIFE_TREATMENT.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_13_DOWNSTREAM_LEASED_ASSETS.value: CategoryRelevance.NOT_RELEVANT.value,
        Scope3Category.CAT_14_FRANCHISES.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_15_INVESTMENTS.value: CategoryRelevance.NOT_RELEVANT.value,
    },
    CompanyType.ENERGY.value: {
        Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES.value: CategoryRelevance.MATERIAL.value,
        Scope3Category.CAT_2_CAPITAL_GOODS.value: CategoryRelevance.MATERIAL.value,
        Scope3Category.CAT_3_FUEL_ENERGY_ACTIVITIES.value: CategoryRelevance.MATERIAL.value,
        Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION.value: CategoryRelevance.MATERIAL.value,
        Scope3Category.CAT_5_WASTE_GENERATED.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_6_BUSINESS_TRAVEL.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_7_EMPLOYEE_COMMUTING.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_8_UPSTREAM_LEASED_ASSETS.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_9_DOWNSTREAM_TRANSPORTATION.value: CategoryRelevance.MATERIAL.value,
        Scope3Category.CAT_10_PROCESSING_SOLD_PRODUCTS.value: CategoryRelevance.MATERIAL.value,
        Scope3Category.CAT_11_USE_OF_SOLD_PRODUCTS.value: CategoryRelevance.MATERIAL.value,
        Scope3Category.CAT_12_END_OF_LIFE_TREATMENT.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_13_DOWNSTREAM_LEASED_ASSETS.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_14_FRANCHISES.value: CategoryRelevance.NOT_RELEVANT.value,
        Scope3Category.CAT_15_INVESTMENTS.value: CategoryRelevance.RELEVANT.value,
    },
    CompanyType.MINING.value: {
        Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES.value: CategoryRelevance.MATERIAL.value,
        Scope3Category.CAT_2_CAPITAL_GOODS.value: CategoryRelevance.MATERIAL.value,
        Scope3Category.CAT_3_FUEL_ENERGY_ACTIVITIES.value: CategoryRelevance.MATERIAL.value,
        Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION.value: CategoryRelevance.MATERIAL.value,
        Scope3Category.CAT_5_WASTE_GENERATED.value: CategoryRelevance.MATERIAL.value,
        Scope3Category.CAT_6_BUSINESS_TRAVEL.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_7_EMPLOYEE_COMMUTING.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_8_UPSTREAM_LEASED_ASSETS.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_9_DOWNSTREAM_TRANSPORTATION.value: CategoryRelevance.MATERIAL.value,
        Scope3Category.CAT_10_PROCESSING_SOLD_PRODUCTS.value: CategoryRelevance.MATERIAL.value,
        Scope3Category.CAT_11_USE_OF_SOLD_PRODUCTS.value: CategoryRelevance.MATERIAL.value,
        Scope3Category.CAT_12_END_OF_LIFE_TREATMENT.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_13_DOWNSTREAM_LEASED_ASSETS.value: CategoryRelevance.NOT_RELEVANT.value,
        Scope3Category.CAT_14_FRANCHISES.value: CategoryRelevance.NOT_RELEVANT.value,
        Scope3Category.CAT_15_INVESTMENTS.value: CategoryRelevance.NOT_RELEVANT.value,
    },
    CompanyType.AGRICULTURE.value: {
        Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES.value: CategoryRelevance.MATERIAL.value,
        Scope3Category.CAT_2_CAPITAL_GOODS.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_3_FUEL_ENERGY_ACTIVITIES.value: CategoryRelevance.MATERIAL.value,
        Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION.value: CategoryRelevance.MATERIAL.value,
        Scope3Category.CAT_5_WASTE_GENERATED.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_6_BUSINESS_TRAVEL.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_7_EMPLOYEE_COMMUTING.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_8_UPSTREAM_LEASED_ASSETS.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_9_DOWNSTREAM_TRANSPORTATION.value: CategoryRelevance.MATERIAL.value,
        Scope3Category.CAT_10_PROCESSING_SOLD_PRODUCTS.value: CategoryRelevance.MATERIAL.value,
        Scope3Category.CAT_11_USE_OF_SOLD_PRODUCTS.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_12_END_OF_LIFE_TREATMENT.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_13_DOWNSTREAM_LEASED_ASSETS.value: CategoryRelevance.NOT_RELEVANT.value,
        Scope3Category.CAT_14_FRANCHISES.value: CategoryRelevance.NOT_RELEVANT.value,
        Scope3Category.CAT_15_INVESTMENTS.value: CategoryRelevance.NOT_RELEVANT.value,
    },
    CompanyType.TRANSPORT.value: {
        Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES.value: CategoryRelevance.MATERIAL.value,
        Scope3Category.CAT_2_CAPITAL_GOODS.value: CategoryRelevance.MATERIAL.value,
        Scope3Category.CAT_3_FUEL_ENERGY_ACTIVITIES.value: CategoryRelevance.MATERIAL.value,
        Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_5_WASTE_GENERATED.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_6_BUSINESS_TRAVEL.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_7_EMPLOYEE_COMMUTING.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_8_UPSTREAM_LEASED_ASSETS.value: CategoryRelevance.MATERIAL.value,
        Scope3Category.CAT_9_DOWNSTREAM_TRANSPORTATION.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_10_PROCESSING_SOLD_PRODUCTS.value: CategoryRelevance.NOT_RELEVANT.value,
        Scope3Category.CAT_11_USE_OF_SOLD_PRODUCTS.value: CategoryRelevance.NOT_RELEVANT.value,
        Scope3Category.CAT_12_END_OF_LIFE_TREATMENT.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_13_DOWNSTREAM_LEASED_ASSETS.value: CategoryRelevance.RELEVANT.value,
        Scope3Category.CAT_14_FRANCHISES.value: CategoryRelevance.NOT_RELEVANT.value,
        Scope3Category.CAT_15_INVESTMENTS.value: CategoryRelevance.NOT_RELEVANT.value,
    },
}

# Compliance framework required categories
COMPLIANCE_REQUIRED_CATEGORIES: Dict[str, Dict[str, Any]] = {
    ComplianceFramework.GHG_PROTOCOL.value: {
        "name": "GHG Protocol Scope 3 Standard",
        "min_categories": 1,
        "required_if_material": True,
        "version": "2011 (with 2013 amendments)",
    },
    ComplianceFramework.ISO_14064.value: {
        "name": "ISO 14064-1:2018",
        "min_categories": 1,
        "required_if_material": True,
        "version": "2018",
    },
    ComplianceFramework.CSRD_ESRS.value: {
        "name": "CSRD ESRS E1 Climate Change",
        "min_categories": 15,
        "required_if_material": True,
        "version": "ESRS E1 (2024)",
    },
    ComplianceFramework.CDP.value: {
        "name": "CDP Climate Change Questionnaire",
        "min_categories": 1,
        "required_if_material": True,
        "version": "2024",
    },
    ComplianceFramework.SBTI.value: {
        "name": "Science Based Targets initiative",
        "min_categories": 1,
        "required_if_material": True,
        "version": "SBTi v5.1 (2024)",
    },
    ComplianceFramework.SB_253.value: {
        "name": "California SB 253",
        "min_categories": 15,
        "required_if_material": True,
        "version": "2023",
    },
    ComplianceFramework.SEC_CLIMATE.value: {
        "name": "SEC Climate Disclosure Rule",
        "min_categories": 1,
        "required_if_material": False,
        "version": "2024",
    },
    ComplianceFramework.EU_TAXONOMY.value: {
        "name": "EU Taxonomy Regulation",
        "min_categories": 1,
        "required_if_material": False,
        "version": "2020/852",
    },
}


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def confidence_to_level(confidence: Decimal) -> ConfidenceLevel:
    """Convert a numeric confidence score to a ConfidenceLevel enum.

    Args:
        confidence: Decimal value between 0 and 1.

    Returns:
        ConfidenceLevel corresponding to the numeric value.

    Example:
        >>> confidence_to_level(Decimal("0.92"))
        <ConfidenceLevel.VERY_HIGH: 'very_high'>
    """
    if confidence >= Decimal("0.90"):
        return ConfidenceLevel.VERY_HIGH
    if confidence >= Decimal("0.75"):
        return ConfidenceLevel.HIGH
    if confidence >= Decimal("0.50"):
        return ConfidenceLevel.MEDIUM
    if confidence >= Decimal("0.25"):
        return ConfidenceLevel.LOW
    return ConfidenceLevel.VERY_LOW


def category_number(category: Scope3Category) -> int:
    """Extract the integer category number (1-15) from a Scope3Category enum.

    Args:
        category: A Scope3Category enum member.

    Returns:
        Integer 1-15.

    Example:
        >>> category_number(Scope3Category.CAT_6_BUSINESS_TRAVEL)
        6
    """
    return int(category.value.split("_")[0])


def category_name(category: Scope3Category) -> str:
    """Get the human-readable name for a Scope3Category.

    Args:
        category: A Scope3Category enum member.

    Returns:
        Human-readable category name string.

    Example:
        >>> category_name(Scope3Category.CAT_6_BUSINESS_TRAVEL)
        'Business Travel'
    """
    metadata = SCOPE3_CATEGORY_METADATA.get(category.value, {})
    return metadata.get("name", category.value)


# ==============================================================================
# PYDANTIC MODELS -- INPUT RECORD TYPES (10)
# ==============================================================================


class SpendRecord(GreenLangBase):
    """Spend-based record for classification.

    Represents a financial transaction from the general ledger, accounts
    payable, or procurement system. Spend records are the most common
    input type and default to Category 1 unless further classified.

    Example:
        >>> record = SpendRecord(
        ...     amount=Decimal("15000.00"),
        ...     currency=CurrencyCode.USD,
        ...     description="Office furniture",
        ...     supplier_name="OfficeMax",
        ...     transaction_date=date(2025, 3, 15),
        ... )
    """

    amount: Decimal = Field(
        ..., gt=0,
        description="Transaction amount in the specified currency",
    )
    currency: CurrencyCode = Field(
        default=CurrencyCode.USD,
        description="ISO 4217 currency code",
    )
    gl_account: Optional[str] = Field(
        default=None, max_length=50,
        description="General ledger account code",
    )
    naics_code: Optional[str] = Field(
        default=None, max_length=6,
        description="NAICS industry code (2-6 digits)",
    )
    isic_code: Optional[str] = Field(
        default=None, max_length=4,
        description="ISIC industry code (1-4 characters)",
    )
    unspsc_code: Optional[str] = Field(
        default=None, max_length=8,
        description="UNSPSC commodity code (2-8 digits)",
    )
    description: Optional[str] = Field(
        default=None, max_length=1000,
        description="Transaction description for keyword classification",
    )
    supplier_name: Optional[str] = Field(
        default=None, max_length=500,
        description="Supplier or vendor name",
    )
    transaction_date: Optional[date] = Field(
        default=None,
        description="Date of the transaction",
    )

    model_config = ConfigDict(frozen=True)

    @validator("naics_code")
    def validate_naics_code(cls, v: Optional[str]) -> Optional[str]:
        """Validate NAICS code is 2-6 digits."""
        if v is not None:
            if not v.isdigit() or len(v) < 2 or len(v) > 6:
                raise ValueError(
                    f"NAICS code must be 2-6 digits, got '{v}'"
                )
        return v

    @validator("isic_code")
    def validate_isic_code(cls, v: Optional[str]) -> Optional[str]:
        """Validate ISIC code is 1-4 alphanumeric characters."""
        if v is not None and (len(v) < 1 or len(v) > 4):
            raise ValueError(
                f"ISIC code must be 1-4 characters, got '{v}'"
            )
        return v

    @validator("unspsc_code")
    def validate_unspsc_code(cls, v: Optional[str]) -> Optional[str]:
        """Validate UNSPSC code is 2-8 digits."""
        if v is not None:
            if not v.isdigit() or len(v) < 2 or len(v) > 8:
                raise ValueError(
                    f"UNSPSC code must be 2-8 digits, got '{v}'"
                )
        return v


class PurchaseOrderRecord(GreenLangBase):
    """Purchase order record for classification.

    Contains line-item detail enabling more precise category determination
    than aggregated spend data.

    Example:
        >>> record = PurchaseOrderRecord(
        ...     po_number="PO-2025-001234",
        ...     amount=Decimal("50000.00"),
        ...     currency=CurrencyCode.USD,
        ...     supplier_name="Industrial Parts Co",
        ...     category="Raw Materials",
        ... )
    """

    po_number: str = Field(
        ..., min_length=1, max_length=100,
        description="Purchase order number",
    )
    amount: Decimal = Field(
        ..., gt=0,
        description="PO total amount",
    )
    currency: CurrencyCode = Field(
        default=CurrencyCode.USD,
        description="ISO 4217 currency code",
    )
    line_items: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="PO line item details",
    )
    supplier_name: Optional[str] = Field(
        default=None, max_length=500,
        description="Supplier name",
    )
    supplier_naics: Optional[str] = Field(
        default=None, max_length=6,
        description="Supplier NAICS code",
    )
    category: Optional[str] = Field(
        default=None, max_length=200,
        description="Procurement category",
    )
    incoterm: Optional[IncotermsRule] = Field(
        default=None,
        description="Incoterms 2020 rule for delivery",
    )

    model_config = ConfigDict(frozen=True)


class BOMRecord(GreenLangBase):
    """Bill of materials record for classification.

    BOM records represent individual material inputs to production and
    typically map to Category 1 (purchased goods).

    Example:
        >>> record = BOMRecord(
        ...     item_code="STEEL-PLATE-10MM",
        ...     description="Carbon steel plate 10mm",
        ...     quantity=Decimal("500"),
        ...     unit="kg",
        ...     material_type="steel",
        ...     weight_kg=Decimal("500.0"),
        ... )
    """

    item_code: str = Field(
        ..., min_length=1, max_length=100,
        description="Material or part code",
    )
    description: Optional[str] = Field(
        default=None, max_length=1000,
        description="Material description",
    )
    quantity: Decimal = Field(
        ..., gt=0,
        description="Quantity of the material",
    )
    unit: str = Field(
        ..., min_length=1, max_length=50,
        description="Unit of measure",
    )
    material_type: Optional[str] = Field(
        default=None, max_length=200,
        description="Material type classification",
    )
    weight_kg: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Weight in kilograms",
    )
    supplier_naics: Optional[str] = Field(
        default=None, max_length=6,
        description="Supplier NAICS code",
    )

    model_config = ConfigDict(frozen=True)


class TravelRecord(GreenLangBase):
    """Business travel record for classification.

    Travel records map to Category 6 (business travel) and may include
    air, rail, car rental, taxi, bus, ferry, and hotel stays.

    Example:
        >>> record = TravelRecord(
        ...     travel_type="air",
        ...     origin="SFO",
        ...     destination="JFK",
        ...     distance_km=Decimal("4150"),
        ...     mode="air",
        ...     class_of_service="economy",
        ... )
    """

    travel_type: str = Field(
        ..., min_length=1, max_length=50,
        description="Travel type (air, rail, car, taxi, bus, ferry)",
    )
    origin: Optional[str] = Field(
        default=None, max_length=200,
        description="Origin city or airport code",
    )
    destination: Optional[str] = Field(
        default=None, max_length=200,
        description="Destination city or airport code",
    )
    distance_km: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Travel distance in kilometers",
    )
    mode: Optional[str] = Field(
        default=None, max_length=50,
        description="Transportation mode",
    )
    class_of_service: Optional[str] = Field(
        default=None, max_length=50,
        description="Class of service (economy, business, first)",
    )
    hotel_nights: Optional[int] = Field(
        default=None, ge=0,
        description="Number of hotel nights",
    )

    model_config = ConfigDict(frozen=True)


class WasteRecord(GreenLangBase):
    """Waste generation record for classification.

    Waste records map to Category 5 (waste generated in operations).

    Example:
        >>> record = WasteRecord(
        ...     waste_type="mixed_municipal",
        ...     quantity_kg=Decimal("1500.0"),
        ...     treatment_method="landfill",
        ...     hazardous=False,
        ... )
    """

    waste_type: str = Field(
        ..., min_length=1, max_length=200,
        description="Waste type classification",
    )
    quantity_kg: Decimal = Field(
        ..., gt=0,
        description="Waste quantity in kilograms",
    )
    treatment_method: Optional[str] = Field(
        default=None, max_length=100,
        description="Waste treatment method (landfill, incineration, recycling, etc.)",
    )
    hazardous: bool = Field(
        default=False,
        description="Whether the waste is classified as hazardous",
    )

    model_config = ConfigDict(frozen=True)


class LeaseRecord(GreenLangBase):
    """Lease record for classification.

    Lease records map to Category 8 (upstream leased assets) or
    Category 13 (downstream leased assets) depending on whether the
    reporting company is lessee or lessor.

    Example:
        >>> record = LeaseRecord(
        ...     asset_type="office_building",
        ...     lease_classification=LeaseClassification.OPERATING_LEASE,
        ...     annual_value=Decimal("120000.00"),
        ...     area_sqm=Decimal("500.0"),
        ... )
    """

    asset_type: str = Field(
        ..., min_length=1, max_length=200,
        description="Leased asset type (building, vehicle, equipment, IT)",
    )
    lease_classification: LeaseClassification = Field(
        ...,
        description="Lease classification per IFRS 16 / ASC 842",
    )
    annual_value: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Annual lease value",
    )
    area_sqm: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Leased area in square meters (for buildings)",
    )
    energy_use_kwh: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Annual energy use in kWh",
    )

    model_config = ConfigDict(frozen=True)


class LogisticsRecord(GreenLangBase):
    """Logistics/transportation record for classification.

    Maps to Category 4 (upstream transportation) or Category 9
    (downstream transportation) based on Incoterm and direction.

    Example:
        >>> record = LogisticsRecord(
        ...     shipment_id="SHP-2025-001234",
        ...     weight_kg=Decimal("5000.0"),
        ...     distance_km=Decimal("1200.0"),
        ...     mode="road",
        ...     incoterm=IncotermsRule.FOB,
        ...     direction=ValueChainPosition.UPSTREAM,
        ... )
    """

    shipment_id: Optional[str] = Field(
        default=None, max_length=100,
        description="Shipment identifier",
    )
    weight_kg: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Shipment weight in kilograms",
    )
    distance_km: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Transportation distance in kilometers",
    )
    mode: Optional[str] = Field(
        default=None, max_length=50,
        description="Transport mode (road, rail, sea, air, inland_waterway)",
    )
    incoterm: Optional[IncotermsRule] = Field(
        default=None,
        description="Incoterms 2020 delivery rule",
    )
    direction: Optional[ValueChainPosition] = Field(
        default=None,
        description="Direction: upstream (inbound) or downstream (outbound)",
    )

    model_config = ConfigDict(frozen=True)


class EnergyRecord(GreenLangBase):
    """Energy purchase record for classification.

    Maps to Category 3 (fuel- and energy-related activities) for WTT,
    upstream, and T&D loss components not covered by Scope 2.

    Example:
        >>> record = EnergyRecord(
        ...     energy_type="electricity",
        ...     quantity=Decimal("500000"),
        ...     unit="kWh",
        ...     grid_region="US-CAMX",
        ... )
    """

    energy_type: str = Field(
        ..., min_length=1, max_length=100,
        description="Energy type (electricity, natural_gas, diesel, etc.)",
    )
    quantity: Decimal = Field(
        ..., gt=0,
        description="Quantity of energy purchased",
    )
    unit: str = Field(
        ..., min_length=1, max_length=50,
        description="Unit of measure (kWh, MWh, therms, GJ, etc.)",
    )
    grid_region: Optional[str] = Field(
        default=None, max_length=50,
        description="Grid region code (e.g., eGRID subregion)",
    )

    model_config = ConfigDict(frozen=True)


class InvestmentRecord(GreenLangBase):
    """Investment holding record for classification.

    Maps to Category 15 (investments), primarily relevant for financial
    institutions.

    Example:
        >>> record = InvestmentRecord(
        ...     asset_class="listed_equity",
        ...     holding_value=Decimal("10000000.00"),
        ...     investee_name="Acme Corp",
        ... )
    """

    asset_class: str = Field(
        ..., min_length=1, max_length=100,
        description="PCAF asset class (listed_equity, corporate_bond, etc.)",
    )
    holding_value: Decimal = Field(
        ..., gt=0,
        description="Outstanding investment value",
    )
    investee_name: Optional[str] = Field(
        default=None, max_length=500,
        description="Investee company or entity name",
    )

    model_config = ConfigDict(frozen=True)


class FranchiseRecord(GreenLangBase):
    """Franchise record for classification.

    Maps to Category 14 (franchises) for franchisor reporting.

    Example:
        >>> record = FranchiseRecord(
        ...     franchise_type="quick_service_restaurant",
        ...     unit_count=150,
        ...     revenue=Decimal("45000000.00"),
        ... )
    """

    franchise_type: str = Field(
        ..., min_length=1, max_length=200,
        description="Franchise type/format",
    )
    unit_count: int = Field(
        ..., ge=1,
        description="Number of franchise units",
    )
    revenue: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Total franchise revenue",
    )

    model_config = ConfigDict(frozen=True)


# ==============================================================================
# PYDANTIC MODELS -- CLASSIFICATION INPUT/OUTPUT (4)
# ==============================================================================


class ClassificationInput(GreenLangBase):
    """Input for classifying a single record to a Scope 3 category.

    The ``record`` field is a generic dictionary to accommodate diverse
    source types. The mapper determines the correct Scope 3 category
    based on source_type, industry codes, GL accounts, and keywords.

    Example:
        >>> inp = ClassificationInput(
        ...     record={"amount": "1500.00", "description": "Office supplies"},
        ...     source_type=DataSourceType.SPEND_DATA,
        ...     organization_id="org-001",
        ...     reporting_year=2025,
        ... )
    """

    record: Dict[str, Any] = Field(
        ...,
        description="Source record data as a dictionary",
    )
    source_type: DataSourceType = Field(
        ...,
        description="Type of data source for classification routing",
    )
    organization_id: str = Field(
        ..., min_length=1, max_length=200,
        description="Organization identifier",
    )
    reporting_year: int = Field(
        ..., ge=2015, le=2030,
        description="GHG reporting year",
    )

    model_config = ConfigDict(frozen=True)


class BatchClassificationInput(GreenLangBase):
    """Input for batch classification of multiple records.

    Supports up to max_batch_size records (default 50,000) in a single
    batch for high-throughput classification.

    Example:
        >>> batch = BatchClassificationInput(
        ...     records=[{"amount": "100.00"}, {"amount": "200.00"}],
        ...     source_type=DataSourceType.SPEND_DATA,
        ...     organization_id="org-001",
        ...     reporting_year=2025,
        ... )
    """

    records: List[Dict[str, Any]] = Field(
        ..., min_length=1,
        description="List of source records to classify",
    )
    source_type: DataSourceType = Field(
        ...,
        description="Type of data source for all records in the batch",
    )
    organization_id: str = Field(
        ..., min_length=1, max_length=200,
        description="Organization identifier",
    )
    reporting_year: int = Field(
        ..., ge=2015, le=2030,
        description="GHG reporting year",
    )
    max_batch_size: int = Field(
        default=50000, ge=1, le=500000,
        description="Maximum number of records in a batch",
    )

    model_config = ConfigDict(frozen=True)

    @validator("records")
    def validate_batch_size(cls, v: List[Dict[str, Any]], values: dict) -> List[Dict[str, Any]]:
        """Validate batch does not exceed max_batch_size."""
        max_size = values.get("max_batch_size", 50000)
        if len(v) > max_size:
            raise ValueError(
                f"Batch size {len(v)} exceeds max_batch_size {max_size}"
            )
        return v


class ClassificationResult(GreenLangBase):
    """Result of classifying a single record to a Scope 3 category.

    Contains the mapped category, confidence score, classification method,
    recommended calculation approach, and provenance hash for audit trail.

    Example:
        >>> result = ClassificationResult(
        ...     source_type=DataSourceType.SPEND_DATA,
        ...     source_id="txn-001",
        ...     mapped_category=Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES,
        ...     category_number=1,
        ...     category_name="Purchased Goods and Services",
        ...     confidence=Decimal("0.95"),
        ...     confidence_level=ConfidenceLevel.VERY_HIGH,
        ...     classification_method=ClassificationMethod.NAICS,
        ...     mapping_rule="NAICS 322211 -> Cat 1",
        ...     recommended_approach=CalculationApproach.SPEND_BASED,
        ...     value_chain_position=ValueChainPosition.UPSTREAM,
        ...     calculation_trace=["source_type=spend_data", "naics=322211"],
        ...     provenance_hash="abc123def456",
        ... )
    """

    source_type: DataSourceType = Field(
        ...,
        description="Original data source type",
    )
    source_id: Optional[str] = Field(
        default=None, max_length=200,
        description="Unique identifier of the source record",
    )
    mapped_category: Scope3Category = Field(
        ...,
        description="Determined Scope 3 category",
    )
    category_number: int = Field(
        ..., ge=1, le=15,
        description="Category number (1-15)",
    )
    category_name: str = Field(
        ..., min_length=1,
        description="Human-readable category name",
    )
    confidence: Decimal = Field(
        ..., ge=0, le=1,
        description="Classification confidence score (0.0-1.0)",
    )
    confidence_level: ConfidenceLevel = Field(
        ...,
        description="Confidence level bucket",
    )
    classification_method: ClassificationMethod = Field(
        ...,
        description="Method used for classification",
    )
    mapping_rule: str = Field(
        ..., min_length=1,
        description="Specific rule or code that determined the mapping",
    )
    recommended_approach: CalculationApproach = Field(
        ...,
        description="Recommended calculation approach for the target agent",
    )
    value_chain_position: ValueChainPosition = Field(
        ...,
        description="Upstream or downstream value chain position",
    )
    calculation_trace: List[str] = Field(
        default_factory=list,
        description="Step-by-step classification trace for auditing",
    )
    provenance_hash: str = Field(
        ..., min_length=1,
        description="SHA-256 provenance hash",
    )
    double_counting_flags: List[str] = Field(
        default_factory=list,
        description="List of triggered double-counting rule IDs",
    )
    mapping_status: MappingStatus = Field(
        default=MappingStatus.MAPPED,
        description="Status of the mapping",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Classification timestamp",
    )

    model_config = ConfigDict(frozen=True)


class BatchClassificationResult(GreenLangBase):
    """Result of batch classification of multiple records.

    Contains aggregated statistics, per-category summary, and the full
    list of individual classification results.

    Example:
        >>> batch_result = BatchClassificationResult(
        ...     success=True,
        ...     results=[],
        ...     summary_by_category={},
        ...     total_records=0,
        ...     mapped_count=0,
        ...     unmapped_count=0,
        ...     average_confidence=Decimal("0.0"),
        ...     processing_time_ms=Decimal("0.0"),
        ...     provenance_hash="abc123def456",
        ... )
    """

    success: bool = Field(
        ...,
        description="Whether the batch was processed successfully",
    )
    results: List[ClassificationResult] = Field(
        default_factory=list,
        description="Individual classification results",
    )
    summary_by_category: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of records per Scope 3 category",
    )
    total_records: int = Field(
        ..., ge=0,
        description="Total number of records processed",
    )
    mapped_count: int = Field(
        ..., ge=0,
        description="Number of successfully mapped records",
    )
    unmapped_count: int = Field(
        ..., ge=0,
        description="Number of records that could not be mapped",
    )
    average_confidence: Decimal = Field(
        ..., ge=0, le=1,
        description="Average confidence score across all mapped records",
    )
    processing_time_ms: Decimal = Field(
        ..., ge=0,
        description="Total batch processing time in milliseconds",
    )
    provenance_hash: str = Field(
        ..., min_length=1,
        description="SHA-256 provenance hash for the batch",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Batch completion timestamp",
    )

    model_config = ConfigDict(frozen=True)


# ==============================================================================
# PYDANTIC MODELS -- ROUTING (2)
# ==============================================================================


class RoutingInstruction(GreenLangBase):
    """Instruction for routing a classified record to a target agent.

    After classification, the mapper generates routing instructions
    that direct records to the appropriate category-specific agent API.

    Example:
        >>> instruction = RoutingInstruction(
        ...     category=Scope3Category.CAT_6_BUSINESS_TRAVEL,
        ...     target_agent_id="GL-MRV-S3-006",
        ...     target_api_endpoint="/api/v1/business-travel",
        ...     transformed_input={"travel_type": "air", "distance_km": 4150},
        ... )
    """

    category: Scope3Category = Field(
        ...,
        description="Target Scope 3 category",
    )
    target_agent_id: str = Field(
        ..., min_length=1,
        description="Target agent identifier (e.g., GL-MRV-S3-006)",
    )
    target_api_endpoint: str = Field(
        ..., min_length=1,
        description="Target agent API endpoint",
    )
    transformed_input: Dict[str, Any] = Field(
        default_factory=dict,
        description="Input data transformed for the target agent schema",
    )
    batch_size: int = Field(
        default=1, ge=1,
        description="Number of records in this routing instruction",
    )

    model_config = ConfigDict(frozen=True)


class RoutingPlan(GreenLangBase):
    """Complete routing plan for a classified batch.

    Aggregates all routing instructions and provides a summary of
    the categories targeted and total record counts.

    Example:
        >>> plan = RoutingPlan(
        ...     instructions=[],
        ...     total_records=100,
        ...     categories_targeted=5,
        ...     dry_run=False,
        ...     provenance_hash="abc123def456",
        ... )
    """

    instructions: List[RoutingInstruction] = Field(
        default_factory=list,
        description="List of routing instructions per category",
    )
    total_records: int = Field(
        ..., ge=0,
        description="Total records in the routing plan",
    )
    categories_targeted: int = Field(
        ..., ge=0, le=15,
        description="Number of distinct categories targeted",
    )
    dry_run: bool = Field(
        default=False,
        description="If True, plan is generated but not executed",
    )
    provenance_hash: str = Field(
        ..., min_length=1,
        description="SHA-256 provenance hash for the routing plan",
    )

    model_config = ConfigDict(frozen=True)


# ==============================================================================
# PYDANTIC MODELS -- BOUNDARY DETERMINATION (1)
# ==============================================================================


class BoundaryDetermination(GreenLangBase):
    """Boundary determination for a Scope 3 category.

    Determines whether a data record falls within a specific category's
    boundary based on consolidation approach, Incoterms, lease type, etc.

    Example:
        >>> boundary = BoundaryDetermination(
        ...     category=Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION,
        ...     value_chain_position=ValueChainPosition.UPSTREAM,
        ...     consolidation_approach=ConsolidationApproach.OPERATIONAL_CONTROL,
        ...     determination_rule="Incoterm FOB -> buyer pays freight",
        ...     confidence=Decimal("0.90"),
        ... )
    """

    category: Scope3Category = Field(
        ...,
        description="Scope 3 category for this boundary determination",
    )
    value_chain_position: ValueChainPosition = Field(
        ...,
        description="Upstream or downstream position",
    )
    consolidation_approach: ConsolidationApproach = Field(
        ...,
        description="Organizational boundary approach",
    )
    determination_rule: str = Field(
        ..., min_length=1,
        description="Rule or logic used for boundary determination",
    )
    confidence: Decimal = Field(
        ..., ge=0, le=1,
        description="Confidence in the boundary determination",
    )
    notes: Optional[str] = Field(
        default=None, max_length=2000,
        description="Additional notes or rationale",
    )

    model_config = ConfigDict(frozen=True)


# ==============================================================================
# PYDANTIC MODELS -- COMPLETENESS (2)
# ==============================================================================


class CategoryCompletenessEntry(GreenLangBase):
    """Completeness assessment for a single Scope 3 category.

    Evaluates whether a given category has data available, its quality
    tier, estimated materiality, and screening status.

    Example:
        >>> entry = CategoryCompletenessEntry(
        ...     category=Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES,
        ...     relevance=CategoryRelevance.MATERIAL,
        ...     data_available=True,
        ...     data_quality_tier=DataQualityTier.TIER_3,
        ...     estimated_materiality_pct=Decimal("45.0"),
        ...     screening_result=ScreeningResult.COMPLETE,
        ...     recommended_action="Improve data quality to Tier 2",
        ... )
    """

    category: Scope3Category = Field(
        ...,
        description="Scope 3 category being assessed",
    )
    relevance: CategoryRelevance = Field(
        ...,
        description="Relevance assessment for this category",
    )
    data_available: bool = Field(
        ...,
        description="Whether data is available for this category",
    )
    data_quality_tier: Optional[DataQualityTier] = Field(
        default=None,
        description="Data quality tier if data is available",
    )
    estimated_materiality_pct: Optional[Decimal] = Field(
        default=None, ge=0, le=100,
        description="Estimated percentage of total Scope 3 emissions",
    )
    screening_result: ScreeningResult = Field(
        ...,
        description="Screening completeness result",
    )
    recommended_action: Optional[str] = Field(
        default=None, max_length=1000,
        description="Recommended next step for this category",
    )

    model_config = ConfigDict(frozen=True)


class CompletenessReport(GreenLangBase):
    """Full completeness report across all 15 Scope 3 categories.

    Provides an overall completeness score and identifies gaps where
    material categories lack data coverage.

    Example:
        >>> report = CompletenessReport(
        ...     company_type=CompanyType.MANUFACTURER,
        ...     entries=[],
        ...     overall_score=Decimal("72.5"),
        ...     categories_reported=8,
        ...     categories_material=5,
        ...     gaps=["Cat 10 -- no processing data"],
        ...     provenance_hash="abc123def456",
        ... )
    """

    company_type: CompanyType = Field(
        ...,
        description="Company type for industry-specific relevance",
    )
    entries: List[CategoryCompletenessEntry] = Field(
        default_factory=list,
        description="Per-category completeness entries",
    )
    overall_score: Decimal = Field(
        ..., ge=0, le=100,
        description="Overall completeness score (0-100)",
    )
    categories_reported: int = Field(
        ..., ge=0, le=15,
        description="Number of categories with data reported",
    )
    categories_material: int = Field(
        ..., ge=0, le=15,
        description="Number of categories assessed as material",
    )
    gaps: List[str] = Field(
        default_factory=list,
        description="List of identified gaps (material categories without data)",
    )
    provenance_hash: str = Field(
        ..., min_length=1,
        description="SHA-256 provenance hash",
    )

    model_config = ConfigDict(frozen=True)


# ==============================================================================
# PYDANTIC MODELS -- DOUBLE-COUNTING (1)
# ==============================================================================


class DoubleCountingCheck(GreenLangBase):
    """Result of a double-counting prevention check.

    Checks for overlap between two categories or between Scope 3 and
    Scope 1/2, and provides a resolution action.

    Example:
        >>> check = DoubleCountingCheck(
        ...     rule_id=DoubleCountingRule.DC_SCM_001,
        ...     category_a=Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES,
        ...     category_b=Scope3Category.CAT_2_CAPITAL_GOODS,
        ...     overlap_detected=True,
        ...     resolution="Reclassify capitalized item from Cat 1 to Cat 2",
        ...     affected_records=3,
        ...     provenance_hash="abc123def456",
        ... )
    """

    rule_id: DoubleCountingRule = Field(
        ...,
        description="Double-counting rule identifier",
    )
    category_a: Scope3Category = Field(
        ...,
        description="First category in the overlap check",
    )
    category_b: Scope3Category = Field(
        ...,
        description="Second category (or Scope 1/2 proxy)",
    )
    overlap_detected: bool = Field(
        ...,
        description="Whether an overlap was detected",
    )
    resolution: str = Field(
        ..., min_length=1,
        description="Resolution action taken or recommended",
    )
    affected_records: int = Field(
        ..., ge=0,
        description="Number of records affected by this overlap",
    )
    provenance_hash: str = Field(
        ..., min_length=1,
        description="SHA-256 provenance hash",
    )

    model_config = ConfigDict(frozen=True)


# ==============================================================================
# PYDANTIC MODELS -- COMPLIANCE (1)
# ==============================================================================


class ComplianceAssessment(GreenLangBase):
    """Compliance assessment against a regulatory framework.

    Evaluates whether the organization's Scope 3 category coverage
    meets the requirements of a specific reporting framework.

    Example:
        >>> assessment = ComplianceAssessment(
        ...     framework=ComplianceFramework.GHG_PROTOCOL,
        ...     categories_required=[
        ...         Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES,
        ...     ],
        ...     categories_reported=[
        ...         Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES,
        ...     ],
        ...     compliant=True,
        ...     gaps=[],
        ...     recommendations=[],
        ...     score=Decimal("100.0"),
        ... )
    """

    framework: ComplianceFramework = Field(
        ...,
        description="Compliance framework being assessed",
    )
    categories_required: List[Scope3Category] = Field(
        default_factory=list,
        description="Categories required by the framework",
    )
    categories_reported: List[Scope3Category] = Field(
        default_factory=list,
        description="Categories with data reported",
    )
    compliant: bool = Field(
        ...,
        description="Whether the organization meets compliance requirements",
    )
    gaps: List[str] = Field(
        default_factory=list,
        description="Compliance gaps identified",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations to achieve compliance",
    )
    score: Decimal = Field(
        ..., ge=0, le=100,
        description="Compliance score (0-100)",
    )

    model_config = ConfigDict(frozen=True)


# ==============================================================================
# ADDITIONAL ENUMERATIONS (Engine 5 & 6)
# ==============================================================================


class ComplianceStatus(str, Enum):
    """Compliance check status for framework assessments."""

    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ComplianceSeverity(str, Enum):
    """Severity level for compliance findings."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


# ==============================================================================
# ADDITIONAL PYDANTIC MODELS (Engine 5 & 6)
# ==============================================================================


class ComplianceFinding(GreenLangBase):
    """Single compliance finding with rule code and severity.

    Represents one discrete compliance check result, typically a failure
    or warning, with context about which framework and regulation it
    relates to, plus a recommendation for remediation.

    Example:
        >>> finding = ComplianceFinding(
        ...     rule_code="GHG-SCR-001",
        ...     description="Not all 15 categories screened",
        ...     severity=ComplianceSeverity.CRITICAL,
        ...     framework="ghg_protocol",
        ...     status=ComplianceStatus.FAIL,
        ...     recommendation="Complete screening of all 15 categories",
        ... )
    """

    rule_code: str = Field(
        ..., description="Compliance rule code (e.g., GHG-SCR-001)"
    )
    description: str = Field(
        ..., description="Description of the finding"
    )
    severity: ComplianceSeverity = Field(
        ..., description="Severity level"
    )
    framework: str = Field(
        ..., description="Framework that produced this finding"
    )
    status: ComplianceStatus = Field(
        default=ComplianceStatus.FAIL, description="Finding status"
    )
    recommendation: Optional[str] = Field(
        None, description="Recommended corrective action"
    )
    regulation_reference: Optional[str] = Field(
        None, description="Regulatory reference (e.g., chapter/clause)"
    )
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional details"
    )

    model_config = ConfigDict(frozen=True)


class DetailedComplianceAssessment(GreenLangBase):
    """Detailed compliance assessment with findings and check counts.

    Extends the base ComplianceAssessment with granular findings,
    individual check pass/fail/warning counts, and provenance tracking.
    Used by ComplianceCheckerEngine (Engine 6) for framework-level
    compliance validation.

    Example:
        >>> assessment = DetailedComplianceAssessment(
        ...     framework=ComplianceFramework.GHG_PROTOCOL,
        ...     framework_description="GHG Protocol Scope 3 Standard",
        ...     status=ComplianceStatus.PASS,
        ...     score=Decimal("95.00"),
        ...     passed_checks=10,
        ...     total_checks=10,
        ...     provenance_hash="abc123...",
        ...     assessed_at="2026-03-01T00:00:00Z",
        ... )
    """

    framework: ComplianceFramework = Field(
        ..., description="Compliance framework assessed"
    )
    framework_description: str = Field(
        default="", description="Human-readable framework name"
    )
    status: ComplianceStatus = Field(
        ..., description="Overall compliance status"
    )
    score: Decimal = Field(
        ..., ge=0, le=100, description="Compliance score (0-100)"
    )
    findings: List[ComplianceFinding] = Field(
        default_factory=list, description="All findings"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Improvement recommendations"
    )
    passed_checks: int = Field(
        default=0, ge=0, description="Number of passed checks"
    )
    failed_checks: int = Field(
        default=0, ge=0, description="Number of failed checks"
    )
    warning_checks: int = Field(
        default=0, ge=0, description="Number of warning checks"
    )
    total_checks: int = Field(
        default=0, ge=0, description="Total checks performed"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )
    assessed_at: str = Field(
        default="", description="ISO 8601 timestamp of assessment"
    )
    processing_time_ms: float = Field(
        default=0.0, description="Processing time in milliseconds"
    )

    model_config = ConfigDict(frozen=True)


class BenchmarkComparison(GreenLangBase):
    """Result of comparing actual emissions distribution against benchmarks.

    Compares the actual reported percentage for a category against the
    industry benchmark percentage, computing deviation and flagging
    significant outliers.

    Example:
        >>> comparison = BenchmarkComparison(
        ...     category=Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES,
        ...     benchmark_pct=Decimal("60.00"),
        ...     actual_pct=Decimal("45.00"),
        ...     deviation_pct=Decimal("-15.00"),
        ...     within_tolerance=False,
        ...     flag="Below benchmark",
        ... )
    """

    category: Scope3Category = Field(
        ..., description="Scope 3 category"
    )
    benchmark_pct: Decimal = Field(
        ..., ge=0, le=100,
        description="Industry benchmark percentage"
    )
    actual_pct: Decimal = Field(
        ..., ge=0, le=100,
        description="Actual reported percentage"
    )
    deviation_pct: Decimal = Field(
        ..., description="Deviation from benchmark (actual - benchmark)"
    )
    within_tolerance: bool = Field(
        ..., description="Whether deviation is within acceptable tolerance"
    )
    flag: Optional[str] = Field(
        None, description="Warning flag if deviation is significant"
    )

    model_config = ConfigDict(frozen=True)


# ==============================================================================
# CONVENIENCE CONSTANTS
# ==============================================================================

ALL_SCOPE3_CATEGORIES: List[Scope3Category] = list(Scope3Category)

SCOPE3_CATEGORY_NAMES: Dict[Scope3Category, str] = {
    cat: SCOPE3_CATEGORY_METADATA[cat.value]["name"]
    for cat in Scope3Category
}

SCOPE3_CATEGORY_NUMBERS: Dict[Scope3Category, int] = {
    cat: int(SCOPE3_CATEGORY_METADATA[cat.value]["number"])
    for cat in Scope3Category
}
