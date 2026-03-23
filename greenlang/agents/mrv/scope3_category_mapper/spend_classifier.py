# -*- coding: utf-8 -*-
"""
SpendClassifierEngine - Deterministic spend classification for Scope 3 categories.

This module implements the SpendClassifierEngine for AGENT-MRV-029
(Scope 3 Category Mapper, GL-MRV-X-040). It classifies organisational records
(spend data, purchase orders, bills of materials, travel records, waste manifests,
lease agreements, logistics data, energy invoices, investments, and franchise
agreements) into the correct GHG Protocol Scope 3 category (1-15) using a
five-level priority hierarchy.

Classification Priority Hierarchy:
    1. Industry code (NAICS/ISIC) -- highest confidence (0.90-0.95)
    2. GL account code -- high confidence (0.85)
    3. Procurement category -- medium confidence (0.70)
    4. Keyword analysis -- lower confidence (0.40-0.60)
    5. Default fallback -- Cat 1 with confidence 0.30

Multi-Category Splitting:
    When a record description contains split indicators ("and", "with",
    "including", "plus"), the engine attempts to classify each segment
    independently and produces split results with allocation ratios.

Batch Processing:
    Supports up to 50,000 records per batch with chunked processing for
    memory efficiency.

Zero-Hallucination Guarantee:
    All classification uses deterministic lookup tables from
    CategoryDatabaseEngine. NO LLM, ML, or probabilistic models are used.
    Every classification includes a SHA-256 provenance hash and confidence
    score that reflects the mapping method quality.

Thread Safety:
    Uses the __new__ singleton pattern with threading.Lock to ensure only
    one instance is created across all threads.

Example:
    >>> engine = SpendClassifierEngine()
    >>> result = engine.classify_spend(SpendRecord(
    ...     description="Office supplies and stationery",
    ...     amount=Decimal("1500.00"),
    ...     gl_account_code="6100",
    ... ))
    >>> result.primary_category
    <Scope3Category.CAT_1: 1>

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-X-040
"""

import hashlib
import json
import logging
import re
import threading
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, validator
from pydantic import ConfigDict

from greenlang.agents.mrv.scope3_category_mapper.category_database import (
    AGENT_ID,
    AGENT_COMPONENT,
    VERSION,
    Scope3Category,
    ValueChainDirection,
    CategoryDatabaseEngine,
    get_category_database_engine,
    NAICSLookupResult,
    ISICLookupResult,
    GLLookupResult,
    KeywordLookupResult,
)

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

_QUANT_8DP = Decimal("0.00000001")
_QUANT_2DP = Decimal("0.01")
_MAX_BATCH_SIZE = 50_000
_DEFAULT_CHUNK_SIZE = 1_000

# Split indicator patterns for multi-category splitting
_SPLIT_PATTERNS = re.compile(
    r"\b(?:and|with|including|plus|&|\+)\b",
    re.IGNORECASE,
)


# =============================================================================
# ENUMERATIONS
# =============================================================================


class DataSourceType(str, Enum):
    """Input data source type for classification routing."""

    SPEND = "spend"                    # GL exports, AP ledger
    PURCHASE_ORDER = "purchase_order"  # PO lines with item descriptions
    BOM = "bom"                        # Bill of materials items
    TRAVEL = "travel"                  # Travel booking / expense reports
    FLEET = "fleet"                    # Vehicle / fleet data
    WASTE = "waste"                    # Waste manifests / disposal records
    LEASE = "lease"                    # Operating / finance leases
    LOGISTICS = "logistics"            # Freight / shipping records
    PRODUCT_SALES = "product_sales"    # Revenue by product
    INVESTMENT = "investment"          # Investment holdings
    FRANCHISE = "franchise"            # Franchise agreements
    ENERGY = "energy"                  # Energy invoices
    SUPPLIER = "supplier"             # Supplier emissions data


class ClassificationMethod(str, Enum):
    """Method used to classify a record into a Scope 3 category."""

    NAICS = "naics"            # NAICS industry code lookup
    ISIC = "isic"              # ISIC code lookup
    GL_ACCOUNT = "gl_account"  # GL account range lookup
    PROCUREMENT = "procurement"  # Procurement category mapping
    KEYWORD = "keyword"        # Keyword-based text analysis
    SOURCE_TYPE = "source_type"  # Inferred from data source type
    DEFAULT = "default"        # Default fallback


class CalculationApproach(str, Enum):
    """Recommended calculation approach per GHG Protocol hierarchy."""

    SUPPLIER_SPECIFIC = "supplier_specific"  # Primary data from supplier
    HYBRID = "hybrid"                        # Mix of primary + secondary
    AVERAGE_DATA = "average_data"            # Industry average emission factors
    SPEND_BASED = "spend_based"              # EEIO spend-based factors


class ConfidenceLevel(str, Enum):
    """Confidence level classification for mapping quality."""

    VERY_HIGH = "very_high"  # >= 0.90
    HIGH = "high"            # >= 0.75
    MEDIUM = "medium"        # >= 0.55
    LOW = "low"              # >= 0.35
    VERY_LOW = "very_low"    # < 0.35


# =============================================================================
# INPUT MODELS
# =============================================================================


class SpendRecord(BaseModel):
    """Spend data record for classification."""

    description: str = Field(..., min_length=1, description="Transaction description")
    amount: Decimal = Field(..., gt=0, description="Spend amount")
    currency: str = Field(default="USD", description="ISO 4217 currency code")
    gl_account_code: Optional[str] = Field(default=None, description="GL account code")
    naics_code: Optional[str] = Field(default=None, description="NAICS code")
    isic_code: Optional[str] = Field(default=None, description="ISIC code")
    vendor_name: Optional[str] = Field(default=None, description="Vendor/supplier name")
    procurement_category: Optional[str] = Field(
        default=None, description="Procurement category"
    )
    cost_center: Optional[str] = Field(default=None, description="Cost center")
    tenant_id: Optional[str] = Field(default=None, description="Tenant ID")

    model_config = ConfigDict(frozen=True)


class PurchaseOrderRecord(BaseModel):
    """Purchase order record for classification."""

    description: str = Field(..., min_length=1, description="PO line description")
    amount: Decimal = Field(..., gt=0, description="PO line amount")
    currency: str = Field(default="USD", description="ISO 4217 currency code")
    naics_code: Optional[str] = Field(default=None, description="Supplier NAICS code")
    isic_code: Optional[str] = Field(default=None, description="Supplier ISIC code")
    gl_account_code: Optional[str] = Field(default=None, description="GL account code")
    item_category: Optional[str] = Field(default=None, description="Item category")
    vendor_name: Optional[str] = Field(default=None, description="Vendor name")
    is_capex: Optional[bool] = Field(default=None, description="Capital expenditure flag")
    tenant_id: Optional[str] = Field(default=None, description="Tenant ID")

    model_config = ConfigDict(frozen=True)


class BOMRecord(BaseModel):
    """Bill of materials record for classification."""

    description: str = Field(..., min_length=1, description="Material description")
    material_type: Optional[str] = Field(default=None, description="Material type/group")
    quantity: Decimal = Field(..., gt=0, description="Quantity")
    unit: str = Field(default="unit", description="Unit of measure")
    unit_cost: Optional[Decimal] = Field(default=None, description="Unit cost")
    naics_code: Optional[str] = Field(default=None, description="Material NAICS code")
    is_intermediate: bool = Field(
        default=False,
        description="Whether the product is sold as intermediate (Cat 10 relevance)"
    )
    tenant_id: Optional[str] = Field(default=None, description="Tenant ID")

    model_config = ConfigDict(frozen=True)


class TravelRecord(BaseModel):
    """Travel record for classification (Cat 6 or Cat 7)."""

    description: str = Field(..., min_length=1, description="Travel description")
    travel_type: str = Field(
        default="business",
        description="Travel type: business, commuting"
    )
    amount: Optional[Decimal] = Field(default=None, description="Travel cost")
    currency: str = Field(default="USD", description="ISO 4217 currency code")
    mode: Optional[str] = Field(
        default=None,
        description="Transport mode: air, rail, car, taxi, bus, ferry"
    )
    tenant_id: Optional[str] = Field(default=None, description="Tenant ID")

    model_config = ConfigDict(frozen=True)


class WasteRecord(BaseModel):
    """Waste manifest record for classification (always Cat 5)."""

    description: str = Field(..., min_length=1, description="Waste description")
    waste_type: Optional[str] = Field(default=None, description="Waste type/category")
    quantity: Decimal = Field(..., gt=0, description="Waste quantity")
    unit: str = Field(default="kg", description="Unit of measure")
    treatment_method: Optional[str] = Field(
        default=None, description="Treatment method (landfill, recycling, etc.)"
    )
    tenant_id: Optional[str] = Field(default=None, description="Tenant ID")

    model_config = ConfigDict(frozen=True)


class LeaseRecord(BaseModel):
    """Lease agreement record for classification (Cat 8 or Cat 13)."""

    description: str = Field(..., min_length=1, description="Lease description")
    asset_type: Optional[str] = Field(
        default=None,
        description="Asset type: building, vehicle, equipment, IT"
    )
    annual_cost: Optional[Decimal] = Field(default=None, description="Annual lease cost")
    currency: str = Field(default="USD", description="ISO 4217 currency code")
    reporter_is_lessee: bool = Field(
        default=True,
        description="True if reporter is lessee (Cat 8), False if lessor (Cat 13)"
    )
    lease_type: Optional[str] = Field(
        default=None,
        description="Lease type: operating, finance, short_term, low_value"
    )
    tenant_id: Optional[str] = Field(default=None, description="Tenant ID")

    model_config = ConfigDict(frozen=True)


class LogisticsRecord(BaseModel):
    """Logistics/freight record for classification (Cat 4 or Cat 9)."""

    description: str = Field(..., min_length=1, description="Shipment description")
    direction: str = Field(
        default="inbound",
        description="Freight direction: inbound (Cat 4), outbound (Cat 9)"
    )
    amount: Optional[Decimal] = Field(default=None, description="Freight cost")
    currency: str = Field(default="USD", description="ISO 4217 currency code")
    mode: Optional[str] = Field(
        default=None,
        description="Transport mode: road, rail, ocean, air, multimodal"
    )
    incoterm: Optional[str] = Field(
        default=None,
        description="Incoterm (EXW, FOB, CIF, DDP, etc.)"
    )
    tenant_id: Optional[str] = Field(default=None, description="Tenant ID")

    model_config = ConfigDict(frozen=True)


class EnergyRecord(BaseModel):
    """Energy invoice record for classification (always Cat 3)."""

    description: str = Field(..., min_length=1, description="Energy description")
    energy_type: Optional[str] = Field(
        default=None,
        description="Energy type: electricity, natural_gas, steam, cooling"
    )
    amount: Optional[Decimal] = Field(default=None, description="Invoice amount")
    currency: str = Field(default="USD", description="ISO 4217 currency code")
    quantity: Optional[Decimal] = Field(default=None, description="Energy quantity")
    unit: Optional[str] = Field(default=None, description="Unit: kWh, therms, MJ, etc.")
    tenant_id: Optional[str] = Field(default=None, description="Tenant ID")

    model_config = ConfigDict(frozen=True)


class InvestmentRecord(BaseModel):
    """Investment record for classification (always Cat 15)."""

    description: str = Field(..., min_length=1, description="Investment description")
    asset_class: Optional[str] = Field(
        default=None,
        description=(
            "Asset class: listed_equity, corporate_bond, private_equity, "
            "project_finance, cre, mortgage, motor_vehicle_loan, sovereign_bond"
        )
    )
    outstanding_amount: Optional[Decimal] = Field(
        default=None, description="Outstanding amount"
    )
    currency: str = Field(default="USD", description="ISO 4217 currency code")
    tenant_id: Optional[str] = Field(default=None, description="Tenant ID")

    model_config = ConfigDict(frozen=True)


class FranchiseRecord(BaseModel):
    """Franchise agreement record for classification (always Cat 14)."""

    description: str = Field(..., min_length=1, description="Franchise description")
    franchise_type: Optional[str] = Field(
        default=None,
        description="Franchise type: qsr, hotel, convenience, retail, etc."
    )
    annual_revenue: Optional[Decimal] = Field(
        default=None, description="Annual franchise revenue"
    )
    currency: str = Field(default="USD", description="ISO 4217 currency code")
    tenant_id: Optional[str] = Field(default=None, description="Tenant ID")

    model_config = ConfigDict(frozen=True)


# =============================================================================
# OUTPUT MODELS
# =============================================================================


class SplitAllocation(BaseModel):
    """Allocation for a single category in a multi-category split."""

    category: Scope3Category = Field(..., description="Scope 3 category")
    allocation_ratio: Decimal = Field(..., description="Allocation ratio (0.0-1.0)")
    description_segment: str = Field(..., description="Text segment for this split")

    model_config = ConfigDict(frozen=True)


class ClassificationResult(BaseModel):
    """Result from classifying a single record."""

    primary_category: Scope3Category = Field(..., description="Primary Scope 3 category")
    secondary_categories: List[Scope3Category] = Field(
        default_factory=list,
        description="Secondary categories that may also apply"
    )
    confidence: Decimal = Field(..., description="Classification confidence (0.0-1.0)")
    confidence_level: ConfidenceLevel = Field(
        ..., description="Human-readable confidence level"
    )
    classification_method: ClassificationMethod = Field(
        ..., description="Method used for classification"
    )
    recommended_approach: CalculationApproach = Field(
        ..., description="Recommended calculation approach"
    )
    is_split: bool = Field(
        default=False,
        description="Whether the record was split across multiple categories"
    )
    split_allocations: List[SplitAllocation] = Field(
        default_factory=list,
        description="Split allocations (populated only if is_split=True)"
    )
    source_type: DataSourceType = Field(..., description="Input data source type")
    description: str = Field(..., description="Classification reasoning")
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    processing_time_ms: Decimal = Field(..., description="Processing time in ms")

    model_config = ConfigDict(frozen=True)


class BatchClassificationResult(BaseModel):
    """Result from batch classification of multiple records."""

    results: List[ClassificationResult] = Field(
        ..., description="Individual classification results"
    )
    total_records: int = Field(..., description="Total records submitted")
    classified_count: int = Field(..., description="Successfully classified count")
    failed_count: int = Field(..., description="Failed classification count")
    category_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of records per category"
    )
    average_confidence: Decimal = Field(..., description="Mean confidence score")
    processing_time_ms: Decimal = Field(..., description="Total processing time in ms")
    errors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Error details for failed records"
    )
    provenance_hash: str = Field(..., description="SHA-256 batch provenance hash")

    model_config = ConfigDict(frozen=True)


# =============================================================================
# PROCUREMENT CATEGORY MAPPING
# =============================================================================

# Maps common procurement category codes/names to Scope 3 categories
_PROCUREMENT_CATEGORY_MAP: Dict[str, Tuple[Scope3Category, Decimal]] = {
    # Direct materials
    "raw_materials": (Scope3Category.CAT_1, Decimal("0.72")),
    "raw materials": (Scope3Category.CAT_1, Decimal("0.72")),
    "direct_materials": (Scope3Category.CAT_1, Decimal("0.72")),
    "direct materials": (Scope3Category.CAT_1, Decimal("0.72")),
    "components": (Scope3Category.CAT_1, Decimal("0.72")),
    "packaging": (Scope3Category.CAT_1, Decimal("0.70")),
    "ingredients": (Scope3Category.CAT_1, Decimal("0.72")),
    # Indirect materials
    "office_supplies": (Scope3Category.CAT_1, Decimal("0.70")),
    "office supplies": (Scope3Category.CAT_1, Decimal("0.70")),
    "it_supplies": (Scope3Category.CAT_1, Decimal("0.70")),
    "it supplies": (Scope3Category.CAT_1, Decimal("0.70")),
    "mro": (Scope3Category.CAT_1, Decimal("0.68")),
    "mro supplies": (Scope3Category.CAT_1, Decimal("0.68")),
    # Services
    "professional_services": (Scope3Category.CAT_1, Decimal("0.70")),
    "professional services": (Scope3Category.CAT_1, Decimal("0.70")),
    "consulting": (Scope3Category.CAT_1, Decimal("0.70")),
    "it_services": (Scope3Category.CAT_1, Decimal("0.70")),
    "it services": (Scope3Category.CAT_1, Decimal("0.70")),
    "marketing": (Scope3Category.CAT_1, Decimal("0.68")),
    "advertising": (Scope3Category.CAT_1, Decimal("0.68")),
    "legal": (Scope3Category.CAT_1, Decimal("0.70")),
    "insurance": (Scope3Category.CAT_1, Decimal("0.68")),
    # Capital
    "capital_equipment": (Scope3Category.CAT_2, Decimal("0.75")),
    "capital equipment": (Scope3Category.CAT_2, Decimal("0.75")),
    "machinery": (Scope3Category.CAT_2, Decimal("0.75")),
    "construction": (Scope3Category.CAT_2, Decimal("0.72")),
    "it_hardware": (Scope3Category.CAT_2, Decimal("0.72")),
    "it hardware": (Scope3Category.CAT_2, Decimal("0.72")),
    "vehicles": (Scope3Category.CAT_2, Decimal("0.72")),
    "furniture_fixtures": (Scope3Category.CAT_2, Decimal("0.70")),
    "furniture and fixtures": (Scope3Category.CAT_2, Decimal("0.70")),
    # Energy
    "electricity": (Scope3Category.CAT_3, Decimal("0.78")),
    "natural_gas": (Scope3Category.CAT_3, Decimal("0.78")),
    "natural gas": (Scope3Category.CAT_3, Decimal("0.78")),
    "utilities": (Scope3Category.CAT_3, Decimal("0.75")),
    "fuel": (Scope3Category.CAT_3, Decimal("0.75")),
    "energy": (Scope3Category.CAT_3, Decimal("0.75")),
    # Transport
    "freight": (Scope3Category.CAT_4, Decimal("0.72")),
    "logistics": (Scope3Category.CAT_4, Decimal("0.72")),
    "shipping": (Scope3Category.CAT_4, Decimal("0.70")),
    "courier": (Scope3Category.CAT_4, Decimal("0.70")),
    "warehousing": (Scope3Category.CAT_4, Decimal("0.68")),
    "distribution": (Scope3Category.CAT_9, Decimal("0.68")),
    # Waste
    "waste_management": (Scope3Category.CAT_5, Decimal("0.75")),
    "waste management": (Scope3Category.CAT_5, Decimal("0.75")),
    "waste_disposal": (Scope3Category.CAT_5, Decimal("0.75")),
    "waste disposal": (Scope3Category.CAT_5, Decimal("0.75")),
    "recycling": (Scope3Category.CAT_5, Decimal("0.72")),
    # Travel
    "travel": (Scope3Category.CAT_6, Decimal("0.72")),
    "business_travel": (Scope3Category.CAT_6, Decimal("0.75")),
    "business travel": (Scope3Category.CAT_6, Decimal("0.75")),
    "accommodation": (Scope3Category.CAT_6, Decimal("0.72")),
    "flights": (Scope3Category.CAT_6, Decimal("0.75")),
    "car_rental": (Scope3Category.CAT_6, Decimal("0.72")),
    "car rental": (Scope3Category.CAT_6, Decimal("0.72")),
    # Commuting
    "commuting": (Scope3Category.CAT_7, Decimal("0.72")),
    "employee_commuting": (Scope3Category.CAT_7, Decimal("0.72")),
    "employee commuting": (Scope3Category.CAT_7, Decimal("0.72")),
    "transit_benefits": (Scope3Category.CAT_7, Decimal("0.70")),
    "transit benefits": (Scope3Category.CAT_7, Decimal("0.70")),
    # Leases
    "rent": (Scope3Category.CAT_8, Decimal("0.72")),
    "lease": (Scope3Category.CAT_8, Decimal("0.72")),
    "leasing": (Scope3Category.CAT_8, Decimal("0.72")),
    "real_estate": (Scope3Category.CAT_8, Decimal("0.70")),
    "real estate": (Scope3Category.CAT_8, Decimal("0.70")),
    # Franchise
    "franchise": (Scope3Category.CAT_14, Decimal("0.75")),
    "franchise_fees": (Scope3Category.CAT_14, Decimal("0.75")),
    "franchise fees": (Scope3Category.CAT_14, Decimal("0.75")),
    "royalties": (Scope3Category.CAT_14, Decimal("0.70")),
    # Investment
    "investments": (Scope3Category.CAT_15, Decimal("0.75")),
    "portfolio": (Scope3Category.CAT_15, Decimal("0.72")),
    "lending": (Scope3Category.CAT_15, Decimal("0.72")),
    "asset_management": (Scope3Category.CAT_15, Decimal("0.70")),
    "asset management": (Scope3Category.CAT_15, Decimal("0.70")),
}


# =============================================================================
# PROVENANCE HELPER
# =============================================================================


def _calculate_hash(*parts: Any) -> str:
    """
    Calculate a SHA-256 provenance hash from variable inputs.

    Args:
        *parts: Variable number of input values to hash.

    Returns:
        Hexadecimal SHA-256 hash string (64 characters).
    """
    hash_input = ""
    for part in parts:
        if isinstance(part, Decimal):
            hash_input += str(part.quantize(_QUANT_8DP, rounding=ROUND_HALF_UP))
        elif isinstance(part, BaseModel):
            hash_input += json.dumps(
                part.model_dump(mode="json"), sort_keys=True, default=str
            )
        elif isinstance(part, (list, dict)):
            hash_input += json.dumps(part, sort_keys=True, default=str)
        elif isinstance(part, Enum):
            hash_input += str(part.value)
        else:
            hash_input += str(part)
    return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()


# =============================================================================
# ENGINE CLASS
# =============================================================================


class SpendClassifierEngine:
    """
    Thread-safe singleton engine for deterministic spend classification.

    Classifies organisational records into GHG Protocol Scope 3 categories
    using a five-level priority hierarchy:
        1. Industry code (NAICS/ISIC) -- 0.90-0.95 confidence
        2. GL account code -- 0.85 confidence
        3. Procurement category -- 0.70 confidence
        4. Keyword analysis -- 0.40-0.60 confidence
        5. Default fallback -- 0.30 confidence (Cat 1)

    Multi-category splitting is performed when descriptions contain split
    indicators ("and", "with", "including", "plus"). Batch processing
    handles up to 50,000 records with chunked execution.

    This engine does NOT perform any LLM or ML calls. All classification
    is deterministic and based on lookup tables from CategoryDatabaseEngine.

    Thread Safety:
        Uses the __new__ singleton pattern with threading.Lock.

    Example:
        >>> engine = SpendClassifierEngine()
        >>> result = engine.classify_spend(SpendRecord(
        ...     description="Air travel to London",
        ...     amount=Decimal("2000.00"),
        ... ))
        >>> result.primary_category
        <Scope3Category.CAT_6: 6>
    """

    _instance: Optional["SpendClassifierEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "SpendClassifierEngine":
        """Thread-safe singleton instantiation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the classifier engine (only once due to singleton)."""
        if hasattr(self, "_initialized"):
            return

        self._initialized: bool = True
        self._db: CategoryDatabaseEngine = get_category_database_engine()
        self._classify_count: int = 0
        self._classify_lock: threading.Lock = threading.Lock()

        logger.info(
            "SpendClassifierEngine initialized: "
            "procurement_categories=%d, max_batch=%d, chunk_size=%d",
            len(_PROCUREMENT_CATEGORY_MAP),
            _MAX_BATCH_SIZE,
            _DEFAULT_CHUNK_SIZE,
        )

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _increment_count(self) -> None:
        """Increment the classification counter thread-safely."""
        with self._classify_lock:
            self._classify_count += 1

    def _quantize(self, value: Decimal) -> Decimal:
        """Quantize a Decimal to 8 decimal places."""
        return value.quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)

    def _determine_confidence_level(self, confidence: Decimal) -> ConfidenceLevel:
        """
        Classify a confidence score into a human-readable level.

        Args:
            confidence: Confidence value between 0.0 and 1.0.

        Returns:
            ConfidenceLevel enum value.
        """
        if confidence >= Decimal("0.90"):
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= Decimal("0.75"):
            return ConfidenceLevel.HIGH
        elif confidence >= Decimal("0.55"):
            return ConfidenceLevel.MEDIUM
        elif confidence >= Decimal("0.35"):
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def recommend_approach(
        self, confidence: Decimal, data_source: DataSourceType
    ) -> CalculationApproach:
        """
        Recommend a GHG Protocol calculation approach based on confidence
        and data source type.

        Decision logic:
        - confidence >= 0.85 and primary data available -> supplier_specific
        - confidence >= 0.70 -> hybrid
        - confidence >= 0.50 -> average_data
        - confidence < 0.50 -> spend_based

        Args:
            confidence: Classification confidence score (0.0-1.0).
            data_source: Type of input data source.

        Returns:
            CalculationApproach enum value.

        Example:
            >>> engine = SpendClassifierEngine()
            >>> engine.recommend_approach(Decimal("0.92"), DataSourceType.SUPPLIER)
            <CalculationApproach.SUPPLIER_SPECIFIC: 'supplier_specific'>
        """
        primary_data_sources = {
            DataSourceType.SUPPLIER,
            DataSourceType.ENERGY,
            DataSourceType.WASTE,
        }

        if confidence >= Decimal("0.85") and data_source in primary_data_sources:
            return CalculationApproach.SUPPLIER_SPECIFIC
        elif confidence >= Decimal("0.70"):
            return CalculationApproach.HYBRID
        elif confidence >= Decimal("0.50"):
            return CalculationApproach.AVERAGE_DATA
        else:
            return CalculationApproach.SPEND_BASED

    def _try_industry_code(
        self,
        naics_code: Optional[str],
        isic_code: Optional[str],
    ) -> Optional[Tuple[Scope3Category, List[Scope3Category], Decimal, ClassificationMethod, str]]:
        """
        Attempt classification via industry code lookup (Level 1).

        Returns:
            Tuple of (primary, secondary, confidence, method, description)
            or None if no industry code is available.
        """
        # Try NAICS first (higher resolution)
        if naics_code:
            try:
                result = self._db.lookup_naics(naics_code)
                return (
                    result.primary_category,
                    result.secondary_categories,
                    result.confidence,
                    ClassificationMethod.NAICS,
                    f"NAICS {result.matched_code}: {result.description}",
                )
            except ValueError:
                logger.debug("NAICS lookup failed for code '%s'", naics_code)

        # Fall back to ISIC
        if isic_code:
            try:
                result = self._db.lookup_isic(isic_code)
                return (
                    result.primary_category,
                    result.secondary_categories,
                    result.confidence,
                    ClassificationMethod.ISIC,
                    f"ISIC {result.matched_code}: {result.description}",
                )
            except ValueError:
                logger.debug("ISIC lookup failed for code '%s'", isic_code)

        return None

    def _try_gl_account(
        self, gl_account_code: Optional[str]
    ) -> Optional[Tuple[Scope3Category, List[Scope3Category], Decimal, ClassificationMethod, str]]:
        """
        Attempt classification via GL account code lookup (Level 2).

        Returns:
            Tuple of (primary, secondary, confidence, method, description)
            or None if no GL code or match found.
        """
        if not gl_account_code:
            return None

        try:
            result = self._db.lookup_gl_account(gl_account_code)
            return (
                result.primary_category,
                result.secondary_categories,
                result.confidence,
                ClassificationMethod.GL_ACCOUNT,
                f"GL {result.matched_range}: {result.description}",
            )
        except ValueError:
            logger.debug(
                "GL account lookup failed for code '%s'", gl_account_code
            )
            return None

    def _try_procurement_category(
        self, procurement_category: Optional[str]
    ) -> Optional[Tuple[Scope3Category, List[Scope3Category], Decimal, ClassificationMethod, str]]:
        """
        Attempt classification via procurement category lookup (Level 3).

        Returns:
            Tuple of (primary, secondary, confidence, method, description)
            or None if no category or match found.
        """
        if not procurement_category:
            return None

        key = procurement_category.strip().lower()
        match = _PROCUREMENT_CATEGORY_MAP.get(key)
        if match:
            category, confidence = match
            return (
                category,
                [],
                confidence,
                ClassificationMethod.PROCUREMENT,
                f"Procurement category '{procurement_category}' -> Cat {category.value}",
            )

        return None

    def _try_keyword(
        self, description: str
    ) -> Optional[Tuple[Scope3Category, List[Scope3Category], Decimal, ClassificationMethod, str]]:
        """
        Attempt classification via keyword analysis (Level 4).

        Returns:
            Tuple of (primary, secondary, confidence, method, description_text)
            or None if no keyword match.
        """
        if not description:
            return None

        try:
            result = self._db.lookup_keyword(description)
            return (
                result.primary_category,
                [],
                result.confidence,
                ClassificationMethod.KEYWORD,
                f"Keyword '{result.matched_keyword}' ({result.keyword_group})",
            )
        except ValueError:
            logger.debug(
                "Keyword lookup failed for text '%s'", description[:50]
            )
            return None

    def _apply_hierarchy(
        self,
        naics_code: Optional[str],
        isic_code: Optional[str],
        gl_account_code: Optional[str],
        procurement_category: Optional[str],
        description: str,
    ) -> Tuple[Scope3Category, List[Scope3Category], Decimal, ClassificationMethod, str]:
        """
        Apply the five-level classification priority hierarchy.

        Attempts each level in order and returns the first successful match.
        Falls back to Cat 1 with confidence 0.30 if all levels fail.

        Returns:
            Tuple of (primary, secondary, confidence, method, reason).
        """
        # Level 1: Industry code
        result = self._try_industry_code(naics_code, isic_code)
        if result:
            return result

        # Level 2: GL account
        result = self._try_gl_account(gl_account_code)
        if result:
            return result

        # Level 3: Procurement category
        result = self._try_procurement_category(procurement_category)
        if result:
            return result

        # Level 4: Keyword analysis
        result = self._try_keyword(description)
        if result:
            return result

        # Level 5: Default fallback
        return (
            Scope3Category.CAT_1,
            [],
            Decimal("0.30"),
            ClassificationMethod.DEFAULT,
            "Default fallback: no classification signals found, assigned to Cat 1",
        )

    def _detect_split(self, description: str) -> List[str]:
        """
        Detect multi-category split indicators in a description.

        Splits the description on "and", "with", "including", "plus", "&", "+".
        Only returns multiple segments if at least two meaningful segments
        are found (each >= 3 characters after stripping).

        Args:
            description: Transaction description text.

        Returns:
            List of text segments. Returns single-element list if no split.
        """
        segments = _SPLIT_PATTERNS.split(description)
        meaningful = [s.strip() for s in segments if len(s.strip()) >= 3]

        if len(meaningful) >= 2:
            return meaningful

        return [description]

    def _build_classification_result(
        self,
        primary: Scope3Category,
        secondary: List[Scope3Category],
        confidence: Decimal,
        method: ClassificationMethod,
        reason: str,
        source_type: DataSourceType,
        is_split: bool,
        split_allocations: List[SplitAllocation],
        start_time: datetime,
    ) -> ClassificationResult:
        """Build a ClassificationResult with provenance hash."""
        elapsed_ms = self._quantize(
            Decimal(str(
                (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            ))
        )

        confidence_q = self._quantize(confidence)
        confidence_level = self._determine_confidence_level(confidence_q)
        recommended = self.recommend_approach(confidence_q, source_type)

        provenance_hash = _calculate_hash(
            "classify", primary, confidence_q, method, source_type, reason,
        )

        return ClassificationResult(
            primary_category=primary,
            secondary_categories=secondary,
            confidence=confidence_q,
            confidence_level=confidence_level,
            classification_method=method,
            recommended_approach=recommended,
            is_split=is_split,
            split_allocations=split_allocations,
            source_type=source_type,
            description=reason,
            provenance_hash=provenance_hash,
            processing_time_ms=elapsed_ms,
        )

    # =========================================================================
    # PUBLIC CLASSIFICATION METHODS
    # =========================================================================

    def classify_record(
        self, record: dict, source_type: DataSourceType
    ) -> ClassificationResult:
        """
        Classify a single generic record into a Scope 3 category.

        Extracts classification signals from the record dict and applies
        the five-level priority hierarchy. Supports multi-category splitting
        when split indicators are detected in the description.

        Args:
            record: Dictionary containing record fields (description, naics_code,
                gl_account_code, procurement_category, etc.)
            source_type: Data source type for routing context.

        Returns:
            ClassificationResult with category, confidence, and provenance hash.

        Example:
            >>> engine = SpendClassifierEngine()
            >>> result = engine.classify_record(
            ...     {"description": "Office supplies", "gl_account_code": "6100"},
            ...     DataSourceType.SPEND,
            ... )
            >>> result.primary_category
            <Scope3Category.CAT_1: 1>
        """
        self._increment_count()
        start_time = datetime.now(timezone.utc)

        description = str(record.get("description", ""))
        naics_code = record.get("naics_code")
        isic_code = record.get("isic_code")
        gl_account_code = record.get("gl_account_code")
        procurement_category = record.get("procurement_category")

        # Check for multi-category split
        segments = self._detect_split(description)
        if len(segments) > 1:
            return self._classify_split(
                segments, naics_code, isic_code,
                gl_account_code, procurement_category,
                source_type, start_time,
            )

        primary, secondary, confidence, method, reason = self._apply_hierarchy(
            naics_code, isic_code, gl_account_code,
            procurement_category, description,
        )

        return self._build_classification_result(
            primary, secondary, confidence, method, reason,
            source_type, False, [], start_time,
        )

    def _classify_split(
        self,
        segments: List[str],
        naics_code: Optional[str],
        isic_code: Optional[str],
        gl_account_code: Optional[str],
        procurement_category: Optional[str],
        source_type: DataSourceType,
        start_time: datetime,
    ) -> ClassificationResult:
        """
        Classify a split record with multiple description segments.

        Each segment is classified independently. The primary category is
        the one assigned to the first (or highest-confidence) segment.
        Allocation ratios are split equally across segments.
        """
        allocations: List[SplitAllocation] = []
        categories_seen: List[Scope3Category] = []
        total_confidence = Decimal("0")
        best_primary = Scope3Category.CAT_1
        best_confidence = Decimal("0")
        best_reason = "Split classification"
        best_method = ClassificationMethod.KEYWORD

        ratio = self._quantize(
            Decimal("1") / Decimal(str(len(segments)))
        )

        for segment in segments:
            primary, _, confidence, method, reason = self._apply_hierarchy(
                naics_code, isic_code, gl_account_code,
                procurement_category, segment,
            )

            allocations.append(SplitAllocation(
                category=primary,
                allocation_ratio=ratio,
                description_segment=segment,
            ))

            if primary not in categories_seen:
                categories_seen.append(primary)

            total_confidence += confidence

            if confidence > best_confidence:
                best_confidence = confidence
                best_primary = primary
                best_method = method
                best_reason = f"Split: {reason}"

        # Average confidence across segments
        avg_confidence = self._quantize(
            total_confidence / Decimal(str(len(segments)))
        )

        # Secondary categories are all non-primary categories seen
        secondary = [c for c in categories_seen if c != best_primary]

        return self._build_classification_result(
            best_primary, secondary, avg_confidence, best_method,
            best_reason, source_type, True, allocations, start_time,
        )

    def classify_spend(self, spend: SpendRecord) -> ClassificationResult:
        """
        Classify a spend data record.

        Uses spend-specific fields (gl_account_code, naics_code, isic_code,
        procurement_category, description) through the priority hierarchy.

        Args:
            spend: SpendRecord with transaction details.

        Returns:
            ClassificationResult with category assignment.

        Example:
            >>> engine = SpendClassifierEngine()
            >>> result = engine.classify_spend(SpendRecord(
            ...     description="Raw material purchase - steel",
            ...     amount=Decimal("50000.00"),
            ...     naics_code="331110",
            ... ))
            >>> result.primary_category
            <Scope3Category.CAT_1: 1>
        """
        return self.classify_record(
            {
                "description": spend.description,
                "naics_code": spend.naics_code,
                "isic_code": spend.isic_code,
                "gl_account_code": spend.gl_account_code,
                "procurement_category": spend.procurement_category,
            },
            DataSourceType.SPEND,
        )

    def classify_purchase_order(
        self, po: PurchaseOrderRecord
    ) -> ClassificationResult:
        """
        Classify a purchase order record.

        If the PO is flagged as capex, overrides the classification to Cat 2
        (Capital Goods) with high confidence.

        Args:
            po: PurchaseOrderRecord with PO line details.

        Returns:
            ClassificationResult with category assignment.

        Example:
            >>> engine = SpendClassifierEngine()
            >>> result = engine.classify_purchase_order(PurchaseOrderRecord(
            ...     description="CNC milling machine",
            ...     amount=Decimal("250000.00"),
            ...     is_capex=True,
            ... ))
            >>> result.primary_category
            <Scope3Category.CAT_2: 2>
        """
        self._increment_count()
        start_time = datetime.now(timezone.utc)

        # Capex override: if explicitly flagged as capital expenditure
        if po.is_capex is True:
            return self._build_classification_result(
                Scope3Category.CAT_2, [],
                Decimal("0.90"), ClassificationMethod.SOURCE_TYPE,
                "Purchase order flagged as capex -> Cat 2 (Capital Goods)",
                DataSourceType.PURCHASE_ORDER, False, [], start_time,
            )

        return self.classify_record(
            {
                "description": po.description,
                "naics_code": po.naics_code,
                "isic_code": po.isic_code,
                "gl_account_code": po.gl_account_code,
                "procurement_category": po.item_category,
            },
            DataSourceType.PURCHASE_ORDER,
        )

    def classify_bom(self, bom: BOMRecord) -> ClassificationResult:
        """
        Classify a bill of materials record.

        BOM items are typically Cat 1 (purchased inputs). If the item is
        flagged as intermediate (sold for further processing), it may map
        to Cat 10.

        Args:
            bom: BOMRecord with material details.

        Returns:
            ClassificationResult with category assignment.

        Example:
            >>> engine = SpendClassifierEngine()
            >>> result = engine.classify_bom(BOMRecord(
            ...     description="Polypropylene resin pellets",
            ...     quantity=Decimal("5000"),
            ...     unit="kg",
            ... ))
            >>> result.primary_category
            <Scope3Category.CAT_1: 1>
        """
        self._increment_count()
        start_time = datetime.now(timezone.utc)

        if bom.is_intermediate:
            return self._build_classification_result(
                Scope3Category.CAT_10,
                [Scope3Category.CAT_1],
                Decimal("0.80"),
                ClassificationMethod.SOURCE_TYPE,
                "BOM item flagged as intermediate -> Cat 10 (Processing of Sold Products)",
                DataSourceType.BOM, False, [], start_time,
            )

        return self.classify_record(
            {
                "description": bom.description,
                "naics_code": bom.naics_code,
                "procurement_category": bom.material_type,
            },
            DataSourceType.BOM,
        )

    def classify_travel(self, travel: TravelRecord) -> ClassificationResult:
        """
        Classify a travel record (always Cat 6 or Cat 7).

        Business travel maps to Cat 6; commuting maps to Cat 7.
        Travel records always receive source-type-based confidence since
        the data type unambiguously determines the category.

        Args:
            travel: TravelRecord with travel details.

        Returns:
            ClassificationResult with Cat 6 or Cat 7.

        Example:
            >>> engine = SpendClassifierEngine()
            >>> result = engine.classify_travel(TravelRecord(
            ...     description="Flight JFK-LHR",
            ...     travel_type="business",
            ... ))
            >>> result.primary_category
            <Scope3Category.CAT_6: 6>
        """
        self._increment_count()
        start_time = datetime.now(timezone.utc)

        travel_type = travel.travel_type.strip().lower()

        if travel_type == "commuting":
            return self._build_classification_result(
                Scope3Category.CAT_7, [],
                Decimal("0.92"), ClassificationMethod.SOURCE_TYPE,
                "Travel record type=commuting -> Cat 7 (Employee Commuting)",
                DataSourceType.TRAVEL, False, [], start_time,
            )

        return self._build_classification_result(
            Scope3Category.CAT_6, [],
            Decimal("0.92"), ClassificationMethod.SOURCE_TYPE,
            "Travel record type=business -> Cat 6 (Business Travel)",
            DataSourceType.TRAVEL, False, [], start_time,
        )

    def classify_waste(self, waste: WasteRecord) -> ClassificationResult:
        """
        Classify a waste manifest record (always Cat 5).

        Waste records unambiguously map to Cat 5 (Waste Generated in
        Operations) with high confidence.

        Args:
            waste: WasteRecord with waste details.

        Returns:
            ClassificationResult with Cat 5.

        Example:
            >>> engine = SpendClassifierEngine()
            >>> result = engine.classify_waste(WasteRecord(
            ...     description="General office waste",
            ...     quantity=Decimal("500"),
            ...     unit="kg",
            ... ))
            >>> result.primary_category
            <Scope3Category.CAT_5: 5>
        """
        self._increment_count()
        start_time = datetime.now(timezone.utc)

        return self._build_classification_result(
            Scope3Category.CAT_5,
            [Scope3Category.CAT_12],
            Decimal("0.92"),
            ClassificationMethod.SOURCE_TYPE,
            "Waste manifest record -> Cat 5 (Waste Generated in Operations)",
            DataSourceType.WASTE, False, [], start_time,
        )

    def classify_lease(self, lease: LeaseRecord) -> ClassificationResult:
        """
        Classify a lease agreement record (Cat 8 or Cat 13).

        The category depends on the reporter's role:
        - Lessee (tenant): Cat 8 (Upstream Leased Assets)
        - Lessor (owner): Cat 13 (Downstream Leased Assets)

        Args:
            lease: LeaseRecord with lease details.

        Returns:
            ClassificationResult with Cat 8 or Cat 13.

        Example:
            >>> engine = SpendClassifierEngine()
            >>> result = engine.classify_lease(LeaseRecord(
            ...     description="Office space lease",
            ...     reporter_is_lessee=True,
            ... ))
            >>> result.primary_category
            <Scope3Category.CAT_8: 8>
        """
        self._increment_count()
        start_time = datetime.now(timezone.utc)

        if lease.reporter_is_lessee:
            return self._build_classification_result(
                Scope3Category.CAT_8,
                [Scope3Category.CAT_13],
                Decimal("0.90"),
                ClassificationMethod.SOURCE_TYPE,
                "Lease record: reporter is lessee -> Cat 8 (Upstream Leased Assets)",
                DataSourceType.LEASE, False, [], start_time,
            )

        return self._build_classification_result(
            Scope3Category.CAT_13,
            [Scope3Category.CAT_8],
            Decimal("0.90"),
            ClassificationMethod.SOURCE_TYPE,
            "Lease record: reporter is lessor -> Cat 13 (Downstream Leased Assets)",
            DataSourceType.LEASE, False, [], start_time,
        )

    def classify_logistics(
        self, logistics: LogisticsRecord
    ) -> ClassificationResult:
        """
        Classify a logistics/freight record (Cat 4 or Cat 9).

        The category depends on freight direction:
        - Inbound: Cat 4 (Upstream Transportation)
        - Outbound: Cat 9 (Downstream Transportation)

        Args:
            logistics: LogisticsRecord with shipment details.

        Returns:
            ClassificationResult with Cat 4 or Cat 9.

        Example:
            >>> engine = SpendClassifierEngine()
            >>> result = engine.classify_logistics(LogisticsRecord(
            ...     description="Inbound raw material shipment",
            ...     direction="inbound",
            ... ))
            >>> result.primary_category
            <Scope3Category.CAT_4: 4>
        """
        self._increment_count()
        start_time = datetime.now(timezone.utc)

        direction = logistics.direction.strip().lower()

        if direction == "outbound":
            return self._build_classification_result(
                Scope3Category.CAT_9,
                [Scope3Category.CAT_4],
                Decimal("0.88"),
                ClassificationMethod.SOURCE_TYPE,
                "Logistics record direction=outbound -> Cat 9 (Downstream Transportation)",
                DataSourceType.LOGISTICS, False, [], start_time,
            )

        return self._build_classification_result(
            Scope3Category.CAT_4,
            [Scope3Category.CAT_9],
            Decimal("0.88"),
            ClassificationMethod.SOURCE_TYPE,
            "Logistics record direction=inbound -> Cat 4 (Upstream Transportation)",
            DataSourceType.LOGISTICS, False, [], start_time,
        )

    def classify_energy(self, energy: EnergyRecord) -> ClassificationResult:
        """
        Classify an energy invoice record (always Cat 3).

        Energy records unambiguously map to Cat 3 (Fuel and Energy-Related
        Activities not included in Scope 1 or Scope 2).

        Args:
            energy: EnergyRecord with energy invoice details.

        Returns:
            ClassificationResult with Cat 3.

        Example:
            >>> engine = SpendClassifierEngine()
            >>> result = engine.classify_energy(EnergyRecord(
            ...     description="Electricity invoice Q4 2025",
            ... ))
            >>> result.primary_category
            <Scope3Category.CAT_3: 3>
        """
        self._increment_count()
        start_time = datetime.now(timezone.utc)

        return self._build_classification_result(
            Scope3Category.CAT_3, [],
            Decimal("0.92"),
            ClassificationMethod.SOURCE_TYPE,
            "Energy invoice record -> Cat 3 (Fuel & Energy Activities)",
            DataSourceType.ENERGY, False, [], start_time,
        )

    def classify_investment(
        self, investment: InvestmentRecord
    ) -> ClassificationResult:
        """
        Classify an investment record (always Cat 15).

        Investment records unambiguously map to Cat 15 (Investments).

        Args:
            investment: InvestmentRecord with holding details.

        Returns:
            ClassificationResult with Cat 15.

        Example:
            >>> engine = SpendClassifierEngine()
            >>> result = engine.classify_investment(InvestmentRecord(
            ...     description="Listed equity portfolio - Tech sector",
            ... ))
            >>> result.primary_category
            <Scope3Category.CAT_15: 15>
        """
        self._increment_count()
        start_time = datetime.now(timezone.utc)

        return self._build_classification_result(
            Scope3Category.CAT_15, [],
            Decimal("0.92"),
            ClassificationMethod.SOURCE_TYPE,
            "Investment record -> Cat 15 (Investments)",
            DataSourceType.INVESTMENT, False, [], start_time,
        )

    def classify_franchise(
        self, franchise: FranchiseRecord
    ) -> ClassificationResult:
        """
        Classify a franchise agreement record (always Cat 14).

        Franchise records unambiguously map to Cat 14 (Franchises).

        Args:
            franchise: FranchiseRecord with franchise details.

        Returns:
            ClassificationResult with Cat 14.

        Example:
            >>> engine = SpendClassifierEngine()
            >>> result = engine.classify_franchise(FranchiseRecord(
            ...     description="QSR franchise unit #1234",
            ... ))
            >>> result.primary_category
            <Scope3Category.CAT_14: 14>
        """
        self._increment_count()
        start_time = datetime.now(timezone.utc)

        return self._build_classification_result(
            Scope3Category.CAT_14, [],
            Decimal("0.92"),
            ClassificationMethod.SOURCE_TYPE,
            "Franchise agreement record -> Cat 14 (Franchises)",
            DataSourceType.FRANCHISE, False, [], start_time,
        )

    # =========================================================================
    # BATCH PROCESSING
    # =========================================================================

    def classify_batch(
        self,
        records: List[dict],
        source_type: DataSourceType,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
    ) -> BatchClassificationResult:
        """
        Classify a batch of records with chunked processing.

        Processes up to 50,000 records in chunks for memory efficiency.
        Records that fail classification are logged and counted but do
        not halt the batch.

        Args:
            records: List of record dicts to classify.
            source_type: Data source type for all records in the batch.
            chunk_size: Number of records per processing chunk (default 1000).

        Returns:
            BatchClassificationResult with individual results, aggregation
            statistics, and batch provenance hash.

        Raises:
            ValueError: If batch size exceeds 50,000 records.

        Example:
            >>> engine = SpendClassifierEngine()
            >>> batch_result = engine.classify_batch(
            ...     [{"description": "Office supplies"}, {"description": "Air travel"}],
            ...     DataSourceType.SPEND,
            ... )
            >>> batch_result.classified_count
            2
        """
        start_time = datetime.now(timezone.utc)

        if len(records) > _MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(records)} exceeds maximum {_MAX_BATCH_SIZE}. "
                f"Please split into smaller batches."
            )

        results: List[ClassificationResult] = []
        errors: List[Dict[str, Any]] = []
        category_counts: Dict[str, int] = {}
        total_confidence = Decimal("0")

        # Process in chunks
        for chunk_start in range(0, len(records), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(records))
            chunk = records[chunk_start:chunk_end]

            for idx, record in enumerate(chunk):
                record_idx = chunk_start + idx
                try:
                    result = self.classify_record(record, source_type)
                    results.append(result)

                    cat_key = f"cat_{result.primary_category.value}"
                    category_counts[cat_key] = category_counts.get(cat_key, 0) + 1
                    total_confidence += result.confidence

                except Exception as exc:
                    logger.warning(
                        "Batch classification failed for record %d: %s",
                        record_idx, str(exc),
                    )
                    errors.append({
                        "record_index": record_idx,
                        "error": str(exc),
                        "description": str(record.get("description", ""))[:100],
                    })

            logger.debug(
                "Batch chunk processed: %d-%d of %d",
                chunk_start, chunk_end, len(records),
            )

        classified_count = len(results)
        avg_confidence = self._quantize(
            total_confidence / Decimal(str(max(classified_count, 1)))
        )

        elapsed_ms = self._quantize(
            Decimal(str(
                (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            ))
        )

        batch_hash = _calculate_hash(
            "batch_classify", len(records), classified_count,
            len(errors), avg_confidence,
        )

        batch_result = BatchClassificationResult(
            results=results,
            total_records=len(records),
            classified_count=classified_count,
            failed_count=len(errors),
            category_distribution=category_counts,
            average_confidence=avg_confidence,
            processing_time_ms=elapsed_ms,
            errors=errors,
            provenance_hash=batch_hash,
        )

        logger.info(
            "Batch classification complete: total=%d, classified=%d, "
            "failed=%d, avg_confidence=%s, elapsed_ms=%s",
            len(records), classified_count, len(errors),
            avg_confidence, elapsed_ms,
        )

        return batch_result

    # =========================================================================
    # SUMMARY AND STATS
    # =========================================================================

    def get_classify_count(self) -> int:
        """
        Get total number of classifications performed.

        Returns:
            Integer count of classifications.
        """
        with self._classify_lock:
            return self._classify_count

    def get_engine_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the classifier engine state.

        Returns:
            Dict with classification statistics and configuration.

        Example:
            >>> engine = SpendClassifierEngine()
            >>> summary = engine.get_engine_summary()
            >>> summary["max_batch_size"]
            50000
        """
        return {
            "agent_id": AGENT_ID,
            "agent_component": AGENT_COMPONENT,
            "version": VERSION,
            "total_classifications": self.get_classify_count(),
            "procurement_categories": len(_PROCUREMENT_CATEGORY_MAP),
            "max_batch_size": _MAX_BATCH_SIZE,
            "chunk_size": _DEFAULT_CHUNK_SIZE,
            "hierarchy_levels": 5,
            "db_summary": self._db.get_database_summary(),
        }

    @classmethod
    def reset(cls) -> None:
        """
        Reset the singleton instance (for testing only).

        Warning: This method is intended for use in test fixtures only.
        Do not call in production code.
        """
        with cls._lock:
            cls._instance = None


# =============================================================================
# MODULE-LEVEL ACCESSOR
# =============================================================================

_engine_instance: Optional[SpendClassifierEngine] = None
_engine_lock: threading.Lock = threading.Lock()


def get_spend_classifier_engine() -> SpendClassifierEngine:
    """
    Get the singleton SpendClassifierEngine instance.

    Thread-safe accessor for the global classifier engine instance.

    Returns:
        SpendClassifierEngine singleton instance.

    Example:
        >>> engine = get_spend_classifier_engine()
        >>> result = engine.classify_spend(...)
    """
    global _engine_instance
    with _engine_lock:
        if _engine_instance is None:
            _engine_instance = SpendClassifierEngine()
        return _engine_instance


def reset_spend_classifier_engine() -> None:
    """
    Reset the module-level engine instance (for testing only).

    Warning: This function is intended for use in test fixtures only.
    Do not call in production code.
    """
    global _engine_instance
    with _engine_lock:
        _engine_instance = None
    SpendClassifierEngine.reset()


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enumerations
    "DataSourceType",
    "ClassificationMethod",
    "CalculationApproach",
    "ConfidenceLevel",
    # Input models
    "SpendRecord",
    "PurchaseOrderRecord",
    "BOMRecord",
    "TravelRecord",
    "WasteRecord",
    "LeaseRecord",
    "LogisticsRecord",
    "EnergyRecord",
    "InvestmentRecord",
    "FranchiseRecord",
    # Output models
    "SplitAllocation",
    "ClassificationResult",
    "BatchClassificationResult",
    # Engine class
    "SpendClassifierEngine",
    "get_spend_classifier_engine",
    "reset_spend_classifier_engine",
]
