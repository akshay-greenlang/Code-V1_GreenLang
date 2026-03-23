# -*- coding: utf-8 -*-
"""
BoundaryDeterminerEngine - Engine 4: Upstream/downstream boundary analysis.

This module implements the BoundaryDeterminerEngine for AGENT-MRV-029
(Scope 3 Category Mapper, GL-MRV-X-040). It determines upstream/downstream
boundaries and resolves category boundary ambiguities using the 10
double-counting prevention rules (DC-SCM-001 through DC-SCM-010).

Boundary Rules:
    DC-SCM-001: Cat 1 vs Cat 2 -- Opex vs Capex (capitalization threshold)
    DC-SCM-002: Cat 1 vs Cat 4 -- Goods vs Freight (Incoterm split)
    DC-SCM-003: Cat 3 vs Scope 2 -- WTT/T&D exclusion
    DC-SCM-004: Cat 4 vs Cat 9 -- Upstream vs Downstream transport (Incoterm)
    DC-SCM-005: Cat 6 vs Cat 7 -- Business travel vs commuting (routine flag)
    DC-SCM-006: Cat 8 vs Scope 1/2 -- Leased assets (consolidation approach)
    DC-SCM-007: Cat 10 vs Cat 11 -- Processing vs Use (sequential boundary)
    DC-SCM-008: Cat 11 vs Cat 12 -- Use vs End-of-Life (product lifetime)
    DC-SCM-009: Cat 13 vs Scope 1/2 -- Downstream leased assets (consolidation)
    DC-SCM-010: Cat 14 vs Cat 15 -- Franchise vs Investment (agreement type)

Zero-Hallucination Guarantee:
    All boundary determinations use deterministic rule-based logic.
    No LLM or ML models are used. Every determination is based on explicit
    input parameters (capitalization thresholds, Incoterms, consolidation
    approach, agreement type, etc.) and traceable via SHA-256 provenance.

Thread Safety:
    Uses __new__ singleton pattern with threading.Lock for thread-safe
    instantiation. Determination counters are protected by a dedicated lock.

Example:
    >>> from greenlang.agents.mrv.scope3_category_mapper.boundary_determiner import (
    ...     BoundaryDeterminerEngine,
    ... )
    >>> engine = BoundaryDeterminerEngine()
    >>> category = engine.resolve_cat1_vs_cat2(
    ...     amount=Decimal("12000"),
    ...     capitalization_threshold=Decimal("5000"),
    ... )
    >>> category
    <Scope3Category.CAT_2_CAPITAL_GOODS: 'cat_2_capital_goods'>

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-X-040
"""

import hashlib
import json
import logging
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from pydantic import ConfigDict

from greenlang.agents.mrv.scope3_category_mapper.models import (
    Scope3Category,
    SCOPE3_CATEGORY_NAMES,
    SCOPE3_CATEGORY_NUMBERS,
    CategoryRelevance,
)

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

AGENT_ID: str = "GL-MRV-X-040"
AGENT_COMPONENT: str = "AGENT-MRV-029"
ENGINE_ID: str = "gl_scm_boundary_determiner_engine"
ENGINE_VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_scm_"

# Quantization constants
_QUANT_8DP = Decimal("0.00000001")
_QUANT_2DP = Decimal("0.01")
_ZERO = Decimal("0")
_ONE = Decimal("1")
_HUNDRED = Decimal("100")

# Default capitalization threshold for Cat 1 vs Cat 2 (DC-SCM-001)
DEFAULT_CAPITALIZATION_THRESHOLD = Decimal("5000")


# =============================================================================
# ENUMERATIONS
# =============================================================================


class IncotermsRule(str, Enum):
    """ICC Incoterms 2020 rules for transport boundary determination.

    Buyer-pays-freight Incoterms (seller delivers early, buyer arranges
    onward transport): EXW, FCA, FAS, FOB.

    Seller-pays-freight Incoterms (seller arranges main carriage):
    CFR, CIF, CPT, CIP, DAP, DPU, DDP.
    """

    EXW = "exw"    # Ex Works
    FCA = "fca"    # Free Carrier
    CPT = "cpt"    # Carriage Paid To
    CIP = "cip"    # Carriage and Insurance Paid To
    DAP = "dap"    # Delivered At Place
    DPU = "dpu"    # Delivered at Place Unloaded
    DDP = "ddp"    # Delivered Duty Paid
    FAS = "fas"    # Free Alongside Ship
    FOB = "fob"    # Free On Board
    CFR = "cfr"    # Cost and Freight
    CIF = "cif"    # Cost, Insurance, Freight


class ConsolidationApproach(str, Enum):
    """GHG Protocol organizational boundary consolidation approaches.

    Determines how leased assets and investments are reported
    between Scope 1/2 and Scope 3.
    """

    OPERATIONAL_CONTROL = "operational_control"
    FINANCIAL_CONTROL = "financial_control"
    EQUITY_SHARE = "equity_share"


class LeaseClassification(str, Enum):
    """Lease classification for Cat 8 vs Scope 1/2 boundary.

    Under IFRS 16 all leases are on-balance-sheet (right-of-use asset);
    under ASC 842 the distinction between operating and finance lease
    affects consolidation approach and scope determination.
    """

    OPERATING_LEASE = "operating_lease"
    FINANCE_LEASE = "finance_lease"
    SHORT_TERM = "short_term"    # <= 12 months
    LOW_VALUE = "low_value"      # Asset value below threshold


class DoubleCountingRule(str, Enum):
    """Double-counting prevention rules (DC-SCM-001 through DC-SCM-010).

    Each rule addresses a specific boundary ambiguity between two
    Scope 3 categories or between Scope 3 and Scope 1/2.
    """

    DC_SCM_001 = "DC-SCM-001"  # Cat 1 vs Cat 2: Opex vs Capex
    DC_SCM_002 = "DC-SCM-002"  # Cat 1 vs Cat 4: Goods vs Freight
    DC_SCM_003 = "DC-SCM-003"  # Cat 3 vs Scope 2: WTT/T&D
    DC_SCM_004 = "DC-SCM-004"  # Cat 4 vs Cat 9: Upstream vs Downstream Transport
    DC_SCM_005 = "DC-SCM-005"  # Cat 6 vs Cat 7: Travel vs Commuting
    DC_SCM_006 = "DC-SCM-006"  # Cat 8 vs Scope 1/2: Leased Assets
    DC_SCM_007 = "DC-SCM-007"  # Cat 10 vs Cat 11: Processing vs Use
    DC_SCM_008 = "DC-SCM-008"  # Cat 11 vs Cat 12: Use vs End-of-Life
    DC_SCM_009 = "DC-SCM-009"  # Cat 13 vs Scope 1/2: Downstream Leased Assets
    DC_SCM_010 = "DC-SCM-010"  # Cat 14 vs Cat 15: Franchise vs Investment


class DataSourceType(str, Enum):
    """Input data source types for boundary determination."""

    SPEND = "spend"
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


class ValueChainPosition(str, Enum):
    """Position in the value chain relative to the reporting company."""

    UPSTREAM = "upstream"      # Categories 1-8
    DOWNSTREAM = "downstream"  # Categories 9-15


# =============================================================================
# INCOTERM CLASSIFICATION SETS
# =============================================================================

# Buyer-pays-freight: seller delivers early, buyer arranges onward transport
# For Cat 4 vs Cat 9: these Incoterms mean buyer pays, so from buyer's
# perspective freight is upstream (Cat 4).
_BUYER_PAYS_FREIGHT: frozenset = frozenset({
    IncotermsRule.EXW,
    IncotermsRule.FCA,
    IncotermsRule.FAS,
    IncotermsRule.FOB,
})

# Seller-pays-freight: seller arranges and pays for main carriage
# For Cat 4 vs Cat 9: from seller's perspective freight is downstream (Cat 9).
_SELLER_PAYS_FREIGHT: frozenset = frozenset({
    IncotermsRule.CFR,
    IncotermsRule.CIF,
    IncotermsRule.CPT,
    IncotermsRule.CIP,
    IncotermsRule.DAP,
    IncotermsRule.DPU,
    IncotermsRule.DDP,
})


# =============================================================================
# DOUBLE-COUNTING RULE DETAILS
# =============================================================================

_DC_RULE_DETAILS: Dict[DoubleCountingRule, Dict[str, Any]] = {
    DoubleCountingRule.DC_SCM_001: {
        "rule_id": "DC-SCM-001",
        "name": "Cat 1 vs Cat 2: Opex vs Capex",
        "categories": ["cat_1_purchased_goods", "cat_2_capital_goods"],
        "description": (
            "If amount >= capitalization_threshold, classify as Cat 2 "
            "(Capital Goods). If amount < threshold, classify as Cat 1 "
            "(Purchased Goods). Default threshold: $5,000."
        ),
        "resolution_method": "capitalization_threshold",
        "ghg_protocol_ref": "Chapter 2, Appendix A Category 1 & 2 boundary",
    },
    DoubleCountingRule.DC_SCM_002: {
        "rule_id": "DC-SCM-002",
        "name": "Cat 1 vs Cat 4: Goods vs Freight",
        "categories": ["cat_1_purchased_goods", "cat_4_upstream_transport"],
        "description": (
            "Split by Incoterm: FOB/FCA -> buyer pays freight (Cat 4 separate). "
            "CIF/DDP -> freight included in goods price (Cat 1 only). "
            "If freight separately invoiced -> always Cat 4."
        ),
        "resolution_method": "incoterm_split",
        "ghg_protocol_ref": "Chapter 2, Appendix A Category 1 & 4 boundary",
    },
    DoubleCountingRule.DC_SCM_003: {
        "rule_id": "DC-SCM-003",
        "name": "Cat 3 vs Scope 2: WTT/T&D",
        "categories": ["cat_3_fuel_energy", "scope_2"],
        "description": (
            "Cat 3 = WTT of fuels + upstream electricity + T&D losses. "
            "Exclude amounts already in Scope 2 reported figures."
        ),
        "resolution_method": "scope2_exclusion",
        "ghg_protocol_ref": "Chapter 2, Appendix A Category 3 boundary",
    },
    DoubleCountingRule.DC_SCM_004: {
        "rule_id": "DC-SCM-004",
        "name": "Cat 4 vs Cat 9: Upstream vs Downstream Transport",
        "categories": ["cat_4_upstream_transport", "cat_9_downstream_transport"],
        "description": (
            "Point of sale determines boundary. Incoterm determines who pays: "
            "EXW/FCA/FOB -> buyer (Cat 4); CIF/DDP -> seller (Cat 9)."
        ),
        "resolution_method": "incoterm_direction",
        "ghg_protocol_ref": "Chapter 2, Appendix A Category 4 & 9 boundary",
    },
    DoubleCountingRule.DC_SCM_005: {
        "rule_id": "DC-SCM-005",
        "name": "Cat 6 vs Cat 7: Travel vs Commuting",
        "categories": ["cat_6_business_travel", "cat_7_employee_commuting"],
        "description": (
            "Business travel = non-routine journeys for work purposes. "
            "Commuting = routine home-to-office travel. "
            "Travel days exclude commuting (no double count)."
        ),
        "resolution_method": "routine_flag",
        "ghg_protocol_ref": "Chapter 2, Appendix A Category 6 & 7 boundary",
    },
    DoubleCountingRule.DC_SCM_006: {
        "rule_id": "DC-SCM-006",
        "name": "Cat 8 vs Scope 1/2: Leased Assets",
        "categories": ["cat_8_upstream_leased", "scope_1_2"],
        "description": (
            "Operational control: lessee includes in Scope 1/2, not Cat 8. "
            "Financial control: may be Cat 8 for lessee. "
            "Equity share: proportional allocation."
        ),
        "resolution_method": "consolidation_approach",
        "ghg_protocol_ref": "Chapter 2, Appendix A Category 8 boundary",
    },
    DoubleCountingRule.DC_SCM_007: {
        "rule_id": "DC-SCM-007",
        "name": "Cat 10 vs Cat 11: Processing vs Use",
        "categories": ["cat_10_processing_sold", "cat_11_use_sold"],
        "description": (
            "Cat 10 = intermediate processing by third parties. "
            "Cat 11 = final use by end consumers. "
            "Sequential boundary: processing comes before use."
        ),
        "resolution_method": "product_lifecycle_stage",
        "ghg_protocol_ref": "Chapter 2, Appendix A Category 10 & 11 boundary",
    },
    DoubleCountingRule.DC_SCM_008: {
        "rule_id": "DC-SCM-008",
        "name": "Cat 11 vs Cat 12: Use vs End-of-Life",
        "categories": ["cat_11_use_sold", "cat_12_end_of_life"],
        "description": (
            "Product lifetime determines boundary. "
            "Cat 11 = during useful life. "
            "Cat 12 = after useful life ends."
        ),
        "resolution_method": "product_lifetime",
        "ghg_protocol_ref": "Chapter 2, Appendix A Category 11 & 12 boundary",
    },
    DoubleCountingRule.DC_SCM_009: {
        "rule_id": "DC-SCM-009",
        "name": "Cat 13 vs Scope 1/2: Downstream Leased Assets",
        "categories": ["cat_13_downstream_leased", "scope_1_2"],
        "description": (
            "Lessor reports in Cat 13 only if NOT consolidated in Scope 1/2. "
            "Consolidation approach determines boundary."
        ),
        "resolution_method": "consolidation_approach",
        "ghg_protocol_ref": "Chapter 2, Appendix A Category 13 boundary",
    },
    DoubleCountingRule.DC_SCM_010: {
        "rule_id": "DC-SCM-010",
        "name": "Cat 14 vs Cat 15: Franchise vs Investment",
        "categories": ["cat_14_franchises", "cat_15_investments"],
        "description": (
            "Franchise agreement -> Cat 14. "
            "Equity/debt investment -> Cat 15. "
            "If both exist, franchise takes precedence for franchise-related "
            "emissions."
        ),
        "resolution_method": "agreement_type",
        "ghg_protocol_ref": "Chapter 2, Appendix A Category 14 & 15 boundary",
    },
}


# =============================================================================
# VALUE CHAIN MAPPING
# =============================================================================

_CATEGORY_VALUE_CHAIN: Dict[Scope3Category, ValueChainPosition] = {
    Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES: ValueChainPosition.UPSTREAM,
    Scope3Category.CAT_2_CAPITAL_GOODS: ValueChainPosition.UPSTREAM,
    Scope3Category.CAT_3_FUEL_ENERGY_ACTIVITIES: ValueChainPosition.UPSTREAM,
    Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION: ValueChainPosition.UPSTREAM,
    Scope3Category.CAT_5_WASTE_GENERATED: ValueChainPosition.UPSTREAM,
    Scope3Category.CAT_6_BUSINESS_TRAVEL: ValueChainPosition.UPSTREAM,
    Scope3Category.CAT_7_EMPLOYEE_COMMUTING: ValueChainPosition.UPSTREAM,
    Scope3Category.CAT_8_UPSTREAM_LEASED_ASSETS: ValueChainPosition.UPSTREAM,
    Scope3Category.CAT_9_DOWNSTREAM_TRANSPORTATION: ValueChainPosition.DOWNSTREAM,
    Scope3Category.CAT_10_PROCESSING_SOLD_PRODUCTS: ValueChainPosition.DOWNSTREAM,
    Scope3Category.CAT_11_USE_OF_SOLD_PRODUCTS: ValueChainPosition.DOWNSTREAM,
    Scope3Category.CAT_12_END_OF_LIFE_TREATMENT: ValueChainPosition.DOWNSTREAM,
    Scope3Category.CAT_13_DOWNSTREAM_LEASED_ASSETS: ValueChainPosition.DOWNSTREAM,
    Scope3Category.CAT_14_FRANCHISES: ValueChainPosition.DOWNSTREAM,
    Scope3Category.CAT_15_INVESTMENTS: ValueChainPosition.DOWNSTREAM,
}


# =============================================================================
# DATA MODELS
# =============================================================================


class CategoryAllocation(BaseModel):
    """Allocation of a record portion to a specific Scope 3 category.

    Used when a record needs to be split across multiple categories
    (e.g., goods portion to Cat 1 and freight portion to Cat 4).

    Attributes:
        category: Target Scope 3 category.
        allocation_pct: Percentage allocated (0-100).
        amount_allocated: Monetary amount allocated.
        rationale: Explanation of the allocation.
    """

    model_config = ConfigDict(frozen=True)

    category: Scope3Category = Field(..., description="Target category")
    allocation_pct: Decimal = Field(
        ..., ge=_ZERO, le=_HUNDRED, description="Allocation percentage"
    )
    amount_allocated: Optional[Decimal] = Field(
        default=None, ge=_ZERO, description="Amount allocated"
    )
    rationale: str = Field(default="", description="Allocation rationale")


class BoundaryDetermination(BaseModel):
    """Result of boundary determination for a single record.

    Captures the final category assignment after applying boundary rules,
    any double-counting rules that influenced the decision, and whether
    a multi-category split is required.

    Attributes:
        record_id: Original record identifier.
        determined_category: Final category after boundary analysis.
        original_category: Category before boundary adjustment.
        value_chain_position: Upstream or downstream.
        consolidation_approach: Consolidation approach used.
        dc_rules_applied: Double-counting rules that influenced the decision.
        boundary_rationale: Explanation of boundary decision.
        split_required: Whether record needs multi-category split.
        split_allocations: Category allocations if split is required.
        scope_destination: Where emissions are reported (scope_1_2 or scope_3).
        incoterm_applied: Incoterm used for transport boundary.
        capitalization_applied: Whether capex threshold was applied.
        provenance_hash: SHA-256 hash of boundary determination.
        determined_at: ISO 8601 timestamp of determination.
    """

    model_config = ConfigDict(frozen=True)

    record_id: str = Field(default="", description="Record ID")
    determined_category: Scope3Category = Field(..., description="Final category")
    original_category: Optional[Scope3Category] = Field(
        default=None, description="Category before adjustment"
    )
    value_chain_position: ValueChainPosition = Field(
        default=ValueChainPosition.UPSTREAM, description="Value chain position"
    )
    consolidation_approach: Optional[ConsolidationApproach] = Field(
        default=None, description="Consolidation approach"
    )
    dc_rules_applied: List[DoubleCountingRule] = Field(
        default_factory=list, description="DC rules applied"
    )
    boundary_rationale: str = Field(default="", description="Boundary rationale")
    split_required: bool = Field(default=False, description="Split required")
    split_allocations: List[CategoryAllocation] = Field(
        default_factory=list, description="Split allocations"
    )
    scope_destination: str = Field(
        default="scope_3", description="scope_1_2 or scope_3"
    )
    incoterm_applied: Optional[IncotermsRule] = Field(
        default=None, description="Incoterm applied"
    )
    capitalization_applied: bool = Field(
        default=False, description="Capex threshold applied"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")
    determined_at: Optional[str] = Field(
        default=None, description="ISO 8601 timestamp"
    )


class DoubleCountingCheck(BaseModel):
    """Result of a double-counting check between categories.

    Identifies potential overlap between two categories for a set of
    records and describes how the overlap was resolved.

    Attributes:
        rule_id: Double-counting rule identifier.
        categories_checked: Categories involved in the check.
        overlap_detected: Whether potential double-counting was found.
        records_affected: Number of records potentially double-counted.
        record_ids: Affected record identifiers.
        resolution: How the overlap was resolved.
        provenance_hash: SHA-256 hash of the check result.
    """

    model_config = ConfigDict(frozen=True)

    rule_id: DoubleCountingRule = Field(..., description="DC rule ID")
    categories_checked: List[Scope3Category] = Field(
        default_factory=list, description="Categories checked"
    )
    overlap_detected: bool = Field(default=False, description="Overlap detected")
    records_affected: int = Field(default=0, ge=0, description="Records affected")
    record_ids: List[str] = Field(
        default_factory=list, description="Affected record IDs"
    )
    resolution: str = Field(default="", description="Resolution description")
    provenance_hash: str = Field(default="", description="SHA-256 hash")


# =============================================================================
# CLASSIFICATION RESULT (intake model for this engine)
# =============================================================================


class ClassificationResult(BaseModel):
    """Simplified classification result consumed by the boundary determiner.

    Attributes:
        record_id: Unique record identifier.
        primary_category: Primary Scope 3 category assigned.
        secondary_categories: Additional categories that may apply.
        confidence: Classification confidence (0.0-1.0).
        amount: Original monetary amount.
        currency: Currency code.
        description: Item or transaction description.
        source_type: Data source type.
        incoterm: Incoterm for transport records.
        is_capital: Capital expenditure flag.
        capitalization_threshold: Organization's capex threshold.
        is_routine: Routine travel flag (for Cat 6 vs Cat 7).
        product_lifecycle_stage: Lifecycle stage (processing, use, end_of_life).
        agreement_type: Agreement type (franchise, investment, both).
        consolidation_approach: GHG Protocol consolidation approach.
        lease_classification: Lease type classification.
        is_scope2_reported: Whether already reported in Scope 2.
        freight_separately_invoiced: Whether freight is on separate invoice.
        metadata: Additional metadata.
    """

    model_config = ConfigDict(frozen=True)

    record_id: str = Field(..., min_length=1, description="Record ID")
    primary_category: Scope3Category = Field(..., description="Primary category")
    secondary_categories: List[Scope3Category] = Field(
        default_factory=list, description="Secondary categories"
    )
    confidence: Decimal = Field(
        default=_ZERO, ge=_ZERO, le=_ONE, description="Confidence"
    )
    amount: Optional[Decimal] = Field(default=None, description="Amount")
    currency: str = Field(default="USD", description="Currency")
    description: str = Field(default="", description="Description")
    source_type: str = Field(default="spend", description="Source type")
    incoterm: Optional[IncotermsRule] = Field(default=None, description="Incoterm")
    is_capital: Optional[bool] = Field(default=None, description="Is capital")
    capitalization_threshold: Optional[Decimal] = Field(
        default=None, description="Capex threshold"
    )
    is_routine: Optional[bool] = Field(
        default=None, description="Routine travel flag"
    )
    product_lifecycle_stage: Optional[str] = Field(
        default=None, description="processing, use, end_of_life"
    )
    agreement_type: Optional[str] = Field(
        default=None, description="franchise, investment, both"
    )
    consolidation_approach: Optional[ConsolidationApproach] = Field(
        default=None, description="Consolidation approach"
    )
    lease_classification: Optional[LeaseClassification] = Field(
        default=None, description="Lease classification"
    )
    is_scope2_reported: Optional[bool] = Field(
        default=None, description="Already in Scope 2"
    )
    freight_separately_invoiced: Optional[bool] = Field(
        default=None, description="Freight on separate invoice"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Metadata"
    )


# =============================================================================
# PROVENANCE HELPER
# =============================================================================


def _calculate_provenance_hash(*parts: Any) -> str:
    """Calculate SHA-256 provenance hash from arbitrary parts.

    Args:
        *parts: Values to include in the hash.

    Returns:
        Hexadecimal SHA-256 hash string (64 characters).
    """
    hasher = hashlib.sha256()
    for part in parts:
        if isinstance(part, BaseModel):
            serialized = part.model_dump_json(exclude_none=False)
        elif isinstance(part, Decimal):
            serialized = str(part)
        elif isinstance(part, datetime):
            serialized = part.isoformat()
        elif isinstance(part, Enum):
            serialized = part.value
        elif isinstance(part, (dict, list)):
            serialized = json.dumps(part, sort_keys=True, default=str)
        else:
            serialized = str(part)
        hasher.update(serialized.encode("utf-8"))
    return hasher.hexdigest()


# =============================================================================
# ENGINE CLASS
# =============================================================================


class BoundaryDeterminerEngine:
    """
    Thread-safe singleton engine for Scope 3 category boundary determination.

    Implements the 10 double-counting prevention rules (DC-SCM-001 through
    DC-SCM-010) to resolve category boundary ambiguities. Each rule is a
    deterministic decision based on explicit input parameters (capitalization
    thresholds, Incoterms, consolidation approach, agreement type, etc.).

    This engine is ZERO-HALLUCINATION: all boundary determinations use
    hardcoded rules. No LLM or ML models are involved.

    Thread Safety:
        Uses __new__ singleton pattern with threading.Lock. The determination
        counter is protected by a dedicated lock.

    Attributes:
        _determination_count: Total determinations performed.

    Example:
        >>> engine = BoundaryDeterminerEngine()
        >>> cat = engine.resolve_cat1_vs_cat2(
        ...     amount=Decimal("12000"),
        ...     capitalization_threshold=Decimal("5000"),
        ... )
        >>> cat
        <Scope3Category.CAT_2_CAPITAL_GOODS: 'cat_2_capital_goods'>
    """

    _instance: Optional["BoundaryDeterminerEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "BoundaryDeterminerEngine":
        """Thread-safe singleton instantiation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the boundary determiner engine (only once)."""
        if hasattr(self, "_initialized"):
            return

        self._initialized: bool = True
        self._determination_count: int = 0
        self._count_lock: threading.Lock = threading.Lock()

        logger.info(
            "BoundaryDeterminerEngine initialized: agent_id=%s, "
            "dc_rules=%d, engine_version=%s",
            AGENT_ID,
            len(_DC_RULE_DETAILS),
            ENGINE_VERSION,
        )

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _increment_determination_count(self) -> int:
        """Increment determination counter in a thread-safe manner.

        Returns:
            Updated total determination count.
        """
        with self._count_lock:
            self._determination_count += 1
            return self._determination_count

    # =========================================================================
    # DC-SCM-001: Cat 1 vs Cat 2 (Opex vs Capex)
    # =========================================================================

    def resolve_cat1_vs_cat2(
        self,
        amount: Decimal,
        capitalization_threshold: Decimal = DEFAULT_CAPITALIZATION_THRESHOLD,
    ) -> Scope3Category:
        """Resolve Cat 1 (Purchased Goods) vs Cat 2 (Capital Goods).

        Applies the organization's capitalization policy threshold.
        Items at or above the threshold are capital goods (Cat 2);
        items below are purchased goods and services (Cat 1).

        Rule: DC-SCM-001

        Args:
            amount: Monetary amount of the purchase.
            capitalization_threshold: Organization's capitalization threshold.
                Defaults to $5,000.

        Returns:
            Scope3Category.CAT_2_CAPITAL_GOODS if amount >= threshold,
            Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES otherwise.

        Example:
            >>> engine = BoundaryDeterminerEngine()
            >>> engine.resolve_cat1_vs_cat2(Decimal("3000"))
            <Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES: ...>
            >>> engine.resolve_cat1_vs_cat2(Decimal("10000"))
            <Scope3Category.CAT_2_CAPITAL_GOODS: ...>
        """
        if amount >= capitalization_threshold:
            result = Scope3Category.CAT_2_CAPITAL_GOODS
            rationale = (
                f"Amount {amount} >= threshold {capitalization_threshold}: "
                f"classified as Cat 2 (Capital Goods)"
            )
        else:
            result = Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES
            rationale = (
                f"Amount {amount} < threshold {capitalization_threshold}: "
                f"classified as Cat 1 (Purchased Goods)"
            )

        logger.debug(
            "DC-SCM-001 resolved: amount=%s, threshold=%s -> %s",
            amount, capitalization_threshold, result.value,
        )

        return result

    # =========================================================================
    # DC-SCM-004: Cat 4 vs Cat 9 (Upstream vs Downstream Transport)
    # =========================================================================

    def resolve_cat4_vs_cat9(
        self,
        incoterm: IncotermsRule,
        direction: str = "buyer",
    ) -> Scope3Category:
        """Resolve Cat 4 (Upstream Transport) vs Cat 9 (Downstream Transport).

        The Incoterm determines who pays for freight. From the BUYER's
        perspective, freight they pay for is Cat 4 (upstream). From the
        SELLER's perspective, freight they pay for is Cat 9 (downstream).

        Rule: DC-SCM-004

        Args:
            incoterm: ICC Incoterm determining transport responsibility.
            direction: Reporter's role -- "buyer" or "seller".

        Returns:
            Scope3Category for the transport boundary.

        Raises:
            ValueError: If direction is not "buyer" or "seller".

        Example:
            >>> engine = BoundaryDeterminerEngine()
            >>> engine.resolve_cat4_vs_cat9(IncotermsRule.FOB, "buyer")
            <Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION: ...>
            >>> engine.resolve_cat4_vs_cat9(IncotermsRule.CIF, "seller")
            <Scope3Category.CAT_9_DOWNSTREAM_TRANSPORTATION: ...>
        """
        direction_lower = direction.lower().strip()
        if direction_lower not in ("buyer", "seller"):
            raise ValueError(
                f"Direction must be 'buyer' or 'seller', got '{direction}'"
            )

        if direction_lower == "buyer":
            # Buyer perspective: buyer pays freight => Cat 4
            if incoterm in _BUYER_PAYS_FREIGHT:
                result = Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION
            else:
                # Seller pays freight; from buyer's view, freight is in goods price
                # so it stays in Cat 1 (no separate Cat 4 for buyer)
                result = Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES
        else:
            # Seller perspective: seller pays freight => Cat 9
            if incoterm in _SELLER_PAYS_FREIGHT:
                result = Scope3Category.CAT_9_DOWNSTREAM_TRANSPORTATION
            else:
                # Buyer pays freight; from seller's view, no downstream transport
                # cost to report
                result = Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION

        logger.debug(
            "DC-SCM-004 resolved: incoterm=%s, direction=%s -> %s",
            incoterm.value, direction_lower, result.value,
        )

        return result

    # =========================================================================
    # DC-SCM-005: Cat 6 vs Cat 7 (Travel vs Commuting)
    # =========================================================================

    def resolve_cat6_vs_cat7(
        self,
        travel_type: str,
        is_routine: bool,
    ) -> Scope3Category:
        """Resolve Cat 6 (Business Travel) vs Cat 7 (Employee Commuting).

        Non-routine journeys for work purposes are Cat 6 (business travel).
        Routine home-to-office travel is Cat 7 (employee commuting).
        On business travel days, commuting is excluded to prevent double-counting.

        Rule: DC-SCM-005

        Args:
            travel_type: Type of travel (e.g., "flight", "train", "car",
                "commute", "work_from_home").
            is_routine: Whether this is routine daily travel (True = commuting).

        Returns:
            Scope3Category for the travel type.

        Example:
            >>> engine = BoundaryDeterminerEngine()
            >>> engine.resolve_cat6_vs_cat7("flight", is_routine=False)
            <Scope3Category.CAT_6_BUSINESS_TRAVEL: ...>
            >>> engine.resolve_cat6_vs_cat7("car", is_routine=True)
            <Scope3Category.CAT_7_EMPLOYEE_COMMUTING: ...>
        """
        if is_routine:
            result = Scope3Category.CAT_7_EMPLOYEE_COMMUTING
        else:
            result = Scope3Category.CAT_6_BUSINESS_TRAVEL

        logger.debug(
            "DC-SCM-005 resolved: travel_type=%s, is_routine=%s -> %s",
            travel_type, is_routine, result.value,
        )

        return result

    # =========================================================================
    # DC-SCM-006: Cat 8 vs Scope 1/2 (Leased Assets -- Lessee)
    # =========================================================================

    def resolve_cat8_scope(
        self,
        consolidation: ConsolidationApproach,
        lease_type: LeaseClassification,
    ) -> str:
        """Resolve Cat 8 (Upstream Leased Assets) vs Scope 1/2.

        Under operational control: lessee has operational control over
        leased asset, so emissions go to Scope 1/2 (not Cat 8).

        Under financial control: operating leases may be Cat 8;
        finance leases go to Scope 1/2.

        Under equity share: proportional allocation; typically Cat 8 for
        non-consolidated portions.

        Rule: DC-SCM-006

        Args:
            consolidation: GHG Protocol consolidation approach.
            lease_type: Lease classification (operating, finance, short-term).

        Returns:
            "scope_1_2" if included in Scope 1/2,
            "cat_8" if reported as Cat 8 (Upstream Leased Assets).

        Example:
            >>> engine = BoundaryDeterminerEngine()
            >>> engine.resolve_cat8_scope(
            ...     ConsolidationApproach.OPERATIONAL_CONTROL,
            ...     LeaseClassification.OPERATING_LEASE,
            ... )
            'scope_1_2'
        """
        if consolidation == ConsolidationApproach.OPERATIONAL_CONTROL:
            # Lessee has operational control => Scope 1/2
            result = "scope_1_2"
            rationale = (
                "Operational control: lessee has operational control "
                "over leased asset; emissions included in Scope 1/2"
            )

        elif consolidation == ConsolidationApproach.FINANCIAL_CONTROL:
            if lease_type == LeaseClassification.FINANCE_LEASE:
                # Finance lease = on balance sheet => Scope 1/2
                result = "scope_1_2"
                rationale = (
                    "Financial control + finance lease: asset is on "
                    "balance sheet; emissions included in Scope 1/2"
                )
            elif lease_type in (
                LeaseClassification.SHORT_TERM,
                LeaseClassification.LOW_VALUE,
            ):
                # Short-term/low-value exemptions => Cat 8
                result = "cat_8"
                rationale = (
                    f"Financial control + {lease_type.value}: exempt "
                    f"from on-balance-sheet treatment; report as Cat 8"
                )
            else:
                # Operating lease under financial control => Cat 8
                result = "cat_8"
                rationale = (
                    "Financial control + operating lease: not on "
                    "balance sheet; report as Cat 8"
                )

        elif consolidation == ConsolidationApproach.EQUITY_SHARE:
            # Equity share: only the non-consolidated portion goes to Cat 8
            result = "cat_8"
            rationale = (
                "Equity share: non-consolidated portion of leased "
                "asset emissions reported as Cat 8"
            )

        else:
            result = "cat_8"
            rationale = (
                f"Unknown consolidation approach '{consolidation}'; "
                f"defaulting to Cat 8"
            )

        logger.debug(
            "DC-SCM-006 resolved: consolidation=%s, lease_type=%s -> %s",
            consolidation.value, lease_type.value, result,
        )

        return result

    # =========================================================================
    # DC-SCM-009: Cat 13 vs Scope 1/2 (Downstream Leased Assets -- Lessor)
    # =========================================================================

    def resolve_cat13_scope(
        self,
        consolidation: ConsolidationApproach,
    ) -> str:
        """Resolve Cat 13 (Downstream Leased Assets) vs Scope 1/2.

        The lessor reports leased-out asset emissions in Cat 13 ONLY if
        the asset is NOT already consolidated in the lessor's Scope 1/2.

        Under operational control: if lessor does NOT have operational
        control (tenant operates), report as Cat 13.

        Under financial control: if asset is off balance sheet for
        lessor, report as Cat 13.

        Under equity share: non-consolidated portion goes to Cat 13.

        Rule: DC-SCM-009

        Args:
            consolidation: GHG Protocol consolidation approach.

        Returns:
            "scope_1_2" if included in lessor's Scope 1/2,
            "cat_13" if reported as Cat 13 (Downstream Leased Assets).

        Example:
            >>> engine = BoundaryDeterminerEngine()
            >>> engine.resolve_cat13_scope(
            ...     ConsolidationApproach.OPERATIONAL_CONTROL,
            ... )
            'cat_13'
        """
        if consolidation == ConsolidationApproach.OPERATIONAL_CONTROL:
            # Lessor does NOT operate the leased asset => Cat 13
            result = "cat_13"
            rationale = (
                "Operational control: lessor does not have operational "
                "control over leased-out asset; report as Cat 13"
            )

        elif consolidation == ConsolidationApproach.FINANCIAL_CONTROL:
            # Under financial control, if asset is an operating lease
            # for the lessor (off balance sheet for the lessor's boundary),
            # it's Cat 13
            result = "cat_13"
            rationale = (
                "Financial control: leased-out asset typically not "
                "consolidated by lessor; report as Cat 13"
            )

        elif consolidation == ConsolidationApproach.EQUITY_SHARE:
            # Non-consolidated portion
            result = "cat_13"
            rationale = (
                "Equity share: non-consolidated portion of downstream "
                "leased asset reported as Cat 13"
            )

        else:
            result = "cat_13"
            rationale = (
                f"Unknown consolidation approach; defaulting to Cat 13"
            )

        logger.debug(
            "DC-SCM-009 resolved: consolidation=%s -> %s",
            consolidation.value, result,
        )

        return result

    # =========================================================================
    # DC-SCM-010: Cat 14 vs Cat 15 (Franchise vs Investment)
    # =========================================================================

    def resolve_cat14_vs_cat15(
        self,
        agreement_type: str,
    ) -> Scope3Category:
        """Resolve Cat 14 (Franchises) vs Cat 15 (Investments).

        If a franchise agreement exists, franchise-related emissions go
        to Cat 14. If only an equity/debt investment exists, emissions go
        to Cat 15. When both relationships exist, the franchise agreement
        takes precedence for franchise-related emissions.

        Rule: DC-SCM-010

        Args:
            agreement_type: Type of relationship -- "franchise", "investment",
                or "both".

        Returns:
            Scope3Category for the relationship type.

        Raises:
            ValueError: If agreement_type is not recognized.

        Example:
            >>> engine = BoundaryDeterminerEngine()
            >>> engine.resolve_cat14_vs_cat15("franchise")
            <Scope3Category.CAT_14_FRANCHISES: ...>
            >>> engine.resolve_cat14_vs_cat15("investment")
            <Scope3Category.CAT_15_INVESTMENTS: ...>
        """
        agreement_lower = agreement_type.lower().strip()

        if agreement_lower == "franchise":
            result = Scope3Category.CAT_14_FRANCHISES
        elif agreement_lower == "investment":
            result = Scope3Category.CAT_15_INVESTMENTS
        elif agreement_lower == "both":
            # Franchise takes precedence per GHG Protocol guidance
            result = Scope3Category.CAT_14_FRANCHISES
        else:
            raise ValueError(
                f"Agreement type must be 'franchise', 'investment', or "
                f"'both', got '{agreement_type}'"
            )

        logger.debug(
            "DC-SCM-010 resolved: agreement_type=%s -> %s",
            agreement_lower, result.value,
        )

        return result

    # =========================================================================
    # DC-SCM-002: Incoterm Split (Cat 1 goods vs Cat 4 freight)
    # =========================================================================

    def apply_incoterm_split(
        self,
        record: Dict[str, Any],
        incoterm: IncotermsRule,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Split a purchase record into goods and freight portions.

        Under buyer-pays-freight Incoterms (EXW, FCA, FAS, FOB), the freight
        cost should be separated and reported in Cat 4. The goods portion
        stays in Cat 1.

        Under seller-pays-freight Incoterms (CIF, DDP, etc.), freight is
        included in the goods price, so the entire amount stays in Cat 1.

        If freight is separately invoiced (indicated by record metadata),
        the freight portion always goes to Cat 4 regardless of Incoterm.

        Rule: DC-SCM-002

        Args:
            record: Dictionary with at minimum "amount" and optionally
                "freight_amount", "goods_amount", "freight_separately_invoiced".
            incoterm: ICC Incoterm for this purchase.

        Returns:
            Tuple of (goods_portion, freight_portion) dictionaries.
            The freight_portion will have amount=0 if freight is included
            in the goods price.

        Example:
            >>> engine = BoundaryDeterminerEngine()
            >>> goods, freight = engine.apply_incoterm_split(
            ...     {"amount": 10000, "freight_amount": 1500},
            ...     IncotermsRule.FOB,
            ... )
            >>> goods["amount"]
            8500
            >>> freight["amount"]
            1500
        """
        total_amount = Decimal(str(record.get("amount", 0)))
        freight_amount = Decimal(str(record.get("freight_amount", 0)))
        is_freight_separate = record.get("freight_separately_invoiced", False)

        goods_portion = dict(record)
        freight_portion = dict(record)

        # If freight is separately invoiced, always split
        if is_freight_separate and freight_amount > _ZERO:
            goods_amount = total_amount - freight_amount
            if goods_amount < _ZERO:
                goods_amount = _ZERO

            goods_portion["amount"] = goods_amount
            goods_portion["category"] = Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES.value
            goods_portion["split_type"] = "goods"

            freight_portion["amount"] = freight_amount
            freight_portion["category"] = Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION.value
            freight_portion["split_type"] = "freight"

            logger.debug(
                "Incoterm split (separate invoice): goods=%s, freight=%s",
                goods_amount, freight_amount,
            )
            return goods_portion, freight_portion

        # Apply Incoterm-based split
        if incoterm in _BUYER_PAYS_FREIGHT:
            # Buyer pays freight: split if freight amount is known
            if freight_amount > _ZERO:
                goods_amount = total_amount - freight_amount
                if goods_amount < _ZERO:
                    goods_amount = _ZERO

                goods_portion["amount"] = goods_amount
                goods_portion["category"] = Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES.value
                goods_portion["split_type"] = "goods"

                freight_portion["amount"] = freight_amount
                freight_portion["category"] = Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION.value
                freight_portion["split_type"] = "freight"
            else:
                # No freight amount known; entire amount stays in Cat 1
                goods_portion["category"] = Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES.value
                goods_portion["split_type"] = "goods_only"

                freight_portion["amount"] = _ZERO
                freight_portion["category"] = Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION.value
                freight_portion["split_type"] = "freight_unknown"
        else:
            # Seller pays freight: freight included in goods price (Cat 1 only)
            goods_portion["amount"] = total_amount
            goods_portion["category"] = Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES.value
            goods_portion["split_type"] = "goods_inclusive"

            freight_portion["amount"] = _ZERO
            freight_portion["category"] = Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION.value
            freight_portion["split_type"] = "freight_included_in_goods"

        logger.debug(
            "Incoterm split: incoterm=%s, goods=%s, freight=%s",
            incoterm.value,
            goods_portion.get("amount"),
            freight_portion.get("amount"),
        )

        return goods_portion, freight_portion

    # =========================================================================
    # COMPREHENSIVE BOUNDARY DETERMINATION
    # =========================================================================

    def determine_boundary(
        self,
        record: Dict[str, Any],
        source_type: DataSourceType,
        organization_context: Dict[str, Any],
    ) -> BoundaryDetermination:
        """Determine the upstream/downstream boundary for a record.

        Applies the relevant DC-SCM rules based on the record's data source
        type and organization context (consolidation approach, capitalization
        policy, etc.).

        Args:
            record: Dictionary with record data including at minimum
                "record_id" and "primary_category".
            source_type: Type of data source.
            organization_context: Dictionary with organization settings
                including "consolidation_approach", "capitalization_threshold",
                and other boundary-relevant parameters.

        Returns:
            BoundaryDetermination with final category, rules applied,
            and provenance hash.

        Example:
            >>> engine = BoundaryDeterminerEngine()
            >>> result = engine.determine_boundary(
            ...     record={"record_id": "R-001", "primary_category":
            ...         "cat_1_purchased_goods", "amount": "12000"},
            ...     source_type=DataSourceType.SPEND,
            ...     organization_context={
            ...         "capitalization_threshold": "5000",
            ...         "consolidation_approach": "operational_control",
            ...     },
            ... )
            >>> result.determined_category
            <Scope3Category.CAT_2_CAPITAL_GOODS: ...>
        """
        start_time = time.monotonic()
        record_id = str(record.get("record_id", "UNKNOWN"))

        # Parse the primary category from string or enum
        primary_cat_raw = record.get("primary_category", "")
        try:
            if isinstance(primary_cat_raw, Scope3Category):
                primary_category = primary_cat_raw
            else:
                primary_category = Scope3Category(str(primary_cat_raw))
        except ValueError:
            logger.warning(
                "Cannot parse primary_category '%s' for record %s",
                primary_cat_raw, record_id,
            )
            primary_category = Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES

        dc_rules_applied: List[DoubleCountingRule] = []
        determined_category = primary_category
        rationale_parts: List[str] = []
        split_required = False
        split_allocations: List[CategoryAllocation] = []
        scope_destination = "scope_3"
        incoterm_applied: Optional[IncotermsRule] = None
        capitalization_applied = False
        consolidation_approach: Optional[ConsolidationApproach] = None

        # Extract organization context
        cap_threshold = Decimal(
            str(organization_context.get(
                "capitalization_threshold",
                DEFAULT_CAPITALIZATION_THRESHOLD,
            ))
        )
        consolidation_str = organization_context.get(
            "consolidation_approach", ""
        )
        if consolidation_str:
            try:
                consolidation_approach = ConsolidationApproach(consolidation_str)
            except ValueError:
                consolidation_approach = None

        amount = Decimal(str(record.get("amount", 0))) if record.get("amount") else _ZERO

        # -----------------------------------------------------------------
        # Rule DC-SCM-001: Cat 1 vs Cat 2 (Opex vs Capex)
        # -----------------------------------------------------------------
        if primary_category in (
            Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES,
            Scope3Category.CAT_2_CAPITAL_GOODS,
        ):
            is_capital_flag = record.get("is_capital")
            if is_capital_flag is True:
                determined_category = Scope3Category.CAT_2_CAPITAL_GOODS
                rationale_parts.append("Flagged as capital expenditure -> Cat 2")
                dc_rules_applied.append(DoubleCountingRule.DC_SCM_001)
                capitalization_applied = True
            elif is_capital_flag is False:
                determined_category = Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES
                rationale_parts.append("Flagged as operating expenditure -> Cat 1")
                dc_rules_applied.append(DoubleCountingRule.DC_SCM_001)
                capitalization_applied = True
            elif amount > _ZERO:
                resolved = self.resolve_cat1_vs_cat2(amount, cap_threshold)
                determined_category = resolved
                rationale_parts.append(
                    f"Capitalization threshold ({cap_threshold}): "
                    f"amount {amount} -> {resolved.value}"
                )
                dc_rules_applied.append(DoubleCountingRule.DC_SCM_001)
                capitalization_applied = True

        # -----------------------------------------------------------------
        # Rule DC-SCM-002: Cat 1 vs Cat 4 (Goods vs Freight)
        # -----------------------------------------------------------------
        if (primary_category == Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES
                and source_type in (DataSourceType.SPEND, DataSourceType.PURCHASE_ORDER)):
            incoterm_raw = record.get("incoterm")
            freight_separate = record.get("freight_separately_invoiced", False)

            if incoterm_raw or freight_separate:
                parsed_incoterm: Optional[IncotermsRule] = None
                if incoterm_raw:
                    try:
                        if isinstance(incoterm_raw, IncotermsRule):
                            parsed_incoterm = incoterm_raw
                        else:
                            parsed_incoterm = IncotermsRule(str(incoterm_raw).lower())
                    except ValueError:
                        parsed_incoterm = None

                if parsed_incoterm and parsed_incoterm in _BUYER_PAYS_FREIGHT:
                    freight_amt = Decimal(str(record.get("freight_amount", 0)))
                    if freight_amt > _ZERO or freight_separate:
                        split_required = True
                        goods_pct = (
                            ((amount - freight_amt) / amount * _HUNDRED)
                            if amount > _ZERO and freight_amt > _ZERO
                            else Decimal("80")
                        ).quantize(_QUANT_2DP, rounding=ROUND_HALF_UP)
                        freight_pct = (_HUNDRED - goods_pct).quantize(
                            _QUANT_2DP, rounding=ROUND_HALF_UP
                        )

                        split_allocations = [
                            CategoryAllocation(
                                category=Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES,
                                allocation_pct=goods_pct,
                                amount_allocated=(
                                    amount - freight_amt if freight_amt > _ZERO
                                    else None
                                ),
                                rationale="Goods portion",
                            ),
                            CategoryAllocation(
                                category=Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION,
                                allocation_pct=freight_pct,
                                amount_allocated=(
                                    freight_amt if freight_amt > _ZERO else None
                                ),
                                rationale="Freight portion",
                            ),
                        ]
                        rationale_parts.append(
                            f"Incoterm {parsed_incoterm.value}: buyer pays "
                            f"freight -> split Cat 1 ({goods_pct}%) / "
                            f"Cat 4 ({freight_pct}%)"
                        )
                        dc_rules_applied.append(DoubleCountingRule.DC_SCM_002)
                        incoterm_applied = parsed_incoterm

                elif freight_separate:
                    split_required = True
                    rationale_parts.append(
                        "Freight separately invoiced -> Cat 4 for freight"
                    )
                    dc_rules_applied.append(DoubleCountingRule.DC_SCM_002)

        # -----------------------------------------------------------------
        # Rule DC-SCM-003: Cat 3 vs Scope 2 (WTT/T&D exclusion)
        # -----------------------------------------------------------------
        if primary_category == Scope3Category.CAT_3_FUEL_ENERGY_ACTIVITIES:
            is_scope2 = record.get("is_scope2_reported", False)
            if is_scope2:
                scope_destination = "scope_2"
                rationale_parts.append(
                    "Energy already reported in Scope 2 -> excluded from Cat 3"
                )
                dc_rules_applied.append(DoubleCountingRule.DC_SCM_003)
            else:
                rationale_parts.append(
                    "WTT/T&D not in Scope 2 -> included in Cat 3"
                )
                dc_rules_applied.append(DoubleCountingRule.DC_SCM_003)

        # -----------------------------------------------------------------
        # Rule DC-SCM-005: Cat 6 vs Cat 7 (Travel vs Commuting)
        # -----------------------------------------------------------------
        if primary_category in (
            Scope3Category.CAT_6_BUSINESS_TRAVEL,
            Scope3Category.CAT_7_EMPLOYEE_COMMUTING,
        ):
            is_routine = record.get("is_routine")
            travel_type = record.get("travel_type", record.get("description", ""))
            if is_routine is not None:
                resolved = self.resolve_cat6_vs_cat7(
                    str(travel_type), bool(is_routine)
                )
                determined_category = resolved
                rationale_parts.append(
                    f"Routine={is_routine}: "
                    f"{'commuting (Cat 7)' if is_routine else 'travel (Cat 6)'}"
                )
                dc_rules_applied.append(DoubleCountingRule.DC_SCM_005)

        # -----------------------------------------------------------------
        # Rule DC-SCM-006: Cat 8 vs Scope 1/2 (Upstream Leased Assets)
        # -----------------------------------------------------------------
        if primary_category == Scope3Category.CAT_8_UPSTREAM_LEASED_ASSETS:
            if consolidation_approach:
                lease_type_raw = record.get("lease_classification", "operating_lease")
                try:
                    if isinstance(lease_type_raw, LeaseClassification):
                        lease_type = lease_type_raw
                    else:
                        lease_type = LeaseClassification(str(lease_type_raw))
                except ValueError:
                    lease_type = LeaseClassification.OPERATING_LEASE

                scope_result = self.resolve_cat8_scope(
                    consolidation_approach, lease_type
                )
                scope_destination = scope_result
                if scope_result == "scope_1_2":
                    rationale_parts.append(
                        f"Consolidation={consolidation_approach.value}, "
                        f"lease={lease_type.value}: in Scope 1/2"
                    )
                else:
                    rationale_parts.append(
                        f"Consolidation={consolidation_approach.value}, "
                        f"lease={lease_type.value}: report as Cat 8"
                    )
                dc_rules_applied.append(DoubleCountingRule.DC_SCM_006)

        # -----------------------------------------------------------------
        # Rule DC-SCM-007: Cat 10 vs Cat 11 (Processing vs Use)
        # -----------------------------------------------------------------
        if primary_category in (
            Scope3Category.CAT_10_PROCESSING_SOLD_PRODUCTS,
            Scope3Category.CAT_11_USE_OF_SOLD_PRODUCTS,
        ):
            lifecycle_stage = record.get("product_lifecycle_stage", "")
            if lifecycle_stage == "processing":
                determined_category = Scope3Category.CAT_10_PROCESSING_SOLD_PRODUCTS
                rationale_parts.append("Lifecycle=processing -> Cat 10")
            elif lifecycle_stage == "use":
                determined_category = Scope3Category.CAT_11_USE_OF_SOLD_PRODUCTS
                rationale_parts.append("Lifecycle=use -> Cat 11")
            dc_rules_applied.append(DoubleCountingRule.DC_SCM_007)

        # -----------------------------------------------------------------
        # Rule DC-SCM-008: Cat 11 vs Cat 12 (Use vs End-of-Life)
        # -----------------------------------------------------------------
        if primary_category in (
            Scope3Category.CAT_11_USE_OF_SOLD_PRODUCTS,
            Scope3Category.CAT_12_END_OF_LIFE_TREATMENT,
        ):
            lifecycle_stage = record.get("product_lifecycle_stage", "")
            if lifecycle_stage == "end_of_life":
                determined_category = Scope3Category.CAT_12_END_OF_LIFE_TREATMENT
                rationale_parts.append("Lifecycle=end_of_life -> Cat 12")
            elif lifecycle_stage == "use":
                determined_category = Scope3Category.CAT_11_USE_OF_SOLD_PRODUCTS
                rationale_parts.append("Lifecycle=use -> Cat 11")
            dc_rules_applied.append(DoubleCountingRule.DC_SCM_008)

        # -----------------------------------------------------------------
        # Rule DC-SCM-009: Cat 13 vs Scope 1/2 (Downstream Leased Assets)
        # -----------------------------------------------------------------
        if primary_category == Scope3Category.CAT_13_DOWNSTREAM_LEASED_ASSETS:
            if consolidation_approach:
                scope_result = self.resolve_cat13_scope(consolidation_approach)
                scope_destination = scope_result
                rationale_parts.append(
                    f"Consolidation={consolidation_approach.value}: "
                    f"lessor reports as {scope_result}"
                )
                dc_rules_applied.append(DoubleCountingRule.DC_SCM_009)

        # -----------------------------------------------------------------
        # Rule DC-SCM-010: Cat 14 vs Cat 15 (Franchise vs Investment)
        # -----------------------------------------------------------------
        if primary_category in (
            Scope3Category.CAT_14_FRANCHISES,
            Scope3Category.CAT_15_INVESTMENTS,
        ):
            agreement = record.get("agreement_type", "")
            if agreement:
                try:
                    resolved = self.resolve_cat14_vs_cat15(agreement)
                    determined_category = resolved
                    rationale_parts.append(
                        f"Agreement type={agreement}: -> {resolved.value}"
                    )
                    dc_rules_applied.append(DoubleCountingRule.DC_SCM_010)
                except ValueError as e:
                    rationale_parts.append(f"Invalid agreement type: {e}")

        # Build value chain position
        value_chain = _CATEGORY_VALUE_CHAIN.get(
            determined_category, ValueChainPosition.UPSTREAM
        )

        # Build rationale
        boundary_rationale = "; ".join(rationale_parts) if rationale_parts else (
            f"No boundary rules applicable for {determined_category.value}"
        )

        # Provenance hash
        self._increment_determination_count()
        elapsed_ms = (time.monotonic() - start_time) * 1000

        provenance_hash = _calculate_provenance_hash(
            record_id, determined_category.value, dc_rules_applied,
            boundary_rationale, scope_destination,
        )

        now_iso = datetime.now(timezone.utc).isoformat()

        determination = BoundaryDetermination(
            record_id=record_id,
            determined_category=determined_category,
            original_category=(
                primary_category if primary_category != determined_category
                else None
            ),
            value_chain_position=value_chain,
            consolidation_approach=consolidation_approach,
            dc_rules_applied=dc_rules_applied,
            boundary_rationale=boundary_rationale,
            split_required=split_required,
            split_allocations=split_allocations,
            scope_destination=scope_destination,
            incoterm_applied=incoterm_applied,
            capitalization_applied=capitalization_applied,
            provenance_hash=provenance_hash,
            determined_at=now_iso,
        )

        logger.info(
            "Boundary determined: record_id=%s, original=%s, "
            "determined=%s, dc_rules=%s, elapsed_ms=%.2f",
            record_id, primary_category.value,
            determined_category.value,
            [r.value for r in dc_rules_applied],
            elapsed_ms,
        )

        return determination

    # =========================================================================
    # DOUBLE-COUNTING CHECK (batch)
    # =========================================================================

    def check_double_counting(
        self,
        results: List[ClassificationResult],
    ) -> List[DoubleCountingCheck]:
        """Check for potential double-counting across classified records.

        Evaluates all 10 DC-SCM rules against the set of classified records
        and identifies any records that may be counted in multiple categories.

        Args:
            results: List of classification results to check.

        Returns:
            List of DoubleCountingCheck results, one per applicable rule.

        Example:
            >>> engine = BoundaryDeterminerEngine()
            >>> checks = engine.check_double_counting(results)
            >>> for check in checks:
            ...     if check.overlap_detected:
            ...         print(f"OVERLAP: {check.rule_id.value}")
        """
        checks: List[DoubleCountingCheck] = []

        # Group records by category
        by_category: Dict[Scope3Category, List[ClassificationResult]] = {}
        for r in results:
            cat = r.primary_category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(r)

        # DC-SCM-001: Cat 1 vs Cat 2
        checks.append(self._check_cat1_vs_cat2(by_category))

        # DC-SCM-002: Cat 1 vs Cat 4
        checks.append(self._check_cat1_vs_cat4(by_category))

        # DC-SCM-003: Cat 3 vs Scope 2
        checks.append(self._check_cat3_scope2(by_category))

        # DC-SCM-004: Cat 4 vs Cat 9
        checks.append(self._check_cat4_vs_cat9(by_category))

        # DC-SCM-005: Cat 6 vs Cat 7
        checks.append(self._check_cat6_vs_cat7(by_category))

        # DC-SCM-006: Cat 8 vs Scope 1/2
        checks.append(self._check_cat8_scope(by_category))

        # DC-SCM-007: Cat 10 vs Cat 11
        checks.append(self._check_cat10_vs_cat11(by_category))

        # DC-SCM-008: Cat 11 vs Cat 12
        checks.append(self._check_cat11_vs_cat12(by_category))

        # DC-SCM-009: Cat 13 vs Scope 1/2
        checks.append(self._check_cat13_scope(by_category))

        # DC-SCM-010: Cat 14 vs Cat 15
        checks.append(self._check_cat14_vs_cat15(by_category))

        overlap_count = sum(1 for c in checks if c.overlap_detected)
        logger.info(
            "Double-counting check complete: %d rules checked, "
            "%d overlaps detected",
            len(checks), overlap_count,
        )

        return checks

    def _check_cat1_vs_cat2(
        self,
        by_category: Dict[Scope3Category, List[ClassificationResult]],
    ) -> DoubleCountingCheck:
        """Check DC-SCM-001: Cat 1 vs Cat 2 overlap."""
        cat1 = by_category.get(Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES, [])
        cat2 = by_category.get(Scope3Category.CAT_2_CAPITAL_GOODS, [])

        # Check for same vendor/description appearing in both
        cat1_ids = {r.record_id for r in cat1}
        cat2_ids = {r.record_id for r in cat2}
        overlap_ids = cat1_ids & cat2_ids

        return DoubleCountingCheck(
            rule_id=DoubleCountingRule.DC_SCM_001,
            categories_checked=[
                Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES,
                Scope3Category.CAT_2_CAPITAL_GOODS,
            ],
            overlap_detected=len(overlap_ids) > 0,
            records_affected=len(overlap_ids),
            record_ids=sorted(overlap_ids),
            resolution=(
                "Apply capitalization threshold to split records"
                if overlap_ids else "No overlap detected"
            ),
            provenance_hash=_calculate_provenance_hash(
                "DC-SCM-001", len(cat1), len(cat2), sorted(overlap_ids),
            ),
        )

    def _check_cat1_vs_cat4(
        self,
        by_category: Dict[Scope3Category, List[ClassificationResult]],
    ) -> DoubleCountingCheck:
        """Check DC-SCM-002: Cat 1 vs Cat 4 overlap."""
        cat1 = by_category.get(Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES, [])
        cat4 = by_category.get(Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION, [])

        cat1_ids = {r.record_id for r in cat1}
        cat4_ids = {r.record_id for r in cat4}
        overlap_ids = cat1_ids & cat4_ids

        return DoubleCountingCheck(
            rule_id=DoubleCountingRule.DC_SCM_002,
            categories_checked=[
                Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES,
                Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION,
            ],
            overlap_detected=len(overlap_ids) > 0,
            records_affected=len(overlap_ids),
            record_ids=sorted(overlap_ids),
            resolution=(
                "Apply Incoterm split to separate goods from freight"
                if overlap_ids else "No overlap detected"
            ),
            provenance_hash=_calculate_provenance_hash(
                "DC-SCM-002", len(cat1), len(cat4), sorted(overlap_ids),
            ),
        )

    def _check_cat3_scope2(
        self,
        by_category: Dict[Scope3Category, List[ClassificationResult]],
    ) -> DoubleCountingCheck:
        """Check DC-SCM-003: Cat 3 vs Scope 2."""
        cat3 = by_category.get(Scope3Category.CAT_3_FUEL_ENERGY_ACTIVITIES, [])

        scope2_flagged = [
            r for r in cat3
            if r.metadata.get("is_scope2_reported", False)
        ]

        return DoubleCountingCheck(
            rule_id=DoubleCountingRule.DC_SCM_003,
            categories_checked=[Scope3Category.CAT_3_FUEL_ENERGY_ACTIVITIES],
            overlap_detected=len(scope2_flagged) > 0,
            records_affected=len(scope2_flagged),
            record_ids=[r.record_id for r in scope2_flagged],
            resolution=(
                "Exclude Scope 2 reported amounts from Cat 3"
                if scope2_flagged else "No Scope 2 overlap detected"
            ),
            provenance_hash=_calculate_provenance_hash(
                "DC-SCM-003", len(cat3), len(scope2_flagged),
            ),
        )

    def _check_cat4_vs_cat9(
        self,
        by_category: Dict[Scope3Category, List[ClassificationResult]],
    ) -> DoubleCountingCheck:
        """Check DC-SCM-004: Cat 4 vs Cat 9 overlap."""
        cat4 = by_category.get(Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION, [])
        cat9 = by_category.get(Scope3Category.CAT_9_DOWNSTREAM_TRANSPORTATION, [])

        cat4_ids = {r.record_id for r in cat4}
        cat9_ids = {r.record_id for r in cat9}
        overlap_ids = cat4_ids & cat9_ids

        return DoubleCountingCheck(
            rule_id=DoubleCountingRule.DC_SCM_004,
            categories_checked=[
                Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION,
                Scope3Category.CAT_9_DOWNSTREAM_TRANSPORTATION,
            ],
            overlap_detected=len(overlap_ids) > 0,
            records_affected=len(overlap_ids),
            record_ids=sorted(overlap_ids),
            resolution=(
                "Apply Incoterm + direction to determine Cat 4 vs Cat 9"
                if overlap_ids else "No overlap detected"
            ),
            provenance_hash=_calculate_provenance_hash(
                "DC-SCM-004", len(cat4), len(cat9), sorted(overlap_ids),
            ),
        )

    def _check_cat6_vs_cat7(
        self,
        by_category: Dict[Scope3Category, List[ClassificationResult]],
    ) -> DoubleCountingCheck:
        """Check DC-SCM-005: Cat 6 vs Cat 7 overlap."""
        cat6 = by_category.get(Scope3Category.CAT_6_BUSINESS_TRAVEL, [])
        cat7 = by_category.get(Scope3Category.CAT_7_EMPLOYEE_COMMUTING, [])

        cat6_ids = {r.record_id for r in cat6}
        cat7_ids = {r.record_id for r in cat7}
        overlap_ids = cat6_ids & cat7_ids

        return DoubleCountingCheck(
            rule_id=DoubleCountingRule.DC_SCM_005,
            categories_checked=[
                Scope3Category.CAT_6_BUSINESS_TRAVEL,
                Scope3Category.CAT_7_EMPLOYEE_COMMUTING,
            ],
            overlap_detected=len(overlap_ids) > 0,
            records_affected=len(overlap_ids),
            record_ids=sorted(overlap_ids),
            resolution=(
                "Exclude commuting on business travel days"
                if overlap_ids else "No overlap detected"
            ),
            provenance_hash=_calculate_provenance_hash(
                "DC-SCM-005", len(cat6), len(cat7), sorted(overlap_ids),
            ),
        )

    def _check_cat8_scope(
        self,
        by_category: Dict[Scope3Category, List[ClassificationResult]],
    ) -> DoubleCountingCheck:
        """Check DC-SCM-006: Cat 8 vs Scope 1/2."""
        cat8 = by_category.get(Scope3Category.CAT_8_UPSTREAM_LEASED_ASSETS, [])

        consolidation_conflicts = [
            r for r in cat8
            if r.metadata.get("consolidation_approach") == "operational_control"
        ]

        return DoubleCountingCheck(
            rule_id=DoubleCountingRule.DC_SCM_006,
            categories_checked=[Scope3Category.CAT_8_UPSTREAM_LEASED_ASSETS],
            overlap_detected=len(consolidation_conflicts) > 0,
            records_affected=len(consolidation_conflicts),
            record_ids=[r.record_id for r in consolidation_conflicts],
            resolution=(
                "Operational control leased assets should be in Scope 1/2"
                if consolidation_conflicts else "No Scope 1/2 overlap"
            ),
            provenance_hash=_calculate_provenance_hash(
                "DC-SCM-006", len(cat8), len(consolidation_conflicts),
            ),
        )

    def _check_cat10_vs_cat11(
        self,
        by_category: Dict[Scope3Category, List[ClassificationResult]],
    ) -> DoubleCountingCheck:
        """Check DC-SCM-007: Cat 10 vs Cat 11 overlap."""
        cat10 = by_category.get(Scope3Category.CAT_10_PROCESSING_SOLD_PRODUCTS, [])
        cat11 = by_category.get(Scope3Category.CAT_11_USE_OF_SOLD_PRODUCTS, [])

        cat10_ids = {r.record_id for r in cat10}
        cat11_ids = {r.record_id for r in cat11}
        overlap_ids = cat10_ids & cat11_ids

        return DoubleCountingCheck(
            rule_id=DoubleCountingRule.DC_SCM_007,
            categories_checked=[
                Scope3Category.CAT_10_PROCESSING_SOLD_PRODUCTS,
                Scope3Category.CAT_11_USE_OF_SOLD_PRODUCTS,
            ],
            overlap_detected=len(overlap_ids) > 0,
            records_affected=len(overlap_ids),
            record_ids=sorted(overlap_ids),
            resolution=(
                "Sequential boundary: processing before use, no overlap"
                if overlap_ids else "No overlap detected"
            ),
            provenance_hash=_calculate_provenance_hash(
                "DC-SCM-007", len(cat10), len(cat11), sorted(overlap_ids),
            ),
        )

    def _check_cat11_vs_cat12(
        self,
        by_category: Dict[Scope3Category, List[ClassificationResult]],
    ) -> DoubleCountingCheck:
        """Check DC-SCM-008: Cat 11 vs Cat 12 overlap."""
        cat11 = by_category.get(Scope3Category.CAT_11_USE_OF_SOLD_PRODUCTS, [])
        cat12 = by_category.get(Scope3Category.CAT_12_END_OF_LIFE_TREATMENT, [])

        cat11_ids = {r.record_id for r in cat11}
        cat12_ids = {r.record_id for r in cat12}
        overlap_ids = cat11_ids & cat12_ids

        return DoubleCountingCheck(
            rule_id=DoubleCountingRule.DC_SCM_008,
            categories_checked=[
                Scope3Category.CAT_11_USE_OF_SOLD_PRODUCTS,
                Scope3Category.CAT_12_END_OF_LIFE_TREATMENT,
            ],
            overlap_detected=len(overlap_ids) > 0,
            records_affected=len(overlap_ids),
            record_ids=sorted(overlap_ids),
            resolution=(
                "Product lifetime boundary: use during life, EoL after"
                if overlap_ids else "No overlap detected"
            ),
            provenance_hash=_calculate_provenance_hash(
                "DC-SCM-008", len(cat11), len(cat12), sorted(overlap_ids),
            ),
        )

    def _check_cat13_scope(
        self,
        by_category: Dict[Scope3Category, List[ClassificationResult]],
    ) -> DoubleCountingCheck:
        """Check DC-SCM-009: Cat 13 vs Scope 1/2."""
        cat13 = by_category.get(Scope3Category.CAT_13_DOWNSTREAM_LEASED_ASSETS, [])

        consolidated = [
            r for r in cat13
            if r.metadata.get("is_consolidated_scope12", False)
        ]

        return DoubleCountingCheck(
            rule_id=DoubleCountingRule.DC_SCM_009,
            categories_checked=[Scope3Category.CAT_13_DOWNSTREAM_LEASED_ASSETS],
            overlap_detected=len(consolidated) > 0,
            records_affected=len(consolidated),
            record_ids=[r.record_id for r in consolidated],
            resolution=(
                "Exclude assets already consolidated in Scope 1/2"
                if consolidated else "No Scope 1/2 overlap"
            ),
            provenance_hash=_calculate_provenance_hash(
                "DC-SCM-009", len(cat13), len(consolidated),
            ),
        )

    def _check_cat14_vs_cat15(
        self,
        by_category: Dict[Scope3Category, List[ClassificationResult]],
    ) -> DoubleCountingCheck:
        """Check DC-SCM-010: Cat 14 vs Cat 15 overlap."""
        cat14 = by_category.get(Scope3Category.CAT_14_FRANCHISES, [])
        cat15 = by_category.get(Scope3Category.CAT_15_INVESTMENTS, [])

        cat14_ids = {r.record_id for r in cat14}
        cat15_ids = {r.record_id for r in cat15}
        overlap_ids = cat14_ids & cat15_ids

        return DoubleCountingCheck(
            rule_id=DoubleCountingRule.DC_SCM_010,
            categories_checked=[
                Scope3Category.CAT_14_FRANCHISES,
                Scope3Category.CAT_15_INVESTMENTS,
            ],
            overlap_detected=len(overlap_ids) > 0,
            records_affected=len(overlap_ids),
            record_ids=sorted(overlap_ids),
            resolution=(
                "Franchise agreement takes precedence over investment"
                if overlap_ids else "No overlap detected"
            ),
            provenance_hash=_calculate_provenance_hash(
                "DC-SCM-010", len(cat14), len(cat15), sorted(overlap_ids),
            ),
        )

    # =========================================================================
    # PUBLIC METHODS -- DC Rule Information
    # =========================================================================

    def get_dc_rule(self, rule_id: DoubleCountingRule) -> Dict[str, Any]:
        """Get details for a specific double-counting rule.

        Args:
            rule_id: Double-counting rule identifier.

        Returns:
            Dictionary with rule details including name, description,
            categories, resolution method, and GHG Protocol reference.

        Raises:
            ValueError: If rule_id is not recognized.

        Example:
            >>> engine = BoundaryDeterminerEngine()
            >>> rule = engine.get_dc_rule(DoubleCountingRule.DC_SCM_001)
            >>> rule["name"]
            'Cat 1 vs Cat 2: Opex vs Capex'
        """
        if rule_id not in _DC_RULE_DETAILS:
            raise ValueError(f"Unknown DC rule: {rule_id.value}")

        return _DC_RULE_DETAILS[rule_id].copy()

    def get_all_dc_rules(self) -> List[Dict[str, Any]]:
        """Get details for all 10 double-counting rules.

        Returns:
            List of dictionaries, one per DC-SCM rule.

        Example:
            >>> engine = BoundaryDeterminerEngine()
            >>> rules = engine.get_all_dc_rules()
            >>> len(rules)
            10
        """
        return [
            details.copy()
            for details in _DC_RULE_DETAILS.values()
        ]

    # =========================================================================
    # PUBLIC METHODS -- Metrics
    # =========================================================================

    def get_engine_stats(self) -> Dict[str, Any]:
        """Get current engine statistics.

        Returns:
            Dictionary with determination_count, dc_rules count, and metadata.
        """
        with self._count_lock:
            return {
                "engine_id": ENGINE_ID,
                "engine_version": ENGINE_VERSION,
                "agent_id": AGENT_ID,
                "determination_count": self._determination_count,
                "dc_rules_count": len(_DC_RULE_DETAILS),
            }

    # =========================================================================
    # CLASS METHOD -- Reset singleton (for testing only)
    # =========================================================================

    @classmethod
    def _reset_instance(cls) -> None:
        """Reset the singleton instance. FOR TESTING ONLY.

        This method is used exclusively in unit tests to reset the
        singleton between test cases. It must never be called in
        production code.
        """
        with cls._lock:
            cls._instance = None
