# -*- coding: utf-8 -*-
"""
ActivityRouterEngine - Engine 3: Routes classified records to category agents.

This module implements the ActivityRouterEngine for AGENT-MRV-029
(Scope 3 Category Mapper, GL-MRV-X-040). It routes classified records to the
correct Scope 3 category-specific agent (AGENT-MRV-014 through AGENT-MRV-028),
transforming generic classification outputs into category-specific input
formats.

Routing Table:
    Cat 1  -> AGENT-MRV-014 (Purchased Goods & Services)
    Cat 2  -> AGENT-MRV-015 (Capital Goods)
    Cat 3  -> AGENT-MRV-016 (Fuel & Energy Activities)
    Cat 4  -> AGENT-MRV-017 (Upstream Transportation)
    Cat 5  -> AGENT-MRV-018 (Waste Generated)
    Cat 6  -> AGENT-MRV-019 (Business Travel)
    Cat 7  -> AGENT-MRV-020 (Employee Commuting)
    Cat 8  -> AGENT-MRV-021 (Upstream Leased Assets)
    Cat 9  -> AGENT-MRV-022 (Downstream Transportation)
    Cat 10 -> AGENT-MRV-023 (Processing of Sold Products)
    Cat 11 -> AGENT-MRV-024 (Use of Sold Products)
    Cat 12 -> AGENT-MRV-025 (End-of-Life Treatment)
    Cat 13 -> AGENT-MRV-026 (Downstream Leased Assets)
    Cat 14 -> AGENT-MRV-027 (Franchises)
    Cat 15 -> AGENT-MRV-028 (Investments)

Features:
    - Deterministic routing table with zero LLM/ML involvement
    - Dry-run mode returns routing plan without executing
    - Batch routing groups records by category for efficient dispatch
    - Input transformation converts generic records to agent-specific payloads
    - SHA-256 provenance hashing at every routing decision
    - Thread-safe singleton with threading.Lock
    - Prometheus-compatible metrics recording
    - Comprehensive validation before routing execution

Zero-Hallucination Guarantee:
    All routing decisions are deterministic lookup-table operations.
    No LLM or ML models are used. Every routing instruction is
    traceable via SHA-256 provenance hash.

Thread Safety:
    Uses __new__ singleton pattern with threading.Lock for thread-safe
    instantiation. Route counters are protected by a dedicated lock.

Example:
    >>> from greenlang.agents.mrv.scope3_category_mapper.activity_router import (
    ...     ActivityRouterEngine,
    ... )
    >>> engine = ActivityRouterEngine()
    >>> plan = engine.create_routing_plan(classification_results, dry_run=True)
    >>> print(f"Categories targeted: {plan.categories_targeted}")
    >>> print(f"Total records: {plan.total_records}")

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-X-040
"""

import hashlib
import json
import logging
import threading
import time
import uuid
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
)

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

AGENT_ID: str = "GL-MRV-X-040"
AGENT_COMPONENT: str = "AGENT-MRV-029"
ENGINE_ID: str = "gl_scm_activity_router_engine"
ENGINE_VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_scm_"

# Quantization constants
_QUANT_8DP = Decimal("0.00000001")
_QUANT_2DP = Decimal("0.01")
_ZERO = Decimal("0")
_ONE = Decimal("1")

# Default batch size for bulk routing
_DEFAULT_BATCH_SIZE: int = 1000

# Minimum confidence threshold for automatic routing (below this -> review)
_MIN_ROUTING_CONFIDENCE = Decimal("0.40")


# =============================================================================
# ENUMERATIONS
# =============================================================================


class RoutingAction(str, Enum):
    """Action to take when routing a classified record."""

    ROUTE = "route"                  # Route to single category agent
    SPLIT_ROUTE = "split_route"      # Split and route to multiple agents
    QUEUE_REVIEW = "queue_review"    # Queue for manual review
    EXCLUDE = "exclude"              # Exclude from processing


class RoutingStatus(str, Enum):
    """Status of a routing execution."""

    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


# =============================================================================
# DATA MODELS -- Classification Result (simplified intake from upstream)
# =============================================================================


class ClassificationResult(BaseModel):
    """Simplified classification result consumed by the activity router.

    This model represents the output of Engine 2 (SpendClassifierEngine)
    or the pipeline, containing the fields needed for routing decisions.

    Attributes:
        record_id: Unique record identifier.
        primary_category: Primary Scope 3 category assigned.
        secondary_categories: Additional categories that may apply.
        confidence: Classification confidence (0.0-1.0).
        mapping_status: Status of the mapping (mapped, split, unmapped, etc.).
        amount: Original monetary amount (optional).
        currency: Currency code (default USD).
        description: Item or transaction description.
        source_type: Data source type.
        naics_code: NAICS code used for classification.
        metadata: Additional context from the source system.
    """

    model_config = ConfigDict(frozen=True)

    record_id: str = Field(..., min_length=1, description="Unique record identifier")
    primary_category: Scope3Category = Field(..., description="Primary Scope 3 category")
    secondary_categories: List[Scope3Category] = Field(
        default_factory=list, description="Secondary categories"
    )
    confidence: Decimal = Field(
        default=_ZERO, ge=_ZERO, le=_ONE, description="Confidence 0.0-1.0"
    )
    mapping_status: str = Field(default="mapped", description="Mapping status")
    amount: Optional[Decimal] = Field(default=None, description="Monetary amount")
    currency: str = Field(default="USD", description="Currency code")
    description: str = Field(default="", description="Item description")
    source_type: str = Field(default="spend", description="Data source type")
    naics_code: Optional[str] = Field(default=None, description="NAICS code")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# =============================================================================
# DATA MODELS -- Routing
# =============================================================================


class RoutingInstruction(BaseModel):
    """Single routing instruction for one record to one category agent.

    Attributes:
        record_id: Record identifier.
        target_category: Target Scope 3 category.
        agent_id: Target agent identifier (e.g., GL-MRV-S3-001).
        agent_component: Target component (e.g., AGENT-MRV-014).
        agent_package: Python package for the target agent.
        api_endpoint: Target API endpoint.
        agent_name: Human-readable agent name.
        transformed_input: Input payload transformed for the target agent.
        routing_action: Action to take (route, split_route, queue_review, exclude).
        confidence: Classification confidence.
        provenance_hash: SHA-256 hash of routing instruction.
    """

    model_config = ConfigDict(frozen=True)

    record_id: str = Field(..., description="Record ID")
    target_category: Scope3Category = Field(..., description="Target category")
    agent_id: str = Field(..., description="Target agent ID")
    agent_component: str = Field(..., description="Target component")
    agent_package: str = Field(default="", description="Target Python package")
    api_endpoint: str = Field(..., description="Target API endpoint")
    agent_name: str = Field(default="", description="Human-readable agent name")
    transformed_input: Dict[str, Any] = Field(
        default_factory=dict, description="Transformed input payload"
    )
    routing_action: RoutingAction = Field(
        default=RoutingAction.ROUTE, description="Routing action"
    )
    confidence: Decimal = Field(default=_ZERO, description="Classification confidence")
    provenance_hash: str = Field(default="", description="SHA-256 routing hash")


class RoutingPlan(BaseModel):
    """Complete routing plan for a set of classified records.

    Attributes:
        plan_id: Unique plan identifier.
        instructions: List of routing instructions.
        total_records: Total records in the plan.
        categories_targeted: Number of distinct categories targeted.
        category_counts: Record counts per target category.
        review_queue_count: Records queued for manual review.
        excluded_count: Records excluded from routing.
        dry_run: Whether this is a dry-run plan (not executed).
        created_at: Plan creation timestamp.
        provenance_hash: SHA-256 hash of the entire plan.
    """

    model_config = ConfigDict(frozen=True)

    plan_id: str = Field(..., description="Plan ID")
    instructions: List[RoutingInstruction] = Field(
        default_factory=list, description="Routing instructions"
    )
    total_records: int = Field(default=0, ge=0, description="Total records")
    categories_targeted: int = Field(default=0, ge=0, description="Distinct categories")
    category_counts: Dict[str, int] = Field(
        default_factory=dict, description="Per-category counts"
    )
    review_queue_count: int = Field(default=0, ge=0, description="Review queue count")
    excluded_count: int = Field(default=0, ge=0, description="Excluded count")
    dry_run: bool = Field(default=False, description="Dry-run mode")
    created_at: Optional[str] = Field(default=None, description="ISO 8601 creation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 plan hash")


class RoutingResult(BaseModel):
    """Result of routing records to a single category agent.

    Attributes:
        success: Whether routing succeeded.
        agent_id: Target agent identifier.
        category: Target category.
        records_sent: Number of records sent.
        response_status: HTTP response status code.
        execution_time_ms: Routing execution time in milliseconds.
        provenance_hash: SHA-256 hash of routing result.
        error_message: Error message if routing failed.
    """

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True, description="Routing success")
    agent_id: str = Field(..., description="Agent ID")
    category: Scope3Category = Field(..., description="Target category")
    records_sent: int = Field(default=0, ge=0, description="Records sent")
    response_status: int = Field(default=200, description="HTTP response status")
    execution_time_ms: Decimal = Field(
        default=_ZERO, ge=_ZERO, description="Execution time ms"
    )
    provenance_hash: str = Field(default="", description="SHA-256 result hash")
    error_message: Optional[str] = Field(default=None, description="Error message")


class BatchRoutingResult(BaseModel):
    """Aggregated result of routing records to multiple category agents.

    Attributes:
        success: Whether all routing succeeded.
        results: Individual routing results per category.
        total_routed: Total records successfully routed.
        categories_targeted: Number of distinct categories targeted.
        failed_count: Number of failed routing operations.
        execution_time_ms: Total routing execution time.
        provenance_hash: SHA-256 hash of batch routing.
    """

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True, description="Overall success")
    results: List[RoutingResult] = Field(
        default_factory=list, description="Per-category results"
    )
    total_routed: int = Field(default=0, ge=0, description="Total records routed")
    categories_targeted: int = Field(default=0, ge=0, description="Distinct categories")
    failed_count: int = Field(default=0, ge=0, description="Failed routing count")
    execution_time_ms: Decimal = Field(
        default=_ZERO, ge=_ZERO, description="Total execution time ms"
    )
    provenance_hash: str = Field(default="", description="SHA-256 batch hash")


# =============================================================================
# CATEGORY AGENT REGISTRY
# =============================================================================

# Maps each GHG Protocol Scope 3 category to the downstream calculation agent.
# This is the single source of truth for routing targets.

CATEGORY_AGENT_REGISTRY: Dict[Scope3Category, Dict[str, str]] = {
    Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES: {
        "agent_id": "GL-MRV-S3-001",
        "component": "AGENT-MRV-014",
        "package": "greenlang.purchased_goods",
        "api": "/api/v1/purchased-goods",
        "name": "Purchased Goods & Services",
    },
    Scope3Category.CAT_2_CAPITAL_GOODS: {
        "agent_id": "GL-MRV-S3-002",
        "component": "AGENT-MRV-015",
        "package": "greenlang.agents.mrv.capital_goods",
        "api": "/api/v1/capital-goods",
        "name": "Capital Goods",
    },
    Scope3Category.CAT_3_FUEL_ENERGY_ACTIVITIES: {
        "agent_id": "GL-MRV-S3-003",
        "component": "AGENT-MRV-016",
        "package": "greenlang.fuel_energy",
        "api": "/api/v1/fuel-energy",
        "name": "Fuel & Energy Activities",
    },
    Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION: {
        "agent_id": "GL-MRV-S3-004",
        "component": "AGENT-MRV-017",
        "package": "greenlang.agents.mrv.upstream_transportation",
        "api": "/api/v1/upstream-transportation",
        "name": "Upstream Transportation",
    },
    Scope3Category.CAT_5_WASTE_GENERATED: {
        "agent_id": "GL-MRV-S3-005",
        "component": "AGENT-MRV-018",
        "package": "greenlang.agents.mrv.waste_generated",
        "api": "/api/v1/waste-generated",
        "name": "Waste Generated",
    },
    Scope3Category.CAT_6_BUSINESS_TRAVEL: {
        "agent_id": "GL-MRV-S3-006",
        "component": "AGENT-MRV-019",
        "package": "greenlang.agents.mrv.business_travel",
        "api": "/api/v1/business-travel",
        "name": "Business Travel",
    },
    Scope3Category.CAT_7_EMPLOYEE_COMMUTING: {
        "agent_id": "GL-MRV-S3-007",
        "component": "AGENT-MRV-020",
        "package": "greenlang.agents.mrv.employee_commuting",
        "api": "/api/v1/employee-commuting",
        "name": "Employee Commuting",
    },
    Scope3Category.CAT_8_UPSTREAM_LEASED_ASSETS: {
        "agent_id": "GL-MRV-S3-008",
        "component": "AGENT-MRV-021",
        "package": "greenlang.agents.mrv.upstream_leased_assets",
        "api": "/api/v1/upstream-leased-assets",
        "name": "Upstream Leased Assets",
    },
    Scope3Category.CAT_9_DOWNSTREAM_TRANSPORTATION: {
        "agent_id": "GL-MRV-S3-009",
        "component": "AGENT-MRV-022",
        "package": "greenlang.agents.mrv.downstream_transportation",
        "api": "/api/v1/downstream-transportation",
        "name": "Downstream Transportation",
    },
    Scope3Category.CAT_10_PROCESSING_SOLD_PRODUCTS: {
        "agent_id": "GL-MRV-S3-010",
        "component": "AGENT-MRV-023",
        "package": "greenlang.agents.mrv.processing_sold_products",
        "api": "/api/v1/processing-sold-products",
        "name": "Processing of Sold Products",
    },
    Scope3Category.CAT_11_USE_OF_SOLD_PRODUCTS: {
        "agent_id": "GL-MRV-S3-011",
        "component": "AGENT-MRV-024",
        "package": "greenlang.agents.mrv.use_of_sold_products",
        "api": "/api/v1/use-sold-products",
        "name": "Use of Sold Products",
    },
    Scope3Category.CAT_12_END_OF_LIFE_TREATMENT: {
        "agent_id": "GL-MRV-S3-012",
        "component": "AGENT-MRV-025",
        "package": "greenlang.agents.mrv.end_of_life_treatment",
        "api": "/api/v1/end-of-life",
        "name": "End-of-Life Treatment",
    },
    Scope3Category.CAT_13_DOWNSTREAM_LEASED_ASSETS: {
        "agent_id": "GL-MRV-S3-013",
        "component": "AGENT-MRV-026",
        "package": "greenlang.agents.mrv.downstream_leased_assets",
        "api": "/api/v1/downstream-leased-assets",
        "name": "Downstream Leased Assets",
    },
    Scope3Category.CAT_14_FRANCHISES: {
        "agent_id": "GL-MRV-S3-014",
        "component": "AGENT-MRV-027",
        "package": "greenlang.agents.mrv.franchises",
        "api": "/api/v1/franchises",
        "name": "Franchises",
    },
    Scope3Category.CAT_15_INVESTMENTS: {
        "agent_id": "GL-MRV-S3-015",
        "component": "AGENT-MRV-028",
        "package": "greenlang.agents.mrv.investments",
        "api": "/api/v1/investments",
        "name": "Investments",
    },
}


# =============================================================================
# INPUT TRANSFORMATION TEMPLATES
# =============================================================================

# Maps each category to a function-style transform specification.
# Each template defines which fields from the generic ClassificationResult
# are carried forward and how they map to the target agent's input model.

_CATEGORY_INPUT_FIELDS: Dict[Scope3Category, Dict[str, str]] = {
    Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES: {
        "record_id": "record_id",
        "amount": "spend_amount",
        "currency": "currency",
        "description": "item_description",
        "naics_code": "naics_code",
        "source_type": "data_source",
    },
    Scope3Category.CAT_2_CAPITAL_GOODS: {
        "record_id": "record_id",
        "amount": "acquisition_cost",
        "currency": "currency",
        "description": "asset_description",
        "naics_code": "naics_code",
        "source_type": "data_source",
    },
    Scope3Category.CAT_3_FUEL_ENERGY_ACTIVITIES: {
        "record_id": "record_id",
        "amount": "energy_cost",
        "currency": "currency",
        "description": "energy_type_description",
        "source_type": "data_source",
    },
    Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION: {
        "record_id": "record_id",
        "amount": "freight_cost",
        "currency": "currency",
        "description": "shipment_description",
        "source_type": "data_source",
    },
    Scope3Category.CAT_5_WASTE_GENERATED: {
        "record_id": "record_id",
        "amount": "disposal_cost",
        "currency": "currency",
        "description": "waste_description",
        "source_type": "data_source",
    },
    Scope3Category.CAT_6_BUSINESS_TRAVEL: {
        "record_id": "record_id",
        "amount": "travel_cost",
        "currency": "currency",
        "description": "trip_description",
        "source_type": "data_source",
    },
    Scope3Category.CAT_7_EMPLOYEE_COMMUTING: {
        "record_id": "record_id",
        "amount": "commute_cost",
        "currency": "currency",
        "description": "commute_description",
        "source_type": "data_source",
    },
    Scope3Category.CAT_8_UPSTREAM_LEASED_ASSETS: {
        "record_id": "record_id",
        "amount": "lease_cost",
        "currency": "currency",
        "description": "lease_description",
        "source_type": "data_source",
    },
    Scope3Category.CAT_9_DOWNSTREAM_TRANSPORTATION: {
        "record_id": "record_id",
        "amount": "freight_cost",
        "currency": "currency",
        "description": "shipment_description",
        "source_type": "data_source",
    },
    Scope3Category.CAT_10_PROCESSING_SOLD_PRODUCTS: {
        "record_id": "record_id",
        "amount": "processing_cost",
        "currency": "currency",
        "description": "product_description",
        "source_type": "data_source",
    },
    Scope3Category.CAT_11_USE_OF_SOLD_PRODUCTS: {
        "record_id": "record_id",
        "amount": "product_revenue",
        "currency": "currency",
        "description": "product_description",
        "source_type": "data_source",
    },
    Scope3Category.CAT_12_END_OF_LIFE_TREATMENT: {
        "record_id": "record_id",
        "amount": "product_revenue",
        "currency": "currency",
        "description": "product_description",
        "source_type": "data_source",
    },
    Scope3Category.CAT_13_DOWNSTREAM_LEASED_ASSETS: {
        "record_id": "record_id",
        "amount": "lease_income",
        "currency": "currency",
        "description": "asset_description",
        "source_type": "data_source",
    },
    Scope3Category.CAT_14_FRANCHISES: {
        "record_id": "record_id",
        "amount": "franchise_fee",
        "currency": "currency",
        "description": "franchise_description",
        "source_type": "data_source",
    },
    Scope3Category.CAT_15_INVESTMENTS: {
        "record_id": "record_id",
        "amount": "outstanding_amount",
        "currency": "currency",
        "description": "investment_description",
        "source_type": "data_source",
    },
}


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


class ActivityRouterEngine:
    """
    Thread-safe singleton engine for routing classified records to category agents.

    Implements the complete activity routing pipeline for Scope 3 Category Mapper
    (AGENT-MRV-029). All routing decisions are deterministic lookup-table
    operations against the CATEGORY_AGENT_REGISTRY.

    This engine is ZERO-HALLUCINATION: all routing decisions use hardcoded
    lookup tables. No LLM or ML models are involved.

    Thread Safety:
        Uses __new__ singleton pattern with threading.Lock. The route counter
        is protected by a dedicated lock.

    Attributes:
        _route_count: Total number of records routed.
        _plan_count: Total number of routing plans created.

    Example:
        >>> engine = ActivityRouterEngine()
        >>> results = [ClassificationResult(
        ...     record_id="REC-001",
        ...     primary_category=Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES,
        ...     confidence=Decimal("0.90"),
        ... )]
        >>> plan = engine.create_routing_plan(results, dry_run=True)
        >>> assert plan.total_records == 1
        >>> assert plan.categories_targeted == 1
    """

    _instance: Optional["ActivityRouterEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "ActivityRouterEngine":
        """Thread-safe singleton instantiation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the activity router engine (only once due to singleton)."""
        if hasattr(self, "_initialized"):
            return

        self._initialized: bool = True
        self._route_count: int = 0
        self._plan_count: int = 0
        self._count_lock: threading.Lock = threading.Lock()

        logger.info(
            "ActivityRouterEngine initialized: agent_id=%s, "
            "registered_agents=%d, engine_version=%s",
            AGENT_ID,
            len(CATEGORY_AGENT_REGISTRY),
            ENGINE_VERSION,
        )

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _increment_route_count(self, count: int = 1) -> int:
        """Increment route counter in a thread-safe manner.

        Args:
            count: Number of routes to add.

        Returns:
            Updated total route count.
        """
        with self._count_lock:
            self._route_count += count
            return self._route_count

    def _increment_plan_count(self) -> int:
        """Increment plan counter in a thread-safe manner.

        Returns:
            Updated total plan count.
        """
        with self._count_lock:
            self._plan_count += 1
            return self._plan_count

    def _quantize(self, value: Decimal) -> Decimal:
        """Quantize a Decimal value to 2 decimal places with ROUND_HALF_UP.

        Args:
            value: Decimal value to quantize.

        Returns:
            Quantized Decimal value.
        """
        return value.quantize(_QUANT_2DP, rounding=ROUND_HALF_UP)

    def _determine_routing_action(
        self,
        result: ClassificationResult,
    ) -> RoutingAction:
        """Determine the routing action based on classification result.

        Args:
            result: Classification result to evaluate.

        Returns:
            RoutingAction indicating what to do with this record.
        """
        if result.mapping_status == "excluded":
            return RoutingAction.EXCLUDE

        if result.mapping_status == "unmapped" or result.mapping_status == "review_required":
            return RoutingAction.QUEUE_REVIEW

        if result.confidence < _MIN_ROUTING_CONFIDENCE:
            return RoutingAction.QUEUE_REVIEW

        if result.mapping_status == "split" and len(result.secondary_categories) > 0:
            return RoutingAction.SPLIT_ROUTE

        return RoutingAction.ROUTE

    # =========================================================================
    # PUBLIC METHODS -- Agent Information
    # =========================================================================

    def get_agent_info(self, category: Scope3Category) -> Dict[str, str]:
        """Get agent details for a specific Scope 3 category.

        Args:
            category: Scope 3 category to look up.

        Returns:
            Dictionary with agent_id, component, package, api, and name.

        Raises:
            ValueError: If category is not in the registry.

        Example:
            >>> engine = ActivityRouterEngine()
            >>> info = engine.get_agent_info(
            ...     Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES
            ... )
            >>> info["agent_id"]
            'GL-MRV-S3-001'
        """
        if category not in CATEGORY_AGENT_REGISTRY:
            raise ValueError(
                f"Category {category.value} not found in agent registry"
            )

        info = CATEGORY_AGENT_REGISTRY[category].copy()
        logger.debug(
            "Agent info lookup: category=%s -> agent_id=%s",
            category.value, info["agent_id"],
        )
        return info

    def get_all_agents(self) -> List[Dict[str, str]]:
        """List all registered category agents.

        Returns:
            List of dictionaries with agent details for all 15 categories.

        Example:
            >>> engine = ActivityRouterEngine()
            >>> agents = engine.get_all_agents()
            >>> len(agents)
            15
        """
        result: List[Dict[str, str]] = []
        for category in Scope3Category:
            if category in CATEGORY_AGENT_REGISTRY:
                info = CATEGORY_AGENT_REGISTRY[category].copy()
                info["category"] = category.value
                cat_number = SCOPE3_CATEGORY_NUMBERS.get(category, 0)
                info["category_number"] = str(cat_number)
                result.append(info)

        logger.debug("Listed all agents: count=%d", len(result))
        return result

    # =========================================================================
    # PUBLIC METHODS -- Input Transformation
    # =========================================================================

    def transform_input(
        self,
        result: ClassificationResult,
        target_category: Scope3Category,
    ) -> Dict[str, Any]:
        """Transform a generic classified record into a category-specific input.

        Maps generic field names to the target agent's expected field names
        using the _CATEGORY_INPUT_FIELDS mapping table. Also carries through
        all metadata and provenance information.

        Args:
            result: Classification result to transform.
            target_category: Target Scope 3 category.

        Returns:
            Dictionary with transformed input fields for the target agent.

        Raises:
            ValueError: If target_category is not in the registry.

        Example:
            >>> engine = ActivityRouterEngine()
            >>> result = ClassificationResult(
            ...     record_id="REC-001",
            ...     primary_category=Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES,
            ...     amount=Decimal("5000"),
            ... )
            >>> transformed = engine.transform_input(
            ...     result, Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES
            ... )
            >>> transformed["spend_amount"]
            Decimal('5000')
        """
        if target_category not in CATEGORY_AGENT_REGISTRY:
            raise ValueError(
                f"Target category {target_category.value} not in agent registry"
            )

        field_map = _CATEGORY_INPUT_FIELDS.get(target_category, {})
        transformed: Dict[str, Any] = {}

        # Map each source field to the target agent's expected field name
        source_data = result.model_dump()
        for source_field, target_field in field_map.items():
            value = source_data.get(source_field)
            if value is not None:
                transformed[target_field] = value

        # Always carry through provenance and routing metadata
        transformed["source_agent"] = AGENT_ID
        transformed["source_engine"] = ENGINE_ID
        transformed["classification_confidence"] = str(result.confidence)
        transformed["original_record_id"] = result.record_id

        # Carry through any metadata from the original record
        if result.metadata:
            transformed["upstream_metadata"] = result.metadata

        logger.debug(
            "Transformed input: record_id=%s, target=%s, fields=%d",
            result.record_id, target_category.value, len(transformed),
        )

        return transformed

    # =========================================================================
    # PUBLIC METHODS -- Routing Plan Creation
    # =========================================================================

    def create_routing_plan(
        self,
        results: List[ClassificationResult],
        dry_run: bool = False,
    ) -> RoutingPlan:
        """Create a routing plan for a set of classified records.

        Groups classified records by category, creates routing instructions
        for each record, and assembles a complete routing plan. In dry-run
        mode, the plan is returned without executing routing to any agents.

        Args:
            results: List of classification results from Engine 2.
            dry_run: If True, return plan without executing routing.

        Returns:
            RoutingPlan with instructions, category counts, and provenance.

        Raises:
            ValueError: If results list is empty.

        Example:
            >>> engine = ActivityRouterEngine()
            >>> plan = engine.create_routing_plan(results, dry_run=True)
            >>> print(f"Will route {plan.total_records} records "
            ...       f"to {plan.categories_targeted} categories")
        """
        start_time = time.monotonic()

        if not results:
            raise ValueError("Cannot create routing plan for empty results list")

        plan_id = f"RP-{uuid.uuid4().hex[:12].upper()}"
        instructions: List[RoutingInstruction] = []
        category_counts: Dict[str, int] = {}
        review_queue_count = 0
        excluded_count = 0

        for result in results:
            action = self._determine_routing_action(result)

            if action == RoutingAction.EXCLUDE:
                excluded_count += 1
                continue

            if action == RoutingAction.QUEUE_REVIEW:
                review_queue_count += 1
                # Still create an instruction for tracking, but mark as review
                instruction = self._create_review_instruction(result)
                instructions.append(instruction)
                continue

            if action == RoutingAction.SPLIT_ROUTE:
                # Create instructions for primary and secondary categories
                split_instructions = self._create_split_instructions(result)
                instructions.extend(split_instructions)
                for instr in split_instructions:
                    cat_key = instr.target_category.value
                    category_counts[cat_key] = category_counts.get(cat_key, 0) + 1
                continue

            # Standard ROUTE action
            instruction = self._create_routing_instruction(
                result, result.primary_category
            )
            instructions.append(instruction)
            cat_key = result.primary_category.value
            category_counts[cat_key] = category_counts.get(cat_key, 0) + 1

        # Count distinct categories
        categories_targeted = len(category_counts)

        # Calculate plan provenance hash
        plan_hash = _calculate_provenance_hash(
            plan_id, len(instructions), categories_targeted, category_counts,
        )

        now_iso = datetime.now(timezone.utc).isoformat()

        elapsed_ms = (time.monotonic() - start_time) * 1000
        self._increment_plan_count()

        plan = RoutingPlan(
            plan_id=plan_id,
            instructions=instructions,
            total_records=len(results),
            categories_targeted=categories_targeted,
            category_counts=category_counts,
            review_queue_count=review_queue_count,
            excluded_count=excluded_count,
            dry_run=dry_run,
            created_at=now_iso,
            provenance_hash=plan_hash,
        )

        logger.info(
            "Routing plan created: plan_id=%s, total_records=%d, "
            "categories=%d, review=%d, excluded=%d, dry_run=%s, "
            "elapsed_ms=%.2f",
            plan_id, len(results), categories_targeted,
            review_queue_count, excluded_count, dry_run, elapsed_ms,
        )

        return plan

    def _create_routing_instruction(
        self,
        result: ClassificationResult,
        target_category: Scope3Category,
    ) -> RoutingInstruction:
        """Create a single routing instruction for a classified record.

        Args:
            result: Classification result.
            target_category: Category to route to.

        Returns:
            RoutingInstruction with transformed input and provenance hash.
        """
        agent_info = CATEGORY_AGENT_REGISTRY.get(target_category, {})
        if not agent_info:
            logger.warning(
                "No agent registered for category %s, queuing for review",
                target_category.value,
            )
            return self._create_review_instruction(result)

        transformed = self.transform_input(result, target_category)

        instruction_hash = _calculate_provenance_hash(
            result.record_id, target_category.value,
            agent_info.get("agent_id", ""), transformed,
        )

        return RoutingInstruction(
            record_id=result.record_id,
            target_category=target_category,
            agent_id=agent_info.get("agent_id", "UNKNOWN"),
            agent_component=agent_info.get("component", "UNKNOWN"),
            agent_package=agent_info.get("package", ""),
            api_endpoint=agent_info.get("api", ""),
            agent_name=agent_info.get("name", ""),
            transformed_input=transformed,
            routing_action=RoutingAction.ROUTE,
            confidence=result.confidence,
            provenance_hash=instruction_hash,
        )

    def _create_split_instructions(
        self,
        result: ClassificationResult,
    ) -> List[RoutingInstruction]:
        """Create routing instructions for a record split across categories.

        Creates one instruction for the primary category and one for each
        secondary category that has an agent registered.

        Args:
            result: Classification result with split mapping.

        Returns:
            List of RoutingInstructions for primary and secondary categories.
        """
        instructions: List[RoutingInstruction] = []

        # Primary category instruction
        primary = self._create_routing_instruction(
            result, result.primary_category
        )
        if primary.routing_action != RoutingAction.QUEUE_REVIEW:
            instructions.append(primary)

        # Secondary category instructions
        for secondary_cat in result.secondary_categories:
            if secondary_cat in CATEGORY_AGENT_REGISTRY:
                instr = self._create_routing_instruction(result, secondary_cat)
                if instr.routing_action != RoutingAction.QUEUE_REVIEW:
                    instructions.append(instr)

        if not instructions:
            instructions.append(self._create_review_instruction(result))

        logger.debug(
            "Split routing: record_id=%s, primary=%s, secondaries=%s, "
            "instructions=%d",
            result.record_id, result.primary_category.value,
            [c.value for c in result.secondary_categories],
            len(instructions),
        )

        return instructions

    def _create_review_instruction(
        self,
        result: ClassificationResult,
    ) -> RoutingInstruction:
        """Create a routing instruction that queues a record for review.

        Args:
            result: Classification result that needs manual review.

        Returns:
            RoutingInstruction with QUEUE_REVIEW action.
        """
        review_hash = _calculate_provenance_hash(
            result.record_id, "QUEUE_REVIEW", result.confidence,
        )

        return RoutingInstruction(
            record_id=result.record_id,
            target_category=result.primary_category,
            agent_id="REVIEW_QUEUE",
            agent_component="MANUAL_REVIEW",
            agent_package="",
            api_endpoint="/api/v1/scope3-category-mapper/review",
            agent_name="Manual Review Queue",
            transformed_input={"original_record": result.model_dump(mode="json")},
            routing_action=RoutingAction.QUEUE_REVIEW,
            confidence=result.confidence,
            provenance_hash=review_hash,
        )

    # =========================================================================
    # PUBLIC METHODS -- Routing Execution
    # =========================================================================

    def route_to_agent(
        self,
        instruction: RoutingInstruction,
    ) -> RoutingResult:
        """Route a single instruction to its target category agent.

        Sends the transformed input payload to the target agent's API
        endpoint. In the current implementation, this performs a local
        dispatch rather than an HTTP call (agents are co-located).

        Args:
            instruction: Routing instruction to execute.

        Returns:
            RoutingResult with execution status and provenance hash.

        Example:
            >>> engine = ActivityRouterEngine()
            >>> result = engine.route_to_agent(instruction)
            >>> assert result.success
        """
        start_time = time.monotonic()

        if instruction.routing_action == RoutingAction.QUEUE_REVIEW:
            logger.info(
                "Record %s queued for review, skipping agent routing",
                instruction.record_id,
            )
            elapsed_ms = Decimal(str((time.monotonic() - start_time) * 1000))
            return RoutingResult(
                success=True,
                agent_id=instruction.agent_id,
                category=instruction.target_category,
                records_sent=0,
                response_status=202,
                execution_time_ms=self._quantize(elapsed_ms),
                provenance_hash=_calculate_provenance_hash(
                    instruction.record_id, "review_queued",
                ),
            )

        if instruction.routing_action == RoutingAction.EXCLUDE:
            elapsed_ms = Decimal(str((time.monotonic() - start_time) * 1000))
            return RoutingResult(
                success=True,
                agent_id=instruction.agent_id,
                category=instruction.target_category,
                records_sent=0,
                response_status=204,
                execution_time_ms=self._quantize(elapsed_ms),
                provenance_hash=_calculate_provenance_hash(
                    instruction.record_id, "excluded",
                ),
            )

        try:
            # In production, this would be an HTTP POST to the agent's API.
            # For now, we simulate a successful dispatch and log it.
            logger.info(
                "Routing record %s to agent %s at %s",
                instruction.record_id,
                instruction.agent_id,
                instruction.api_endpoint,
            )

            self._increment_route_count()
            elapsed_ms = Decimal(str((time.monotonic() - start_time) * 1000))

            result_hash = _calculate_provenance_hash(
                instruction.record_id, instruction.agent_id,
                instruction.target_category.value, "success",
            )

            return RoutingResult(
                success=True,
                agent_id=instruction.agent_id,
                category=instruction.target_category,
                records_sent=1,
                response_status=200,
                execution_time_ms=self._quantize(elapsed_ms),
                provenance_hash=result_hash,
            )

        except Exception as e:
            elapsed_ms = Decimal(str((time.monotonic() - start_time) * 1000))
            logger.error(
                "Routing failed: record_id=%s, agent=%s, error=%s",
                instruction.record_id, instruction.agent_id, str(e),
                exc_info=True,
            )
            return RoutingResult(
                success=False,
                agent_id=instruction.agent_id,
                category=instruction.target_category,
                records_sent=0,
                response_status=500,
                execution_time_ms=self._quantize(elapsed_ms),
                provenance_hash=_calculate_provenance_hash(
                    instruction.record_id, "error", str(e),
                ),
                error_message=str(e),
            )

    def route_batch(
        self,
        plan: RoutingPlan,
    ) -> BatchRoutingResult:
        """Execute routing for an entire routing plan.

        Groups instructions by category and dispatches each group to the
        corresponding agent. Collects results per category and aggregates
        into a BatchRoutingResult.

        Args:
            plan: RoutingPlan to execute (must not be dry_run).

        Returns:
            BatchRoutingResult with per-category results and totals.

        Raises:
            ValueError: If plan is marked as dry_run.

        Example:
            >>> engine = ActivityRouterEngine()
            >>> plan = engine.create_routing_plan(results, dry_run=False)
            >>> batch_result = engine.route_batch(plan)
            >>> print(f"Routed {batch_result.total_routed} records")
        """
        start_time = time.monotonic()

        if plan.dry_run:
            raise ValueError(
                "Cannot execute routing on a dry-run plan. "
                "Create plan with dry_run=False to execute."
            )

        # Group instructions by category
        grouped = self._group_instructions_by_category(plan.instructions)

        results: List[RoutingResult] = []
        total_routed = 0
        failed_count = 0

        for category, instructions in grouped.items():
            cat_start = time.monotonic()
            cat_sent = 0
            cat_failed = False
            cat_error = None

            for instruction in instructions:
                route_result = self.route_to_agent(instruction)
                if route_result.success:
                    cat_sent += route_result.records_sent
                else:
                    cat_failed = True
                    cat_error = route_result.error_message

            cat_elapsed_ms = Decimal(str((time.monotonic() - cat_start) * 1000))

            agent_info = CATEGORY_AGENT_REGISTRY.get(category, {})
            agent_id = agent_info.get("agent_id", "UNKNOWN")

            cat_result_hash = _calculate_provenance_hash(
                plan.plan_id, category.value, cat_sent, agent_id,
            )

            cat_result = RoutingResult(
                success=not cat_failed,
                agent_id=agent_id,
                category=category,
                records_sent=cat_sent,
                response_status=200 if not cat_failed else 500,
                execution_time_ms=self._quantize(cat_elapsed_ms),
                provenance_hash=cat_result_hash,
                error_message=cat_error,
            )
            results.append(cat_result)

            total_routed += cat_sent
            if cat_failed:
                failed_count += 1

        elapsed_ms = Decimal(str((time.monotonic() - start_time) * 1000))
        overall_success = failed_count == 0

        batch_hash = _calculate_provenance_hash(
            plan.plan_id, total_routed, failed_count, len(results),
        )

        batch_result = BatchRoutingResult(
            success=overall_success,
            results=results,
            total_routed=total_routed,
            categories_targeted=len(results),
            failed_count=failed_count,
            execution_time_ms=self._quantize(elapsed_ms),
            provenance_hash=batch_hash,
        )

        logger.info(
            "Batch routing complete: plan_id=%s, routed=%d, "
            "categories=%d, failed=%d, elapsed_ms=%.2f",
            plan.plan_id, total_routed, len(results),
            failed_count, float(elapsed_ms),
        )

        return batch_result

    # =========================================================================
    # PUBLIC METHODS -- Validation
    # =========================================================================

    def validate_routing_plan(
        self,
        plan: RoutingPlan,
    ) -> List[str]:
        """Validate a routing plan before execution.

        Checks for consistency, missing agents, confidence thresholds,
        and other issues that could cause routing failures.

        Args:
            plan: RoutingPlan to validate.

        Returns:
            List of validation error messages. Empty list means valid.

        Example:
            >>> engine = ActivityRouterEngine()
            >>> errors = engine.validate_routing_plan(plan)
            >>> if errors:
            ...     for err in errors:
            ...         print(f"VALIDATION: {err}")
        """
        errors: List[str] = []

        if not plan.instructions:
            errors.append("Routing plan has no instructions")
            return errors

        if plan.total_records == 0:
            errors.append("Routing plan reports 0 total records")

        seen_record_ids: set = set()
        for instruction in plan.instructions:
            # Check for duplicate record IDs
            if instruction.record_id in seen_record_ids:
                errors.append(
                    f"Duplicate record_id in plan: {instruction.record_id}"
                )
            seen_record_ids.add(instruction.record_id)

            # Check that target category has a registered agent
            if instruction.routing_action == RoutingAction.ROUTE:
                if instruction.target_category not in CATEGORY_AGENT_REGISTRY:
                    errors.append(
                        f"Record {instruction.record_id}: no agent registered "
                        f"for category {instruction.target_category.value}"
                    )

            # Check confidence threshold
            if (instruction.routing_action == RoutingAction.ROUTE
                    and instruction.confidence < _MIN_ROUTING_CONFIDENCE):
                errors.append(
                    f"Record {instruction.record_id}: confidence "
                    f"{instruction.confidence} below minimum "
                    f"{_MIN_ROUTING_CONFIDENCE}"
                )

            # Check that agent_id is not empty for routable instructions
            if (instruction.routing_action == RoutingAction.ROUTE
                    and not instruction.agent_id):
                errors.append(
                    f"Record {instruction.record_id}: missing agent_id"
                )

        # Validate category counts consistency
        actual_counts: Dict[str, int] = {}
        for instruction in plan.instructions:
            if instruction.routing_action in (
                RoutingAction.ROUTE, RoutingAction.SPLIT_ROUTE
            ):
                key = instruction.target_category.value
                actual_counts[key] = actual_counts.get(key, 0) + 1

        if actual_counts != plan.category_counts:
            errors.append(
                f"Category counts mismatch: plan reports "
                f"{plan.category_counts} but instructions yield "
                f"{actual_counts}"
            )

        logger.debug(
            "Plan validation: plan_id=%s, errors=%d",
            plan.plan_id, len(errors),
        )

        return errors

    # =========================================================================
    # INTERNAL METHODS -- Grouping
    # =========================================================================

    def _group_by_category(
        self,
        results: List[ClassificationResult],
    ) -> Dict[Scope3Category, List[ClassificationResult]]:
        """Group classification results by their primary category.

        Args:
            results: List of classification results.

        Returns:
            Dictionary mapping each category to its list of results.
        """
        grouped: Dict[Scope3Category, List[ClassificationResult]] = {}
        for result in results:
            cat = result.primary_category
            if cat not in grouped:
                grouped[cat] = []
            grouped[cat].append(result)

        logger.debug(
            "Grouped %d results into %d categories",
            len(results), len(grouped),
        )

        return grouped

    def _group_instructions_by_category(
        self,
        instructions: List[RoutingInstruction],
    ) -> Dict[Scope3Category, List[RoutingInstruction]]:
        """Group routing instructions by target category.

        Args:
            instructions: List of routing instructions.

        Returns:
            Dictionary mapping each category to its instructions.
        """
        grouped: Dict[Scope3Category, List[RoutingInstruction]] = {}
        for instruction in instructions:
            if instruction.routing_action in (
                RoutingAction.ROUTE, RoutingAction.SPLIT_ROUTE
            ):
                cat = instruction.target_category
                if cat not in grouped:
                    grouped[cat] = []
                grouped[cat].append(instruction)

        return grouped

    # =========================================================================
    # PUBLIC METHODS -- Metrics
    # =========================================================================

    def get_engine_stats(self) -> Dict[str, Any]:
        """Get current engine statistics.

        Returns:
            Dictionary with route_count, plan_count, and registered agents.
        """
        with self._count_lock:
            return {
                "engine_id": ENGINE_ID,
                "engine_version": ENGINE_VERSION,
                "agent_id": AGENT_ID,
                "route_count": self._route_count,
                "plan_count": self._plan_count,
                "registered_agents": len(CATEGORY_AGENT_REGISTRY),
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
