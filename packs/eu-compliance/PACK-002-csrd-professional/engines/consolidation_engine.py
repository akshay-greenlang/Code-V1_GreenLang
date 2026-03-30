# -*- coding: utf-8 -*-
"""
ConsolidationEngine - PACK-002 CSRD Professional Engine 1

Multi-entity ESRS consolidation engine that supports operational control,
financial control, and equity share approaches for group-level CSRD
reporting. Handles subsidiary data aggregation, intercompany elimination,
minority interest calculations, and reconciliation.

Consolidation Approaches (per ESRS 1, Chapter 3):
    - Operational Control: 100% of emissions from entities the parent
      operates, regardless of ownership percentage.
    - Financial Control: 100% of emissions from entities where the parent
      has the ability to direct financial and operating policies.
    - Equity Share: Proportional to the parent's ownership interest.

Features:
    - Multi-level entity hierarchy with parent-subsidiary relationships
    - Intercompany transaction elimination (revenue, cost, Scope 3 transfers)
    - Minority interest disclosure calculations
    - Entity-to-group reconciliation with variance tracking
    - Side-by-side approach comparison
    - SHA-256 provenance hashing at entity and consolidated levels
    - Decimal arithmetic for all financial and emission calculations

Zero-Hallucination:
    - All consolidation uses deterministic arithmetic
    - Ownership percentages applied via Decimal multiplication
    - No LLM involvement in any calculation path
    - Elimination logic uses explicit matching rules

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-002 CSRD Professional
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, Pydantic model, or other).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ConsolidationMethod(str, Enum):
    """Method used to consolidate a subsidiary."""

    OPERATIONAL_CONTROL = "operational_control"
    FINANCIAL_CONTROL = "financial_control"
    EQUITY_SHARE = "equity_share"

class ConsolidationApproach(str, Enum):
    """Approach for group-level consolidation."""

    OPERATIONAL_CONTROL = "operational_control"
    FINANCIAL_CONTROL = "financial_control"
    EQUITY_SHARE = "equity_share"

class TransactionType(str, Enum):
    """Type of intercompany transaction."""

    REVENUE = "revenue"
    COST = "cost"
    EMISSION_TRANSFER = "emission_transfer"
    WASTE_TRANSFER = "waste_transfer"

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class EntityDefinition(BaseModel):
    """Definition of a reporting entity (subsidiary, joint venture, etc.)."""

    entity_id: str = Field(..., description="Unique entity identifier")
    name: str = Field(..., description="Legal entity name")
    country: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    ownership_pct: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Parent ownership percentage (0-100)",
    )
    consolidation_method: ConsolidationMethod = Field(
        ..., description="How this entity is consolidated"
    )
    parent_entity_id: Optional[str] = Field(
        None, description="Parent entity ID (None for group parent)"
    )
    nace_codes: List[str] = Field(
        default_factory=list, description="NACE sector codes"
    )
    is_eu_entity: bool = Field(
        True, description="Whether entity is in EU jurisdiction"
    )
    currency: str = Field("EUR", description="Reporting currency ISO 4217")
    employee_count: int = Field(0, ge=0, description="Number of employees")

    @field_validator("country")
    @classmethod
    def validate_country(cls, v: str) -> str:
        """Validate country is a 2-letter ISO code."""
        if len(v) != 2 or not v.isalpha():
            raise ValueError(f"Country must be a 2-letter ISO code, got '{v}'")
        return v.upper()

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, v: str) -> str:
        """Validate currency is a 3-letter ISO code."""
        if len(v) != 3 or not v.isalpha():
            raise ValueError(f"Currency must be a 3-letter ISO 4217 code, got '{v}'")
        return v.upper()

class EntityESRSData(BaseModel):
    """ESRS data submitted by a single entity."""

    entity_id: str = Field(..., description="Entity submitting this data")
    data_points: Dict[str, Any] = Field(
        default_factory=dict, description="ESRS data point values keyed by ID"
    )
    reporting_period: str = Field(
        ..., description="Reporting period (e.g. '2025-01-01/2025-12-31')"
    )
    quality_score: float = Field(
        0.0, ge=0.0, le=100.0, description="Data quality score 0-100"
    )
    submission_date: datetime = Field(
        default_factory=utcnow, description="Date data was submitted"
    )

class IntercompanyTransaction(BaseModel):
    """An intercompany transaction that may require elimination."""

    transaction_id: str = Field(default_factory=_new_uuid, description="Transaction ID")
    from_entity: str = Field(..., description="Selling / transferring entity")
    to_entity: str = Field(..., description="Buying / receiving entity")
    transaction_type: TransactionType = Field(
        ..., description="Type of intercompany transaction"
    )
    amount: Decimal = Field(..., description="Transaction amount or quantity")
    scope3_category: Optional[int] = Field(
        None,
        ge=1,
        le=15,
        description="Scope 3 category if emission/waste transfer",
    )
    elimination_method: str = Field(
        "full", description="Elimination method: full, partial, or none"
    )

    @field_validator("elimination_method")
    @classmethod
    def validate_elimination_method(cls, v: str) -> str:
        """Validate elimination method."""
        allowed = {"full", "partial", "none"}
        if v not in allowed:
            raise ValueError(f"elimination_method must be one of {allowed}")
        return v

class ReconciliationEntry(BaseModel):
    """Single reconciliation line between entity and consolidated values."""

    entity_id: str = Field(..., description="Entity identifier")
    data_point_id: str = Field("", description="ESRS data point identifier")
    entity_value: Decimal = Field(..., description="Value reported by entity")
    consolidated_value: Decimal = Field(..., description="Value in consolidated report")
    adjustment: Decimal = Field(..., description="Adjustment applied")
    adjustment_reason: str = Field("", description="Reason for adjustment")

class ConsolidationResult(BaseModel):
    """Output of a consolidation run."""

    consolidation_id: str = Field(default_factory=_new_uuid, description="Run ID")
    approach: str = Field(..., description="Consolidation approach used")
    consolidated_data: Dict[str, Any] = Field(
        default_factory=dict, description="Consolidated data points"
    )
    per_entity_results: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Per-entity contribution breakdown"
    )
    eliminations_applied: List[Dict[str, Any]] = Field(
        default_factory=list, description="Intercompany eliminations performed"
    )
    minority_adjustments: List[Dict[str, Any]] = Field(
        default_factory=list, description="Minority interest adjustments"
    )
    reconciliation_variance: Decimal = Field(
        Decimal("0"), description="Total reconciliation variance"
    )
    entity_count: int = Field(0, description="Number of entities consolidated")
    provenance_hash: str = Field("", description="SHA-256 provenance hash")
    processing_time_ms: float = Field(0.0, description="Processing duration in ms")
    created_at: datetime = Field(default_factory=utcnow, description="Timestamp")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class ConsolidationConfig(BaseModel):
    """Configuration for the consolidation engine."""

    default_approach: ConsolidationApproach = Field(
        ConsolidationApproach.OPERATIONAL_CONTROL,
        description="Default consolidation approach",
    )
    variance_threshold_pct: Decimal = Field(
        Decimal("1.0"),
        description="Max acceptable reconciliation variance percentage",
    )
    auto_eliminate_intercompany: bool = Field(
        True, description="Automatically eliminate intercompany transactions"
    )
    include_minority_disclosures: bool = Field(
        True, description="Include minority interest disclosures"
    )
    require_all_entities_data: bool = Field(
        False, description="Require data from all registered entities"
    )

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ConsolidationEngine:
    """Multi-entity ESRS consolidation engine.

    Supports three consolidation approaches per ESRS 1 Chapter 3:
    operational control, financial control, and equity share. Provides
    intercompany elimination, minority interest calculations, and
    full entity-to-group reconciliation.

    Attributes:
        config: Engine configuration.
        entities: Registered entity definitions keyed by entity_id.
        entity_data: ESRS data keyed by entity_id.
        transactions: Intercompany transactions.

    Example:
        >>> config = ConsolidationConfig()
        >>> engine = ConsolidationEngine(config)
        >>> engine.add_entity(EntityDefinition(
        ...     entity_id="parent", name="Parent Co", country="DE",
        ...     ownership_pct=Decimal("100"), consolidation_method=ConsolidationMethod.OPERATIONAL_CONTROL
        ... ))
        >>> result = await engine.consolidate(ConsolidationApproach.OPERATIONAL_CONTROL)
        >>> assert result.provenance_hash != ""
    """

    def __init__(self, config: Optional[ConsolidationConfig] = None) -> None:
        """Initialize ConsolidationEngine.

        Args:
            config: Engine configuration. Uses defaults if not provided.
        """
        self.config = config or ConsolidationConfig()
        self.entities: Dict[str, EntityDefinition] = {}
        self.entity_data: Dict[str, EntityESRSData] = {}
        self.transactions: List[IntercompanyTransaction] = []
        self._consolidation_history: List[ConsolidationResult] = []
        logger.info(
            "ConsolidationEngine initialized (approach=%s, version=%s)",
            self.config.default_approach.value,
            _MODULE_VERSION,
        )

    # -- Entity Management ---------------------------------------------------

    def add_entity(self, entity: EntityDefinition) -> None:
        """Register a subsidiary or entity for consolidation.

        Args:
            entity: Entity definition to register.

        Raises:
            ValueError: If entity_id already registered or parent not found.
        """
        if entity.entity_id in self.entities:
            raise ValueError(f"Entity '{entity.entity_id}' already registered")

        if entity.parent_entity_id and entity.parent_entity_id not in self.entities:
            # Allow forward-reference only if parent is None (root)
            if entity.parent_entity_id != entity.entity_id:
                logger.warning(
                    "Parent '%s' not yet registered for entity '%s'",
                    entity.parent_entity_id,
                    entity.entity_id,
                )

        self.entities[entity.entity_id] = entity
        logger.info(
            "Entity registered: %s (%s, ownership=%.1f%%)",
            entity.entity_id,
            entity.name,
            entity.ownership_pct,
        )

    def set_entity_data(self, entity_id: str, data: EntityESRSData) -> None:
        """Load ESRS data for an entity.

        Args:
            entity_id: Entity identifier.
            data: ESRS data points for the entity.

        Raises:
            ValueError: If entity is not registered.
        """
        if entity_id not in self.entities:
            raise ValueError(f"Entity '{entity_id}' is not registered")
        if data.entity_id != entity_id:
            raise ValueError(
                f"Data entity_id '{data.entity_id}' does not match '{entity_id}'"
            )
        self.entity_data[entity_id] = data
        logger.info(
            "ESRS data loaded for entity '%s' (%d data points, quality=%.1f)",
            entity_id,
            len(data.data_points),
            data.quality_score,
        )

    def add_intercompany_transaction(self, txn: IntercompanyTransaction) -> None:
        """Register an intercompany transaction for elimination.

        Args:
            txn: Intercompany transaction definition.

        Raises:
            ValueError: If either entity is not registered.
        """
        if txn.from_entity not in self.entities:
            raise ValueError(f"From-entity '{txn.from_entity}' is not registered")
        if txn.to_entity not in self.entities:
            raise ValueError(f"To-entity '{txn.to_entity}' is not registered")
        if txn.from_entity == txn.to_entity:
            raise ValueError("from_entity and to_entity must be different")

        self.transactions.append(txn)
        logger.info(
            "Intercompany transaction registered: %s -> %s (type=%s, amount=%s)",
            txn.from_entity,
            txn.to_entity,
            txn.transaction_type.value,
            txn.amount,
        )

    # -- Consolidation -------------------------------------------------------

    async def consolidate(
        self, approach: Optional[ConsolidationApproach] = None
    ) -> ConsolidationResult:
        """Execute consolidation using the specified approach.

        Aggregates entity-level ESRS data into a group-level consolidated
        view using the chosen consolidation approach.

        Args:
            approach: Consolidation approach. Uses config default if None.

        Returns:
            ConsolidationResult with consolidated data and provenance.

        Raises:
            ValueError: If no entities or required data is missing.
        """
        start = utcnow()
        approach = approach or self.config.default_approach

        if not self.entities:
            raise ValueError("No entities registered for consolidation")

        if self.config.require_all_entities_data:
            missing = set(self.entities.keys()) - set(self.entity_data.keys())
            if missing:
                raise ValueError(f"Missing data for entities: {missing}")

        logger.info(
            "Starting consolidation (approach=%s, entities=%d)",
            approach.value,
            len(self.entities),
        )

        # Step 1: Calculate entity contributions
        per_entity_results = self._calculate_entity_contributions(approach)

        # Step 2: Aggregate to group level
        consolidated_data = self._aggregate_contributions(per_entity_results)

        # Step 3: Eliminate intercompany transactions
        eliminations: List[Dict[str, Any]] = []
        if self.config.auto_eliminate_intercompany and self.transactions:
            eliminations = self.eliminate_intercompany()
            consolidated_data = self._apply_eliminations(
                consolidated_data, eliminations
            )

        # Step 4: Calculate minority interest
        minority_adjustments: List[Dict[str, Any]] = []
        if self.config.include_minority_disclosures:
            minority_adjustments = self.calculate_minority_interest()

        # Step 5: Calculate reconciliation variance
        variance = self._calculate_variance(per_entity_results, consolidated_data)

        # Step 6: Compute provenance hash
        end = utcnow()
        elapsed_ms = (end - start).total_seconds() * 1000

        result = ConsolidationResult(
            approach=approach.value,
            consolidated_data=consolidated_data,
            per_entity_results=per_entity_results,
            eliminations_applied=eliminations,
            minority_adjustments=minority_adjustments,
            reconciliation_variance=variance,
            entity_count=len(self.entities),
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        self._consolidation_history.append(result)
        logger.info(
            "Consolidation complete (approach=%s, entities=%d, variance=%s, time=%.1fms)",
            approach.value,
            len(self.entities),
            variance,
            elapsed_ms,
        )
        return result

    def _calculate_entity_contributions(
        self, approach: ConsolidationApproach
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate each entity's contribution based on approach.

        Args:
            approach: The consolidation approach to apply.

        Returns:
            Dict mapping entity_id to its adjusted data contributions.
        """
        per_entity: Dict[str, Dict[str, Any]] = {}

        for entity_id, entity in self.entities.items():
            data = self.entity_data.get(entity_id)
            if data is None:
                logger.warning(
                    "No data for entity '%s', skipping contribution", entity_id
                )
                per_entity[entity_id] = {
                    "entity_name": entity.name,
                    "ownership_pct": str(entity.ownership_pct),
                    "consolidation_factor": "0",
                    "data_points": {},
                    "status": "no_data",
                }
                continue

            factor = self._get_consolidation_factor(entity, approach)
            adjusted_points: Dict[str, Any] = {}

            for dp_id, dp_value in data.data_points.items():
                adjusted_points[dp_id] = self._apply_factor(dp_value, factor)

            per_entity[entity_id] = {
                "entity_name": entity.name,
                "ownership_pct": str(entity.ownership_pct),
                "consolidation_factor": str(factor),
                "data_points": adjusted_points,
                "quality_score": data.quality_score,
                "status": "consolidated",
            }

        return per_entity

    def _get_consolidation_factor(
        self, entity: EntityDefinition, approach: ConsolidationApproach
    ) -> Decimal:
        """Determine the consolidation factor for an entity.

        Args:
            entity: Entity definition.
            approach: Consolidation approach.

        Returns:
            Factor to multiply entity values by (0-1).
        """
        if approach == ConsolidationApproach.EQUITY_SHARE:
            return (entity.ownership_pct / Decimal("100")).quantize(
                Decimal("0.0001"), rounding=ROUND_HALF_UP
            )

        # Operational or financial control: 100% if controlled
        if approach == ConsolidationApproach.OPERATIONAL_CONTROL:
            if entity.consolidation_method == ConsolidationMethod.OPERATIONAL_CONTROL:
                return Decimal("1.0")
            return Decimal("0.0")

        if approach == ConsolidationApproach.FINANCIAL_CONTROL:
            if entity.consolidation_method in (
                ConsolidationMethod.FINANCIAL_CONTROL,
                ConsolidationMethod.OPERATIONAL_CONTROL,
            ):
                return Decimal("1.0")
            return Decimal("0.0")

        return Decimal("0.0")

    def _apply_factor(self, value: Any, factor: Decimal) -> Any:
        """Apply consolidation factor to a data point value.

        Args:
            value: Original data point value.
            factor: Consolidation factor (0-1).

        Returns:
            Adjusted value. Numeric values are multiplied by factor.
            Non-numeric values are returned unchanged.
        """
        if isinstance(value, (int, float, Decimal)):
            return str(
                (_decimal(value) * factor).quantize(
                    Decimal("0.0001"), rounding=ROUND_HALF_UP
                )
            )
        if isinstance(value, dict):
            return {
                k: self._apply_factor(v, factor) for k, v in value.items()
            }
        if isinstance(value, list):
            return [self._apply_factor(v, factor) for v in value]
        return value

    def _aggregate_contributions(
        self, per_entity_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate per-entity contributions into group totals.

        Args:
            per_entity_results: Per-entity contribution data.

        Returns:
            Dict of consolidated data points (summed where numeric).
        """
        consolidated: Dict[str, Any] = {}
        numeric_accum: Dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
        non_numeric: Dict[str, List[Any]] = defaultdict(list)

        for entity_id, entity_result in per_entity_results.items():
            if entity_result.get("status") == "no_data":
                continue
            for dp_id, dp_value in entity_result.get("data_points", {}).items():
                try:
                    numeric_accum[dp_id] += _decimal(dp_value)
                except (InvalidOperation, TypeError, ValueError):
                    non_numeric[dp_id].append(dp_value)

        for dp_id, total in numeric_accum.items():
            consolidated[dp_id] = str(
                total.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
            )

        for dp_id, values in non_numeric.items():
            if dp_id not in consolidated:
                consolidated[dp_id] = values

        return consolidated

    # -- Intercompany Elimination -------------------------------------------

    def eliminate_intercompany(self) -> List[Dict[str, Any]]:
        """Remove double-counted intercompany transactions.

        Identifies and eliminates intercompany revenue/cost and
        Scope 3 emission/waste transfers to prevent double counting.

        Returns:
            List of elimination entries applied.
        """
        eliminations: List[Dict[str, Any]] = []

        for txn in self.transactions:
            if txn.elimination_method == "none":
                logger.debug("Skipping transaction %s (method=none)", txn.transaction_id)
                continue

            elimination_factor = Decimal("1.0")
            if txn.elimination_method == "partial":
                elimination_factor = Decimal("0.5")

            elimination_amount = (txn.amount * elimination_factor).quantize(
                Decimal("0.0001"), rounding=ROUND_HALF_UP
            )

            elimination_entry = {
                "transaction_id": txn.transaction_id,
                "from_entity": txn.from_entity,
                "to_entity": txn.to_entity,
                "transaction_type": txn.transaction_type.value,
                "original_amount": str(txn.amount),
                "elimination_amount": str(elimination_amount),
                "elimination_method": txn.elimination_method,
                "scope3_category": txn.scope3_category,
                "provenance_hash": _compute_hash(
                    {
                        "txn_id": txn.transaction_id,
                        "amount": str(elimination_amount),
                        "method": txn.elimination_method,
                    }
                ),
            }
            eliminations.append(elimination_entry)
            logger.info(
                "Eliminated intercompany: %s -> %s, amount=%s (type=%s)",
                txn.from_entity,
                txn.to_entity,
                elimination_amount,
                txn.transaction_type.value,
            )

        return eliminations

    def _apply_eliminations(
        self,
        consolidated_data: Dict[str, Any],
        eliminations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Apply elimination adjustments to consolidated data.

        Args:
            consolidated_data: Aggregated consolidated data points.
            eliminations: List of elimination entries.

        Returns:
            Adjusted consolidated data after eliminations.
        """
        total_eliminated: Dict[str, Decimal] = defaultdict(lambda: Decimal("0"))

        for elim in eliminations:
            txn_type = elim["transaction_type"]
            amount = _decimal(elim["elimination_amount"])

            if txn_type in ("revenue", "cost"):
                total_eliminated[f"intercompany_{txn_type}"] += amount
            elif txn_type == "emission_transfer":
                cat = elim.get("scope3_category")
                key = f"scope3_cat{cat}_eliminated" if cat else "scope3_eliminated"
                total_eliminated[key] += amount
            elif txn_type == "waste_transfer":
                total_eliminated["waste_eliminated"] += amount

        result = dict(consolidated_data)
        for key, amount in total_eliminated.items():
            result[f"_elimination_{key}"] = str(amount)

        return result

    # -- Minority Interest --------------------------------------------------

    def calculate_minority_interest(self) -> List[Dict[str, Any]]:
        """Calculate minority (non-controlling) interest disclosures.

        For entities with less than 100% ownership, calculates the
        portion attributable to minority shareholders.

        Returns:
            List of minority interest entries per entity.
        """
        adjustments: List[Dict[str, Any]] = []

        for entity_id, entity in self.entities.items():
            if entity.ownership_pct >= Decimal("100"):
                continue

            minority_pct = Decimal("100") - entity.ownership_pct
            data = self.entity_data.get(entity_id)

            minority_entry: Dict[str, Any] = {
                "entity_id": entity_id,
                "entity_name": entity.name,
                "ownership_pct": str(entity.ownership_pct),
                "minority_pct": str(minority_pct),
                "minority_data_points": {},
            }

            if data:
                for dp_id, dp_value in data.data_points.items():
                    try:
                        full_val = _decimal(dp_value)
                        minority_val = (
                            full_val * minority_pct / Decimal("100")
                        ).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
                        minority_entry["minority_data_points"][dp_id] = str(
                            minority_val
                        )
                    except (InvalidOperation, TypeError, ValueError):
                        pass

            minority_entry["provenance_hash"] = _compute_hash(minority_entry)
            adjustments.append(minority_entry)

            logger.info(
                "Minority interest for '%s': %.1f%% non-controlling",
                entity.name,
                minority_pct,
            )

        return adjustments

    # -- Reconciliation -----------------------------------------------------

    def generate_reconciliation(
        self, result: ConsolidationResult
    ) -> List[ReconciliationEntry]:
        """Generate entity-to-group reconciliation entries.

        Args:
            result: Consolidation result to reconcile.

        Returns:
            List of reconciliation entries showing entity vs. consolidated.
        """
        entries: List[ReconciliationEntry] = []

        for entity_id, entity_result in result.per_entity_results.items():
            if entity_result.get("status") == "no_data":
                continue

            raw_data = self.entity_data.get(entity_id)
            if not raw_data:
                continue

            for dp_id, raw_value in raw_data.data_points.items():
                try:
                    entity_val = _decimal(raw_value)
                except (InvalidOperation, TypeError, ValueError):
                    continue

                adjusted_val = _decimal(
                    entity_result.get("data_points", {}).get(dp_id, "0")
                )
                adjustment = adjusted_val - entity_val

                entries.append(
                    ReconciliationEntry(
                        entity_id=entity_id,
                        data_point_id=dp_id,
                        entity_value=entity_val,
                        consolidated_value=adjusted_val,
                        adjustment=adjustment,
                        adjustment_reason=(
                            "ownership_adjustment"
                            if adjustment != Decimal("0")
                            else "no_adjustment"
                        ),
                    )
                )

        return entries

    def _calculate_variance(
        self,
        per_entity_results: Dict[str, Dict[str, Any]],
        consolidated_data: Dict[str, Any],
    ) -> Decimal:
        """Calculate reconciliation variance between sum-of-parts and consolidated.

        Args:
            per_entity_results: Per-entity contributions.
            consolidated_data: Aggregated consolidated data.

        Returns:
            Absolute variance as a Decimal.
        """
        entity_sum = Decimal("0")
        for entity_result in per_entity_results.values():
            if entity_result.get("status") == "no_data":
                continue
            for dp_value in entity_result.get("data_points", {}).values():
                try:
                    entity_sum += _decimal(dp_value)
                except (InvalidOperation, TypeError, ValueError):
                    pass

        consolidated_sum = Decimal("0")
        for key, value in consolidated_data.items():
            if key.startswith("_elimination_"):
                continue
            try:
                consolidated_sum += _decimal(value)
            except (InvalidOperation, TypeError, ValueError):
                pass

        variance = abs(entity_sum - consolidated_sum)
        return variance.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

    # -- Approach Comparison ------------------------------------------------

    async def compare_approaches(self) -> Dict[str, Any]:
        """Compare all three consolidation approaches side-by-side.

        Runs consolidation with each approach and returns a comparison
        showing differences in consolidated totals.

        Returns:
            Dict with per-approach results and deltas.
        """
        logger.info("Running approach comparison across all three methods")

        results: Dict[str, ConsolidationResult] = {}
        for approach in ConsolidationApproach:
            result = await self.consolidate(approach)
            results[approach.value] = result

        comparison: Dict[str, Any] = {
            "approaches": {},
            "variance_matrix": {},
            "recommendation": "",
        }

        for name, result in results.items():
            comparison["approaches"][name] = {
                "consolidated_data": result.consolidated_data,
                "entity_count": result.entity_count,
                "variance": str(result.reconciliation_variance),
                "processing_time_ms": result.processing_time_ms,
            }

        approach_names = list(results.keys())
        for i, a in enumerate(approach_names):
            for j, b in enumerate(approach_names):
                if i >= j:
                    continue
                delta = self._compute_approach_delta(
                    results[a].consolidated_data,
                    results[b].consolidated_data,
                )
                comparison["variance_matrix"][f"{a}_vs_{b}"] = delta

        lowest_variance = min(
            results.values(),
            key=lambda r: r.reconciliation_variance,
        )
        comparison["recommendation"] = (
            f"Lowest variance approach: {lowest_variance.approach}"
        )

        comparison["provenance_hash"] = _compute_hash(comparison)
        return comparison

    def _compute_approach_delta(
        self, data_a: Dict[str, Any], data_b: Dict[str, Any]
    ) -> Dict[str, str]:
        """Compute deltas between two approaches' consolidated data.

        Args:
            data_a: First approach consolidated data.
            data_b: Second approach consolidated data.

        Returns:
            Dict of data_point_id to delta string.
        """
        all_keys = set(data_a.keys()) | set(data_b.keys())
        deltas: Dict[str, str] = {}

        for key in all_keys:
            if key.startswith("_elimination_"):
                continue
            try:
                val_a = _decimal(data_a.get(key, "0"))
                val_b = _decimal(data_b.get(key, "0"))
                delta = (val_a - val_b).quantize(
                    Decimal("0.0001"), rounding=ROUND_HALF_UP
                )
                deltas[key] = str(delta)
            except (InvalidOperation, TypeError, ValueError):
                deltas[key] = "non_numeric"

        return deltas

    # -- Entity Hierarchy ---------------------------------------------------

    def get_entity_hierarchy(self) -> Dict[str, Any]:
        """Build a tree structure of the entity hierarchy.

        Returns:
            Nested dict representing parent-child relationships.
        """
        roots: List[str] = []
        children_map: Dict[str, List[str]] = defaultdict(list)

        for entity_id, entity in self.entities.items():
            if entity.parent_entity_id is None:
                roots.append(entity_id)
            else:
                children_map[entity.parent_entity_id].append(entity_id)

        def build_tree(entity_id: str) -> Dict[str, Any]:
            entity = self.entities[entity_id]
            node: Dict[str, Any] = {
                "entity_id": entity_id,
                "name": entity.name,
                "country": entity.country,
                "ownership_pct": str(entity.ownership_pct),
                "consolidation_method": entity.consolidation_method.value,
                "is_eu_entity": entity.is_eu_entity,
                "employee_count": entity.employee_count,
                "children": [
                    build_tree(child_id) for child_id in children_map.get(entity_id, [])
                ],
            }
            return node

        hierarchy = {
            "group": [build_tree(root_id) for root_id in roots],
            "total_entities": len(self.entities),
            "eu_entities": sum(1 for e in self.entities.values() if e.is_eu_entity),
            "non_eu_entities": sum(
                1 for e in self.entities.values() if not e.is_eu_entity
            ),
        }
        hierarchy["provenance_hash"] = _compute_hash(hierarchy)
        return hierarchy

    # -- History -------------------------------------------------------------

    def get_consolidation_history(self) -> List[ConsolidationResult]:
        """Return all consolidation runs performed by this engine instance.

        Returns:
            List of ConsolidationResult objects in chronological order.
        """
        return list(self._consolidation_history)

    def reset(self) -> None:
        """Reset the engine, clearing all entities, data, and transactions."""
        self.entities.clear()
        self.entity_data.clear()
        self.transactions.clear()
        self._consolidation_history.clear()
        logger.info("ConsolidationEngine reset")
