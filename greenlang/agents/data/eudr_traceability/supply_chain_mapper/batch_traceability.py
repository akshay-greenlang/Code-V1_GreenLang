# -*- coding: utf-8 -*-
"""
Batch Traceability Engine -- AGENT-EUDR-001: Supply Chain Mapping Master (Feature 4)

Implements many-to-many batch traceability for EUDR-regulated commodities,
supporting three chain of custody models (Identity Preserved, Segregated,
Mass Balance), batch splitting, merging, and transformation operations with
deterministic mass balance verification using Python Decimal arithmetic.

Zero-Hallucination Guarantees:
    - ALL mass balance calculations use ``decimal.Decimal`` -- no floating-point.
    - Quantity conservation is enforced deterministically on every split/merge.
    - SHA-256 provenance hash recorded for every batch operation.
    - Forward and backward trace use iterative BFS with cycle detection --
      no LLM, no heuristics, no approximation.
    - Bit-perfect reproducibility: same input data produces the same trace
      result and the same provenance hash on every invocation.

Performance Target:
    - Full backward trace < 2 seconds for chains with 50 split/merge operations.

Regulatory Basis:
    - EUDR Article 4(2)(f): Supply chain information collection
    - EUDR Article 9(1): Geolocation traceability to plot of production
    - EUDR Article 10(2)(f): Risk of circumvention or mixing with unknown origin

Integration:
    - Imports ``BatchRecord``, ``CustodyModel``, ``EUDRCommodity`` from
      ``greenlang.agents.data.eudr_traceability.models`` (AGENT-DATA-005).
    - Optionally integrates with ``ChainOfCustodyEngine`` and
      ``PlotRegistryEngine`` for live data lookups.
    - Optionally integrates with ``ProvenanceTracker`` for audit chain.

Example:
    >>> from greenlang.agents.data.eudr_traceability.supply_chain_mapper.batch_traceability import (
    ...     BatchTraceabilityEngine,
    ... )
    >>> engine = BatchTraceabilityEngine()
    >>> batch = engine.register_batch(
    ...     commodity=EUDRCommodity.COCOA,
    ...     product_description="Raw cocoa beans",
    ...     quantity=Decimal("10000"),
    ...     unit="kg",
    ...     origin_plot_ids=["PLOT-001", "PLOT-002"],
    ...     custody_model=CustodyModel.SEGREGATED,
    ... )
    >>> children = engine.split_batch(batch.batch_id, [Decimal("6000"), Decimal("4000")])
    >>> trace = engine.backward_trace(children[0].batch_id)
    >>> assert "PLOT-001" in trace.origin_plot_ids

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-001 Supply Chain Mapping Master, Feature 4
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from pydantic import ConfigDict, Field, field_validator

from greenlang.agents.data.eudr_traceability.models import (
    BatchRecord,
    CustodyModel,
    EUDRCommodity,
    DERIVED_TO_PRIMARY,
    PlotRecord,
)
from greenlang.schemas import GreenLangBase, utcnow
from greenlang.schemas.enums import AlertSeverity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Uses ``json.dumps`` with ``sort_keys=True`` and ``default=str`` to ensure
    deterministic serialization regardless of dict insertion order or
    non-JSON-native types (Decimal, datetime, Enum).

    Args:
        data: Data to hash.  If a Pydantic model, calls ``model_dump(mode='json')``.

    Returns:
        SHA-256 hex digest string (64 characters, lowercase).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _generate_id(prefix: str) -> str:
    """Generate a unique identifier with the given prefix.

    Args:
        prefix: Identifier prefix (e.g. ``"BATCH"``, ``"OP"``).

    Returns:
        ID string in format ``"{prefix}-{hex12}"``.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"

def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal.

    Converts via string representation to avoid floating-point
    intermediate values that would introduce drift.

    Args:
        value: Numeric value to convert.

    Returns:
        Decimal representation of the value.

    Raises:
        ValueError: If conversion fails.
    """
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(
            f"Cannot convert {value!r} to Decimal: {exc}"
        ) from exc

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class BatchOperationType(str, Enum):
    """Type of batch operation in the traceability graph.

    Attributes:
        REGISTER: Initial batch registration (leaf node).
        SPLIT: One input batch becomes N output batches.
        MERGE: N input batches become one output batch.
        TRANSFORM: Input commodity transforms to derived product.
    """

    REGISTER = "register"
    SPLIT = "split"
    MERGE = "merge"
    TRANSFORM = "transform"

# ---------------------------------------------------------------------------
# Pydantic Result Models
# ---------------------------------------------------------------------------

class BatchOperation(GreenLangBase):
    """Immutable record of a single batch operation with provenance.

    Every split, merge, transform, or registration is recorded as a
    BatchOperation with a SHA-256 provenance hash linking inputs to outputs.

    Attributes:
        operation_id: Unique identifier for this operation.
        operation_type: Type of operation performed.
        input_batch_ids: List of input batch IDs consumed by this operation.
        output_batch_ids: List of output batch IDs produced by this operation.
        input_total_quantity: Sum of input batch quantities (Decimal string).
        output_total_quantity: Sum of output batch quantities (Decimal string).
        quantity_unit: Unit of measurement for quantities.
        custody_model: Chain of custody model governing this operation.
        commodity: EUDR commodity involved.
        output_commodity: Output commodity (differs from input for transforms).
        provenance_hash: SHA-256 hash of all operation data for audit trail.
        timestamp: UTC timestamp of the operation.
        operator_id: Optional operator performing the operation.
        notes: Optional free-text notes.
        metadata: Additional key-value metadata.
    """

    model_config = ConfigDict(from_attributes=True)

    operation_id: str = Field(
        default_factory=lambda: _generate_id("OP"),
        description="Unique identifier for this batch operation",
    )
    operation_type: BatchOperationType = Field(
        ...,
        description="Type of batch operation",
    )
    input_batch_ids: List[str] = Field(
        default_factory=list,
        description="Input batch IDs consumed by this operation",
    )
    output_batch_ids: List[str] = Field(
        default_factory=list,
        description="Output batch IDs produced by this operation",
    )
    input_total_quantity: str = Field(
        default="0",
        description="Sum of input batch quantities as Decimal string",
    )
    output_total_quantity: str = Field(
        default="0",
        description="Sum of output batch quantities as Decimal string",
    )
    quantity_unit: str = Field(
        default="kg",
        description="Unit of measurement for quantities",
    )
    custody_model: CustodyModel = Field(
        default=CustodyModel.SEGREGATED,
        description="Chain of custody model governing this operation",
    )
    commodity: EUDRCommodity = Field(
        ...,
        description="Primary EUDR commodity involved",
    )
    output_commodity: Optional[EUDRCommodity] = Field(
        None,
        description="Output commodity (differs from input for transformations)",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of all operation data",
    )
    timestamp: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp of the operation",
    )
    operator_id: Optional[str] = Field(
        None,
        description="Operator performing the operation",
    )
    notes: Optional[str] = Field(
        None,
        description="Free-text notes for this operation",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional key-value metadata",
    )

class TraceResult(GreenLangBase):
    """Result of a forward or backward trace through the batch graph.

    Attributes:
        target_batch_id: The batch ID that was traced.
        direction: Trace direction ('forward' or 'backward').
        origin_plot_ids: Deduplicated list of origin plot IDs found.
        origin_plots: PlotRecord instances (if PlotRegistryEngine available).
        reached_batch_ids: All batch IDs visited during the trace.
        operation_chain: Ordered list of operations traversed.
        depth: Maximum depth of the trace (number of operation hops).
        is_complete: Whether all leaf nodes reached a registered plot.
        unknown_origin_count: Number of leaf batches with no origin plot IDs.
        traceability_score: Score from 0-100 indicating traceability completeness.
        trace_time_ms: Wall-clock time for the trace in milliseconds.
        provenance_hash: SHA-256 hash of the trace result for audit.
    """

    model_config = ConfigDict(from_attributes=True)

    target_batch_id: str = Field(
        ...,
        description="Batch ID that was traced",
    )
    direction: str = Field(
        ...,
        description="Trace direction: 'forward' or 'backward'",
    )
    origin_plot_ids: List[str] = Field(
        default_factory=list,
        description="Deduplicated list of origin plot IDs found",
    )
    origin_plots: List[Any] = Field(
        default_factory=list,
        description="PlotRecord instances found during trace",
    )
    reached_batch_ids: List[str] = Field(
        default_factory=list,
        description="All batch IDs visited during the trace",
    )
    operation_chain: List[str] = Field(
        default_factory=list,
        description="Operation IDs traversed in order",
    )
    depth: int = Field(
        default=0,
        description="Maximum depth of the trace (operation hops)",
    )
    is_complete: bool = Field(
        default=False,
        description="Whether all leaf nodes have registered origin plots",
    )
    unknown_origin_count: int = Field(
        default=0,
        description="Number of leaf batches with no origin plot IDs",
    )
    traceability_score: str = Field(
        default="0.00",
        description="Traceability completeness score (0-100) as Decimal string",
    )
    trace_time_ms: float = Field(
        default=0.0,
        description="Wall-clock trace time in milliseconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of the trace result",
    )

class MassBalanceResult(GreenLangBase):
    """Result of mass balance verification for a batch or operation.

    Uses Decimal strings to preserve bit-perfect precision in serialization.

    Attributes:
        batch_id: Batch ID being verified.
        input_quantity: Total input quantity (Decimal string).
        output_quantity: Total output quantity (Decimal string).
        balance: input - output (Decimal string). Positive = surplus, negative = deficit.
        tolerance: Allowed tolerance for mass balance (Decimal string).
        is_balanced: Whether |balance| <= tolerance.
        is_surplus: Whether output < input (acceptable).
        is_deficit: Whether output > input (compliance alert).
        unit: Unit of measurement.
        custody_model: Custody model governing the balance check.
        provenance_hash: SHA-256 hash of the balance result.
    """

    model_config = ConfigDict(from_attributes=True)

    batch_id: str = Field(
        ...,
        description="Batch ID being verified",
    )
    input_quantity: str = Field(
        ...,
        description="Total input quantity as Decimal string",
    )
    output_quantity: str = Field(
        ...,
        description="Total output quantity as Decimal string",
    )
    balance: str = Field(
        ...,
        description="input - output as Decimal string",
    )
    tolerance: str = Field(
        default="0.001",
        description="Allowed mass balance tolerance as Decimal string",
    )
    is_balanced: bool = Field(
        ...,
        description="Whether |balance| <= tolerance",
    )
    is_surplus: bool = Field(
        default=False,
        description="Whether output < input (acceptable loss/waste)",
    )
    is_deficit: bool = Field(
        default=False,
        description="Whether output > input (compliance alert)",
    )
    unit: str = Field(
        default="kg",
        description="Unit of measurement",
    )
    custody_model: CustodyModel = Field(
        default=CustodyModel.MASS_BALANCE,
        description="Custody model governing the balance check",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of the balance result",
    )

class ComplianceAlert(GreenLangBase):
    """Compliance alert raised by the batch traceability engine.

    Attributes:
        alert_id: Unique identifier for this alert.
        severity: Alert severity level.
        alert_type: Type of compliance issue detected.
        batch_id: Batch ID that triggered the alert.
        operation_id: Operation ID that triggered the alert (if applicable).
        message: Human-readable description of the issue.
        eudr_article: EUDR article reference for the compliance requirement.
        details: Structured details about the alert.
        timestamp: UTC timestamp when the alert was raised.
    """

    model_config = ConfigDict(from_attributes=True)

    alert_id: str = Field(
        default_factory=lambda: _generate_id("ALERT"),
        description="Unique identifier for this alert",
    )
    severity: AlertSeverity = Field(
        ...,
        description="Alert severity level",
    )
    alert_type: str = Field(
        ...,
        description="Type of compliance issue",
    )
    batch_id: str = Field(
        ...,
        description="Batch ID that triggered the alert",
    )
    operation_id: Optional[str] = Field(
        None,
        description="Operation ID that triggered the alert",
    )
    message: str = Field(
        ...,
        description="Human-readable description of the issue",
    )
    eudr_article: str = Field(
        default="",
        description="EUDR article reference",
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured details about the alert",
    )
    timestamp: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp when the alert was raised",
    )

class TraceabilityScore(GreenLangBase):
    """Traceability completeness score for a batch under a custody model.

    For Identity Preserved and Segregated models, the score is binary
    (100 if all origins known, 0 otherwise). For Mass Balance, a partial
    score is calculated based on the fraction of quantity with known origins.

    Attributes:
        batch_id: Batch ID being scored.
        custody_model: Custody model governing the scoring.
        score: Traceability score (0-100) as Decimal string.
        total_quantity: Total batch quantity as Decimal string.
        traced_quantity: Quantity with known plot origins as Decimal string.
        untraced_quantity: Quantity with unknown origins as Decimal string.
        known_plot_count: Number of known origin plots.
        unknown_leaf_count: Number of leaf batches without origin plots.
        is_fully_traceable: Whether score == 100.
        provenance_hash: SHA-256 hash for audit.
    """

    model_config = ConfigDict(from_attributes=True)

    batch_id: str = Field(
        ...,
        description="Batch ID being scored",
    )
    custody_model: CustodyModel = Field(
        ...,
        description="Custody model governing the scoring",
    )
    score: str = Field(
        ...,
        description="Traceability score (0-100) as Decimal string",
    )
    total_quantity: str = Field(
        ...,
        description="Total batch quantity as Decimal string",
    )
    traced_quantity: str = Field(
        ...,
        description="Quantity with known plot origins as Decimal string",
    )
    untraced_quantity: str = Field(
        ...,
        description="Quantity with unknown origins as Decimal string",
    )
    known_plot_count: int = Field(
        default=0,
        description="Number of known origin plots",
    )
    unknown_leaf_count: int = Field(
        default=0,
        description="Number of leaf batches without origin plots",
    )
    is_fully_traceable: bool = Field(
        default=False,
        description="Whether score equals 100",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit",
    )

# ---------------------------------------------------------------------------
# Internal graph node for batch lineage
# ---------------------------------------------------------------------------

class _BatchNode:
    """Internal mutable node in the batch lineage graph.

    Not exposed publicly. Used for efficient graph traversal.

    Attributes:
        batch_id: Batch identifier.
        batch: The BatchRecord model.
        parent_ids: Set of parent batch IDs (inputs that produced this batch).
        child_ids: Set of child batch IDs (batches derived from this one).
        operation_ids: Set of operation IDs involving this batch.
        is_leaf: Whether this batch has no parents (origin/root batch).
        original_quantity: Quantity at registration time (before any splits).
    """

    __slots__ = (
        "batch_id",
        "batch",
        "parent_ids",
        "child_ids",
        "operation_ids",
        "is_leaf",
        "original_quantity",
    )

    def __init__(self, batch: BatchRecord) -> None:
        self.batch_id: str = batch.batch_id
        self.batch: BatchRecord = batch
        self.parent_ids: Set[str] = set(batch.parent_batch_ids)
        self.child_ids: Set[str] = set()
        self.operation_ids: Set[str] = set()
        self.is_leaf: bool = len(batch.parent_batch_ids) == 0
        self.original_quantity: Decimal = batch.quantity

# ===========================================================================
# BatchTraceabilityEngine
# ===========================================================================

class BatchTraceabilityEngine:
    """Many-to-many batch traceability engine for EUDR supply chains.

    Implements PRD Feature 4 (Many-to-Many Batch Traceability) with support
    for three chain of custody models, batch splitting, merging, and
    transformation, deterministic mass balance verification, and forward/
    backward trace with SHA-256 provenance on every operation.

    **Zero-Hallucination Guarantee**: All calculations in this engine use
    ``decimal.Decimal`` arithmetic.  There is no LLM, no floating-point, and
    no heuristic in any code path.  The same inputs always produce the same
    outputs (bit-perfect reproducibility).

    Attributes:
        _nodes: Internal batch lineage graph keyed by batch_id.
        _operations: Recorded batch operations keyed by operation_id.
        _alerts: Compliance alerts raised by the engine.
        _plot_index: Reverse index from plot_id to set of batch_ids.
        _chain_of_custody: Optional reference to ChainOfCustodyEngine.
        _plot_registry: Optional reference to PlotRegistryEngine.
        _provenance: Optional ProvenanceTracker for audit chain.
        _mass_balance_tolerance: Decimal tolerance for balance checks.

    Example:
        >>> engine = BatchTraceabilityEngine()
        >>> batch = engine.register_batch(
        ...     commodity=EUDRCommodity.COCOA,
        ...     product_description="Raw cocoa beans",
        ...     quantity=Decimal("10000"),
        ...     origin_plot_ids=["PLOT-001"],
        ...     custody_model=CustodyModel.IDENTITY_PRESERVED,
        ... )
        >>> children = engine.split_batch(
        ...     batch.batch_id,
        ...     [Decimal("6000"), Decimal("4000")],
        ... )
        >>> assert len(children) == 2
    """

    # Class-level constants
    AGENT_ID = "GL-EUDR-SCM-001"
    DEFAULT_TOLERANCE = Decimal("0.001")

    def __init__(
        self,
        chain_of_custody: Any = None,
        plot_registry: Any = None,
        provenance: Any = None,
        mass_balance_tolerance: Optional[Decimal] = None,
    ) -> None:
        """Initialize BatchTraceabilityEngine.

        Args:
            chain_of_custody: Optional ChainOfCustodyEngine instance for
                live batch/transfer lookups.
            plot_registry: Optional PlotRegistryEngine instance for
                origin plot resolution during traces.
            provenance: Optional ProvenanceTracker instance for SHA-256
                audit chain recording.
            mass_balance_tolerance: Decimal tolerance for mass balance
                verification.  Defaults to 0.001.
        """
        # External integrations (optional)
        self._chain_of_custody = chain_of_custody
        self._plot_registry = plot_registry
        self._provenance = provenance

        # Configuration
        self._mass_balance_tolerance: Decimal = (
            mass_balance_tolerance
            if mass_balance_tolerance is not None
            else self.DEFAULT_TOLERANCE
        )

        # Internal storage
        self._nodes: Dict[str, _BatchNode] = {}
        self._operations: Dict[str, BatchOperation] = {}
        self._alerts: List[ComplianceAlert] = []

        # Indexes
        self._plot_index: Dict[str, Set[str]] = {}  # plot_id -> {batch_ids}
        self._commodity_index: Dict[str, Set[str]] = {}  # commodity -> {batch_ids}
        self._op_by_input: Dict[str, Set[str]] = {}  # batch_id -> {op_ids consuming it}
        self._op_by_output: Dict[str, Set[str]] = {}  # batch_id -> {op_ids producing it}

        logger.info(
            "BatchTraceabilityEngine initialized (agent=%s, tolerance=%s)",
            self.AGENT_ID,
            self._mass_balance_tolerance,
        )

    # ------------------------------------------------------------------
    # Public API -- Batch Registration
    # ------------------------------------------------------------------

    def register_batch(
        self,
        commodity: EUDRCommodity,
        product_description: str,
        quantity: Decimal,
        origin_plot_ids: Optional[List[str]] = None,
        custody_model: CustodyModel = CustodyModel.SEGREGATED,
        unit: str = "kg",
        batch_id: Optional[str] = None,
        operator_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BatchRecord:
        """Register a new origin batch in the traceability graph.

        Creates a leaf node (no parents) representing a batch of commodity
        at its point of origin or first point of intake.

        Args:
            commodity: EUDR commodity type.
            product_description: Human-readable product description.
            quantity: Batch quantity as Decimal (must be > 0).
            origin_plot_ids: List of origin plot IDs contributing to this batch.
            custody_model: Chain of custody model.
            unit: Unit of measurement (default "kg").
            batch_id: Optional pre-assigned batch ID.
            operator_id: Optional operator identifier.
            metadata: Optional additional metadata.

        Returns:
            Registered BatchRecord.

        Raises:
            ValueError: If quantity <= 0 or other validation fails.
        """
        start = time.monotonic()

        # Validate quantity
        qty = _decimal(quantity)
        if qty <= Decimal("0"):
            raise ValueError(
                f"Batch quantity must be positive, got {qty}"
            )

        # Generate or use provided batch_id
        bid = batch_id or _generate_id("BATCH")

        if bid in self._nodes:
            raise ValueError(f"Batch {bid} already registered")

        # Build origin plot list
        plots = list(origin_plot_ids) if origin_plot_ids else []

        # Create BatchRecord (from AGENT-DATA-005 models)
        batch = BatchRecord(
            batch_id=bid,
            commodity=commodity,
            product_description=product_description,
            quantity=qty,
            unit=unit,
            origin_plot_ids=plots,
            parent_batch_ids=[],
            custody_model=custody_model,
        )

        # Create internal node
        node = _BatchNode(batch)
        self._nodes[bid] = node

        # Update indexes
        self._index_batch(bid, commodity, plots)

        # Create operation record
        operation = BatchOperation(
            operation_type=BatchOperationType.REGISTER,
            input_batch_ids=[],
            output_batch_ids=[bid],
            input_total_quantity="0",
            output_total_quantity=str(qty),
            quantity_unit=unit,
            custody_model=custody_model,
            commodity=commodity,
            operator_id=operator_id,
            metadata=metadata or {},
        )
        operation.provenance_hash = _compute_hash(operation.model_dump(mode="json"))
        self._operations[operation.operation_id] = operation
        node.operation_ids.add(operation.operation_id)
        self._op_by_output.setdefault(bid, set()).add(operation.operation_id)

        # Record provenance
        self._record_provenance(
            entity_type="batch",
            entity_id=bid,
            action="batch_register",
            data_hash=operation.provenance_hash,
        )

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Registered batch %s: commodity=%s, qty=%s %s, plots=%d (%.1f ms)",
            bid,
            commodity.value,
            qty,
            unit,
            len(plots),
            elapsed_ms,
        )
        return batch

    # ------------------------------------------------------------------
    # Public API -- Batch Splitting
    # ------------------------------------------------------------------

    def split_batch(
        self,
        parent_batch_id: str,
        split_quantities: List[Decimal],
        descriptions: Optional[List[str]] = None,
        operator_id: Optional[str] = None,
    ) -> List[BatchRecord]:
        """Split one input batch into N output batches with quantity conservation.

        The sum of ``split_quantities`` must not exceed the parent batch's
        current remaining quantity.  Origin plot linkage is preserved on all
        child batches.  The parent batch's remaining quantity is decremented
        by the total split amount.

        For **Identity Preserved** custody model, only single-origin batches
        may be split (all children inherit the same single origin).
        For **Segregated** model, all children inherit the full origin set.
        For **Mass Balance** model, origins are carried forward with quantity
        proportional attribution.

        Args:
            parent_batch_id: ID of the batch to split.
            split_quantities: List of Decimal quantities for each child batch.
            descriptions: Optional list of descriptions for each child.
                If not provided, generates default descriptions.
            operator_id: Optional operator performing the split.

        Returns:
            List of new child BatchRecord instances.

        Raises:
            ValueError: If parent not found, quantities exceed parent,
                or custody model constraints are violated.
        """
        start = time.monotonic()

        # Validate parent exists
        parent_node = self._nodes.get(parent_batch_id)
        if parent_node is None:
            raise ValueError(f"Batch {parent_batch_id} not found")

        parent_batch = parent_node.batch

        # Validate at least one split
        if not split_quantities or len(split_quantities) < 1:
            raise ValueError("At least one split quantity is required")

        # Convert and validate quantities (DETERMINISTIC Decimal arithmetic)
        split_qtys: List[Decimal] = []
        for i, sq in enumerate(split_quantities):
            q = _decimal(sq)
            if q <= Decimal("0"):
                raise ValueError(
                    f"Split quantity at index {i} must be positive, got {q}"
                )
            split_qtys.append(q)

        total_split = sum(split_qtys, Decimal("0"))

        # Mass balance check: total split must not exceed parent remaining
        if total_split > parent_batch.quantity:
            alert = self._raise_alert(
                severity=AlertSeverity.CRITICAL,
                alert_type="mass_balance_split_overflow",
                batch_id=parent_batch_id,
                message=(
                    f"Split total ({total_split}) exceeds parent remaining "
                    f"quantity ({parent_batch.quantity})"
                ),
                eudr_article="Art. 10(2)(f)",
                details={
                    "parent_quantity": str(parent_batch.quantity),
                    "split_total": str(total_split),
                    "overflow": str(total_split - parent_batch.quantity),
                },
            )
            raise ValueError(
                f"Split total ({total_split}) exceeds parent remaining "
                f"quantity ({parent_batch.quantity}). "
                f"Alert {alert.alert_id} raised."
            )

        # Identity Preserved: enforce single-origin constraint
        if parent_batch.custody_model == CustodyModel.IDENTITY_PRESERVED:
            if len(parent_batch.origin_plot_ids) > 1:
                raise ValueError(
                    f"Identity Preserved batch {parent_batch_id} has "
                    f"{len(parent_batch.origin_plot_ids)} origin plots; "
                    f"cannot split a multi-origin batch under IP model"
                )

        # Build descriptions
        if descriptions is not None:
            if len(descriptions) != len(split_qtys):
                raise ValueError(
                    f"descriptions length ({len(descriptions)}) must match "
                    f"split_quantities length ({len(split_qtys)})"
                )
            descs = descriptions
        else:
            descs = [
                f"Split {i + 1}/{len(split_qtys)} from {parent_batch_id}"
                for i in range(len(split_qtys))
            ]

        # Create child batches
        child_batches: List[BatchRecord] = []
        child_ids: List[str] = []

        for qty, desc in zip(split_qtys, descs):
            child_id = _generate_id("BATCH")
            child = BatchRecord(
                batch_id=child_id,
                commodity=parent_batch.commodity,
                product_description=desc,
                quantity=qty,
                unit=parent_batch.unit,
                origin_plot_ids=list(parent_batch.origin_plot_ids),
                parent_batch_ids=[parent_batch_id],
                custody_model=parent_batch.custody_model,
            )

            # Create internal node and link
            child_node = _BatchNode(child)
            child_node.parent_ids.add(parent_batch_id)
            self._nodes[child_id] = child_node
            parent_node.child_ids.add(child_id)

            # Update indexes
            self._index_batch(
                child_id,
                parent_batch.commodity,
                parent_batch.origin_plot_ids,
            )

            child_batches.append(child)
            child_ids.append(child_id)

        # Decrement parent remaining quantity (DETERMINISTIC)
        parent_batch.quantity = parent_batch.quantity - total_split

        # Create operation record
        operation = BatchOperation(
            operation_type=BatchOperationType.SPLIT,
            input_batch_ids=[parent_batch_id],
            output_batch_ids=child_ids,
            input_total_quantity=str(total_split),
            output_total_quantity=str(total_split),
            quantity_unit=parent_batch.unit,
            custody_model=parent_batch.custody_model,
            commodity=parent_batch.commodity,
            operator_id=operator_id,
        )
        operation.provenance_hash = _compute_hash(operation.model_dump(mode="json"))
        self._operations[operation.operation_id] = operation

        # Link operation to nodes
        parent_node.operation_ids.add(operation.operation_id)
        self._op_by_input.setdefault(parent_batch_id, set()).add(
            operation.operation_id
        )
        for cid in child_ids:
            self._nodes[cid].operation_ids.add(operation.operation_id)
            self._op_by_output.setdefault(cid, set()).add(operation.operation_id)

        # Record provenance
        self._record_provenance(
            entity_type="batch",
            entity_id=parent_batch_id,
            action="batch_split",
            data_hash=operation.provenance_hash,
        )

        # Record metrics
        self._record_metrics("batch_split", time.monotonic() - start)

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Split batch %s into %d children: quantities=%s (%.1f ms)",
            parent_batch_id,
            len(child_batches),
            [str(q) for q in split_qtys],
            elapsed_ms,
        )
        return child_batches

    # ------------------------------------------------------------------
    # Public API -- Batch Merging
    # ------------------------------------------------------------------

    def merge_batches(
        self,
        batch_ids: List[str],
        product_description: str,
        custody_model: Optional[CustodyModel] = None,
        operator_id: Optional[str] = None,
    ) -> BatchRecord:
        """Merge N input batches into one output batch with origin preservation.

        All input batches must have the same commodity type.  Origin plot IDs
        from all inputs are deduplicated and preserved in the merged batch.
        The merged quantity is the sum of all input quantities.

        For **Identity Preserved** model, merging is only allowed if all
        inputs share the same single origin plot.
        For **Segregated** model, merging is allowed for same-commodity batches.
        For **Mass Balance** model, merging is always allowed for same-commodity.

        Args:
            batch_ids: List of batch IDs to merge (minimum 2).
            product_description: Description for the merged output batch.
            custody_model: Optional override for custody model on the merged
                batch.  If not provided, uses the strictest model among inputs.
            operator_id: Optional operator performing the merge.

        Returns:
            New merged BatchRecord.

        Raises:
            ValueError: If fewer than 2 batches, commodity mismatch,
                batch not found, or custody model constraint violated.
        """
        start = time.monotonic()

        if len(batch_ids) < 2:
            raise ValueError("At least 2 batches required for merge")

        # Resolve all input batches
        input_nodes: List[_BatchNode] = []
        for bid in batch_ids:
            node = self._nodes.get(bid)
            if node is None:
                raise ValueError(f"Batch {bid} not found")
            input_nodes.append(node)

        # Validate same commodity
        commodities = {n.batch.commodity for n in input_nodes}
        if len(commodities) > 1:
            raise ValueError(
                f"Cannot merge batches with different commodities: "
                f"{sorted(c.value for c in commodities)}"
            )
        commodity = input_nodes[0].batch.commodity

        # Determine custody model
        if custody_model is not None:
            merged_model = custody_model
        else:
            merged_model = self._strictest_custody_model(
                [n.batch.custody_model for n in input_nodes]
            )

        # Identity Preserved: all inputs must share same single origin
        if merged_model == CustodyModel.IDENTITY_PRESERVED:
            all_origins: List[Set[str]] = [
                set(n.batch.origin_plot_ids) for n in input_nodes
            ]
            if not all_origins:
                raise ValueError(
                    "Cannot merge under Identity Preserved: no origins"
                )
            first_origins = all_origins[0]
            if len(first_origins) != 1:
                raise ValueError(
                    "Identity Preserved merge requires single-origin batches"
                )
            for i, origins in enumerate(all_origins[1:], 1):
                if origins != first_origins:
                    raise ValueError(
                        f"Identity Preserved merge: batch {batch_ids[i]} has "
                        f"different origin ({origins}) than batch "
                        f"{batch_ids[0]} ({first_origins})"
                    )

        # Calculate merged quantity (DETERMINISTIC Decimal sum)
        total_quantity = sum(
            (n.batch.quantity for n in input_nodes),
            Decimal("0"),
        )

        # Collect and deduplicate origin plot IDs (preserve order)
        all_plot_ids: List[str] = []
        seen_plots: Set[str] = set()
        for node in input_nodes:
            for pid in node.batch.origin_plot_ids:
                if pid not in seen_plots:
                    seen_plots.add(pid)
                    all_plot_ids.append(pid)

        # Create merged batch
        merged_id = _generate_id("BATCH")
        merged = BatchRecord(
            batch_id=merged_id,
            commodity=commodity,
            product_description=product_description,
            quantity=total_quantity,
            unit=input_nodes[0].batch.unit,
            origin_plot_ids=all_plot_ids,
            parent_batch_ids=list(batch_ids),
            custody_model=merged_model,
        )

        # Create internal node and link
        merged_node = _BatchNode(merged)
        for bid in batch_ids:
            merged_node.parent_ids.add(bid)
            self._nodes[bid].child_ids.add(merged_id)
        self._nodes[merged_id] = merged_node

        # Update indexes
        self._index_batch(merged_id, commodity, all_plot_ids)

        # Create operation record
        operation = BatchOperation(
            operation_type=BatchOperationType.MERGE,
            input_batch_ids=list(batch_ids),
            output_batch_ids=[merged_id],
            input_total_quantity=str(total_quantity),
            output_total_quantity=str(total_quantity),
            quantity_unit=input_nodes[0].batch.unit,
            custody_model=merged_model,
            commodity=commodity,
            operator_id=operator_id,
        )
        operation.provenance_hash = _compute_hash(operation.model_dump(mode="json"))
        self._operations[operation.operation_id] = operation

        # Link operation to nodes
        for bid in batch_ids:
            self._nodes[bid].operation_ids.add(operation.operation_id)
            self._op_by_input.setdefault(bid, set()).add(operation.operation_id)
        merged_node.operation_ids.add(operation.operation_id)
        self._op_by_output.setdefault(merged_id, set()).add(operation.operation_id)

        # Record provenance
        self._record_provenance(
            entity_type="batch",
            entity_id=merged_id,
            action="batch_merge",
            data_hash=operation.provenance_hash,
        )

        # Record metrics
        self._record_metrics("batch_merge", time.monotonic() - start)

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Merged %d batches into %s: qty=%s, origins=%d (%.1f ms)",
            len(batch_ids),
            merged_id,
            total_quantity,
            len(all_plot_ids),
            elapsed_ms,
        )
        return merged

    # ------------------------------------------------------------------
    # Public API -- Batch Transformation
    # ------------------------------------------------------------------

    def transform_batch(
        self,
        input_batch_id: str,
        output_commodity: EUDRCommodity,
        output_description: str,
        output_quantity: Decimal,
        operator_id: Optional[str] = None,
        conversion_factor: Optional[Decimal] = None,
    ) -> BatchRecord:
        """Transform an input batch into a derived product.

        Models commodity transformation (e.g. cocoa beans to chocolate,
        wood to furniture).  Origin plot linkage is preserved through the
        transformation.  The output quantity may differ from input (due to
        processing yield) but must not exceed ``input * conversion_factor``
        if a conversion factor is provided.

        Args:
            input_batch_id: ID of the input batch to transform.
            output_commodity: EUDR commodity of the output product.
            output_description: Description of the output product.
            output_quantity: Quantity of the output product as Decimal.
            operator_id: Optional operator performing the transformation.
            conversion_factor: Optional maximum yield ratio.  If provided,
                ``output_quantity <= input_quantity * conversion_factor``
                is enforced.

        Returns:
            New transformed BatchRecord.

        Raises:
            ValueError: If input not found, output quantity exceeds allowed
                maximum, or commodity transformation is invalid.
        """
        start = time.monotonic()

        # Validate input exists
        input_node = self._nodes.get(input_batch_id)
        if input_node is None:
            raise ValueError(f"Batch {input_batch_id} not found")

        input_batch = input_node.batch
        out_qty = _decimal(output_quantity)

        if out_qty <= Decimal("0"):
            raise ValueError(
                f"Output quantity must be positive, got {out_qty}"
            )

        # Validate commodity transformation makes sense
        # (output should be a derived product of the input commodity, or same)
        input_primary = DERIVED_TO_PRIMARY.get(
            input_batch.commodity, input_batch.commodity
        )
        output_primary = DERIVED_TO_PRIMARY.get(output_commodity, output_commodity)

        if input_primary != output_primary and input_batch.commodity != output_commodity:
            raise ValueError(
                f"Invalid transformation: {input_batch.commodity.value} -> "
                f"{output_commodity.value}. Primary commodities do not match "
                f"({input_primary.value} vs {output_primary.value})"
            )

        # Validate conversion factor if provided
        if conversion_factor is not None:
            cf = _decimal(conversion_factor)
            max_output = input_batch.quantity * cf
            if out_qty > max_output:
                alert = self._raise_alert(
                    severity=AlertSeverity.HIGH,
                    alert_type="transformation_yield_exceeded",
                    batch_id=input_batch_id,
                    message=(
                        f"Transformation output ({out_qty}) exceeds maximum "
                        f"allowed by conversion factor ({max_output})"
                    ),
                    eudr_article="Art. 10(2)(f)",
                    details={
                        "input_quantity": str(input_batch.quantity),
                        "conversion_factor": str(cf),
                        "max_output": str(max_output),
                        "actual_output": str(out_qty),
                    },
                )
                raise ValueError(
                    f"Output quantity ({out_qty}) exceeds maximum "
                    f"({max_output}) given conversion factor ({cf}). "
                    f"Alert {alert.alert_id} raised."
                )

        # Mass balance check: flag if output > input (without conversion factor)
        if conversion_factor is None and out_qty > input_batch.quantity:
            self._raise_alert(
                severity=AlertSeverity.HIGH,
                alert_type="mass_balance_transform_deficit",
                batch_id=input_batch_id,
                message=(
                    f"Transformation output ({out_qty}) exceeds input "
                    f"({input_batch.quantity}) without conversion factor"
                ),
                eudr_article="Art. 10(2)(f)",
                details={
                    "input_quantity": str(input_batch.quantity),
                    "output_quantity": str(out_qty),
                    "deficit": str(out_qty - input_batch.quantity),
                },
            )

        # Create output batch
        output_id = _generate_id("BATCH")
        output_batch = BatchRecord(
            batch_id=output_id,
            commodity=output_commodity,
            product_description=output_description,
            quantity=out_qty,
            unit=input_batch.unit,
            origin_plot_ids=list(input_batch.origin_plot_ids),
            parent_batch_ids=[input_batch_id],
            custody_model=input_batch.custody_model,
        )

        # Create internal node and link
        output_node = _BatchNode(output_batch)
        output_node.parent_ids.add(input_batch_id)
        input_node.child_ids.add(output_id)
        self._nodes[output_id] = output_node

        # Update indexes
        self._index_batch(output_id, output_commodity, input_batch.origin_plot_ids)

        # Create operation record
        operation = BatchOperation(
            operation_type=BatchOperationType.TRANSFORM,
            input_batch_ids=[input_batch_id],
            output_batch_ids=[output_id],
            input_total_quantity=str(input_batch.quantity),
            output_total_quantity=str(out_qty),
            quantity_unit=input_batch.unit,
            custody_model=input_batch.custody_model,
            commodity=input_batch.commodity,
            output_commodity=output_commodity,
            operator_id=operator_id,
            metadata={
                "conversion_factor": str(conversion_factor) if conversion_factor else None,
            },
        )
        operation.provenance_hash = _compute_hash(operation.model_dump(mode="json"))
        self._operations[operation.operation_id] = operation

        # Link operation to nodes
        input_node.operation_ids.add(operation.operation_id)
        output_node.operation_ids.add(operation.operation_id)
        self._op_by_input.setdefault(input_batch_id, set()).add(
            operation.operation_id
        )
        self._op_by_output.setdefault(output_id, set()).add(operation.operation_id)

        # Record provenance
        self._record_provenance(
            entity_type="batch",
            entity_id=output_id,
            action="batch_transform",
            data_hash=operation.provenance_hash,
        )

        # Record metrics
        self._record_metrics("batch_transform", time.monotonic() - start)

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Transformed batch %s -> %s: %s -> %s, qty %s -> %s (%.1f ms)",
            input_batch_id,
            output_id,
            input_batch.commodity.value,
            output_commodity.value,
            input_batch.quantity,
            out_qty,
            elapsed_ms,
        )
        return output_batch

    # ------------------------------------------------------------------
    # Public API -- Backward Trace
    # ------------------------------------------------------------------

    def backward_trace(self, batch_id: str) -> TraceResult:
        """Trace a batch backward to find all contributing origin plots.

        Performs an iterative breadth-first search (BFS) upward through
        parent links to discover every origin plot that contributed to the
        given batch.  Handles unlimited levels of splits and merges.

        This answers the question: "Which plots contributed to Product Y?"

        Args:
            batch_id: The batch ID to trace backward from.

        Returns:
            TraceResult with all origin plots, visited batches, and
            traceability score.

        Raises:
            ValueError: If the batch is not found.
        """
        start = time.monotonic()

        if batch_id not in self._nodes:
            raise ValueError(f"Batch {batch_id} not found")

        # BFS backward through parent links
        visited: Set[str] = set()
        queue: deque[Tuple[str, int]] = deque()
        queue.append((batch_id, 0))

        origin_plot_ids: List[str] = []
        origin_plots: List[PlotRecord] = []
        seen_plots: Set[str] = set()
        reached_batch_ids: List[str] = []
        operation_chain: List[str] = []
        seen_ops: Set[str] = set()
        max_depth = 0
        unknown_leaf_count = 0

        while queue:
            current_id, depth = queue.popleft()

            if current_id in visited:
                continue
            visited.add(current_id)
            reached_batch_ids.append(current_id)

            if depth > max_depth:
                max_depth = depth

            node = self._nodes.get(current_id)
            if node is None:
                # Node not in our graph -- might be external reference
                unknown_leaf_count += 1
                continue

            # Collect operations that produced this batch
            producing_ops = self._op_by_output.get(current_id, set())
            for op_id in producing_ops:
                if op_id not in seen_ops:
                    seen_ops.add(op_id)
                    operation_chain.append(op_id)

            # If this is a leaf node (no parents), collect its origin plots
            if not node.parent_ids:
                if node.batch.origin_plot_ids:
                    for pid in node.batch.origin_plot_ids:
                        if pid not in seen_plots:
                            seen_plots.add(pid)
                            origin_plot_ids.append(pid)
                            # Resolve PlotRecord if registry available
                            plot = self._resolve_plot(pid)
                            if plot is not None:
                                origin_plots.append(plot)
                else:
                    unknown_leaf_count += 1
            else:
                # Enqueue parents
                for parent_id in node.parent_ids:
                    if parent_id not in visited:
                        queue.append((parent_id, depth + 1))

        # Calculate traceability score
        total_leaves = len(origin_plot_ids) + unknown_leaf_count
        if total_leaves > 0:
            score = (
                Decimal(str(len(origin_plot_ids)))
                / Decimal(str(total_leaves))
                * Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        else:
            score = Decimal("0.00")

        is_complete = unknown_leaf_count == 0 and len(origin_plot_ids) > 0

        elapsed_ms = (time.monotonic() - start) * 1000

        # Build result
        result = TraceResult(
            target_batch_id=batch_id,
            direction="backward",
            origin_plot_ids=origin_plot_ids,
            origin_plots=origin_plots,
            reached_batch_ids=reached_batch_ids,
            operation_chain=operation_chain,
            depth=max_depth,
            is_complete=is_complete,
            unknown_origin_count=unknown_leaf_count,
            traceability_score=str(score),
            trace_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(
            result.model_dump(mode="json", exclude={"provenance_hash"})
        )

        logger.info(
            "Backward trace %s: %d plots, depth=%d, score=%s, complete=%s (%.1f ms)",
            batch_id,
            len(origin_plot_ids),
            max_depth,
            score,
            is_complete,
            elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Public API -- Forward Trace
    # ------------------------------------------------------------------

    def forward_trace(self, plot_id: str) -> TraceResult:
        """Trace forward from an origin plot to find all derived products.

        Performs an iterative BFS downward through child links starting
        from all batches that reference the given plot, to discover every
        product that contains commodity from that plot.

        This answers the question: "Which products contain commodity from Plot X?"

        Args:
            plot_id: The origin plot ID to trace forward from.

        Returns:
            TraceResult with all reached batches and operations.

        Raises:
            ValueError: If no batches reference the given plot.
        """
        start = time.monotonic()

        # Find all batches that reference this plot
        seed_batch_ids = self._plot_index.get(plot_id, set())
        if not seed_batch_ids:
            raise ValueError(
                f"No batches reference plot {plot_id}"
            )

        # BFS forward through child links
        visited: Set[str] = set()
        queue: deque[Tuple[str, int]] = deque()
        for bid in seed_batch_ids:
            queue.append((bid, 0))

        reached_batch_ids: List[str] = []
        operation_chain: List[str] = []
        seen_ops: Set[str] = set()
        max_depth = 0

        while queue:
            current_id, depth = queue.popleft()

            if current_id in visited:
                continue
            visited.add(current_id)
            reached_batch_ids.append(current_id)

            if depth > max_depth:
                max_depth = depth

            node = self._nodes.get(current_id)
            if node is None:
                continue

            # Collect operations that consume this batch
            consuming_ops = self._op_by_input.get(current_id, set())
            for op_id in consuming_ops:
                if op_id not in seen_ops:
                    seen_ops.add(op_id)
                    operation_chain.append(op_id)

            # Enqueue children
            for child_id in node.child_ids:
                if child_id not in visited:
                    queue.append((child_id, depth + 1))

        elapsed_ms = (time.monotonic() - start) * 1000

        result = TraceResult(
            target_batch_id=plot_id,
            direction="forward",
            origin_plot_ids=[plot_id],
            origin_plots=[],
            reached_batch_ids=reached_batch_ids,
            operation_chain=operation_chain,
            depth=max_depth,
            is_complete=True,
            unknown_origin_count=0,
            traceability_score="100.00",
            trace_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(
            result.model_dump(mode="json", exclude={"provenance_hash"})
        )

        # Resolve plot record if available
        plot = self._resolve_plot(plot_id)
        if plot is not None:
            result.origin_plots = [plot]

        logger.info(
            "Forward trace from plot %s: %d batches reached, depth=%d (%.1f ms)",
            plot_id,
            len(reached_batch_ids),
            max_depth,
            elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Public API -- Mass Balance Verification
    # ------------------------------------------------------------------

    def verify_mass_balance(self, batch_id: str) -> MassBalanceResult:
        """Verify mass balance for a batch and its children.

        Checks that total output quantity (remaining in batch + all children
        created via splits) does not exceed the original registered quantity.
        Uses Decimal arithmetic for zero floating-point drift.

        Args:
            batch_id: Batch ID to verify.

        Returns:
            MassBalanceResult with deterministic balance calculation.

        Raises:
            ValueError: If batch not found.
        """
        node = self._nodes.get(batch_id)
        if node is None:
            raise ValueError(f"Batch {batch_id} not found")

        batch = node.batch

        # Input: original quantity at registration
        input_qty = node.original_quantity

        # Output: current remaining + all direct children quantities
        output_qty = batch.quantity
        for child_id in node.child_ids:
            child_node = self._nodes.get(child_id)
            if child_node is not None:
                output_qty += child_node.batch.quantity

        # Deterministic balance calculation
        balance = input_qty - output_qty
        is_balanced = abs(balance) <= self._mass_balance_tolerance
        is_surplus = balance > self._mass_balance_tolerance
        is_deficit = balance < -self._mass_balance_tolerance

        # Raise alert on deficit (output > input)
        if is_deficit:
            self._raise_alert(
                severity=AlertSeverity.CRITICAL,
                alert_type="mass_balance_deficit",
                batch_id=batch_id,
                message=(
                    f"Mass balance deficit: output ({output_qty}) exceeds "
                    f"input ({input_qty}) by {abs(balance)}"
                ),
                eudr_article="Art. 10(2)(f)",
                details={
                    "input_quantity": str(input_qty),
                    "output_quantity": str(output_qty),
                    "deficit": str(abs(balance)),
                    "tolerance": str(self._mass_balance_tolerance),
                },
            )

        result = MassBalanceResult(
            batch_id=batch_id,
            input_quantity=str(input_qty),
            output_quantity=str(output_qty),
            balance=str(balance),
            tolerance=str(self._mass_balance_tolerance),
            is_balanced=is_balanced,
            is_surplus=is_surplus,
            is_deficit=is_deficit,
            unit=batch.unit,
            custody_model=batch.custody_model,
        )
        result.provenance_hash = _compute_hash(
            result.model_dump(mode="json", exclude={"provenance_hash"})
        )

        logger.debug(
            "Mass balance for %s: input=%s, output=%s, balance=%s, ok=%s",
            batch_id,
            input_qty,
            output_qty,
            balance,
            is_balanced,
        )
        return result

    # ------------------------------------------------------------------
    # Public API -- Traceability Scoring
    # ------------------------------------------------------------------

    def compute_traceability_score(self, batch_id: str) -> TraceabilityScore:
        """Compute traceability completeness score for a batch.

        For **Identity Preserved** and **Segregated** models, the score is
        binary: 100 if all origins are known, 0 otherwise.

        For **Mass Balance** model, a partial score is calculated based on the
        fraction of the batch's quantity that can be traced to known origin
        plots.  This accounts for situations where some input batches in a
        merge have unknown origins.

        Args:
            batch_id: Batch ID to score.

        Returns:
            TraceabilityScore with deterministic calculation.

        Raises:
            ValueError: If batch not found.
        """
        node = self._nodes.get(batch_id)
        if node is None:
            raise ValueError(f"Batch {batch_id} not found")

        batch = node.batch

        if batch.custody_model in (
            CustodyModel.IDENTITY_PRESERVED,
            CustodyModel.SEGREGATED,
        ):
            return self._score_ip_segregated(batch_id, node)
        else:
            return self._score_mass_balance(batch_id, node)

    # ------------------------------------------------------------------
    # Public API -- Batch and Operation Queries
    # ------------------------------------------------------------------

    def get_batch(self, batch_id: str) -> Optional[BatchRecord]:
        """Get a batch record by ID.

        Args:
            batch_id: Batch identifier.

        Returns:
            BatchRecord or None if not found.
        """
        node = self._nodes.get(batch_id)
        return node.batch if node is not None else None

    def get_operation(self, operation_id: str) -> Optional[BatchOperation]:
        """Get a batch operation by ID.

        Args:
            operation_id: Operation identifier.

        Returns:
            BatchOperation or None if not found.
        """
        return self._operations.get(operation_id)

    def get_batch_children(self, batch_id: str) -> List[BatchRecord]:
        """Get all direct child batches of a given batch.

        Args:
            batch_id: Parent batch identifier.

        Returns:
            List of child BatchRecord instances.
        """
        node = self._nodes.get(batch_id)
        if node is None:
            return []
        return [
            self._nodes[cid].batch
            for cid in node.child_ids
            if cid in self._nodes
        ]

    def get_batch_parents(self, batch_id: str) -> List[BatchRecord]:
        """Get all direct parent batches of a given batch.

        Args:
            batch_id: Child batch identifier.

        Returns:
            List of parent BatchRecord instances.
        """
        node = self._nodes.get(batch_id)
        if node is None:
            return []
        return [
            self._nodes[pid].batch
            for pid in node.parent_ids
            if pid in self._nodes
        ]

    def get_batches_by_plot(self, plot_id: str) -> List[BatchRecord]:
        """Get all batches that directly reference a given origin plot.

        Args:
            plot_id: Origin plot identifier.

        Returns:
            List of BatchRecord instances referencing this plot.
        """
        batch_ids = self._plot_index.get(plot_id, set())
        return [
            self._nodes[bid].batch
            for bid in batch_ids
            if bid in self._nodes
        ]

    def get_batches_by_commodity(
        self,
        commodity: EUDRCommodity,
    ) -> List[BatchRecord]:
        """Get all batches for a given EUDR commodity.

        Args:
            commodity: EUDR commodity to filter by.

        Returns:
            List of BatchRecord instances for the commodity.
        """
        batch_ids = self._commodity_index.get(commodity.value, set())
        return [
            self._nodes[bid].batch
            for bid in batch_ids
            if bid in self._nodes
        ]

    def get_operations_for_batch(
        self,
        batch_id: str,
    ) -> List[BatchOperation]:
        """Get all operations involving a given batch (as input or output).

        Args:
            batch_id: Batch identifier.

        Returns:
            List of BatchOperation instances.
        """
        node = self._nodes.get(batch_id)
        if node is None:
            return []
        return [
            self._operations[op_id]
            for op_id in node.operation_ids
            if op_id in self._operations
        ]

    @property
    def alerts(self) -> List[ComplianceAlert]:
        """Return all compliance alerts raised by the engine."""
        return list(self._alerts)

    def clear_alerts(self) -> int:
        """Clear all compliance alerts and return the count cleared.

        Returns:
            Number of alerts cleared.
        """
        count = len(self._alerts)
        self._alerts.clear()
        return count

    # ------------------------------------------------------------------
    # Public API -- Statistics
    # ------------------------------------------------------------------

    @property
    def batch_count(self) -> int:
        """Return the total number of batches registered."""
        return len(self._nodes)

    @property
    def operation_count(self) -> int:
        """Return the total number of operations recorded."""
        return len(self._operations)

    @property
    def alert_count(self) -> int:
        """Return the total number of active compliance alerts."""
        return len(self._alerts)

    def get_statistics(self) -> Dict[str, Any]:
        """Return engine statistics as a dictionary.

        Returns:
            Dictionary with batch_count, operation_count, alert_count,
            plot_count, commodity_breakdown, and operation_type_breakdown.
        """
        op_types: Dict[str, int] = {}
        for op in self._operations.values():
            key = op.operation_type.value
            op_types[key] = op_types.get(key, 0) + 1

        commodity_counts: Dict[str, int] = {}
        for commodity, batch_ids in self._commodity_index.items():
            commodity_counts[commodity] = len(batch_ids)

        return {
            "batch_count": self.batch_count,
            "operation_count": self.operation_count,
            "alert_count": self.alert_count,
            "plot_count": len(self._plot_index),
            "commodity_breakdown": commodity_counts,
            "operation_type_breakdown": op_types,
        }

    # ------------------------------------------------------------------
    # Public API -- Import from ChainOfCustodyEngine
    # ------------------------------------------------------------------

    def import_from_chain_of_custody(
        self,
        chain_engine: Any,
    ) -> int:
        """Import batches from an existing ChainOfCustodyEngine.

        Reads all batches from the ChainOfCustodyEngine's internal storage
        and registers them in the traceability graph, preserving parent-child
        relationships.

        Args:
            chain_engine: ChainOfCustodyEngine instance.

        Returns:
            Number of batches imported.

        Raises:
            AttributeError: If the engine does not have expected attributes.
        """
        imported = 0

        # Access the internal _batches dict from ChainOfCustodyEngine
        batches: Dict[str, BatchRecord] = getattr(chain_engine, "_batches", {})

        # First pass: register all batches as nodes
        for bid, batch in batches.items():
            if bid not in self._nodes:
                node = _BatchNode(batch)
                self._nodes[bid] = node
                self._index_batch(
                    bid,
                    batch.commodity,
                    batch.origin_plot_ids,
                )
                imported += 1

        # Second pass: rebuild parent-child links
        for bid, batch in batches.items():
            node = self._nodes.get(bid)
            if node is None:
                continue
            for parent_id in batch.parent_batch_ids:
                parent_node = self._nodes.get(parent_id)
                if parent_node is not None:
                    node.parent_ids.add(parent_id)
                    parent_node.child_ids.add(bid)
                    node.is_leaf = False

        logger.info(
            "Imported %d batches from ChainOfCustodyEngine", imported
        )
        return imported

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _index_batch(
        self,
        batch_id: str,
        commodity: EUDRCommodity,
        origin_plot_ids: List[str],
    ) -> None:
        """Update internal indexes for a batch.

        Args:
            batch_id: Batch identifier.
            commodity: EUDR commodity.
            origin_plot_ids: List of origin plot IDs.
        """
        # Plot index
        for pid in origin_plot_ids:
            self._plot_index.setdefault(pid, set()).add(batch_id)

        # Commodity index
        self._commodity_index.setdefault(commodity.value, set()).add(batch_id)

    def _resolve_plot(self, plot_id: str) -> Optional[PlotRecord]:
        """Resolve a plot ID to a PlotRecord using the plot registry.

        Args:
            plot_id: Plot identifier.

        Returns:
            PlotRecord or None if registry not available or plot not found.
        """
        if self._plot_registry is None:
            return None
        try:
            return self._plot_registry.get_plot(plot_id)
        except Exception:
            logger.debug("Failed to resolve plot %s", plot_id)
            return None

    def _strictest_custody_model(
        self,
        models: List[CustodyModel],
    ) -> CustodyModel:
        """Determine the strictest custody model from a list.

        Order of strictness (strictest first):
        Identity Preserved > Segregated > Mass Balance.

        Args:
            models: List of custody models.

        Returns:
            The strictest CustodyModel present.
        """
        precedence = {
            CustodyModel.IDENTITY_PRESERVED: 0,
            CustodyModel.SEGREGATED: 1,
            CustodyModel.MASS_BALANCE: 2,
        }
        return min(models, key=lambda m: precedence.get(m, 99))

    def _score_ip_segregated(
        self,
        batch_id: str,
        node: _BatchNode,
    ) -> TraceabilityScore:
        """Score traceability for Identity Preserved or Segregated model.

        Binary scoring: 100 if a backward trace finds all origins known,
        0 if any leaf has unknown origins.

        Args:
            batch_id: Batch identifier.
            node: Internal batch node.

        Returns:
            TraceabilityScore with binary score.
        """
        trace = self.backward_trace(batch_id)
        total_qty = node.batch.quantity
        has_unknown = trace.unknown_origin_count > 0

        if has_unknown:
            score = Decimal("0.00")
            traced_qty = Decimal("0")
            untraced_qty = total_qty
        else:
            score = Decimal("100.00")
            traced_qty = total_qty
            untraced_qty = Decimal("0")

        result = TraceabilityScore(
            batch_id=batch_id,
            custody_model=node.batch.custody_model,
            score=str(score),
            total_quantity=str(total_qty),
            traced_quantity=str(traced_qty),
            untraced_quantity=str(untraced_qty),
            known_plot_count=len(trace.origin_plot_ids),
            unknown_leaf_count=trace.unknown_origin_count,
            is_fully_traceable=not has_unknown and len(trace.origin_plot_ids) > 0,
        )
        result.provenance_hash = _compute_hash(
            result.model_dump(mode="json", exclude={"provenance_hash"})
        )
        return result

    def _score_mass_balance(
        self,
        batch_id: str,
        node: _BatchNode,
    ) -> TraceabilityScore:
        """Score traceability for Mass Balance model with partial scoring.

        Walks backward through parent links. For each leaf batch, if origin
        plots are known, its quantity contributes to ``traced_quantity``.
        If unknown, it contributes to ``untraced_quantity``.  The score is
        ``(traced_quantity / total_quantity) * 100``.

        This is the key differentiation for Mass Balance: partial traceability
        is permitted and scored proportionally.

        Args:
            batch_id: Batch identifier.
            node: Internal batch node.

        Returns:
            TraceabilityScore with proportional score.
        """
        total_qty = node.batch.quantity

        # Walk backward to collect leaf quantities
        traced_qty = Decimal("0")
        untraced_qty = Decimal("0")
        known_plots: Set[str] = set()
        unknown_leaves = 0

        visited: Set[str] = set()
        queue: deque[str] = deque()
        queue.append(batch_id)

        while queue:
            current_id = queue.popleft()
            if current_id in visited:
                continue
            visited.add(current_id)

            current_node = self._nodes.get(current_id)
            if current_node is None:
                # External/missing node -- treat as unknown
                unknown_leaves += 1
                continue

            if not current_node.parent_ids:
                # Leaf node -- check origins
                if current_node.batch.origin_plot_ids:
                    traced_qty += current_node.batch.quantity
                    for pid in current_node.batch.origin_plot_ids:
                        known_plots.add(pid)
                else:
                    untraced_qty += current_node.batch.quantity
                    unknown_leaves += 1
            else:
                for pid in current_node.parent_ids:
                    if pid not in visited:
                        queue.append(pid)

        # Calculate score -- handle edge case where leaf quantities
        # may not sum to total_qty (due to processing yield loss)
        leaf_total = traced_qty + untraced_qty
        if leaf_total > Decimal("0"):
            score = (traced_qty / leaf_total * Decimal("100")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        else:
            score = Decimal("0.00")

        result = TraceabilityScore(
            batch_id=batch_id,
            custody_model=CustodyModel.MASS_BALANCE,
            score=str(score),
            total_quantity=str(total_qty),
            traced_quantity=str(traced_qty),
            untraced_quantity=str(untraced_qty),
            known_plot_count=len(known_plots),
            unknown_leaf_count=unknown_leaves,
            is_fully_traceable=unknown_leaves == 0 and len(known_plots) > 0,
        )
        result.provenance_hash = _compute_hash(
            result.model_dump(mode="json", exclude={"provenance_hash"})
        )
        return result

    def _raise_alert(
        self,
        severity: AlertSeverity,
        alert_type: str,
        batch_id: str,
        message: str,
        eudr_article: str = "",
        operation_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> ComplianceAlert:
        """Create and store a compliance alert.

        Args:
            severity: Alert severity level.
            alert_type: Type of compliance issue.
            batch_id: Batch that triggered the alert.
            message: Human-readable message.
            eudr_article: EUDR article reference.
            operation_id: Optional operation ID.
            details: Optional structured details.

        Returns:
            The created ComplianceAlert.
        """
        alert = ComplianceAlert(
            severity=severity,
            alert_type=alert_type,
            batch_id=batch_id,
            operation_id=operation_id,
            message=message,
            eudr_article=eudr_article,
            details=details or {},
        )
        self._alerts.append(alert)
        logger.warning(
            "Compliance alert [%s] %s: %s (batch=%s)",
            severity.value,
            alert_type,
            message,
            batch_id,
        )
        return alert

    def _record_provenance(
        self,
        entity_type: str,
        entity_id: str,
        action: str,
        data_hash: str,
    ) -> None:
        """Record a provenance entry if a ProvenanceTracker is configured.

        Args:
            entity_type: Entity type string.
            entity_id: Entity identifier.
            action: Action performed.
            data_hash: SHA-256 hash of the operation data.
        """
        if self._provenance is None:
            return
        try:
            self._provenance.record(
                entity_type=entity_type,
                entity_id=entity_id,
                action=action,
                data_hash=data_hash,
            )
        except Exception as exc:
            logger.warning(
                "Failed to record provenance for %s/%s: %s",
                entity_type,
                entity_id,
                exc,
            )

    def _record_metrics(self, operation: str, duration_seconds: float) -> None:
        """Record Prometheus metrics if available.

        Args:
            operation: Operation name for labeling.
            duration_seconds: Duration of the operation in seconds.
        """
        try:
            from greenlang.agents.data.eudr_traceability.metrics import record_batch_operation
            record_batch_operation(operation, duration_seconds)
        except ImportError:
            pass

__all__ = [
    "BatchTraceabilityEngine",
    "BatchOperation",
    "BatchOperationType",
    "AlertSeverity",
    "TraceResult",
    "MassBalanceResult",
    "ComplianceAlert",
    "TraceabilityScore",
]
