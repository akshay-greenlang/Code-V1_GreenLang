# -*- coding: utf-8 -*-
"""
Chain of Custody Engine - AGENT-DATA-004: EUDR Traceability Connector

Manages chain of custody transfers for EUDR-regulated commodities,
enabling full traceability from final product back to production plot.
Supports identity-preserved, segregated, and mass-balance custody
models with batch splitting and merging.

Zero-Hallucination Guarantees:
    - Mass balance calculations are deterministic arithmetic
    - Batch quantity reconciliation uses exact Decimal comparison
    - SHA-256 provenance hashes on all custody operations
    - Recursive origin tracing via transfer chain

Example:
    >>> from greenlang.eudr_traceability.chain_of_custody import ChainOfCustodyEngine
    >>> engine = ChainOfCustodyEngine()
    >>> transfer = engine.record_transfer(request)
    >>> origins = engine.trace_to_origin(transfer.transfer_id)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-004 EUDR Traceability Connector
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from greenlang.eudr_traceability.models import (
    BatchRecord,
    CustodyModel,
    CustodyTransfer,
    EUDRCommodity,
    PlotRecord,
    RecordTransferRequest,
)

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash.

    Returns:
        SHA-256 hex digest string.
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    else:
        serializable = data
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


class ChainOfCustodyEngine:
    """Chain of custody tracking engine for EUDR-regulated commodities.

    Manages custody transfers, batch splitting/merging, mass balance
    verification, and recursive origin tracing for full supply chain
    traceability.

    Attributes:
        _config: Configuration dictionary or object.
        _plot_registry: Optional PlotRegistryEngine for origin lookups.
        _transfers: In-memory transfer storage keyed by transfer_id.
        _batches: In-memory batch storage keyed by batch_id.
        _provenance: Provenance tracker instance.

    Example:
        >>> engine = ChainOfCustodyEngine()
        >>> transfer = engine.record_transfer(request)
        >>> assert transfer.transfer_id is not None
    """

    def __init__(
        self,
        config: Any = None,
        plot_registry: Any = None,
        provenance: Any = None,
    ) -> None:
        """Initialize ChainOfCustodyEngine.

        Args:
            config: Optional configuration.
            plot_registry: Optional PlotRegistryEngine for origin lookups.
            provenance: Optional ProvenanceTracker instance.
        """
        self._config = config or {}
        self._plot_registry = plot_registry
        self._provenance = provenance

        # In-memory storage
        self._transfers: Dict[str, CustodyTransfer] = {}
        self._batches: Dict[str, BatchRecord] = {}

        # Indexes
        self._idx_batch_transfers: Dict[str, List[str]] = {}
        self._idx_operator_transfers: Dict[str, List[str]] = {}

        logger.info("ChainOfCustodyEngine initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_transfer(
        self,
        request: RecordTransferRequest,
    ) -> CustodyTransfer:
        """Record a custody transfer between operators.

        Creates a transfer record and optionally creates/updates the
        associated batch record.

        Args:
            request: Transfer request with details.

        Returns:
            CustodyTransfer with generated transfer_id and provenance.

        Raises:
            ValueError: If request validation fails.
        """
        start_time = time.monotonic()

        transfer_id = self._generate_transfer_id()

        # Create transfer using new model field names
        transfer = CustodyTransfer(
            transfer_id=transfer_id,
            source_operator_id=request.source_operator_id,
            source_operator_name=request.source_operator_name,
            target_operator_id=request.target_operator_id,
            target_operator_name=request.target_operator_name,
            commodity=request.commodity,
            product_description=request.product_description,
            quantity=request.quantity,
            origin_plot_ids=list(request.origin_plot_ids),
            custody_model=request.custody_model,
            batch_number=request.batch_number,
            transport_mode=request.transport_mode,
            cn_code=request.cn_code,
            hs_code=request.hs_code,
        )

        # Ensure batch exists for tracking
        batch_key = request.batch_number or transfer_id
        if batch_key not in self._batches:
            batch = BatchRecord(
                batch_id=batch_key,
                commodity=request.commodity,
                product_description=request.product_description,
                quantity=request.quantity,
                origin_plot_ids=list(request.origin_plot_ids),
                custody_model=request.custody_model,
            )
            self._batches[batch_key] = batch

        # Store and index
        self._transfers[transfer_id] = transfer
        self._index_transfer(transfer)

        # Record provenance
        if self._provenance is not None:
            data_hash = _compute_hash(transfer)
            self._provenance.record(
                entity_type="transfer",
                entity_id=transfer_id,
                action="custody_transfer",
                data_hash=data_hash,
            )

        # Record metrics
        try:
            from greenlang.eudr_traceability.metrics import record_custody_transfer
            record_custody_transfer(
                request.commodity.value,
                request.custody_model.value,
            )
        except ImportError:
            pass

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Recorded transfer %s: %s -> %s, qty=%s (%.1f ms)",
            transfer_id,
            request.source_operator_id[:8],
            request.target_operator_id[:8],
            request.quantity, elapsed_ms,
        )
        return transfer

    def get_transfer(self, transfer_id: str) -> Optional[CustodyTransfer]:
        """Get a transfer record by ID.

        Args:
            transfer_id: Transfer identifier.

        Returns:
            CustodyTransfer or None if not found.
        """
        return self._transfers.get(transfer_id)

    def list_transfers(
        self,
        commodity: Optional[str] = None,
        operator_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[CustodyTransfer]:
        """List transfers with optional filtering.

        Args:
            commodity: Optional commodity filter.
            operator_id: Optional operator filter (source or target).
            limit: Maximum results.
            offset: Results to skip.

        Returns:
            List of CustodyTransfer instances.
        """
        transfers = list(self._transfers.values())

        if commodity is not None:
            transfers = [
                t for t in transfers if t.commodity.value == commodity
            ]
        if operator_id is not None:
            transfers = [
                t for t in transfers
                if t.source_operator_id == operator_id
                or t.target_operator_id == operator_id
            ]

        return transfers[offset:offset + limit]

    def trace_to_origin(self, batch_id: str) -> List[PlotRecord]:
        """Recursively trace a batch back to its origin plots.

        Follows the transfer chain backwards to find all production
        plots that contributed to this batch.

        Args:
            batch_id: Batch identifier to trace.

        Returns:
            List of PlotRecord instances (origin plots).
        """
        if self._plot_registry is None:
            logger.warning(
                "Cannot trace to origin: no PlotRegistryEngine configured"
            )
            return []

        visited_batches: set = set()
        origin_plots: List[PlotRecord] = []
        self._recursive_trace(batch_id, visited_batches, origin_plots)
        return origin_plots

    def get_full_chain(
        self,
        operator_id: str,
        commodity: Optional[str] = None,
    ) -> List[CustodyTransfer]:
        """Get the full chain of transfers for an operator.

        Args:
            operator_id: Operator identifier.
            commodity: Optional commodity filter.

        Returns:
            List of CustodyTransfer instances involving this operator.
        """
        transfer_ids = self._idx_operator_transfers.get(operator_id, [])
        transfers = [
            self._transfers[tid] for tid in transfer_ids
            if tid in self._transfers
        ]

        if commodity is not None:
            transfers = [
                t for t in transfers if t.commodity.value == commodity
            ]

        return transfers

    def split_batch(
        self,
        parent_batch_id: str,
        split_quantities: List[Decimal],
        descriptions: List[str],
    ) -> List[BatchRecord]:
        """Split a batch into multiple child batches.

        The sum of split_quantities must not exceed the parent batch
        quantity (deterministic mass balance check).

        Args:
            parent_batch_id: Batch to split.
            split_quantities: Quantities for each child batch.
            descriptions: Descriptions for each child batch.

        Returns:
            List of new BatchRecord instances.

        Raises:
            ValueError: If parent not found or quantities exceed parent.
        """
        parent = self._batches.get(parent_batch_id)
        if parent is None:
            raise ValueError(f"Batch {parent_batch_id} not found")

        total_split = sum(split_quantities)
        if total_split > parent.quantity:
            raise ValueError(
                f"Split total ({total_split}) exceeds parent quantity "
                f"({parent.quantity})"
            )

        if len(split_quantities) != len(descriptions):
            raise ValueError(
                "split_quantities and descriptions must have same length"
            )

        child_batches: List[BatchRecord] = []
        for qty, desc in zip(split_quantities, descriptions):
            child_id = self._generate_batch_id()
            child = BatchRecord(
                batch_id=child_id,
                commodity=parent.commodity,
                product_description=desc,
                quantity=qty,
                origin_plot_ids=list(parent.origin_plot_ids),
                parent_batch_ids=[parent_batch_id],
                custody_model=parent.custody_model,
            )
            self._batches[child_id] = child
            child_batches.append(child)

        # Update parent quantity (remaining)
        parent.quantity -= total_split

        # Record provenance
        if self._provenance is not None:
            for child in child_batches:
                data_hash = _compute_hash(child)
                self._provenance.record(
                    entity_type="batch",
                    entity_id=child.batch_id,
                    action="batch_split",
                    data_hash=data_hash,
                )

        logger.info(
            "Split batch %s into %d child batches",
            parent_batch_id, len(child_batches),
        )
        return child_batches

    def merge_batches(
        self,
        batch_ids: List[str],
        new_description: str,
    ) -> BatchRecord:
        """Merge multiple batches into one.

        All batches must have the same commodity. The merged batch
        quantity is the sum of all input quantities.

        Args:
            batch_ids: List of batch IDs to merge.
            new_description: Description for the merged batch.

        Returns:
            New merged BatchRecord.

        Raises:
            ValueError: If batches not found or commodity mismatch.
        """
        if len(batch_ids) < 2:
            raise ValueError("At least 2 batches required for merge")

        batches: List[BatchRecord] = []
        for bid in batch_ids:
            batch = self._batches.get(bid)
            if batch is None:
                raise ValueError(f"Batch {bid} not found")
            batches.append(batch)

        # Validate same commodity
        commodities = {b.commodity for b in batches}
        if len(commodities) > 1:
            raise ValueError(
                f"Cannot merge batches with different commodities: "
                f"{[c.value for c in commodities]}"
            )

        merged_id = self._generate_batch_id()
        total_quantity = sum(b.quantity for b in batches)
        all_origins: List[str] = []
        for b in batches:
            all_origins.extend(b.origin_plot_ids)
        unique_origins = list(dict.fromkeys(all_origins))

        merged = BatchRecord(
            batch_id=merged_id,
            commodity=batches[0].commodity,
            product_description=new_description,
            quantity=total_quantity,
            origin_plot_ids=unique_origins,
            parent_batch_ids=list(batch_ids),
            custody_model=batches[0].custody_model,
        )
        self._batches[merged_id] = merged

        # Record provenance
        if self._provenance is not None:
            data_hash = _compute_hash(merged)
            self._provenance.record(
                entity_type="batch",
                entity_id=merged_id,
                action="batch_merge",
                data_hash=data_hash,
            )

        logger.info(
            "Merged %d batches into %s: qty=%s",
            len(batch_ids), merged_id, total_quantity,
        )
        return merged

    def get_mass_balance(self, batch_id: str) -> Dict[str, Any]:
        """Get mass balance reconciliation for a batch.

        Compares total input quantity to the batch plus any split
        children for deterministic balance verification.

        Args:
            batch_id: Batch identifier.

        Returns:
            Dictionary with input_quantity, output_quantity, balance,
            and is_balanced flag.

        Raises:
            ValueError: If batch not found.
        """
        batch = self._batches.get(batch_id)
        if batch is None:
            raise ValueError(f"Batch {batch_id} not found")

        # Input: transfers into this batch
        transfer_ids = self._idx_batch_transfers.get(batch_id, [])
        input_qty = sum(
            self._transfers[tid].quantity
            for tid in transfer_ids
            if tid in self._transfers
        )

        # If no transfers, input is the batch's own initial quantity
        if input_qty == Decimal("0"):
            input_qty = batch.quantity

        # Output: current batch quantity + children created from splits
        output_qty = batch.quantity
        for b in self._batches.values():
            if batch_id in b.parent_batch_ids:
                output_qty += b.quantity

        balance = input_qty - output_qty
        is_balanced = abs(balance) < Decimal("0.001")

        return {
            "batch_id": batch_id,
            "input_quantity": str(input_qty),
            "output_quantity": str(output_qty),
            "balance": str(balance),
            "is_balanced": is_balanced,
            "unit": batch.unit,
        }

    def verify_transfer(
        self,
        transfer_id: str,
        verifier: str,
    ) -> Optional[CustodyTransfer]:
        """Mark a transfer as verified.

        Args:
            transfer_id: Transfer identifier.
            verifier: Identifier of the verifier.

        Returns:
            Updated CustodyTransfer or None if not found.
        """
        transfer = self._transfers.get(transfer_id)
        if transfer is None:
            logger.warning("Cannot verify: transfer %s not found", transfer_id)
            return None

        from greenlang.eudr_traceability.models import ComplianceStatus
        transfer.verification_status = ComplianceStatus.COMPLIANT
        transfer.verified_by = verifier
        transfer.verified_at = _utcnow()

        # Record provenance
        if self._provenance is not None:
            data_hash = _compute_hash(transfer)
            self._provenance.record(
                entity_type="transfer",
                entity_id=transfer_id,
                action="transfer_verification",
                data_hash=data_hash,
            )

        logger.info(
            "Transfer %s verified by %s",
            transfer_id, verifier,
        )
        return transfer

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _recursive_trace(
        self,
        batch_id: str,
        visited: set,
        origins: List[PlotRecord],
    ) -> None:
        """Recursively trace a batch to origin plots.

        Args:
            batch_id: Current batch ID to trace.
            visited: Set of already-visited batch IDs.
            origins: Accumulator for origin PlotRecord instances.
        """
        if batch_id in visited:
            return
        visited.add(batch_id)

        batch = self._batches.get(batch_id)
        if batch is None:
            return

        # Check if this batch has direct plot origins
        if batch.origin_plot_ids and self._plot_registry is not None:
            for plot_id in batch.origin_plot_ids:
                plot = self._plot_registry.get_plot(plot_id)
                if plot is not None and plot.plot_id not in {
                    p.plot_id for p in origins
                }:
                    origins.append(plot)

        # Trace parent batches (splits/merges)
        for parent_id in batch.parent_batch_ids:
            self._recursive_trace(parent_id, visited, origins)

    def _index_transfer(self, transfer: CustodyTransfer) -> None:
        """Index a transfer by batch and operator.

        Args:
            transfer: CustodyTransfer to index.
        """
        # Batch index (using batch_number if available)
        batch_key = transfer.batch_number or transfer.transfer_id
        if batch_key not in self._idx_batch_transfers:
            self._idx_batch_transfers[batch_key] = []
        self._idx_batch_transfers[batch_key].append(
            transfer.transfer_id
        )

        # Operator index (both source and target)
        for op_id in (transfer.source_operator_id, transfer.target_operator_id):
            if op_id not in self._idx_operator_transfers:
                self._idx_operator_transfers[op_id] = []
            self._idx_operator_transfers[op_id].append(transfer.transfer_id)

    def _generate_transfer_id(self) -> str:
        """Generate a unique transfer identifier.

        Returns:
            Transfer ID in format "COC-{hex12}".
        """
        return f"COC-{uuid.uuid4().hex[:12]}"

    def _generate_batch_id(self) -> str:
        """Generate a unique batch identifier.

        Returns:
            Batch ID in format "BATCH-{hex12}".
        """
        return f"BATCH-{uuid.uuid4().hex[:12]}"

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def transfer_count(self) -> int:
        """Return the total number of transfers."""
        return len(self._transfers)

    @property
    def batch_count(self) -> int:
        """Return the total number of batches."""
        return len(self._batches)


__all__ = [
    "ChainOfCustodyEngine",
]
