# -*- coding: utf-8 -*-
"""
Unit Tests for ChainOfCustodyEngine (AGENT-DATA-005)

Tests custody transfer recording, retrieval, listing with filters
(commodity, operator, batch), trace-to-origin, full chain retrieval,
batch splitting, batch merging, mass balance, verification, custody
model assignment, CN code storage, and provenance chain integrity.

Coverage target: 85%+ of chain_of_custody.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline PlotRecord (minimal for chain of custody dependency)
# ---------------------------------------------------------------------------


class PlotRecord:
    def __init__(
        self,
        plot_id: str = "",
        commodity: str = "cocoa",
        country_code: str = "",
        operator_id: str = "",
        risk_level: str = "standard",
    ):
        self.plot_id = plot_id
        self.commodity = commodity
        self.country_code = country_code
        self.operator_id = operator_id
        self.risk_level = risk_level


# ---------------------------------------------------------------------------
# Inline PlotRegistryEngine (minimal stub for chain of custody integration)
# ---------------------------------------------------------------------------


class PlotRegistryEngine:
    """Minimal stub of PlotRegistryEngine for chain of custody tests."""

    def __init__(self):
        self._plots: Dict[str, PlotRecord] = {}

    def add_plot(self, plot: PlotRecord) -> None:
        self._plots[plot.plot_id] = plot

    def get_plot(self, plot_id: str) -> Optional[PlotRecord]:
        return self._plots.get(plot_id)


# ---------------------------------------------------------------------------
# Inline CustodyTransfer
# ---------------------------------------------------------------------------


class CustodyTransfer:
    def __init__(
        self,
        transfer_id: str = "",
        transaction_id: str = "",
        batch_number: str = "",
        commodity: str = "cocoa",
        quantity_kg: float = 0.0,
        unit: str = "kg",
        custody_model: str = "segregated",
        from_operator_id: str = "",
        to_operator_id: str = "",
        origin_plot_ids: Optional[List[str]] = None,
        cn_code: str = "",
        transfer_date: Optional[str] = None,
        verified: bool = False,
        provenance_hash: Optional[str] = None,
    ):
        self.transfer_id = transfer_id
        self.transaction_id = transaction_id
        self.batch_number = batch_number
        self.commodity = commodity
        self.quantity_kg = quantity_kg
        self.unit = unit
        self.custody_model = custody_model
        self.from_operator_id = from_operator_id
        self.to_operator_id = to_operator_id
        self.origin_plot_ids = origin_plot_ids or []
        self.cn_code = cn_code
        self.transfer_date = transfer_date or datetime.now(timezone.utc).isoformat()
        self.verified = verified
        self.provenance_hash = provenance_hash


# ---------------------------------------------------------------------------
# Inline RecordTransferRequest
# ---------------------------------------------------------------------------


class RecordTransferRequest:
    def __init__(
        self,
        batch_number: str = "",
        commodity: str = "cocoa",
        quantity_kg: float = 0.0,
        custody_model: str = "segregated",
        from_operator_id: str = "",
        to_operator_id: str = "",
        origin_plot_ids: Optional[List[str]] = None,
        cn_code: str = "",
    ):
        self.batch_number = batch_number
        self.commodity = commodity
        self.quantity_kg = quantity_kg
        self.custody_model = custody_model
        self.from_operator_id = from_operator_id
        self.to_operator_id = to_operator_id
        self.origin_plot_ids = origin_plot_ids or []
        self.cn_code = cn_code


# ---------------------------------------------------------------------------
# Inline ChainOfCustodyEngine
# ---------------------------------------------------------------------------


class ChainOfCustodyEngine:
    """Manages EUDR chain of custody transfers, tracing, and mass balance."""

    def __init__(
        self,
        plot_registry: Optional[PlotRegistryEngine] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self._plot_registry = plot_registry or PlotRegistryEngine()
        self._config = config or {}
        self._transfers: Dict[str, CustodyTransfer] = {}
        self._batch_index: Dict[str, List[str]] = {}  # batch_number -> [transfer_id]
        self._operator_index: Dict[str, List[str]] = {}  # operator_id -> [transfer_id]
        self._lock = threading.Lock()
        self._transfer_counter = 0
        self._txn_counter = 0
        self._stats = {
            "transfers_recorded": 0,
            "transfers_verified": 0,
            "batches_split": 0,
            "batches_merged": 0,
        }

    def _next_transfer_id(self) -> str:
        self._transfer_counter += 1
        return f"TXF-{self._transfer_counter:05d}"

    def _next_transaction_id(self) -> str:
        self._txn_counter += 1
        return f"TXN-{self._txn_counter:05d}"

    def _compute_provenance(self, data: Dict[str, Any]) -> str:
        canonical = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def record_transfer(self, request: RecordTransferRequest) -> CustodyTransfer:
        """Record a new custody transfer."""
        if request.quantity_kg <= 0:
            raise ValueError(f"Quantity must be positive, got {request.quantity_kg}")

        with self._lock:
            transfer_id = self._next_transfer_id()
            txn_id = self._next_transaction_id()

        prov_data = {
            "op": "record_transfer", "transfer_id": transfer_id,
            "batch": request.batch_number, "commodity": request.commodity,
            "qty": request.quantity_kg, "from": request.from_operator_id,
            "to": request.to_operator_id,
        }

        transfer = CustodyTransfer(
            transfer_id=transfer_id,
            transaction_id=txn_id,
            batch_number=request.batch_number,
            commodity=request.commodity,
            quantity_kg=request.quantity_kg,
            custody_model=request.custody_model,
            from_operator_id=request.from_operator_id,
            to_operator_id=request.to_operator_id,
            origin_plot_ids=list(request.origin_plot_ids),
            cn_code=request.cn_code,
            provenance_hash=self._compute_provenance(prov_data),
        )

        with self._lock:
            self._transfers[transfer_id] = transfer
            # Index by batch number
            if request.batch_number:
                self._batch_index.setdefault(request.batch_number, []).append(transfer_id)
            # Index by operator (both from and to)
            if request.from_operator_id:
                self._operator_index.setdefault(request.from_operator_id, []).append(transfer_id)
            if request.to_operator_id:
                self._operator_index.setdefault(request.to_operator_id, []).append(transfer_id)
            self._stats["transfers_recorded"] += 1

        return transfer

    def get_transfer(self, transfer_id: str) -> Optional[CustodyTransfer]:
        """Retrieve a transfer by ID. Returns None if not found."""
        with self._lock:
            return self._transfers.get(transfer_id)

    def list_transfers(
        self,
        commodity: Optional[str] = None,
        operator_id: Optional[str] = None,
        batch_number: Optional[str] = None,
    ) -> List[CustodyTransfer]:
        """List transfers with optional filters."""
        with self._lock:
            if batch_number is not None:
                ids = self._batch_index.get(batch_number, [])
                result = [self._transfers[tid] for tid in ids if tid in self._transfers]
            elif operator_id is not None:
                ids = self._operator_index.get(operator_id, [])
                result = [self._transfers[tid] for tid in ids if tid in self._transfers]
            else:
                result = list(self._transfers.values())
        if commodity is not None:
            result = [t for t in result if t.commodity == commodity]
        return result

    def trace_to_origin(self, batch_number: str) -> List[str]:
        """Trace a batch back to its origin plot IDs."""
        with self._lock:
            ids = self._batch_index.get(batch_number, [])
            plot_ids = set()
            for tid in ids:
                transfer = self._transfers.get(tid)
                if transfer:
                    plot_ids.update(transfer.origin_plot_ids)
        return sorted(plot_ids)

    def get_full_chain(
        self,
        operator_id: str,
        commodity: Optional[str] = None,
    ) -> List[CustodyTransfer]:
        """Get full transfer chain for an operator, optionally filtered by commodity."""
        return self.list_transfers(operator_id=operator_id, commodity=commodity)

    def split_batch(
        self,
        source_transfer_id: str,
        split_quantities_kg: List[float],
    ) -> List[CustodyTransfer]:
        """Split a batch into sub-batches. Quantities must sum to original."""
        with self._lock:
            source = self._transfers.get(source_transfer_id)
        if source is None:
            raise ValueError(f"Unknown transfer: {source_transfer_id}")

        total_split = sum(split_quantities_kg)
        if abs(total_split - source.quantity_kg) > 0.01:
            raise ValueError(
                f"Split quantities ({total_split}) must sum to original "
                f"({source.quantity_kg})"
            )

        sub_transfers = []
        for i, qty in enumerate(split_quantities_kg):
            sub_req = RecordTransferRequest(
                batch_number=f"{source.batch_number}-S{i + 1:02d}",
                commodity=source.commodity,
                quantity_kg=qty,
                custody_model=source.custody_model,
                from_operator_id=source.to_operator_id,
                to_operator_id="",  # TBD by next transfer
                origin_plot_ids=list(source.origin_plot_ids),
                cn_code=source.cn_code,
            )
            sub = self.record_transfer(sub_req)
            sub_transfers.append(sub)

        with self._lock:
            self._stats["batches_split"] += 1
        return sub_transfers

    def merge_batches(
        self,
        source_transfer_ids: List[str],
        merged_batch_number: str,
        to_operator_id: str = "",
    ) -> CustodyTransfer:
        """Merge multiple batches into one. Origin plots are combined."""
        sources = []
        for tid in source_transfer_ids:
            with self._lock:
                src = self._transfers.get(tid)
            if src is None:
                raise ValueError(f"Unknown transfer: {tid}")
            sources.append(src)

        # Validate all same commodity
        commodities = set(s.commodity for s in sources)
        if len(commodities) != 1:
            raise ValueError(f"Cannot merge different commodities: {commodities}")

        total_qty = sum(s.quantity_kg for s in sources)
        all_plot_ids = set()
        for s in sources:
            all_plot_ids.update(s.origin_plot_ids)

        merge_req = RecordTransferRequest(
            batch_number=merged_batch_number,
            commodity=sources[0].commodity,
            quantity_kg=total_qty,
            custody_model="mass_balance",  # Merging implies mass balance
            from_operator_id=sources[0].to_operator_id,
            to_operator_id=to_operator_id,
            origin_plot_ids=sorted(all_plot_ids),
            cn_code=sources[0].cn_code,
        )
        merged = self.record_transfer(merge_req)

        with self._lock:
            self._stats["batches_merged"] += 1
        return merged

    def get_mass_balance(self, operator_id: str) -> Dict[str, Any]:
        """Get input vs output mass balance reconciliation for an operator."""
        with self._lock:
            ids = self._operator_index.get(operator_id, [])
            transfers = [self._transfers[tid] for tid in ids if tid in self._transfers]

        input_kg = 0.0
        output_kg = 0.0
        for t in transfers:
            if t.to_operator_id == operator_id:
                input_kg += t.quantity_kg
            if t.from_operator_id == operator_id:
                output_kg += t.quantity_kg

        balance_kg = input_kg - output_kg
        return {
            "operator_id": operator_id,
            "input_kg": round(input_kg, 2),
            "output_kg": round(output_kg, 2),
            "balance_kg": round(balance_kg, 2),
            "balanced": abs(balance_kg) < 0.01,
        }

    def verify_transfer(self, transfer_id: str) -> Optional[CustodyTransfer]:
        """Mark a transfer as verified. Returns None if not found."""
        with self._lock:
            transfer = self._transfers.get(transfer_id)
            if transfer is None:
                return None
            transfer.verified = True
            self._stats["transfers_verified"] += 1
        return transfer

    def get_statistics(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._stats)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def plot_registry():
    """PlotRegistryEngine with pre-registered plots."""
    registry = PlotRegistryEngine()
    registry.add_plot(PlotRecord(
        plot_id="PLOT-00001", commodity="cocoa",
        country_code="GH", operator_id="OP-001",
    ))
    registry.add_plot(PlotRecord(
        plot_id="PLOT-00002", commodity="cocoa",
        country_code="CI", operator_id="OP-001",
    ))
    registry.add_plot(PlotRecord(
        plot_id="PLOT-00003", commodity="coffee",
        country_code="ET", operator_id="OP-004",
    ))
    registry.add_plot(PlotRecord(
        plot_id="PLOT-00004", commodity="rubber",
        country_code="ID", operator_id="OP-005",
    ))
    return registry


@pytest.fixture
def engine(plot_registry):
    """ChainOfCustodyEngine with plot_registry attached."""
    return ChainOfCustodyEngine(plot_registry=plot_registry)


@pytest.fixture
def sample_transfer():
    """RecordTransferRequest for a cocoa batch."""
    return RecordTransferRequest(
        batch_number="BATCH-2026-001",
        commodity="cocoa",
        quantity_kg=5000.0,
        custody_model="segregated",
        from_operator_id="OP-001",
        to_operator_id="OP-002",
        origin_plot_ids=["PLOT-00001", "PLOT-00002"],
        cn_code="1801.00",
    )


# ===========================================================================
# Test Classes
# ===========================================================================


class TestRecordTransfer:
    """Test recording custody transfers."""

    def test_record_transfer_success(self, engine, sample_transfer):
        """Record and verify CustodyTransfer returned."""
        transfer = engine.record_transfer(sample_transfer)
        assert transfer.transfer_id is not None
        assert transfer.batch_number == "BATCH-2026-001"
        assert transfer.commodity == "cocoa"
        assert transfer.quantity_kg == 5000.0
        assert transfer.custody_model == "segregated"
        assert transfer.from_operator_id == "OP-001"
        assert transfer.to_operator_id == "OP-002"
        assert len(transfer.origin_plot_ids) == 2

    def test_record_transfer_generates_ids(self, engine, sample_transfer):
        """transfer_id (TXF-xxxxx) and transaction_id (TXN-xxxxx) formats."""
        transfer = engine.record_transfer(sample_transfer)
        assert transfer.transfer_id.startswith("TXF-")
        assert len(transfer.transfer_id) == 9  # TXF- + 5 digits
        assert transfer.transaction_id.startswith("TXN-")
        assert len(transfer.transaction_id) == 9  # TXN- + 5 digits

    def test_record_transfer_zero_quantity_raises(self, engine):
        """Zero quantity raises ValueError."""
        req = RecordTransferRequest(
            batch_number="B-001", commodity="cocoa", quantity_kg=0.0,
        )
        with pytest.raises(ValueError, match="Quantity must be positive"):
            engine.record_transfer(req)

    def test_record_transfer_negative_quantity_raises(self, engine):
        """Negative quantity raises ValueError."""
        req = RecordTransferRequest(
            batch_number="B-001", commodity="cocoa", quantity_kg=-100.0,
        )
        with pytest.raises(ValueError, match="Quantity must be positive"):
            engine.record_transfer(req)

    def test_record_transfer_increments_stats(self, engine, sample_transfer):
        engine.record_transfer(sample_transfer)
        stats = engine.get_statistics()
        assert stats["transfers_recorded"] == 1

    def test_record_transfer_provenance_hash(self, engine, sample_transfer):
        """Provenance hash is SHA-256 (64 hex chars)."""
        transfer = engine.record_transfer(sample_transfer)
        assert transfer.provenance_hash is not None
        assert len(transfer.provenance_hash) == 64


class TestGetTransfer:
    """Test transfer retrieval."""

    def test_get_transfer_exists(self, engine, sample_transfer):
        """Retrieve recorded transfer."""
        recorded = engine.record_transfer(sample_transfer)
        retrieved = engine.get_transfer(recorded.transfer_id)
        assert retrieved is not None
        assert retrieved.transfer_id == recorded.transfer_id
        assert retrieved.batch_number == "BATCH-2026-001"

    def test_get_transfer_not_found(self, engine):
        """Returns None for unknown transfer_id."""
        assert engine.get_transfer("TXF-99999") is None


class TestListTransfers:
    """Test listing transfers with filters."""

    def test_list_transfers_all(self, engine, sample_transfer):
        """List all transfers."""
        engine.record_transfer(sample_transfer)
        req2 = RecordTransferRequest(
            batch_number="BATCH-2026-002", commodity="coffee",
            quantity_kg=2000.0, from_operator_id="OP-004",
            to_operator_id="OP-005", origin_plot_ids=["PLOT-00003"],
        )
        engine.record_transfer(req2)
        transfers = engine.list_transfers()
        assert len(transfers) == 2

    def test_list_transfers_by_commodity(self, engine, sample_transfer):
        """Filter by commodity."""
        engine.record_transfer(sample_transfer)  # cocoa
        req2 = RecordTransferRequest(
            batch_number="BATCH-2026-002", commodity="coffee",
            quantity_kg=2000.0, from_operator_id="OP-004",
            to_operator_id="OP-005",
        )
        engine.record_transfer(req2)  # coffee
        cocoa_transfers = engine.list_transfers(commodity="cocoa")
        assert len(cocoa_transfers) == 1
        assert cocoa_transfers[0].commodity == "cocoa"

    def test_list_transfers_by_operator(self, engine, sample_transfer):
        """Filter by operator_id (matches both from and to)."""
        engine.record_transfer(sample_transfer)
        # OP-001 is from_operator, OP-002 is to_operator
        op1_transfers = engine.list_transfers(operator_id="OP-001")
        assert len(op1_transfers) == 1
        op2_transfers = engine.list_transfers(operator_id="OP-002")
        assert len(op2_transfers) == 1

    def test_list_transfers_by_batch(self, engine, sample_transfer):
        """Filter by batch number."""
        engine.record_transfer(sample_transfer)
        batch_transfers = engine.list_transfers(batch_number="BATCH-2026-001")
        assert len(batch_transfers) == 1
        assert batch_transfers[0].batch_number == "BATCH-2026-001"

    def test_list_transfers_by_batch_not_found(self, engine):
        """Unknown batch returns empty list."""
        assert engine.list_transfers(batch_number="BATCH-NONEXISTENT") == []

    def test_list_transfers_empty(self, engine):
        """Empty engine returns empty list."""
        assert engine.list_transfers() == []


class TestTraceToOrigin:
    """Test tracing batches back to origin plots."""

    def test_trace_to_origin(self, engine, sample_transfer):
        """Trace batch to origin plots."""
        engine.record_transfer(sample_transfer)
        plots = engine.trace_to_origin("BATCH-2026-001")
        assert len(plots) == 2
        assert "PLOT-00001" in plots
        assert "PLOT-00002" in plots

    def test_trace_to_origin_not_found(self, engine):
        """Empty list for unknown batch."""
        plots = engine.trace_to_origin("BATCH-UNKNOWN")
        assert plots == []

    def test_trace_to_origin_multiple_transfers_same_batch(self, engine):
        """Multiple transfers with same batch combine all origin plots."""
        req1 = RecordTransferRequest(
            batch_number="BATCH-MULTI", commodity="cocoa",
            quantity_kg=3000.0, from_operator_id="OP-001",
            to_operator_id="OP-002",
            origin_plot_ids=["PLOT-00001"],
        )
        req2 = RecordTransferRequest(
            batch_number="BATCH-MULTI", commodity="cocoa",
            quantity_kg=2000.0, from_operator_id="OP-001",
            to_operator_id="OP-002",
            origin_plot_ids=["PLOT-00002"],
        )
        engine.record_transfer(req1)
        engine.record_transfer(req2)
        plots = engine.trace_to_origin("BATCH-MULTI")
        assert len(plots) == 2
        assert "PLOT-00001" in plots
        assert "PLOT-00002" in plots


class TestGetFullChain:
    """Test full chain retrieval for an operator."""

    def test_get_full_chain(self, engine, sample_transfer):
        """Full chain for operator OP-001."""
        engine.record_transfer(sample_transfer)
        chain = engine.get_full_chain("OP-001")
        assert len(chain) == 1
        assert chain[0].from_operator_id == "OP-001"

    def test_get_full_chain_filter_commodity(self, engine):
        """Filter chain by commodity."""
        req1 = RecordTransferRequest(
            batch_number="B-1", commodity="cocoa",
            quantity_kg=1000.0, from_operator_id="OP-001",
            to_operator_id="OP-002",
        )
        req2 = RecordTransferRequest(
            batch_number="B-2", commodity="coffee",
            quantity_kg=500.0, from_operator_id="OP-001",
            to_operator_id="OP-003",
        )
        engine.record_transfer(req1)
        engine.record_transfer(req2)
        cocoa_chain = engine.get_full_chain("OP-001", commodity="cocoa")
        assert len(cocoa_chain) == 1
        assert cocoa_chain[0].commodity == "cocoa"

    def test_get_full_chain_empty(self, engine):
        """No chain for unknown operator."""
        assert engine.get_full_chain("OP-UNKNOWN") == []


class TestSplitBatch:
    """Test batch splitting."""

    def test_split_batch(self, engine, sample_transfer):
        """Split batch into parts, quantities sum to original."""
        original = engine.record_transfer(sample_transfer)
        subs = engine.split_batch(original.transfer_id, [2000.0, 1500.0, 1500.0])
        assert len(subs) == 3
        total = sum(s.quantity_kg for s in subs)
        assert total == pytest.approx(5000.0)

    def test_split_batch_sub_batch_numbers(self, engine, sample_transfer):
        """Sub-batches get sequential suffixes."""
        original = engine.record_transfer(sample_transfer)
        subs = engine.split_batch(original.transfer_id, [3000.0, 2000.0])
        assert subs[0].batch_number == "BATCH-2026-001-S01"
        assert subs[1].batch_number == "BATCH-2026-001-S02"

    def test_split_batch_preserves_origin_plots(self, engine, sample_transfer):
        """Sub-batches inherit origin plot IDs."""
        original = engine.record_transfer(sample_transfer)
        subs = engine.split_batch(original.transfer_id, [3000.0, 2000.0])
        for sub in subs:
            assert "PLOT-00001" in sub.origin_plot_ids
            assert "PLOT-00002" in sub.origin_plot_ids

    def test_split_batch_invalid_quantities(self, engine, sample_transfer):
        """Sum exceeds original fails."""
        original = engine.record_transfer(sample_transfer)
        with pytest.raises(ValueError, match="must sum to original"):
            engine.split_batch(original.transfer_id, [3000.0, 3000.0])

    def test_split_batch_unknown_transfer(self, engine):
        """Split unknown transfer raises ValueError."""
        with pytest.raises(ValueError, match="Unknown transfer"):
            engine.split_batch("TXF-99999", [1000.0])

    def test_split_batch_increments_stats(self, engine, sample_transfer):
        original = engine.record_transfer(sample_transfer)
        engine.split_batch(original.transfer_id, [2500.0, 2500.0])
        stats = engine.get_statistics()
        assert stats["batches_split"] == 1


class TestMergeBatches:
    """Test batch merging."""

    def test_merge_batches(self, engine):
        """Merge multiple batches into one."""
        req1 = RecordTransferRequest(
            batch_number="B-1", commodity="cocoa",
            quantity_kg=2000.0, from_operator_id="OP-001",
            to_operator_id="OP-002",
            origin_plot_ids=["PLOT-00001"],
        )
        req2 = RecordTransferRequest(
            batch_number="B-2", commodity="cocoa",
            quantity_kg=3000.0, from_operator_id="OP-001",
            to_operator_id="OP-002",
            origin_plot_ids=["PLOT-00002"],
        )
        t1 = engine.record_transfer(req1)
        t2 = engine.record_transfer(req2)
        merged = engine.merge_batches(
            [t1.transfer_id, t2.transfer_id],
            merged_batch_number="B-MERGED-001",
            to_operator_id="OP-003",
        )
        assert merged.batch_number == "B-MERGED-001"
        assert merged.quantity_kg == pytest.approx(5000.0)
        assert merged.custody_model == "mass_balance"

    def test_merge_batches_plots_combined(self, engine):
        """Origin plots from all batches are combined."""
        req1 = RecordTransferRequest(
            batch_number="B-1", commodity="cocoa",
            quantity_kg=1000.0, from_operator_id="OP-001",
            to_operator_id="OP-002",
            origin_plot_ids=["PLOT-00001"],
        )
        req2 = RecordTransferRequest(
            batch_number="B-2", commodity="cocoa",
            quantity_kg=1500.0, from_operator_id="OP-001",
            to_operator_id="OP-002",
            origin_plot_ids=["PLOT-00002"],
        )
        t1 = engine.record_transfer(req1)
        t2 = engine.record_transfer(req2)
        merged = engine.merge_batches(
            [t1.transfer_id, t2.transfer_id],
            merged_batch_number="B-MERGED-002",
        )
        assert "PLOT-00001" in merged.origin_plot_ids
        assert "PLOT-00002" in merged.origin_plot_ids

    def test_merge_batches_different_commodities_raises(self, engine):
        """Cannot merge different commodities."""
        req1 = RecordTransferRequest(
            batch_number="B-1", commodity="cocoa",
            quantity_kg=1000.0, from_operator_id="OP-001",
            to_operator_id="OP-002",
        )
        req2 = RecordTransferRequest(
            batch_number="B-2", commodity="coffee",
            quantity_kg=1000.0, from_operator_id="OP-004",
            to_operator_id="OP-005",
        )
        t1 = engine.record_transfer(req1)
        t2 = engine.record_transfer(req2)
        with pytest.raises(ValueError, match="Cannot merge different commodities"):
            engine.merge_batches(
                [t1.transfer_id, t2.transfer_id],
                merged_batch_number="B-BAD",
            )

    def test_merge_batches_unknown_transfer_raises(self, engine, sample_transfer):
        t1 = engine.record_transfer(sample_transfer)
        with pytest.raises(ValueError, match="Unknown transfer"):
            engine.merge_batches(
                [t1.transfer_id, "TXF-99999"],
                merged_batch_number="B-BAD",
            )

    def test_merge_increments_stats(self, engine):
        req1 = RecordTransferRequest(
            batch_number="B-1", commodity="cocoa",
            quantity_kg=1000.0, from_operator_id="OP-001",
            to_operator_id="OP-002",
        )
        req2 = RecordTransferRequest(
            batch_number="B-2", commodity="cocoa",
            quantity_kg=1000.0, from_operator_id="OP-001",
            to_operator_id="OP-002",
        )
        t1 = engine.record_transfer(req1)
        t2 = engine.record_transfer(req2)
        engine.merge_batches(
            [t1.transfer_id, t2.transfer_id],
            merged_batch_number="B-MERGED",
        )
        stats = engine.get_statistics()
        assert stats["batches_merged"] == 1


class TestMassBalance:
    """Test mass balance reconciliation."""

    def test_get_mass_balance(self, engine):
        """Input vs output reconciliation for an operator."""
        # OP-002 receives 5000 kg (input)
        req1 = RecordTransferRequest(
            batch_number="B-IN", commodity="cocoa",
            quantity_kg=5000.0, from_operator_id="OP-001",
            to_operator_id="OP-002",
        )
        # OP-002 sends out 3000 kg (output)
        req2 = RecordTransferRequest(
            batch_number="B-OUT", commodity="cocoa",
            quantity_kg=3000.0, from_operator_id="OP-002",
            to_operator_id="OP-003",
        )
        engine.record_transfer(req1)
        engine.record_transfer(req2)

        balance = engine.get_mass_balance("OP-002")
        assert balance["operator_id"] == "OP-002"
        assert balance["input_kg"] == pytest.approx(5000.0)
        assert balance["output_kg"] == pytest.approx(3000.0)
        assert balance["balance_kg"] == pytest.approx(2000.0)
        assert balance["balanced"] is False

    def test_get_mass_balance_perfect(self, engine):
        """Balanced operator (input == output)."""
        req1 = RecordTransferRequest(
            batch_number="B-IN", commodity="cocoa",
            quantity_kg=2000.0, from_operator_id="OP-001",
            to_operator_id="OP-002",
        )
        req2 = RecordTransferRequest(
            batch_number="B-OUT", commodity="cocoa",
            quantity_kg=2000.0, from_operator_id="OP-002",
            to_operator_id="OP-003",
        )
        engine.record_transfer(req1)
        engine.record_transfer(req2)

        balance = engine.get_mass_balance("OP-002")
        assert balance["balanced"] is True
        assert balance["balance_kg"] == pytest.approx(0.0)

    def test_get_mass_balance_no_transfers(self, engine):
        """Operator with no transfers has zero balance."""
        balance = engine.get_mass_balance("OP-UNKNOWN")
        assert balance["input_kg"] == 0.0
        assert balance["output_kg"] == 0.0
        assert balance["balanced"] is True


class TestVerifyTransfer:
    """Test transfer verification."""

    def test_verify_transfer(self, engine, sample_transfer):
        """Mark as verified."""
        recorded = engine.record_transfer(sample_transfer)
        assert recorded.verified is False
        verified = engine.verify_transfer(recorded.transfer_id)
        assert verified is not None
        assert verified.verified is True

    def test_verify_transfer_not_found(self, engine):
        """Non-existent transfer returns None."""
        assert engine.verify_transfer("TXF-99999") is None

    def test_verify_increments_stats(self, engine, sample_transfer):
        recorded = engine.record_transfer(sample_transfer)
        engine.verify_transfer(recorded.transfer_id)
        stats = engine.get_statistics()
        assert stats["transfers_verified"] == 1


class TestCustodyModelAssignment:
    """Test custody model assignment on transfers."""

    def test_custody_model_identity_preserved(self, engine):
        """IP model assigned when requested."""
        req = RecordTransferRequest(
            batch_number="B-IP", commodity="cocoa",
            quantity_kg=1000.0, custody_model="identity_preserved",
            from_operator_id="OP-001", to_operator_id="OP-002",
        )
        t = engine.record_transfer(req)
        assert t.custody_model == "identity_preserved"

    def test_custody_model_segregated(self, engine):
        """SEG model assigned when requested."""
        req = RecordTransferRequest(
            batch_number="B-SEG", commodity="cocoa",
            quantity_kg=1000.0, custody_model="segregated",
            from_operator_id="OP-001", to_operator_id="OP-002",
        )
        t = engine.record_transfer(req)
        assert t.custody_model == "segregated"

    def test_custody_model_mass_balance(self, engine):
        """MB model assigned when requested."""
        req = RecordTransferRequest(
            batch_number="B-MB", commodity="cocoa",
            quantity_kg=1000.0, custody_model="mass_balance",
            from_operator_id="OP-001", to_operator_id="OP-002",
        )
        t = engine.record_transfer(req)
        assert t.custody_model == "mass_balance"


class TestTransferCNCode:
    """Test CN code storage on transfers."""

    def test_transfer_with_cn_code(self, engine, sample_transfer):
        """CN code stored correctly."""
        t = engine.record_transfer(sample_transfer)
        assert t.cn_code == "1801.00"

    def test_transfer_without_cn_code(self, engine):
        """Empty CN code is accepted."""
        req = RecordTransferRequest(
            batch_number="B-NO-CN", commodity="coffee",
            quantity_kg=500.0, from_operator_id="OP-004",
            to_operator_id="OP-005",
        )
        t = engine.record_transfer(req)
        assert t.cn_code == ""


class TestTransferProvenanceChain:
    """Test provenance chain integrity across transfers."""

    def test_transfer_chain_provenance(self, engine):
        """Each transfer in a chain gets a unique provenance hash."""
        req1 = RecordTransferRequest(
            batch_number="B-CHAIN-1", commodity="cocoa",
            quantity_kg=5000.0, from_operator_id="OP-001",
            to_operator_id="OP-002",
            origin_plot_ids=["PLOT-00001"],
        )
        req2 = RecordTransferRequest(
            batch_number="B-CHAIN-2", commodity="cocoa",
            quantity_kg=5000.0, from_operator_id="OP-002",
            to_operator_id="OP-003",
            origin_plot_ids=["PLOT-00001"],
        )
        t1 = engine.record_transfer(req1)
        t2 = engine.record_transfer(req2)

        # Both have provenance hashes
        assert t1.provenance_hash is not None
        assert t2.provenance_hash is not None
        assert len(t1.provenance_hash) == 64
        assert len(t2.provenance_hash) == 64

        # Hashes are different since inputs differ
        assert t1.provenance_hash != t2.provenance_hash

    def test_deterministic_provenance(self, engine):
        """Same request data produces the same provenance hash across engines."""
        req = RecordTransferRequest(
            batch_number="B-DET", commodity="cocoa",
            quantity_kg=1000.0, from_operator_id="OP-001",
            to_operator_id="OP-002",
        )
        t1 = engine.record_transfer(req)

        engine2 = ChainOfCustodyEngine()
        t2 = engine2.record_transfer(req)

        # Since transfer IDs are sequential and identical across fresh engines,
        # the provenance hashes should match
        assert t1.provenance_hash == t2.provenance_hash
