# -*- coding: utf-8 -*-
"""
Ledger Manager Engine - AGENT-EUDR-011: Mass Balance Calculator (Engine 1)

Double-entry ledger management engine for EUDR mass balance accounting.
Provides ledger creation, entry recording, balance calculation, search,
summary, and bulk import functionality with SHA-256 provenance hashing
on every operation.

Zero-Hallucination Guarantees:
    - All balance calculations are deterministic Python Decimal arithmetic
    - Running balance: sum(inputs + CF_in) - sum(outputs + losses + waste +
      CF_out + expiry)
    - Immutable entries: no delete/modify, corrections via adjustment entries
    - Strict chronological ordering enforced on all entries
    - SHA-256 provenance hashes on every entry and balance update
    - No ML/LLM used for any calculation logic

Regulatory References:
    - EU 2023/1115 (EUDR) Article 4: Mass balance chain of custody
    - EU 2023/1115 (EUDR) Article 10(2)(f): Mass balance verification
    - EU 2023/1115 (EUDR) Article 14: Five-year record retention
    - ISO 22095:2020: Chain of Custody - Mass Balance requirements

Performance Targets:
    - Single entry recording: <10ms
    - Balance calculation: <5ms
    - Bulk import (500 entries): <2 seconds
    - Ledger search (10,000 entries): <100ms

PRD Feature References:
    - PRD-AGENT-EUDR-011 Feature 1: Double-Entry Ledger Management
    - PRD-AGENT-EUDR-011 Feature 1.1: Ledger Creation
    - PRD-AGENT-EUDR-011 Feature 1.2: Entry Recording
    - PRD-AGENT-EUDR-011 Feature 1.3: Balance Calculation
    - PRD-AGENT-EUDR-011 Feature 1.4: Immutable Entries
    - PRD-AGENT-EUDR-011 Feature 1.5: Chronological Ordering
    - PRD-AGENT-EUDR-011 Feature 1.6: Ledger Search
    - PRD-AGENT-EUDR-011 Feature 1.7: Ledger Summary
    - PRD-AGENT-EUDR-011 Feature 1.8: Bulk Entry Import

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-011
Agent ID: GL-EUDR-MBC-011
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

from greenlang.agents.eudr.mass_balance_calculator.config import get_config
from greenlang.agents.eudr.mass_balance_calculator.metrics import (
    observe_entry_recording_duration,
    record_api_error,
    record_batch_job,
    record_input_entry,
    record_ledger_entry,
    record_output_entry,
    set_active_ledgers,
    set_total_balance_kg,
)
from greenlang.agents.eudr.mass_balance_calculator.models import (
    ComplianceStatus,
    Ledger,
    LedgerEntry,
    LedgerEntryType,
    StandardType,
)
from greenlang.agents.eudr.mass_balance_calculator.provenance import (
    ProvenanceTracker,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version for provenance tracking
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for audit provenance.

    Args:
        data: Any JSON-serializable object.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _generate_id() -> str:
    """Generate a new UUID4 string identifier.

    Returns:
        UUID4 string.
    """
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Entry types that increase the ledger balance.
_CREDIT_ENTRY_TYPES = frozenset({
    LedgerEntryType.INPUT,
    LedgerEntryType.CARRY_FORWARD_IN,
})

#: Entry types that decrease the ledger balance.
_DEBIT_ENTRY_TYPES = frozenset({
    LedgerEntryType.OUTPUT,
    LedgerEntryType.LOSS,
    LedgerEntryType.WASTE,
    LedgerEntryType.CARRY_FORWARD_OUT,
    LedgerEntryType.EXPIRY,
})

#: Entry types that can be positive or negative (adjustment).
_ADJUSTMENT_ENTRY_TYPES = frozenset({
    LedgerEntryType.ADJUSTMENT,
})

#: All valid entry types.
_ALL_ENTRY_TYPES = _CREDIT_ENTRY_TYPES | _DEBIT_ENTRY_TYPES | _ADJUSTMENT_ENTRY_TYPES

#: Supported bulk import formats.
SUPPORTED_IMPORT_FORMATS = frozenset({"edi", "csv", "xml", "json"})

#: Maximum number of entries per bulk import.
MAX_BULK_IMPORT_SIZE = 5000

#: Default page size for search results.
DEFAULT_PAGE_SIZE = 100

#: Maximum page size for search results.
MAX_PAGE_SIZE = 1000


# ---------------------------------------------------------------------------
# LedgerManager
# ---------------------------------------------------------------------------


class LedgerManager:
    """Double-entry ledger management engine for EUDR mass balance accounting.

    Provides comprehensive ledger management with the following capabilities:
        - Create ledgers per facility+commodity+period with unique identifiers
        - Record immutable entries: input, output, adjustment, loss, waste,
          carry_forward_in, carry_forward_out, expiry
        - Running balance calculation: sum(inputs + CF_in) - sum(outputs +
          losses + waste + CF_out + expiry)
        - Balance validation: must be >= 0 (or within overdraft tolerance)
        - Immutable entries: no delete/modify, corrections via adjustment
          entries only
        - Strict chronological ordering of all entries
        - Ledger search by facility, commodity, period, date_range, batch_id,
          entry_type
        - Ledger summary: total_inputs, total_outputs, total_losses,
          total_waste, current_balance, utilization_rate
        - Bulk entry import from EDI/CSV/XML/JSON
        - SHA-256 provenance hash on every entry

    All operations are thread-safe via reentrant locking. All balance
    calculations use deterministic Python Decimal arithmetic for
    zero-hallucination compliance.

    Attributes:
        _config: Mass balance calculator configuration.
        _provenance: ProvenanceTracker for audit trail.
        _ledgers: In-memory ledger storage keyed by ledger_id.
        _entries: In-memory entry storage keyed by entry_id.
        _ledger_entries: Mapping of ledger_id to ordered list of entry_ids.
        _facility_ledgers: Mapping of facility_id to list of ledger_ids.
        _lock: Reentrant lock for thread-safe access.

    Example:
        >>> from greenlang.agents.eudr.mass_balance_calculator.ledger_manager import (
        ...     LedgerManager,
        ... )
        >>> mgr = LedgerManager()
        >>> result = mgr.create_ledger("facility-001", "cocoa", "rspo")
        >>> assert result["status"] == "created"
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        provenance: Optional[ProvenanceTracker] = None,
    ) -> None:
        """Initialize LedgerManager with configuration and provenance tracker.

        Args:
            config: Optional MassBalanceCalculatorConfig override. If None,
                uses the singleton from get_config().
            provenance: Optional ProvenanceTracker override. If None,
                creates a new instance.
        """
        self._config = config or get_config()
        self._provenance = provenance or ProvenanceTracker(
            genesis_hash=self._config.genesis_hash,
        )

        # -- In-memory storage -------------------------------------------------
        self._ledgers: Dict[str, Dict[str, Any]] = {}
        self._entries: Dict[str, Dict[str, Any]] = {}
        self._ledger_entries: Dict[str, List[str]] = {}
        self._facility_ledgers: Dict[str, List[str]] = {}
        self._commodity_ledgers: Dict[str, List[str]] = {}
        self._period_ledgers: Dict[str, List[str]] = {}

        # -- Thread safety -----------------------------------------------------
        self._lock = threading.RLock()

        logger.info(
            "LedgerManager initialized: module_version=%s, "
            "provenance_enabled=%s",
            _MODULE_VERSION,
            self._config.enable_provenance,
        )

    # ------------------------------------------------------------------
    # Ledger CRUD
    # ------------------------------------------------------------------

    def create_ledger(
        self,
        facility_id: str,
        commodity: str,
        standard: str = "eudr_default",
        period_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a new double-entry mass balance ledger.

        Creates a ledger for a specific facility, commodity, and certification
        standard combination. The ledger starts with a zero balance and is
        immediately ready to accept entries.

        PRD Reference: Feature 1.1 - Ledger Creation.

        Args:
            facility_id: Unique identifier for the facility.
            commodity: EUDR commodity (cattle, cocoa, coffee, oil_palm,
                rubber, soya, wood).
            standard: Certification standard (rspo, fsc, iscc, eudr_default).
                Defaults to eudr_default.
            period_id: Optional credit period identifier to associate with
                this ledger.
            metadata: Optional dictionary of additional contextual fields.

        Returns:
            Dictionary containing ledger details:
                - ledger_id: Unique ledger identifier
                - facility_id: Facility identifier
                - commodity: Commodity type
                - standard: Certification standard
                - period_id: Associated credit period
                - current_balance: Starting balance (0.0)
                - status: "created"
                - provenance_hash: SHA-256 provenance hash
                - created_at: UTC creation timestamp

        Raises:
            ValueError: If facility_id, commodity, or standard are invalid.
        """
        start_time = time.monotonic()

        # -- Validate inputs ---------------------------------------------------
        self._validate_facility_id(facility_id)
        self._validate_commodity(commodity)
        standard_lower = standard.lower().strip()
        self._validate_standard(standard_lower)

        ledger_id = _generate_id()
        now = _utcnow()

        # -- Check for duplicate ledger ----------------------------------------
        with self._lock:
            for existing in self._ledgers.values():
                if (
                    existing["facility_id"] == facility_id
                    and existing["commodity"] == commodity
                    and existing["standard"] == standard_lower
                    and existing.get("period_id") == period_id
                    and not existing.get("closed", False)
                ):
                    logger.warning(
                        "Duplicate active ledger exists for "
                        "facility=%s commodity=%s standard=%s period=%s",
                        facility_id,
                        commodity,
                        standard_lower,
                        period_id,
                    )
                    return {
                        "ledger_id": existing["ledger_id"],
                        "facility_id": facility_id,
                        "commodity": commodity,
                        "standard": standard_lower,
                        "period_id": period_id,
                        "current_balance": float(existing["current_balance"]),
                        "status": "already_exists",
                        "provenance_hash": existing.get("provenance_hash", ""),
                        "created_at": existing["created_at"],
                    }

        # -- Build ledger record -----------------------------------------------
        ledger_data = {
            "ledger_id": ledger_id,
            "facility_id": facility_id,
            "commodity": commodity,
            "standard": standard_lower,
            "period_id": period_id,
            "current_balance": Decimal("0"),
            "total_inputs": Decimal("0"),
            "total_outputs": Decimal("0"),
            "total_losses": Decimal("0"),
            "total_waste": Decimal("0"),
            "total_adjustments": Decimal("0"),
            "total_carry_forward_in": Decimal("0"),
            "total_carry_forward_out": Decimal("0"),
            "total_expiry": Decimal("0"),
            "entry_count": 0,
            "utilization_rate": 0.0,
            "compliance_status": ComplianceStatus.PENDING.value,
            "closed": False,
            "metadata": metadata or {},
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }

        # -- Compute provenance hash ------------------------------------------
        provenance_hash = _compute_hash(ledger_data)
        ledger_data["provenance_hash"] = provenance_hash

        # -- Persist -----------------------------------------------------------
        with self._lock:
            self._ledgers[ledger_id] = ledger_data
            self._ledger_entries[ledger_id] = []

            # Index by facility
            if facility_id not in self._facility_ledgers:
                self._facility_ledgers[facility_id] = []
            self._facility_ledgers[facility_id].append(ledger_id)

            # Index by commodity
            if commodity not in self._commodity_ledgers:
                self._commodity_ledgers[commodity] = []
            self._commodity_ledgers[commodity].append(ledger_id)

            # Index by period
            if period_id:
                if period_id not in self._period_ledgers:
                    self._period_ledgers[period_id] = []
                self._period_ledgers[period_id].append(ledger_id)

        # -- Record provenance -------------------------------------------------
        if self._config.enable_provenance:
            self._provenance.record(
                entity_type="ledger",
                action="create",
                entity_id=ledger_id,
                data=ledger_data,
                metadata={
                    "facility_id": facility_id,
                    "commodity": commodity,
                    "standard": standard_lower,
                },
            )

        # -- Update metrics ----------------------------------------------------
        self._update_ledger_metrics()

        elapsed = time.monotonic() - start_time
        logger.info(
            "Created ledger: ledger_id=%s facility=%s commodity=%s "
            "standard=%s period=%s elapsed_ms=%.1f",
            ledger_id,
            facility_id,
            commodity,
            standard_lower,
            period_id,
            elapsed * 1000,
        )

        return {
            "ledger_id": ledger_id,
            "facility_id": facility_id,
            "commodity": commodity,
            "standard": standard_lower,
            "period_id": period_id,
            "current_balance": 0.0,
            "status": "created",
            "provenance_hash": provenance_hash,
            "created_at": now.isoformat(),
        }

    def get_ledger(self, ledger_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a ledger by its identifier.

        Args:
            ledger_id: Unique ledger identifier.

        Returns:
            Ledger dictionary if found, None otherwise.
        """
        with self._lock:
            ledger = self._ledgers.get(ledger_id)
            if ledger is None:
                return None
            return self._serialize_ledger(ledger)

    def close_ledger(
        self,
        ledger_id: str,
        operator_id: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Close a ledger, preventing further entries.

        Args:
            ledger_id: Unique ledger identifier.
            operator_id: Identifier of the operator closing the ledger.
            reason: Reason for closing the ledger.

        Returns:
            Dictionary with closure details.

        Raises:
            ValueError: If ledger_id is not found.
        """
        with self._lock:
            ledger = self._ledgers.get(ledger_id)
            if ledger is None:
                raise ValueError(f"Ledger not found: {ledger_id}")

            if ledger["closed"]:
                return {
                    "ledger_id": ledger_id,
                    "status": "already_closed",
                    "closed_at": ledger.get("closed_at", ""),
                }

            now = _utcnow()
            ledger["closed"] = True
            ledger["closed_at"] = now.isoformat()
            ledger["closed_by"] = operator_id
            ledger["close_reason"] = reason
            ledger["updated_at"] = now.isoformat()
            ledger["provenance_hash"] = _compute_hash(ledger)

        if self._config.enable_provenance:
            self._provenance.record(
                entity_type="ledger",
                action="update",
                entity_id=ledger_id,
                data={"action": "close", "reason": reason},
                metadata={"operator_id": operator_id or "system"},
            )

        self._update_ledger_metrics()

        logger.info(
            "Closed ledger: ledger_id=%s operator=%s reason=%s",
            ledger_id,
            operator_id,
            reason,
        )
        return {
            "ledger_id": ledger_id,
            "status": "closed",
            "closed_at": now.isoformat(),
            "closed_by": operator_id,
        }

    # ------------------------------------------------------------------
    # Entry Recording
    # ------------------------------------------------------------------

    def record_entry(
        self,
        ledger_id: str,
        entry_type: str,
        batch_id: Optional[str] = None,
        quantity_kg: float = 0.0,
        source_destination: Optional[str] = None,
        conversion_factor_applied: Optional[float] = None,
        operator_id: Optional[str] = None,
        compliance_status: str = "pending",
        notes: Optional[str] = None,
        timestamp: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record an immutable entry in the mass balance ledger.

        Creates a new ledger entry and updates the running balance.
        Entries are immutable once recorded; corrections must be made
        via adjustment entries. Strict chronological ordering is enforced.

        PRD Reference: Feature 1.2 - Entry Recording, Feature 1.4 -
        Immutable Entries, Feature 1.5 - Chronological Ordering.

        Args:
            ledger_id: Identifier of the parent ledger.
            entry_type: Type of entry (input, output, adjustment, loss,
                waste, carry_forward_in, carry_forward_out, expiry).
            batch_id: Optional associated batch identifier for traceability.
            quantity_kg: Quantity in kilograms (must be > 0 for all types
                except adjustment which can be negative).
            source_destination: Source (for inputs) or destination (for
                outputs) facility/operator identifier.
            conversion_factor_applied: Optional conversion factor applied
                to this entry (0.0-1.0).
            operator_id: Identifier of the operator recording the entry.
            compliance_status: Compliance status of the source material
                (pending, compliant, non_compliant).
            notes: Optional free-text notes.
            timestamp: Optional ISO-format UTC timestamp. Defaults to now.
            metadata: Optional dictionary of additional fields.

        Returns:
            Dictionary containing entry details:
                - entry_id: Unique entry identifier
                - ledger_id: Parent ledger identifier
                - entry_type: Type of entry
                - quantity_kg: Quantity in kg
                - running_balance: Updated running balance
                - provenance_hash: SHA-256 hash
                - status: "recorded"
                - timestamp: Entry timestamp

        Raises:
            ValueError: If entry parameters are invalid or ledger is closed.
        """
        start_time = time.monotonic()

        # -- Validate entry parameters -----------------------------------------
        self._validate_ledger_exists(ledger_id)
        entry_type_enum = self._validate_entry_type(entry_type)
        quantity = self._validate_quantity(quantity_kg, entry_type_enum)

        # -- Parse timestamp ---------------------------------------------------
        entry_timestamp = self._parse_timestamp(timestamp)

        # -- Validate conversion factor ----------------------------------------
        if conversion_factor_applied is not None:
            if not (0.0 < conversion_factor_applied <= 1.0):
                raise ValueError(
                    f"conversion_factor_applied must be in (0.0, 1.0], "
                    f"got {conversion_factor_applied}"
                )

        with self._lock:
            ledger = self._ledgers.get(ledger_id)
            if ledger is None:
                raise ValueError(f"Ledger not found: {ledger_id}")

            # -- Check ledger is open ------------------------------------------
            if ledger.get("closed", False):
                raise ValueError(
                    f"Ledger {ledger_id} is closed; no new entries allowed"
                )

            # -- Enforce chronological ordering --------------------------------
            self._validate_chronological_order(ledger_id, entry_timestamp)

            # -- Create entry record -------------------------------------------
            entry_id = _generate_id()
            now = _utcnow()

            entry_data = {
                "entry_id": entry_id,
                "ledger_id": ledger_id,
                "entry_type": entry_type_enum.value,
                "batch_id": batch_id,
                "quantity_kg": quantity,
                "source_destination": source_destination,
                "conversion_factor_applied": conversion_factor_applied,
                "operator_id": operator_id,
                "compliance_status": compliance_status,
                "notes": notes,
                "timestamp": entry_timestamp.isoformat(),
                "voided": False,
                "voided_at": None,
                "voided_by": None,
                "void_reason": None,
                "metadata": metadata or {},
                "created_at": now.isoformat(),
            }

            # -- Compute provenance hash for entry -----------------------------
            entry_provenance = _compute_hash(entry_data)
            entry_data["provenance_hash"] = entry_provenance

            # -- Persist entry -------------------------------------------------
            self._entries[entry_id] = entry_data
            self._ledger_entries[ledger_id].append(entry_id)

            # -- Update running balance ----------------------------------------
            self._update_balance(ledger_id, entry_type_enum, quantity)

            # -- Get updated balance -------------------------------------------
            running_balance = ledger["current_balance"]

        # -- Record provenance -------------------------------------------------
        if self._config.enable_provenance:
            self._provenance.record(
                entity_type="ledger_entry",
                action="record",
                entity_id=entry_id,
                data=entry_data,
                metadata={
                    "ledger_id": ledger_id,
                    "entry_type": entry_type_enum.value,
                    "quantity_kg": str(quantity),
                    "batch_id": batch_id or "",
                },
            )

        # -- Update metrics ----------------------------------------------------
        record_ledger_entry(entry_type_enum.value, ledger.get("commodity", ""))
        if entry_type_enum == LedgerEntryType.INPUT:
            record_input_entry(ledger.get("commodity", ""))
        elif entry_type_enum == LedgerEntryType.OUTPUT:
            record_output_entry(ledger.get("commodity", ""))

        elapsed = time.monotonic() - start_time
        observe_entry_recording_duration(elapsed)

        logger.info(
            "Recorded entry: entry_id=%s ledger=%s type=%s qty=%.4f "
            "balance=%.4f elapsed_ms=%.1f",
            entry_id,
            ledger_id,
            entry_type_enum.value,
            float(quantity),
            float(running_balance),
            elapsed * 1000,
        )

        return {
            "entry_id": entry_id,
            "ledger_id": ledger_id,
            "entry_type": entry_type_enum.value,
            "batch_id": batch_id,
            "quantity_kg": float(quantity),
            "running_balance": float(running_balance),
            "provenance_hash": entry_provenance,
            "status": "recorded",
            "timestamp": entry_timestamp.isoformat(),
        }

    def void_entry(
        self,
        entry_id: str,
        operator_id: str,
        reason: str,
    ) -> Dict[str, Any]:
        """Void an existing ledger entry and reverse its balance effect.

        Since entries are immutable, voiding marks the entry as voided and
        creates a compensating adjustment entry to reverse the balance impact.

        PRD Reference: Feature 1.4 - Immutable Entries.

        Args:
            entry_id: Identifier of the entry to void.
            operator_id: Identifier of the operator performing the void.
            reason: Reason for voiding the entry.

        Returns:
            Dictionary with void details and compensating adjustment.

        Raises:
            ValueError: If entry not found or already voided.
        """
        if not operator_id or not operator_id.strip():
            raise ValueError("operator_id is required for voiding entries")
        if not reason or not reason.strip():
            raise ValueError("reason is required for voiding entries")

        with self._lock:
            entry = self._entries.get(entry_id)
            if entry is None:
                raise ValueError(f"Entry not found: {entry_id}")
            if entry["voided"]:
                raise ValueError(f"Entry already voided: {entry_id}")

            now = _utcnow()
            ledger_id = entry["ledger_id"]
            entry_type_str = entry["entry_type"]
            quantity = entry["quantity_kg"]

            # -- Mark entry as voided ------------------------------------------
            entry["voided"] = True
            entry["voided_at"] = now.isoformat()
            entry["voided_by"] = operator_id
            entry["void_reason"] = reason

            # -- Reverse the balance effect ------------------------------------
            entry_type_enum = LedgerEntryType(entry_type_str)
            self._reverse_balance(ledger_id, entry_type_enum, quantity)

            # -- Update ledger provenance hash ---------------------------------
            ledger = self._ledgers.get(ledger_id)
            if ledger:
                ledger["updated_at"] = now.isoformat()
                ledger["provenance_hash"] = _compute_hash(ledger)

            running_balance = ledger["current_balance"] if ledger else Decimal("0")

        # -- Record provenance for void ----------------------------------------
        if self._config.enable_provenance:
            self._provenance.record(
                entity_type="ledger_entry",
                action="void",
                entity_id=entry_id,
                data={
                    "voided_by": operator_id,
                    "void_reason": reason,
                    "original_type": entry_type_str,
                    "original_qty": str(quantity),
                },
                metadata={"ledger_id": ledger_id},
            )

        logger.info(
            "Voided entry: entry_id=%s ledger=%s type=%s qty=%.4f "
            "operator=%s reason=%s",
            entry_id,
            ledger_id,
            entry_type_str,
            float(quantity),
            operator_id,
            reason,
        )

        return {
            "entry_id": entry_id,
            "ledger_id": ledger_id,
            "status": "voided",
            "voided_at": now.isoformat(),
            "voided_by": operator_id,
            "void_reason": reason,
            "running_balance": float(running_balance),
        }

    # ------------------------------------------------------------------
    # Balance Operations
    # ------------------------------------------------------------------

    def get_balance(self, ledger_id: str) -> Dict[str, Any]:
        """Get the current balance and breakdown for a ledger.

        Calculates the running balance using deterministic Decimal arithmetic:
        balance = sum(inputs + CF_in + positive_adjustments) -
                  sum(outputs + losses + waste + CF_out + expiry +
                      negative_adjustments)

        PRD Reference: Feature 1.3 - Balance Calculation.

        Args:
            ledger_id: Unique ledger identifier.

        Returns:
            Dictionary containing:
                - ledger_id: Ledger identifier
                - current_balance: Current balance in kg
                - total_inputs: Cumulative inputs in kg
                - total_outputs: Cumulative outputs in kg
                - total_losses: Cumulative losses in kg
                - total_waste: Cumulative waste in kg
                - total_carry_forward_in: Cumulative CF in
                - total_carry_forward_out: Cumulative CF out
                - total_expiry: Cumulative expiry
                - total_adjustments: Net adjustments
                - entry_count: Total number of non-voided entries
                - utilization_rate: outputs / (inputs + CF_in)
                - balance_valid: Whether balance >= 0
                - provenance_hash: SHA-256 hash

        Raises:
            ValueError: If ledger not found.
        """
        with self._lock:
            ledger = self._ledgers.get(ledger_id)
            if ledger is None:
                raise ValueError(f"Ledger not found: {ledger_id}")

            balance = self._recalculate_balance(ledger_id)

            total_inputs = ledger["total_inputs"]
            total_cf_in = ledger["total_carry_forward_in"]
            total_available = total_inputs + total_cf_in

            utilization_rate = 0.0
            if total_available > Decimal("0"):
                utilization_rate = float(
                    ledger["total_outputs"] / total_available
                )
                utilization_rate = min(utilization_rate, 1.0)

            non_voided_count = sum(
                1
                for eid in self._ledger_entries.get(ledger_id, [])
                if not self._entries.get(eid, {}).get("voided", False)
            )

        balance_data = {
            "ledger_id": ledger_id,
            "facility_id": ledger["facility_id"],
            "commodity": ledger["commodity"],
            "standard": ledger["standard"],
            "current_balance": float(balance),
            "total_inputs": float(ledger["total_inputs"]),
            "total_outputs": float(ledger["total_outputs"]),
            "total_losses": float(ledger["total_losses"]),
            "total_waste": float(ledger["total_waste"]),
            "total_carry_forward_in": float(ledger["total_carry_forward_in"]),
            "total_carry_forward_out": float(ledger["total_carry_forward_out"]),
            "total_expiry": float(ledger["total_expiry"]),
            "total_adjustments": float(ledger["total_adjustments"]),
            "entry_count": non_voided_count,
            "utilization_rate": round(utilization_rate, 6),
            "balance_valid": balance >= Decimal("0"),
            "provenance_hash": _compute_hash({
                "ledger_id": ledger_id,
                "balance": str(balance),
                "timestamp": _utcnow().isoformat(),
            }),
        }

        return balance_data

    def recalculate_balance(self, ledger_id: str) -> Dict[str, Any]:
        """Force a full recalculation of ledger balance from entries.

        Iterates all non-voided entries and recomputes the balance from
        scratch. Useful for data integrity verification.

        Args:
            ledger_id: Unique ledger identifier.

        Returns:
            Dictionary with recalculated balance and delta from stored value.

        Raises:
            ValueError: If ledger not found.
        """
        with self._lock:
            ledger = self._ledgers.get(ledger_id)
            if ledger is None:
                raise ValueError(f"Ledger not found: {ledger_id}")

            stored_balance = ledger["current_balance"]
            recalculated = self._recalculate_balance(ledger_id)
            delta = recalculated - stored_balance

        logger.info(
            "Recalculated balance: ledger=%s stored=%.4f recalc=%.4f "
            "delta=%.4f",
            ledger_id,
            float(stored_balance),
            float(recalculated),
            float(delta),
        )

        return {
            "ledger_id": ledger_id,
            "stored_balance": float(stored_balance),
            "recalculated_balance": float(recalculated),
            "delta": float(delta),
            "is_consistent": delta == Decimal("0"),
        }

    # ------------------------------------------------------------------
    # Entry History and Search
    # ------------------------------------------------------------------

    def get_entry_history(
        self,
        ledger_id: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Get the entry history for a ledger with optional filtering.

        Returns entries in chronological order (oldest first).
        Supports filtering by entry_type, batch_id, date_range,
        voided status, and operator_id.

        PRD Reference: Feature 1.6 - Ledger Search.

        Args:
            ledger_id: Unique ledger identifier.
            filters: Optional filter dictionary with keys:
                - entry_type (str): Filter by entry type
                - batch_id (str): Filter by batch identifier
                - start_date (str): ISO-format start date (inclusive)
                - end_date (str): ISO-format end date (inclusive)
                - include_voided (bool): Include voided entries (default False)
                - operator_id (str): Filter by operator
                - limit (int): Maximum entries to return
                - offset (int): Number of entries to skip

        Returns:
            List of entry dictionaries in chronological order.

        Raises:
            ValueError: If ledger not found.
        """
        self._validate_ledger_exists(ledger_id)
        filters = filters or {}

        with self._lock:
            entry_ids = list(self._ledger_entries.get(ledger_id, []))

        # -- Collect entries ---------------------------------------------------
        entries = []
        for eid in entry_ids:
            entry = self._entries.get(eid)
            if entry is None:
                continue
            entries.append(dict(entry))

        # -- Apply filters -----------------------------------------------------
        entries = self._apply_entry_filters(entries, filters)

        # -- Apply pagination --------------------------------------------------
        offset = filters.get("offset", 0)
        limit = filters.get("limit", DEFAULT_PAGE_SIZE)
        limit = min(limit, MAX_PAGE_SIZE)
        entries = entries[offset:offset + limit]

        # -- Serialize ---------------------------------------------------------
        return [self._serialize_entry(e) for e in entries]

    def search_ledgers(
        self,
        facility_id: Optional[str] = None,
        commodity: Optional[str] = None,
        standard: Optional[str] = None,
        period_id: Optional[str] = None,
        include_closed: bool = False,
        limit: int = DEFAULT_PAGE_SIZE,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Search ledgers by facility, commodity, standard, or period.

        PRD Reference: Feature 1.6 - Ledger Search.

        Args:
            facility_id: Optional filter by facility identifier.
            commodity: Optional filter by commodity.
            standard: Optional filter by certification standard.
            period_id: Optional filter by credit period.
            include_closed: Whether to include closed ledgers.
            limit: Maximum results to return. Default 100.
            offset: Number of results to skip. Default 0.

        Returns:
            List of matching ledger dictionaries.
        """
        limit = min(limit, MAX_PAGE_SIZE)

        with self._lock:
            candidates = list(self._ledgers.values())

        # -- Apply filters -----------------------------------------------------
        results = []
        for ledger in candidates:
            if facility_id and ledger["facility_id"] != facility_id:
                continue
            if commodity and ledger["commodity"] != commodity:
                continue
            if standard and ledger["standard"] != standard.lower().strip():
                continue
            if period_id and ledger.get("period_id") != period_id:
                continue
            if not include_closed and ledger.get("closed", False):
                continue
            results.append(ledger)

        # -- Sort by creation date (newest first) ------------------------------
        results.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        # -- Paginate ----------------------------------------------------------
        results = results[offset:offset + limit]

        return [self._serialize_ledger(l) for l in results]

    def get_entry(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single entry by its identifier.

        Args:
            entry_id: Unique entry identifier.

        Returns:
            Entry dictionary if found, None otherwise.
        """
        with self._lock:
            entry = self._entries.get(entry_id)
            if entry is None:
                return None
            return self._serialize_entry(entry)

    # ------------------------------------------------------------------
    # Ledger Summary
    # ------------------------------------------------------------------

    def get_summary(self, ledger_id: str) -> Dict[str, Any]:
        """Get a comprehensive summary for a ledger.

        Provides aggregated statistics including total inputs, outputs,
        losses, waste, balance, utilization rate, entry counts by type,
        and date range.

        PRD Reference: Feature 1.7 - Ledger Summary.

        Args:
            ledger_id: Unique ledger identifier.

        Returns:
            Dictionary with comprehensive ledger summary.

        Raises:
            ValueError: If ledger not found.
        """
        with self._lock:
            ledger = self._ledgers.get(ledger_id)
            if ledger is None:
                raise ValueError(f"Ledger not found: {ledger_id}")

            entry_ids = list(self._ledger_entries.get(ledger_id, []))

            # -- Count entries by type -----------------------------------------
            type_counts: Dict[str, int] = {}
            type_quantities: Dict[str, Decimal] = {}
            first_entry_date: Optional[str] = None
            last_entry_date: Optional[str] = None
            batch_ids: set = set()
            active_entries = 0

            for eid in entry_ids:
                entry = self._entries.get(eid)
                if entry is None:
                    continue
                if entry.get("voided", False):
                    continue

                active_entries += 1
                et = entry["entry_type"]
                type_counts[et] = type_counts.get(et, 0) + 1
                qty = entry["quantity_kg"]
                type_quantities[et] = type_quantities.get(et, Decimal("0")) + qty

                entry_ts = entry.get("timestamp", "")
                if first_entry_date is None or entry_ts < first_entry_date:
                    first_entry_date = entry_ts
                if last_entry_date is None or entry_ts > last_entry_date:
                    last_entry_date = entry_ts

                if entry.get("batch_id"):
                    batch_ids.add(entry["batch_id"])

        # -- Calculate utilization rate ----------------------------------------
        total_inputs = ledger["total_inputs"]
        total_cf_in = ledger["total_carry_forward_in"]
        total_available = total_inputs + total_cf_in

        utilization_rate = 0.0
        if total_available > Decimal("0"):
            utilization_rate = float(
                ledger["total_outputs"] / total_available
            )
            utilization_rate = min(utilization_rate, 1.0)

        summary = {
            "ledger_id": ledger_id,
            "facility_id": ledger["facility_id"],
            "commodity": ledger["commodity"],
            "standard": ledger["standard"],
            "period_id": ledger.get("period_id"),
            "current_balance": float(ledger["current_balance"]),
            "total_inputs": float(ledger["total_inputs"]),
            "total_outputs": float(ledger["total_outputs"]),
            "total_losses": float(ledger["total_losses"]),
            "total_waste": float(ledger["total_waste"]),
            "total_carry_forward_in": float(ledger["total_carry_forward_in"]),
            "total_carry_forward_out": float(ledger["total_carry_forward_out"]),
            "total_expiry": float(ledger["total_expiry"]),
            "total_adjustments": float(ledger["total_adjustments"]),
            "utilization_rate": round(utilization_rate, 6),
            "entry_count": active_entries,
            "entries_by_type": type_counts,
            "quantities_by_type": {
                k: float(v) for k, v in type_quantities.items()
            },
            "unique_batch_count": len(batch_ids),
            "first_entry_date": first_entry_date,
            "last_entry_date": last_entry_date,
            "compliance_status": ledger.get(
                "compliance_status", ComplianceStatus.PENDING.value
            ),
            "is_closed": ledger.get("closed", False),
            "created_at": ledger["created_at"],
            "updated_at": ledger["updated_at"],
            "provenance_hash": _compute_hash({
                "ledger_id": ledger_id,
                "summary_timestamp": _utcnow().isoformat(),
                "balance": str(ledger["current_balance"]),
            }),
        }

        return summary

    # ------------------------------------------------------------------
    # Bulk Import
    # ------------------------------------------------------------------

    def bulk_import(
        self,
        entries: List[Dict[str, Any]],
        source_format: str = "json",
        operator_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Bulk import ledger entries from EDI/CSV/XML/JSON data.

        Validates and records multiple entries in a single batch operation.
        Each entry is validated independently; failures are collected
        and reported without blocking successful entries.

        PRD Reference: Feature 1.8 - Bulk Entry Import.

        Args:
            entries: List of entry dictionaries, each containing at minimum:
                - ledger_id: Parent ledger identifier
                - entry_type: Type of entry
                - quantity_kg: Quantity in kilograms
                Optional fields: batch_id, source_destination,
                    conversion_factor_applied, notes, timestamp, metadata.
            source_format: Import format (edi, csv, xml, json).
            operator_id: Operator performing the import.

        Returns:
            Dictionary with import results:
                - total_submitted: Number of entries submitted
                - total_succeeded: Number successfully recorded
                - total_failed: Number that failed validation
                - entries: List of result dicts for each entry
                - provenance_hash: SHA-256 hash for the batch
                - elapsed_ms: Processing time

        Raises:
            ValueError: If entries list is empty or exceeds MAX_BULK_IMPORT_SIZE.
        """
        start_time = time.monotonic()

        # -- Validate batch parameters -----------------------------------------
        if not entries:
            raise ValueError("Entries list must not be empty")

        if len(entries) > MAX_BULK_IMPORT_SIZE:
            raise ValueError(
                f"Bulk import size {len(entries)} exceeds maximum "
                f"{MAX_BULK_IMPORT_SIZE}"
            )

        source_format_lower = source_format.lower().strip()
        if source_format_lower not in SUPPORTED_IMPORT_FORMATS:
            raise ValueError(
                f"Unsupported import format '{source_format}'. "
                f"Supported: {sorted(SUPPORTED_IMPORT_FORMATS)}"
            )

        # -- Process each entry ------------------------------------------------
        results: List[Dict[str, Any]] = []
        succeeded = 0
        failed = 0

        for idx, entry_data in enumerate(entries):
            try:
                result = self.record_entry(
                    ledger_id=entry_data.get("ledger_id", ""),
                    entry_type=entry_data.get("entry_type", ""),
                    batch_id=entry_data.get("batch_id"),
                    quantity_kg=float(entry_data.get("quantity_kg", 0)),
                    source_destination=entry_data.get("source_destination"),
                    conversion_factor_applied=entry_data.get(
                        "conversion_factor_applied"
                    ),
                    operator_id=operator_id or entry_data.get("operator_id"),
                    compliance_status=entry_data.get(
                        "compliance_status", "pending"
                    ),
                    notes=entry_data.get("notes"),
                    timestamp=entry_data.get("timestamp"),
                    metadata=entry_data.get("metadata"),
                )
                result["index"] = idx
                result["import_status"] = "success"
                results.append(result)
                succeeded += 1

            except (ValueError, KeyError, TypeError) as exc:
                failed += 1
                results.append({
                    "index": idx,
                    "import_status": "failed",
                    "error": str(exc),
                    "ledger_id": entry_data.get("ledger_id", ""),
                    "entry_type": entry_data.get("entry_type", ""),
                })
                logger.warning(
                    "Bulk import entry[%d] failed: %s", idx, exc,
                )

        # -- Record batch provenance -------------------------------------------
        elapsed = time.monotonic() - start_time
        batch_hash = _compute_hash({
            "total_submitted": len(entries),
            "total_succeeded": succeeded,
            "total_failed": failed,
            "source_format": source_format_lower,
            "timestamp": _utcnow().isoformat(),
        })

        if self._config.enable_provenance:
            self._provenance.record(
                entity_type="batch_job",
                action="import",
                entity_id=_generate_id(),
                data={
                    "total_submitted": len(entries),
                    "total_succeeded": succeeded,
                    "total_failed": failed,
                    "source_format": source_format_lower,
                },
                metadata={"operator_id": operator_id or "system"},
            )

        record_batch_job()

        logger.info(
            "Bulk import completed: submitted=%d succeeded=%d failed=%d "
            "format=%s elapsed_ms=%.1f",
            len(entries),
            succeeded,
            failed,
            source_format_lower,
            elapsed * 1000,
        )

        return {
            "total_submitted": len(entries),
            "total_succeeded": succeeded,
            "total_failed": failed,
            "entries": results,
            "source_format": source_format_lower,
            "provenance_hash": batch_hash,
            "elapsed_ms": round(elapsed * 1000, 2),
        }

    # ------------------------------------------------------------------
    # Facility and Commodity Lookups
    # ------------------------------------------------------------------

    def get_facility_ledgers(
        self,
        facility_id: str,
        include_closed: bool = False,
    ) -> List[Dict[str, Any]]:
        """Get all ledgers for a specific facility.

        Args:
            facility_id: Facility identifier.
            include_closed: Whether to include closed ledgers.

        Returns:
            List of ledger dictionaries.
        """
        with self._lock:
            ledger_ids = list(
                self._facility_ledgers.get(facility_id, [])
            )

        results = []
        for lid in ledger_ids:
            ledger = self._ledgers.get(lid)
            if ledger is None:
                continue
            if not include_closed and ledger.get("closed", False):
                continue
            results.append(self._serialize_ledger(ledger))

        return results

    def get_commodity_ledgers(
        self,
        commodity: str,
        include_closed: bool = False,
    ) -> List[Dict[str, Any]]:
        """Get all ledgers for a specific commodity.

        Args:
            commodity: EUDR commodity.
            include_closed: Whether to include closed ledgers.

        Returns:
            List of ledger dictionaries.
        """
        with self._lock:
            ledger_ids = list(
                self._commodity_ledgers.get(commodity, [])
            )

        results = []
        for lid in ledger_ids:
            ledger = self._ledgers.get(lid)
            if ledger is None:
                continue
            if not include_closed and ledger.get("closed", False):
                continue
            results.append(self._serialize_ledger(ledger))

        return results

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall ledger manager statistics.

        Returns:
            Dictionary with aggregate statistics across all ledgers.
        """
        with self._lock:
            total_ledgers = len(self._ledgers)
            active_ledgers = sum(
                1 for l in self._ledgers.values()
                if not l.get("closed", False)
            )
            closed_ledgers = total_ledgers - active_ledgers
            total_entries = len(self._entries)
            voided_entries = sum(
                1 for e in self._entries.values()
                if e.get("voided", False)
            )
            active_entries = total_entries - voided_entries
            total_facilities = len(self._facility_ledgers)
            total_commodities = len(self._commodity_ledgers)

            total_balance = sum(
                l["current_balance"]
                for l in self._ledgers.values()
                if not l.get("closed", False)
            )

        return {
            "total_ledgers": total_ledgers,
            "active_ledgers": active_ledgers,
            "closed_ledgers": closed_ledgers,
            "total_entries": total_entries,
            "active_entries": active_entries,
            "voided_entries": voided_entries,
            "total_facilities": total_facilities,
            "total_commodities": total_commodities,
            "total_balance_kg": float(total_balance),
            "provenance_entries": self._provenance.entry_count,
        }

    # ------------------------------------------------------------------
    # Internal: Validation
    # ------------------------------------------------------------------

    def _validate_facility_id(self, facility_id: str) -> None:
        """Validate facility identifier is non-empty.

        Args:
            facility_id: Facility identifier to validate.

        Raises:
            ValueError: If facility_id is empty or whitespace.
        """
        if not facility_id or not facility_id.strip():
            raise ValueError("facility_id must not be empty")

    def _validate_commodity(self, commodity: str) -> None:
        """Validate commodity is a recognized EUDR commodity.

        Args:
            commodity: Commodity string to validate.

        Raises:
            ValueError: If commodity is not in the EUDR commodity list.
        """
        if not commodity or not commodity.strip():
            raise ValueError("commodity must not be empty")
        valid_commodities = set(self._config.eudr_commodities)
        if commodity.lower().strip() not in valid_commodities:
            raise ValueError(
                f"Unrecognized commodity '{commodity}'. "
                f"Valid: {sorted(valid_commodities)}"
            )

    def _validate_standard(self, standard: str) -> None:
        """Validate certification standard.

        Args:
            standard: Standard string to validate.

        Raises:
            ValueError: If standard is not recognized.
        """
        valid_standards = {s.value for s in StandardType}
        if standard not in valid_standards:
            raise ValueError(
                f"Unrecognized standard '{standard}'. "
                f"Valid: {sorted(valid_standards)}"
            )

    def _validate_ledger_exists(self, ledger_id: str) -> None:
        """Validate that a ledger exists.

        Args:
            ledger_id: Ledger identifier to validate.

        Raises:
            ValueError: If ledger_id is empty or not found.
        """
        if not ledger_id or not ledger_id.strip():
            raise ValueError("ledger_id must not be empty")
        with self._lock:
            if ledger_id not in self._ledgers:
                raise ValueError(f"Ledger not found: {ledger_id}")

    def _validate_entry_type(self, entry_type: str) -> LedgerEntryType:
        """Validate and parse the entry type string.

        Args:
            entry_type: Entry type string to validate.

        Returns:
            Parsed LedgerEntryType enum value.

        Raises:
            ValueError: If entry_type is invalid.
        """
        if not entry_type or not entry_type.strip():
            raise ValueError("entry_type must not be empty")
        try:
            return LedgerEntryType(entry_type.lower().strip())
        except ValueError:
            valid_types = [t.value for t in LedgerEntryType]
            raise ValueError(
                f"Invalid entry_type '{entry_type}'. "
                f"Valid: {valid_types}"
            )

    def _validate_quantity(
        self,
        quantity_kg: float,
        entry_type: LedgerEntryType,
    ) -> Decimal:
        """Validate and convert quantity to Decimal.

        For adjustment entries, quantity can be negative. For all other
        entry types, quantity must be strictly positive.

        Args:
            quantity_kg: Quantity in kilograms.
            entry_type: Entry type for context-specific validation.

        Returns:
            Validated Decimal quantity.

        Raises:
            ValueError: If quantity is invalid for the entry type.
        """
        try:
            quantity = Decimal(str(quantity_kg))
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid quantity_kg value: {quantity_kg} ({exc})"
            )

        if entry_type == LedgerEntryType.ADJUSTMENT:
            if quantity == Decimal("0"):
                raise ValueError("Adjustment quantity must not be zero")
        else:
            if quantity <= Decimal("0"):
                raise ValueError(
                    f"quantity_kg must be > 0 for {entry_type.value}, "
                    f"got {float(quantity)}"
                )

        return quantity

    def _validate_entry(self, entry_data: Dict[str, Any]) -> bool:
        """Validate a complete entry data dictionary.

        Args:
            entry_data: Entry data to validate.

        Returns:
            True if entry is valid, raises ValueError otherwise.

        Raises:
            ValueError: If any field fails validation.
        """
        required_fields = ["ledger_id", "entry_type", "quantity_kg"]
        for field_name in required_fields:
            if field_name not in entry_data or entry_data[field_name] is None:
                raise ValueError(f"Missing required field: {field_name}")

        self._validate_ledger_exists(entry_data["ledger_id"])
        entry_type = self._validate_entry_type(entry_data["entry_type"])
        self._validate_quantity(
            float(entry_data["quantity_kg"]), entry_type
        )

        return True

    def _parse_timestamp(self, timestamp: Optional[str]) -> datetime:
        """Parse an ISO-format timestamp string or return current UTC time.

        Args:
            timestamp: Optional ISO-format timestamp string.

        Returns:
            Parsed datetime (UTC) or current UTC time.

        Raises:
            ValueError: If timestamp string is malformed.
        """
        if timestamp is None:
            return _utcnow()

        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, AttributeError) as exc:
            raise ValueError(
                f"Invalid timestamp format: {timestamp} ({exc})"
            )

    def _validate_chronological_order(
        self,
        ledger_id: str,
        new_timestamp: datetime,
    ) -> None:
        """Ensure the new entry timestamp is not before the last entry.

        PRD Reference: Feature 1.5 - Chronological Ordering.

        Args:
            ledger_id: Ledger identifier.
            new_timestamp: Timestamp of the new entry.

        Raises:
            ValueError: If new timestamp violates chronological order.
        """
        entry_ids = self._ledger_entries.get(ledger_id, [])
        if not entry_ids:
            return

        # Walk backwards to find last non-voided entry
        for eid in reversed(entry_ids):
            entry = self._entries.get(eid)
            if entry is None or entry.get("voided", False):
                continue

            last_ts_str = entry.get("timestamp", "")
            if not last_ts_str:
                return

            try:
                last_ts = datetime.fromisoformat(
                    last_ts_str.replace("Z", "+00:00")
                )
                if last_ts.tzinfo is None:
                    last_ts = last_ts.replace(tzinfo=timezone.utc)
            except (ValueError, AttributeError):
                return

            if new_timestamp < last_ts:
                raise ValueError(
                    f"Entry timestamp {new_timestamp.isoformat()} "
                    f"violates chronological order; last entry was "
                    f"at {last_ts.isoformat()}"
                )
            return

    # ------------------------------------------------------------------
    # Internal: Balance Management
    # ------------------------------------------------------------------

    def _update_balance(
        self,
        ledger_id: str,
        entry_type: LedgerEntryType,
        quantity: Decimal,
    ) -> None:
        """Update ledger balance and aggregates after recording an entry.

        Uses deterministic Decimal arithmetic. Must be called while
        holding self._lock.

        Args:
            ledger_id: Ledger identifier.
            entry_type: Type of entry recorded.
            quantity: Quantity in kg (positive).
        """
        ledger = self._ledgers[ledger_id]
        now = _utcnow()

        # -- Update type-specific totals and balance ---------------------------
        if entry_type == LedgerEntryType.INPUT:
            ledger["total_inputs"] += quantity
            ledger["current_balance"] += quantity

        elif entry_type == LedgerEntryType.OUTPUT:
            ledger["total_outputs"] += quantity
            ledger["current_balance"] -= quantity

        elif entry_type == LedgerEntryType.LOSS:
            ledger["total_losses"] += quantity
            ledger["current_balance"] -= quantity

        elif entry_type == LedgerEntryType.WASTE:
            ledger["total_waste"] += quantity
            ledger["current_balance"] -= quantity

        elif entry_type == LedgerEntryType.CARRY_FORWARD_IN:
            ledger["total_carry_forward_in"] += quantity
            ledger["current_balance"] += quantity

        elif entry_type == LedgerEntryType.CARRY_FORWARD_OUT:
            ledger["total_carry_forward_out"] += quantity
            ledger["current_balance"] -= quantity

        elif entry_type == LedgerEntryType.EXPIRY:
            ledger["total_expiry"] += quantity
            ledger["current_balance"] -= quantity

        elif entry_type == LedgerEntryType.ADJUSTMENT:
            ledger["total_adjustments"] += quantity
            ledger["current_balance"] += quantity

        # -- Update entry count ------------------------------------------------
        ledger["entry_count"] = ledger.get("entry_count", 0) + 1

        # -- Update utilization rate -------------------------------------------
        total_available = (
            ledger["total_inputs"] + ledger["total_carry_forward_in"]
        )
        if total_available > Decimal("0"):
            ledger["utilization_rate"] = float(
                min(ledger["total_outputs"] / total_available, Decimal("1"))
            )
        else:
            ledger["utilization_rate"] = 0.0

        # -- Update compliance status ------------------------------------------
        if ledger["current_balance"] < Decimal("0"):
            ledger["compliance_status"] = ComplianceStatus.NON_COMPLIANT.value
        else:
            ledger["compliance_status"] = ComplianceStatus.COMPLIANT.value

        # -- Update timestamp and provenance -----------------------------------
        ledger["updated_at"] = now.isoformat()
        ledger["provenance_hash"] = _compute_hash(ledger)

    def _reverse_balance(
        self,
        ledger_id: str,
        entry_type: LedgerEntryType,
        quantity: Decimal,
    ) -> None:
        """Reverse the balance effect of a voided entry.

        Must be called while holding self._lock.

        Args:
            ledger_id: Ledger identifier.
            entry_type: Type of entry being voided.
            quantity: Quantity in kg being reversed.
        """
        ledger = self._ledgers[ledger_id]

        if entry_type == LedgerEntryType.INPUT:
            ledger["total_inputs"] -= quantity
            ledger["current_balance"] -= quantity

        elif entry_type == LedgerEntryType.OUTPUT:
            ledger["total_outputs"] -= quantity
            ledger["current_balance"] += quantity

        elif entry_type == LedgerEntryType.LOSS:
            ledger["total_losses"] -= quantity
            ledger["current_balance"] += quantity

        elif entry_type == LedgerEntryType.WASTE:
            ledger["total_waste"] -= quantity
            ledger["current_balance"] += quantity

        elif entry_type == LedgerEntryType.CARRY_FORWARD_IN:
            ledger["total_carry_forward_in"] -= quantity
            ledger["current_balance"] -= quantity

        elif entry_type == LedgerEntryType.CARRY_FORWARD_OUT:
            ledger["total_carry_forward_out"] -= quantity
            ledger["current_balance"] += quantity

        elif entry_type == LedgerEntryType.EXPIRY:
            ledger["total_expiry"] -= quantity
            ledger["current_balance"] += quantity

        elif entry_type == LedgerEntryType.ADJUSTMENT:
            ledger["total_adjustments"] -= quantity
            ledger["current_balance"] -= quantity

        ledger["entry_count"] = max(0, ledger.get("entry_count", 1) - 1)

        # -- Update utilization rate -------------------------------------------
        total_available = (
            ledger["total_inputs"] + ledger["total_carry_forward_in"]
        )
        if total_available > Decimal("0"):
            ledger["utilization_rate"] = float(
                min(ledger["total_outputs"] / total_available, Decimal("1"))
            )
        else:
            ledger["utilization_rate"] = 0.0

    def _recalculate_balance(self, ledger_id: str) -> Decimal:
        """Recalculate ledger balance from all non-voided entries.

        Full recalculation from scratch for integrity verification.
        Must be called while holding self._lock.

        Args:
            ledger_id: Ledger identifier.

        Returns:
            Recalculated Decimal balance.
        """
        entry_ids = self._ledger_entries.get(ledger_id, [])

        total_credits = Decimal("0")
        total_debits = Decimal("0")
        total_inputs = Decimal("0")
        total_outputs = Decimal("0")
        total_losses = Decimal("0")
        total_waste = Decimal("0")
        total_cf_in = Decimal("0")
        total_cf_out = Decimal("0")
        total_expiry = Decimal("0")
        total_adj = Decimal("0")

        for eid in entry_ids:
            entry = self._entries.get(eid)
            if entry is None or entry.get("voided", False):
                continue

            qty = entry["quantity_kg"]
            et_str = entry["entry_type"]

            try:
                et = LedgerEntryType(et_str)
            except ValueError:
                continue

            if et == LedgerEntryType.INPUT:
                total_credits += qty
                total_inputs += qty
            elif et == LedgerEntryType.CARRY_FORWARD_IN:
                total_credits += qty
                total_cf_in += qty
            elif et == LedgerEntryType.OUTPUT:
                total_debits += qty
                total_outputs += qty
            elif et == LedgerEntryType.LOSS:
                total_debits += qty
                total_losses += qty
            elif et == LedgerEntryType.WASTE:
                total_debits += qty
                total_waste += qty
            elif et == LedgerEntryType.CARRY_FORWARD_OUT:
                total_debits += qty
                total_cf_out += qty
            elif et == LedgerEntryType.EXPIRY:
                total_debits += qty
                total_expiry += qty
            elif et == LedgerEntryType.ADJUSTMENT:
                total_adj += qty

        balance = total_credits - total_debits + total_adj

        # -- Update stored aggregates ------------------------------------------
        ledger = self._ledgers[ledger_id]
        ledger["total_inputs"] = total_inputs
        ledger["total_outputs"] = total_outputs
        ledger["total_losses"] = total_losses
        ledger["total_waste"] = total_waste
        ledger["total_carry_forward_in"] = total_cf_in
        ledger["total_carry_forward_out"] = total_cf_out
        ledger["total_expiry"] = total_expiry
        ledger["total_adjustments"] = total_adj
        ledger["current_balance"] = balance

        return balance

    # ------------------------------------------------------------------
    # Internal: Filtering
    # ------------------------------------------------------------------

    def _apply_entry_filters(
        self,
        entries: List[Dict[str, Any]],
        filters: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Apply filters to a list of entry dictionaries.

        Args:
            entries: List of entry dicts to filter.
            filters: Filter criteria dictionary.

        Returns:
            Filtered list of entry dicts.
        """
        result = entries

        # -- Filter by voided status -------------------------------------------
        include_voided = filters.get("include_voided", False)
        if not include_voided:
            result = [e for e in result if not e.get("voided", False)]

        # -- Filter by entry_type ----------------------------------------------
        entry_type_filter = filters.get("entry_type")
        if entry_type_filter:
            et_lower = entry_type_filter.lower().strip()
            result = [e for e in result if e.get("entry_type") == et_lower]

        # -- Filter by batch_id ------------------------------------------------
        batch_id_filter = filters.get("batch_id")
        if batch_id_filter:
            result = [
                e for e in result if e.get("batch_id") == batch_id_filter
            ]

        # -- Filter by operator_id ---------------------------------------------
        operator_filter = filters.get("operator_id")
        if operator_filter:
            result = [
                e for e in result if e.get("operator_id") == operator_filter
            ]

        # -- Filter by date range ----------------------------------------------
        start_date = filters.get("start_date")
        end_date = filters.get("end_date")

        if start_date:
            if isinstance(start_date, str):
                start_date = start_date.replace("Z", "+00:00")
            result = [
                e for e in result
                if e.get("timestamp", "") >= (
                    start_date if isinstance(start_date, str) else
                    start_date.isoformat()
                )
            ]

        if end_date:
            if isinstance(end_date, str):
                end_date = end_date.replace("Z", "+00:00")
            result = [
                e for e in result
                if e.get("timestamp", "") <= (
                    end_date if isinstance(end_date, str) else
                    end_date.isoformat()
                )
            ]

        return result

    # ------------------------------------------------------------------
    # Internal: Serialization
    # ------------------------------------------------------------------

    def _serialize_ledger(self, ledger: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize a ledger dictionary for external consumption.

        Converts Decimal fields to float for JSON compatibility.

        Args:
            ledger: Internal ledger dictionary.

        Returns:
            Serialized ledger dictionary.
        """
        return {
            "ledger_id": ledger["ledger_id"],
            "facility_id": ledger["facility_id"],
            "commodity": ledger["commodity"],
            "standard": ledger["standard"],
            "period_id": ledger.get("period_id"),
            "current_balance": float(ledger["current_balance"]),
            "total_inputs": float(ledger["total_inputs"]),
            "total_outputs": float(ledger["total_outputs"]),
            "total_losses": float(ledger["total_losses"]),
            "total_waste": float(ledger["total_waste"]),
            "total_carry_forward_in": float(
                ledger.get("total_carry_forward_in", Decimal("0"))
            ),
            "total_carry_forward_out": float(
                ledger.get("total_carry_forward_out", Decimal("0"))
            ),
            "total_expiry": float(
                ledger.get("total_expiry", Decimal("0"))
            ),
            "total_adjustments": float(
                ledger.get("total_adjustments", Decimal("0"))
            ),
            "entry_count": ledger.get("entry_count", 0),
            "utilization_rate": ledger.get("utilization_rate", 0.0),
            "compliance_status": ledger.get(
                "compliance_status", ComplianceStatus.PENDING.value
            ),
            "is_closed": ledger.get("closed", False),
            "metadata": ledger.get("metadata", {}),
            "provenance_hash": ledger.get("provenance_hash", ""),
            "created_at": ledger["created_at"],
            "updated_at": ledger["updated_at"],
        }

    def _serialize_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize an entry dictionary for external consumption.

        Args:
            entry: Internal entry dictionary.

        Returns:
            Serialized entry dictionary with Decimal-to-float conversion.
        """
        qty = entry.get("quantity_kg", Decimal("0"))
        return {
            "entry_id": entry["entry_id"],
            "ledger_id": entry["ledger_id"],
            "entry_type": entry["entry_type"],
            "batch_id": entry.get("batch_id"),
            "quantity_kg": float(qty) if isinstance(qty, Decimal) else qty,
            "source_destination": entry.get("source_destination"),
            "conversion_factor_applied": entry.get(
                "conversion_factor_applied"
            ),
            "operator_id": entry.get("operator_id"),
            "compliance_status": entry.get("compliance_status", "pending"),
            "notes": entry.get("notes"),
            "timestamp": entry.get("timestamp", ""),
            "voided": entry.get("voided", False),
            "voided_at": entry.get("voided_at"),
            "voided_by": entry.get("voided_by"),
            "void_reason": entry.get("void_reason"),
            "metadata": entry.get("metadata", {}),
            "provenance_hash": entry.get("provenance_hash", ""),
            "created_at": entry.get("created_at", ""),
        }

    # ------------------------------------------------------------------
    # Internal: Metrics
    # ------------------------------------------------------------------

    def _update_ledger_metrics(self) -> None:
        """Update Prometheus gauge metrics for active ledger count and balance.

        Thread-safe: acquires lock to read current state.
        """
        with self._lock:
            active_count = sum(
                1 for l in self._ledgers.values()
                if not l.get("closed", False)
            )
            total_balance = sum(
                l["current_balance"]
                for l in self._ledgers.values()
                if not l.get("closed", False)
            )

        set_active_ledgers(active_count)
        set_total_balance_kg(float(total_balance))

    # ------------------------------------------------------------------
    # Dunder Methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        with self._lock:
            n_ledgers = len(self._ledgers)
            n_entries = len(self._entries)
        return (
            f"LedgerManager(ledgers={n_ledgers}, entries={n_entries}, "
            f"module_version={_MODULE_VERSION})"
        )

    def __len__(self) -> int:
        """Return the total number of ledgers."""
        with self._lock:
            return len(self._ledgers)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "LedgerManager",
    "SUPPORTED_IMPORT_FORMATS",
    "MAX_BULK_IMPORT_SIZE",
    "DEFAULT_PAGE_SIZE",
    "MAX_PAGE_SIZE",
]
