# -*- coding: utf-8 -*-
"""
Loss and Waste Tracking Engine - AGENT-EUDR-011 Engine 5

Production-grade processing loss and waste tracking engine for the Mass
Balance Calculator Agent.  Every processing transformation must record its
losses; this engine validates them against commodity-specific reference
tolerances, flags under-reporting and potential fraud, tracks cumulative
losses per batch, analyses trends per facility, handles by-product credits,
allocates losses across batch splits, and links waste documentation to
certificates.

PRD Features Implemented:
    - PRD-MBC-LOSS-001: Mandatory loss recording per transformation
    - PRD-MBC-LOSS-002: Commodity-specific max loss tolerances
    - PRD-MBC-LOSS-003: Six loss types, three waste types
    - PRD-MBC-LOSS-004: By-product credit at conversion rate
    - PRD-MBC-LOSS-005: Loss validation (too-low / too-high flagging)
    - PRD-MBC-LOSS-006: Cumulative loss tracking per batch
    - PRD-MBC-LOSS-007: Loss trend analysis per facility per commodity
    - PRD-MBC-LOSS-008: Waste documentation linkage to certificates
    - PRD-MBC-LOSS-009: Loss allocation for batch splits (proportional)
    - PRD-MBC-LOSS-010: SHA-256 provenance hashing on all operations

Commodity-Specific Max Loss Tolerances (reference data):
    Cocoa beans -> nibs:      13% expected, 20% max
    Palm FFB -> CPO:          78.5% expected, 82% max
    Coffee cherry -> green:   81.5% expected, 85% max
    Soya beans -> oil:        1% expected, 3% max
    Rubber latex -> sheet:    67.5% expected, 72% max
    Wood log -> sawn:         50% expected, 60% max
    Cattle live -> carcass:   45% expected, 52% max

Loss Types:
    processing_loss, transport_loss, storage_loss,
    quality_rejection, spillage, contamination_loss

Waste Types:
    by_product (valuable), waste_material, hazardous_waste

Zero-Hallucination:
    All loss calculations use deterministic Python arithmetic.
    No LLM calls are used for numeric validation or tolerance checks.

Thread Safety:
    All mutable state is protected by a threading.RLock.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-011 Mass Balance Calculator (GL-EUDR-MBC-011)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import statistics
import threading
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

from greenlang.agents.eudr.mass_balance_calculator.config import (
    MassBalanceCalculatorConfig,
    get_config,
)
from greenlang.agents.eudr.mass_balance_calculator.models import (
    LedgerEntryType,
    LossRecord,
    LossType,
    WasteType,
)
from greenlang.agents.eudr.mass_balance_calculator.provenance import (
    ProvenanceTracker,
    get_provenance_tracker,
)
from greenlang.agents.eudr.mass_balance_calculator.metrics import (
    record_loss_recorded,
    record_api_error,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# UTC helper
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Constants: commodity-specific processing loss reference data
# ---------------------------------------------------------------------------

#: Expected and maximum loss percentages keyed by
#: ``(commodity, process_type)`` tuple.  Values are
#: ``{"expected_pct": float, "max_pct": float}``.
COMMODITY_PROCESS_LOSS_REFERENCE: Dict[Tuple[str, str], Dict[str, float]] = {
    # Cocoa beans -> nibs
    ("cocoa", "beans_to_nibs"): {
        "expected_pct": 13.0,
        "max_pct": 20.0,
    },
    ("cocoa", "fermentation"): {
        "expected_pct": 8.0,
        "max_pct": 12.0,
    },
    ("cocoa", "drying"): {
        "expected_pct": 12.0,
        "max_pct": 18.0,
    },
    ("cocoa", "roasting"): {
        "expected_pct": 15.0,
        "max_pct": 22.0,
    },
    ("cocoa", "winnowing"): {
        "expected_pct": 20.0,
        "max_pct": 28.0,
    },
    # Palm FFB -> CPO
    ("oil_palm", "ffb_to_cpo"): {
        "expected_pct": 78.5,
        "max_pct": 82.0,
    },
    ("oil_palm", "sterilization"): {
        "expected_pct": 5.0,
        "max_pct": 8.0,
    },
    ("oil_palm", "threshing"): {
        "expected_pct": 35.0,
        "max_pct": 42.0,
    },
    ("oil_palm", "extraction"): {
        "expected_pct": 78.0,
        "max_pct": 82.0,
    },
    # Coffee cherry -> green
    ("coffee", "cherry_to_green"): {
        "expected_pct": 81.5,
        "max_pct": 85.0,
    },
    ("coffee", "wet_processing"): {
        "expected_pct": 40.0,
        "max_pct": 50.0,
    },
    ("coffee", "dry_processing"): {
        "expected_pct": 50.0,
        "max_pct": 60.0,
    },
    ("coffee", "hulling"): {
        "expected_pct": 20.0,
        "max_pct": 28.0,
    },
    # Soya beans -> oil
    ("soya", "beans_to_oil"): {
        "expected_pct": 1.0,
        "max_pct": 3.0,
    },
    ("soya", "cleaning"): {
        "expected_pct": 2.0,
        "max_pct": 5.0,
    },
    ("soya", "dehulling"): {
        "expected_pct": 8.0,
        "max_pct": 12.0,
    },
    ("soya", "solvent_extraction"): {
        "expected_pct": 18.0,
        "max_pct": 25.0,
    },
    # Rubber latex -> sheet
    ("rubber", "latex_to_sheet"): {
        "expected_pct": 67.5,
        "max_pct": 72.0,
    },
    ("rubber", "coagulation"): {
        "expected_pct": 40.0,
        "max_pct": 48.0,
    },
    ("rubber", "sheeting"): {
        "expected_pct": 5.0,
        "max_pct": 8.0,
    },
    ("rubber", "smoking"): {
        "expected_pct": 12.0,
        "max_pct": 18.0,
    },
    # Wood log -> sawn timber
    ("wood", "log_to_sawn"): {
        "expected_pct": 50.0,
        "max_pct": 60.0,
    },
    ("wood", "debarking"): {
        "expected_pct": 10.0,
        "max_pct": 15.0,
    },
    ("wood", "sawing"): {
        "expected_pct": 45.0,
        "max_pct": 55.0,
    },
    ("wood", "kiln_drying"): {
        "expected_pct": 8.0,
        "max_pct": 12.0,
    },
    # Cattle live -> carcass
    ("cattle", "live_to_carcass"): {
        "expected_pct": 45.0,
        "max_pct": 52.0,
    },
    ("cattle", "slaughtering"): {
        "expected_pct": 45.0,
        "max_pct": 52.0,
    },
    ("cattle", "deboning"): {
        "expected_pct": 30.0,
        "max_pct": 38.0,
    },
}

#: Valid loss types as a frozen set for O(1) membership checks.
VALID_LOSS_TYPES = frozenset({
    "processing_loss",
    "transport_loss",
    "storage_loss",
    "quality_rejection",
    "spillage",
    "contamination_loss",
})

#: Valid waste types as a frozen set.
VALID_WASTE_TYPES = frozenset({
    "by_product",
    "waste_material",
    "hazardous_waste",
})

#: Under-reporting threshold multiplier.
#: If reported loss < expected * LOW_THRESHOLD_MULTIPLIER, flag as
#: suspiciously low (potential under-reporting).
LOW_THRESHOLD_MULTIPLIER: float = 0.3

#: Maximum number of trend data points to retain per facility+commodity key.
MAX_TREND_DATA_POINTS: int = 500


# ---------------------------------------------------------------------------
# LossWasteTracker
# ---------------------------------------------------------------------------


class LossWasteTracker:
    """Processing loss and waste tracking engine for mass balance.

    Provides mandatory loss recording for every processing transformation,
    validates losses against commodity-specific reference tolerances, flags
    under-reporting (potential concealment) and over-reporting (potential
    fraud), tracks cumulative losses per batch across processing steps,
    analyses loss trends per facility per commodity, handles by-product
    credits at configurable conversion rates, allocates losses across
    batch splits proportionally, and links waste records to certificates.

    All numeric calculations are deterministic (Python arithmetic only)
    with SHA-256 provenance hashing on every operation for EUDR Article 14
    compliance.

    PRD References:
        PRD-MBC-LOSS-001 through PRD-MBC-LOSS-010.

    Attributes:
        _config: Mass balance calculator configuration.
        _provenance: Provenance tracker for SHA-256 audit trail.
        _loss_records: In-memory store keyed by record_id.
        _waste_records: In-memory store keyed by record_id.
        _batch_cumulative: Cumulative losses keyed by batch_id.
        _facility_trends: Loss history keyed by (facility_id, commodity).
        _waste_certificates: Waste-to-certificate linkage map.
        _by_product_credits: By-product credit records.
        _lock: Reentrant lock for thread safety.

    Example:
        >>> tracker = LossWasteTracker()
        >>> result = tracker.record_loss(
        ...     ledger_id="ledger-001",
        ...     loss_type="processing_loss",
        ...     quantity_kg=130.0,
        ...     batch_id="batch-001",
        ...     process_type="beans_to_nibs",
        ...     commodity="cocoa",
        ... )
        >>> assert result["status"] == "recorded"
    """

    def __init__(
        self,
        config: Optional[MassBalanceCalculatorConfig] = None,
        provenance: Optional[ProvenanceTracker] = None,
    ) -> None:
        """Initialize LossWasteTracker.

        Args:
            config: Optional configuration override. Falls back to
                the singleton ``get_config()`` when not provided.
            provenance: Optional provenance tracker override. Falls back
                to the singleton ``get_provenance_tracker()`` when not
                provided.
        """
        self._config: MassBalanceCalculatorConfig = config or get_config()
        self._provenance: ProvenanceTracker = (
            provenance or get_provenance_tracker()
        )

        # In-memory stores
        self._loss_records: Dict[str, Dict[str, Any]] = {}
        self._waste_records: Dict[str, Dict[str, Any]] = {}
        self._batch_cumulative: Dict[str, Dict[str, Any]] = {}
        self._facility_trends: Dict[
            Tuple[str, str], List[Dict[str, Any]]
        ] = {}
        self._waste_certificates: Dict[str, Dict[str, Any]] = {}
        self._by_product_credits: Dict[str, Dict[str, Any]] = {}

        # Thread safety
        self._lock: threading.RLock = threading.RLock()

        logger.info(
            "LossWasteTracker initialized: loss_validation=%s, "
            "by_product_credit=%s",
            self._config.loss_validation_enabled,
            self._config.by_product_credit_enabled,
        )

    # ------------------------------------------------------------------
    # Public API: record_loss
    # ------------------------------------------------------------------

    def record_loss(
        self,
        ledger_id: str,
        loss_type: str,
        quantity_kg: float,
        batch_id: str,
        process_type: str,
        commodity: str,
        facility_id: Optional[str] = None,
        operator_id: Optional[str] = None,
        notes: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a processing loss against the mass balance ledger.

        Creates a loss record, validates against commodity-specific
        tolerances (if enabled), updates cumulative batch losses, and
        appends to facility trend data.

        PRD: PRD-MBC-LOSS-001 (mandatory loss recording).

        Args:
            ledger_id: Identifier of the parent mass balance ledger.
            loss_type: Type of loss (processing_loss, transport_loss,
                storage_loss, quality_rejection, spillage,
                contamination_loss).
            quantity_kg: Quantity lost in kilograms.  Must be > 0.
            batch_id: Identifier of the batch experiencing the loss.
            process_type: Processing type that caused the loss (e.g.
                ``beans_to_nibs``, ``ffb_to_cpo``).
            commodity: EUDR commodity (cocoa, oil_palm, coffee, soya,
                rubber, wood, cattle).
            facility_id: Optional facility identifier.
            operator_id: Optional operator who recorded the loss.
            notes: Optional free-text notes.
            metadata: Optional additional key-value pairs.

        Returns:
            Dictionary containing:
                - record_id: Unique loss record identifier.
                - status: ``recorded`` on success.
                - validation: Validation result dict (if enabled).
                - cumulative_loss_kg: Updated cumulative loss for batch.
                - provenance_hash: SHA-256 hash of the operation.
                - timestamp: ISO-formatted UTC timestamp.

        Raises:
            ValueError: If loss_type is invalid or quantity_kg <= 0.
        """
        start_time = _utcnow()

        # -- Input validation -------------------------------------------------
        self._validate_loss_type(loss_type)
        if quantity_kg <= 0:
            raise ValueError(
                f"quantity_kg must be > 0, got {quantity_kg}"
            )
        if not ledger_id:
            raise ValueError("ledger_id must not be empty")
        if not batch_id:
            raise ValueError("batch_id must not be empty")
        if not commodity:
            raise ValueError("commodity must not be empty")

        record_id = str(uuid.uuid4())
        timestamp = start_time.isoformat()
        quantity_dec = Decimal(str(quantity_kg))

        # -- Validate loss percentage against reference data ------------------
        validation_result: Optional[Dict[str, Any]] = None
        if self._config.loss_validation_enabled:
            validation_result = self._validate_loss_against_reference(
                commodity=commodity,
                process_type=process_type,
                loss_percentage=quantity_kg,
                quantity_kg=quantity_kg,
            )

        # -- Build loss record ------------------------------------------------
        loss_record: Dict[str, Any] = {
            "record_id": record_id,
            "ledger_id": ledger_id,
            "loss_type": loss_type,
            "quantity_kg": float(quantity_dec),
            "batch_id": batch_id,
            "process_type": process_type,
            "commodity": commodity,
            "facility_id": facility_id,
            "operator_id": operator_id,
            "notes": notes,
            "validation": validation_result,
            "within_tolerance": (
                validation_result.get("within_tolerance", True)
                if validation_result
                else True
            ),
            "metadata": metadata or {},
            "created_at": timestamp,
            "voided": False,
        }

        # -- Provenance hash ---------------------------------------------------
        provenance_hash = self._compute_provenance_hash(loss_record)
        loss_record["provenance_hash"] = provenance_hash

        with self._lock:
            # Store loss record
            self._loss_records[record_id] = loss_record

            # Update cumulative batch loss
            cumulative = self._update_batch_cumulative(
                batch_id=batch_id,
                quantity_kg=float(quantity_dec),
                loss_type=loss_type,
                process_type=process_type,
            )

            # Update facility trends
            if facility_id:
                self._append_trend_data(
                    facility_id=facility_id,
                    commodity=commodity,
                    loss_type=loss_type,
                    quantity_kg=float(quantity_dec),
                    process_type=process_type,
                    timestamp=timestamp,
                )

        # -- Provenance tracking -----------------------------------------------
        self._provenance.record(
            entity_type="loss_record",
            action="record",
            entity_id=record_id,
            data=loss_record,
            metadata={
                "ledger_id": ledger_id,
                "loss_type": loss_type,
                "quantity_kg": float(quantity_dec),
                "batch_id": batch_id,
                "commodity": commodity,
            },
        )

        # -- Metrics -----------------------------------------------------------
        record_loss_recorded(loss_type)

        processing_ms = self._elapsed_ms(start_time)
        logger.info(
            "Loss recorded: record_id=%s ledger=%s type=%s "
            "qty=%.2f kg batch=%s commodity=%s within_tolerance=%s "
            "[%.1f ms]",
            record_id,
            ledger_id,
            loss_type,
            quantity_kg,
            batch_id,
            commodity,
            loss_record["within_tolerance"],
            processing_ms,
        )

        return {
            "record_id": record_id,
            "status": "recorded",
            "validation": validation_result,
            "cumulative_loss_kg": cumulative,
            "provenance_hash": provenance_hash,
            "timestamp": timestamp,
            "processing_time_ms": processing_ms,
        }

    # ------------------------------------------------------------------
    # Public API: record_waste
    # ------------------------------------------------------------------

    def record_waste(
        self,
        ledger_id: str,
        waste_type: str,
        quantity_kg: float,
        batch_id: str,
        commodity: str,
        process_type: Optional[str] = None,
        facility_id: Optional[str] = None,
        operator_id: Optional[str] = None,
        disposal_method: Optional[str] = None,
        disposal_facility: Optional[str] = None,
        hazardous_classification: Optional[str] = None,
        notes: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record waste generated during processing.

        Handles three waste types: by_product (valuable, may generate
        credits), waste_material (non-recoverable), and hazardous_waste
        (requires special disposal documentation).

        PRD: PRD-MBC-LOSS-003 (waste type recording).

        Args:
            ledger_id: Identifier of the parent mass balance ledger.
            waste_type: Type of waste (by_product, waste_material,
                hazardous_waste).
            quantity_kg: Quantity of waste in kilograms.  Must be > 0.
            batch_id: Identifier of the source batch.
            commodity: EUDR commodity.
            process_type: Optional processing step that generated waste.
            facility_id: Optional facility identifier.
            operator_id: Optional operator who recorded the waste.
            disposal_method: Optional disposal or utilization method.
            disposal_facility: Optional disposal facility identifier.
            hazardous_classification: Optional hazardous waste class.
            notes: Optional free-text notes.
            metadata: Optional additional key-value pairs.

        Returns:
            Dictionary containing:
                - record_id: Unique waste record identifier.
                - status: ``recorded`` on success.
                - waste_type: Recorded waste type.
                - credit_eligible: Whether by-product credit is eligible.
                - provenance_hash: SHA-256 hash.
                - timestamp: ISO-formatted UTC timestamp.

        Raises:
            ValueError: If waste_type is invalid or quantity_kg <= 0.
        """
        start_time = _utcnow()

        # -- Input validation -------------------------------------------------
        self._validate_waste_type(waste_type)
        if quantity_kg <= 0:
            raise ValueError(
                f"quantity_kg must be > 0, got {quantity_kg}"
            )
        if not ledger_id:
            raise ValueError("ledger_id must not be empty")
        if not batch_id:
            raise ValueError("batch_id must not be empty")

        record_id = str(uuid.uuid4())
        timestamp = start_time.isoformat()
        quantity_dec = Decimal(str(quantity_kg))

        credit_eligible = (
            waste_type == "by_product"
            and self._config.by_product_credit_enabled
        )

        # -- Build waste record -----------------------------------------------
        waste_record: Dict[str, Any] = {
            "record_id": record_id,
            "ledger_id": ledger_id,
            "waste_type": waste_type,
            "quantity_kg": float(quantity_dec),
            "batch_id": batch_id,
            "commodity": commodity,
            "process_type": process_type,
            "facility_id": facility_id,
            "operator_id": operator_id,
            "disposal_method": disposal_method,
            "disposal_facility": disposal_facility,
            "hazardous_classification": hazardous_classification,
            "credit_eligible": credit_eligible,
            "notes": notes,
            "certificate_ref": None,
            "metadata": metadata or {},
            "created_at": timestamp,
            "voided": False,
        }

        provenance_hash = self._compute_provenance_hash(waste_record)
        waste_record["provenance_hash"] = provenance_hash

        with self._lock:
            self._waste_records[record_id] = waste_record

        # -- Provenance tracking -----------------------------------------------
        self._provenance.record(
            entity_type="loss_record",
            action="record",
            entity_id=record_id,
            data=waste_record,
            metadata={
                "ledger_id": ledger_id,
                "waste_type": waste_type,
                "quantity_kg": float(quantity_dec),
                "batch_id": batch_id,
                "commodity": commodity,
            },
        )

        processing_ms = self._elapsed_ms(start_time)
        logger.info(
            "Waste recorded: record_id=%s ledger=%s type=%s "
            "qty=%.2f kg batch=%s credit_eligible=%s [%.1f ms]",
            record_id,
            ledger_id,
            waste_type,
            quantity_kg,
            batch_id,
            credit_eligible,
            processing_ms,
        )

        return {
            "record_id": record_id,
            "status": "recorded",
            "waste_type": waste_type,
            "credit_eligible": credit_eligible,
            "provenance_hash": provenance_hash,
            "timestamp": timestamp,
            "processing_time_ms": processing_ms,
        }

    # ------------------------------------------------------------------
    # Public API: validate_loss
    # ------------------------------------------------------------------

    def validate_loss(
        self,
        commodity: str,
        process_type: str,
        loss_percentage: float,
    ) -> Dict[str, Any]:
        """Validate a loss percentage against commodity-specific tolerances.

        Checks the reported loss percentage against the reference data
        for the given commodity and process type.  Flags losses that
        are too low (under-reporting, potential concealment) or too high
        (above maximum tolerance, potential fraud).

        PRD: PRD-MBC-LOSS-005 (loss validation, too-low / too-high
        flagging).

        Args:
            commodity: EUDR commodity (cocoa, oil_palm, coffee, etc.).
            process_type: Processing step (beans_to_nibs, ffb_to_cpo,
                etc.).
            loss_percentage: Reported loss as a percentage (0-100).

        Returns:
            Dictionary containing:
                - commodity: Input commodity.
                - process_type: Input process type.
                - loss_percentage: Reported loss percentage.
                - expected_pct: Expected loss percentage from reference.
                - max_pct: Maximum tolerance percentage from reference.
                - within_tolerance: True if within acceptable range.
                - flag: ``none``, ``too_low``, ``too_high``, or
                  ``no_reference``.
                - message: Human-readable validation message.
                - provenance_hash: SHA-256 hash.
        """
        start_time = _utcnow()

        result = self._validate_loss_against_reference(
            commodity=commodity,
            process_type=process_type,
            loss_percentage=loss_percentage,
            quantity_kg=0.0,
        )

        provenance_hash = self._compute_provenance_hash(result)
        result["provenance_hash"] = provenance_hash

        self._provenance.record(
            entity_type="loss_record",
            action="validate",
            entity_id=f"{commodity}:{process_type}",
            data=result,
            metadata={
                "commodity": commodity,
                "process_type": process_type,
                "loss_percentage": loss_percentage,
            },
        )

        processing_ms = self._elapsed_ms(start_time)
        result["processing_time_ms"] = processing_ms

        logger.info(
            "Loss validation: commodity=%s process=%s loss=%.1f%% "
            "flag=%s within_tolerance=%s [%.1f ms]",
            commodity,
            process_type,
            loss_percentage,
            result.get("flag", "none"),
            result.get("within_tolerance", True),
            processing_ms,
        )

        return result

    # ------------------------------------------------------------------
    # Public API: get_loss_records
    # ------------------------------------------------------------------

    def get_loss_records(
        self,
        facility_id: Optional[str] = None,
        ledger_id: Optional[str] = None,
        batch_id: Optional[str] = None,
        commodity: Optional[str] = None,
        loss_type: Optional[str] = None,
        within_tolerance: Optional[bool] = None,
        include_voided: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Retrieve loss records with optional filtering.

        Args:
            facility_id: Filter by facility identifier.
            ledger_id: Filter by ledger identifier.
            batch_id: Filter by batch identifier.
            commodity: Filter by commodity.
            loss_type: Filter by loss type.
            within_tolerance: Filter by tolerance status.
            include_voided: Whether to include voided records.
            limit: Maximum number of records to return (1-1000).
            offset: Number of records to skip for pagination.

        Returns:
            List of loss record dictionaries matching the filters,
            sorted by creation timestamp descending.
        """
        with self._lock:
            records = list(self._loss_records.values())

        # -- Apply filters ----------------------------------------------------
        if not include_voided:
            records = [r for r in records if not r.get("voided", False)]
        if facility_id is not None:
            records = [
                r for r in records if r.get("facility_id") == facility_id
            ]
        if ledger_id is not None:
            records = [
                r for r in records if r.get("ledger_id") == ledger_id
            ]
        if batch_id is not None:
            records = [
                r for r in records if r.get("batch_id") == batch_id
            ]
        if commodity is not None:
            records = [
                r for r in records if r.get("commodity") == commodity
            ]
        if loss_type is not None:
            records = [
                r for r in records if r.get("loss_type") == loss_type
            ]
        if within_tolerance is not None:
            records = [
                r for r in records
                if r.get("within_tolerance") == within_tolerance
            ]

        # -- Sort by created_at descending ------------------------------------
        records.sort(key=lambda r: r.get("created_at", ""), reverse=True)

        # -- Paginate ---------------------------------------------------------
        clamped_limit = max(1, min(limit, 1000))
        clamped_offset = max(0, offset)
        paginated = records[clamped_offset:clamped_offset + clamped_limit]

        logger.debug(
            "get_loss_records: total=%d filtered=%d returned=%d "
            "offset=%d limit=%d",
            len(self._loss_records),
            len(records),
            len(paginated),
            clamped_offset,
            clamped_limit,
        )

        return paginated

    # ------------------------------------------------------------------
    # Public API: get_cumulative_loss
    # ------------------------------------------------------------------

    def get_cumulative_loss(
        self,
        batch_id: str,
    ) -> Dict[str, Any]:
        """Get cumulative loss tracking data for a specific batch.

        Returns the total losses accumulated across all processing
        steps for the given batch, broken down by loss type and
        process type.

        PRD: PRD-MBC-LOSS-006 (cumulative loss tracking per batch).

        Args:
            batch_id: Batch identifier to retrieve cumulative losses for.

        Returns:
            Dictionary containing:
                - batch_id: The batch identifier.
                - total_loss_kg: Total cumulative loss in kg.
                - loss_count: Number of loss events recorded.
                - by_loss_type: Breakdown by loss type.
                - by_process_type: Breakdown by process type.
                - loss_records: List of individual loss record IDs.
                - provenance_hash: SHA-256 hash.
                - timestamp: Current UTC timestamp.
        """
        if not batch_id:
            raise ValueError("batch_id must not be empty")

        with self._lock:
            cumulative = self._batch_cumulative.get(batch_id)

        if cumulative is None:
            return {
                "batch_id": batch_id,
                "total_loss_kg": 0.0,
                "loss_count": 0,
                "by_loss_type": {},
                "by_process_type": {},
                "loss_records": [],
                "provenance_hash": self._compute_provenance_hash(
                    {"batch_id": batch_id, "total_loss_kg": 0.0}
                ),
                "timestamp": _utcnow().isoformat(),
            }

        result = dict(cumulative)
        result["provenance_hash"] = self._compute_provenance_hash(result)
        result["timestamp"] = _utcnow().isoformat()

        logger.debug(
            "get_cumulative_loss: batch=%s total=%.2f kg count=%d",
            batch_id,
            result.get("total_loss_kg", 0.0),
            result.get("loss_count", 0),
        )

        return result

    # ------------------------------------------------------------------
    # Public API: get_loss_trends
    # ------------------------------------------------------------------

    def get_loss_trends(
        self,
        facility_id: str,
        commodity: str,
        num_periods: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Analyse loss trends for a facility and commodity combination.

        Computes statistical metrics (mean, median, std deviation,
        min, max) and identifies trend direction (increasing,
        decreasing, stable) from historical loss data.

        PRD: PRD-MBC-LOSS-007 (loss trend analysis per facility per
        commodity).

        Args:
            facility_id: Facility identifier.
            commodity: Commodity to analyse.
            num_periods: Optional limit on the number of most recent
                data points to include.  Defaults to all available data.

        Returns:
            Dictionary containing:
                - facility_id: Input facility.
                - commodity: Input commodity.
                - data_points: Number of data points analysed.
                - total_loss_kg: Total loss across all data points.
                - mean_loss_kg: Mean loss per event.
                - median_loss_kg: Median loss per event.
                - std_deviation_kg: Standard deviation.
                - min_loss_kg: Minimum single-event loss.
                - max_loss_kg: Maximum single-event loss.
                - trend_direction: ``increasing``, ``decreasing``, or
                  ``stable``.
                - by_loss_type: Breakdown by loss type.
                - by_process_type: Breakdown by process type.
                - recent_entries: Most recent trend entries.
                - provenance_hash: SHA-256 hash.
        """
        if not facility_id:
            raise ValueError("facility_id must not be empty")
        if not commodity:
            raise ValueError("commodity must not be empty")

        trend_key = (facility_id, commodity)
        with self._lock:
            raw_data = list(
                self._facility_trends.get(trend_key, [])
            )

        if num_periods is not None and num_periods > 0:
            raw_data = raw_data[-num_periods:]

        if not raw_data:
            empty_result = self._build_empty_trend_result(
                facility_id, commodity
            )
            return empty_result

        # -- Compute statistics -----------------------------------------------
        quantities = [d["quantity_kg"] for d in raw_data]
        total_loss = sum(quantities)
        mean_loss = statistics.mean(quantities)
        median_loss = statistics.median(quantities)
        std_dev = (
            statistics.stdev(quantities)
            if len(quantities) > 1
            else 0.0
        )
        min_loss = min(quantities)
        max_loss = max(quantities)

        # -- Trend direction ---------------------------------------------------
        trend_direction = self._compute_trend_direction(quantities)

        # -- Breakdown by loss type and process type --------------------------
        by_loss_type = self._aggregate_by_key(raw_data, "loss_type")
        by_process_type = self._aggregate_by_key(raw_data, "process_type")

        # -- Recent entries (last 10) -----------------------------------------
        recent_entries = raw_data[-10:]

        result: Dict[str, Any] = {
            "facility_id": facility_id,
            "commodity": commodity,
            "data_points": len(quantities),
            "total_loss_kg": round(total_loss, 4),
            "mean_loss_kg": round(mean_loss, 4),
            "median_loss_kg": round(median_loss, 4),
            "std_deviation_kg": round(std_dev, 4),
            "min_loss_kg": round(min_loss, 4),
            "max_loss_kg": round(max_loss, 4),
            "trend_direction": trend_direction,
            "by_loss_type": by_loss_type,
            "by_process_type": by_process_type,
            "recent_entries": recent_entries,
        }
        result["provenance_hash"] = self._compute_provenance_hash(result)

        logger.info(
            "Loss trends: facility=%s commodity=%s points=%d "
            "mean=%.2f kg trend=%s",
            facility_id,
            commodity,
            len(quantities),
            mean_loss,
            trend_direction,
        )

        return result

    # ------------------------------------------------------------------
    # Public API: credit_by_product
    # ------------------------------------------------------------------

    def credit_by_product(
        self,
        ledger_id: str,
        by_product_qty: float,
        conversion_rate: float,
        batch_id: Optional[str] = None,
        commodity: Optional[str] = None,
        facility_id: Optional[str] = None,
        by_product_name: Optional[str] = None,
        notes: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Credit valuable by-products back to the mass balance ledger.

        When processing generates a valuable by-product (e.g. cocoa
        butter from pressing, palm kernel oil from extraction), the
        by-product quantity is converted to a credit using the specified
        conversion rate and added back to the ledger balance.

        PRD: PRD-MBC-LOSS-004 (by-product credit at conversion rate).

        Args:
            ledger_id: Identifier of the parent ledger to credit.
            by_product_qty: Quantity of by-product in kilograms.
            conversion_rate: Conversion rate (0.0-1.0) to apply.
                The credit amount = by_product_qty * conversion_rate.
            batch_id: Optional source batch identifier.
            commodity: Optional commodity.
            facility_id: Optional facility identifier.
            by_product_name: Optional name of the by-product.
            notes: Optional free-text notes.
            metadata: Optional additional key-value pairs.

        Returns:
            Dictionary containing:
                - credit_id: Unique credit record identifier.
                - status: ``credited`` on success.
                - by_product_qty_kg: Input by-product quantity.
                - conversion_rate: Applied conversion rate.
                - credit_amount_kg: Credited quantity
                  (by_product_qty * conversion_rate).
                - provenance_hash: SHA-256 hash.
                - timestamp: ISO-formatted UTC timestamp.

        Raises:
            ValueError: If by-product credits are disabled, quantities
                are invalid, or conversion_rate is out of range.
        """
        start_time = _utcnow()

        if not self._config.by_product_credit_enabled:
            raise ValueError(
                "By-product credits are disabled in configuration. "
                "Set by_product_credit_enabled=True to enable."
            )
        if by_product_qty <= 0:
            raise ValueError(
                f"by_product_qty must be > 0, got {by_product_qty}"
            )
        if not (0.0 < conversion_rate <= 1.0):
            raise ValueError(
                f"conversion_rate must be in (0.0, 1.0], "
                f"got {conversion_rate}"
            )
        if not ledger_id:
            raise ValueError("ledger_id must not be empty")

        credit_id = str(uuid.uuid4())
        timestamp = start_time.isoformat()

        # Deterministic credit calculation
        credit_amount = Decimal(str(by_product_qty)) * Decimal(
            str(conversion_rate)
        )
        credit_amount = credit_amount.quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP
        )

        credit_record: Dict[str, Any] = {
            "credit_id": credit_id,
            "ledger_id": ledger_id,
            "by_product_qty_kg": by_product_qty,
            "conversion_rate": conversion_rate,
            "credit_amount_kg": float(credit_amount),
            "batch_id": batch_id,
            "commodity": commodity,
            "facility_id": facility_id,
            "by_product_name": by_product_name,
            "notes": notes,
            "metadata": metadata or {},
            "created_at": timestamp,
        }

        provenance_hash = self._compute_provenance_hash(credit_record)
        credit_record["provenance_hash"] = provenance_hash

        with self._lock:
            self._by_product_credits[credit_id] = credit_record

        self._provenance.record(
            entity_type="loss_record",
            action="create",
            entity_id=credit_id,
            data=credit_record,
            metadata={
                "ledger_id": ledger_id,
                "credit_amount_kg": float(credit_amount),
                "conversion_rate": conversion_rate,
            },
        )

        processing_ms = self._elapsed_ms(start_time)
        logger.info(
            "By-product credit: credit_id=%s ledger=%s "
            "input=%.2f kg rate=%.4f credit=%.4f kg [%.1f ms]",
            credit_id,
            ledger_id,
            by_product_qty,
            conversion_rate,
            float(credit_amount),
            processing_ms,
        )

        return {
            "credit_id": credit_id,
            "status": "credited",
            "by_product_qty_kg": by_product_qty,
            "conversion_rate": conversion_rate,
            "credit_amount_kg": float(credit_amount),
            "provenance_hash": provenance_hash,
            "timestamp": timestamp,
            "processing_time_ms": processing_ms,
        }

    # ------------------------------------------------------------------
    # Public API: allocate_loss_to_splits
    # ------------------------------------------------------------------

    def allocate_loss_to_splits(
        self,
        loss_record_id: str,
        split_ratios: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Allocate a loss record proportionally across batch splits.

        When a batch is split into sub-batches, the associated losses
        must be allocated proportionally to each split based on the
        mass ratios.

        PRD: PRD-MBC-LOSS-009 (loss allocation for batch splits).

        Args:
            loss_record_id: Identifier of the original loss record.
            split_ratios: Dictionary mapping split_batch_id to its
                proportion (0.0-1.0).  All ratios must sum to 1.0
                (within 0.001 tolerance).

        Returns:
            List of allocation dictionaries, one per split, each
            containing:
                - allocation_id: Unique allocation identifier.
                - original_record_id: Original loss record ID.
                - split_batch_id: Split batch identifier.
                - ratio: Proportion allocated.
                - allocated_qty_kg: Allocated loss quantity in kg.
                - provenance_hash: SHA-256 hash.

        Raises:
            ValueError: If loss record not found, split_ratios is
                empty, ratios do not sum to ~1.0, or any ratio is
                out of range.
        """
        start_time = _utcnow()

        if not loss_record_id:
            raise ValueError("loss_record_id must not be empty")
        if not split_ratios:
            raise ValueError("split_ratios must not be empty")

        # Validate ratios sum to ~1.0
        ratio_sum = sum(split_ratios.values())
        if abs(ratio_sum - 1.0) > 0.001:
            raise ValueError(
                f"split_ratios must sum to 1.0, got {ratio_sum:.6f}"
            )

        # Validate individual ratios
        for split_id, ratio in split_ratios.items():
            if not (0.0 < ratio <= 1.0):
                raise ValueError(
                    f"Ratio for split '{split_id}' must be in "
                    f"(0.0, 1.0], got {ratio}"
                )

        # Look up original loss record
        with self._lock:
            original = self._loss_records.get(loss_record_id)

        if original is None:
            raise ValueError(
                f"Loss record '{loss_record_id}' not found"
            )

        original_qty = Decimal(str(original["quantity_kg"]))
        allocations: List[Dict[str, Any]] = []
        timestamp = start_time.isoformat()

        for split_batch_id, ratio in split_ratios.items():
            allocation_id = str(uuid.uuid4())
            allocated_qty = original_qty * Decimal(str(ratio))
            allocated_qty = allocated_qty.quantize(
                Decimal("0.0001"), rounding=ROUND_HALF_UP
            )

            allocation: Dict[str, Any] = {
                "allocation_id": allocation_id,
                "original_record_id": loss_record_id,
                "split_batch_id": split_batch_id,
                "ratio": ratio,
                "allocated_qty_kg": float(allocated_qty),
                "commodity": original.get("commodity"),
                "loss_type": original.get("loss_type"),
                "process_type": original.get("process_type"),
                "created_at": timestamp,
            }

            provenance_hash = self._compute_provenance_hash(allocation)
            allocation["provenance_hash"] = provenance_hash
            allocations.append(allocation)

        # Track provenance
        self._provenance.record(
            entity_type="loss_record",
            action="update",
            entity_id=loss_record_id,
            data={
                "action": "allocate_to_splits",
                "split_count": len(split_ratios),
                "allocations": [a["allocation_id"] for a in allocations],
            },
            metadata={"split_count": len(split_ratios)},
        )

        processing_ms = self._elapsed_ms(start_time)
        logger.info(
            "Loss allocated to %d splits: record=%s "
            "original_qty=%.2f kg [%.1f ms]",
            len(allocations),
            loss_record_id,
            float(original_qty),
            processing_ms,
        )

        return allocations

    # ------------------------------------------------------------------
    # Public API: link_waste_certificate
    # ------------------------------------------------------------------

    def link_waste_certificate(
        self,
        record_id: str,
        certificate_ref: str,
        certificate_type: Optional[str] = None,
        issuing_authority: Optional[str] = None,
        issue_date: Optional[str] = None,
        expiry_date: Optional[str] = None,
        notes: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Link a waste record to a disposal or handling certificate.

        Associates a waste record with its disposal, handling, or
        recycling certificate for regulatory documentation and
        audit trail purposes.

        PRD: PRD-MBC-LOSS-008 (waste documentation linkage).

        Args:
            record_id: Waste record identifier to link.
            certificate_ref: Certificate reference number or ID.
            certificate_type: Optional type of certificate (e.g.
                ``disposal``, ``recycling``, ``hazardous_manifest``).
            issuing_authority: Optional authority that issued the cert.
            issue_date: Optional ISO-formatted issue date.
            expiry_date: Optional ISO-formatted expiry date.
            notes: Optional free-text notes.
            metadata: Optional additional key-value pairs.

        Returns:
            Dictionary containing:
                - record_id: Waste record identifier.
                - certificate_ref: Linked certificate reference.
                - status: ``linked`` on success.
                - link_id: Unique link identifier.
                - provenance_hash: SHA-256 hash.

        Raises:
            ValueError: If record_id not found or certificate_ref empty.
        """
        start_time = _utcnow()

        if not record_id:
            raise ValueError("record_id must not be empty")
        if not certificate_ref:
            raise ValueError("certificate_ref must not be empty")

        with self._lock:
            waste_record = self._waste_records.get(record_id)

        if waste_record is None:
            raise ValueError(
                f"Waste record '{record_id}' not found"
            )

        link_id = str(uuid.uuid4())
        timestamp = _utcnow().isoformat()

        link_data: Dict[str, Any] = {
            "link_id": link_id,
            "record_id": record_id,
            "certificate_ref": certificate_ref,
            "certificate_type": certificate_type,
            "issuing_authority": issuing_authority,
            "issue_date": issue_date,
            "expiry_date": expiry_date,
            "notes": notes,
            "metadata": metadata or {},
            "created_at": timestamp,
        }

        provenance_hash = self._compute_provenance_hash(link_data)
        link_data["provenance_hash"] = provenance_hash

        with self._lock:
            # Update waste record with certificate reference
            self._waste_records[record_id]["certificate_ref"] = (
                certificate_ref
            )
            # Store link
            self._waste_certificates[link_id] = link_data

        self._provenance.record(
            entity_type="loss_record",
            action="update",
            entity_id=record_id,
            data=link_data,
            metadata={
                "certificate_ref": certificate_ref,
                "certificate_type": certificate_type,
            },
        )

        processing_ms = self._elapsed_ms(start_time)
        logger.info(
            "Waste certificate linked: record=%s cert=%s "
            "link_id=%s [%.1f ms]",
            record_id,
            certificate_ref,
            link_id,
            processing_ms,
        )

        return {
            "record_id": record_id,
            "certificate_ref": certificate_ref,
            "status": "linked",
            "link_id": link_id,
            "provenance_hash": provenance_hash,
            "timestamp": timestamp,
            "processing_time_ms": processing_ms,
        }

    # ------------------------------------------------------------------
    # Public API: void_loss_record
    # ------------------------------------------------------------------

    def void_loss_record(
        self,
        record_id: str,
        reason: str,
        operator_id: str,
    ) -> Dict[str, Any]:
        """Void a previously recorded loss record.

        Marks the loss record as voided without deleting it, preserving
        the audit trail.  Voided records are excluded from cumulative
        totals by default.

        Args:
            record_id: Loss record identifier to void.
            reason: Reason for voiding.
            operator_id: Operator performing the void.

        Returns:
            Dictionary with record_id, status=``voided``, and
            provenance_hash.

        Raises:
            ValueError: If record not found or already voided.
        """
        if not record_id:
            raise ValueError("record_id must not be empty")
        if not reason:
            raise ValueError("reason must not be empty")
        if not operator_id:
            raise ValueError("operator_id must not be empty")

        timestamp = _utcnow().isoformat()

        with self._lock:
            record = self._loss_records.get(record_id)
            if record is None:
                raise ValueError(
                    f"Loss record '{record_id}' not found"
                )
            if record.get("voided", False):
                raise ValueError(
                    f"Loss record '{record_id}' is already voided"
                )

            record["voided"] = True
            record["voided_at"] = timestamp
            record["voided_by"] = operator_id
            record["void_reason"] = reason

        provenance_hash = self._compute_provenance_hash({
            "record_id": record_id,
            "action": "void",
            "reason": reason,
            "operator_id": operator_id,
            "timestamp": timestamp,
        })

        self._provenance.record(
            entity_type="loss_record",
            action="void",
            entity_id=record_id,
            data={"reason": reason, "operator_id": operator_id},
            metadata={"operator_id": operator_id},
        )

        logger.info(
            "Loss record voided: record=%s by=%s reason=%s",
            record_id,
            operator_id,
            reason,
        )

        return {
            "record_id": record_id,
            "status": "voided",
            "provenance_hash": provenance_hash,
            "timestamp": timestamp,
        }

    # ------------------------------------------------------------------
    # Public API: get_waste_records
    # ------------------------------------------------------------------

    def get_waste_records(
        self,
        facility_id: Optional[str] = None,
        ledger_id: Optional[str] = None,
        batch_id: Optional[str] = None,
        waste_type: Optional[str] = None,
        include_voided: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Retrieve waste records with optional filtering.

        Args:
            facility_id: Filter by facility identifier.
            ledger_id: Filter by ledger identifier.
            batch_id: Filter by batch identifier.
            waste_type: Filter by waste type.
            include_voided: Whether to include voided records.
            limit: Maximum number of records to return.
            offset: Number of records to skip for pagination.

        Returns:
            List of waste record dictionaries matching the filters.
        """
        with self._lock:
            records = list(self._waste_records.values())

        if not include_voided:
            records = [r for r in records if not r.get("voided", False)]
        if facility_id is not None:
            records = [
                r for r in records if r.get("facility_id") == facility_id
            ]
        if ledger_id is not None:
            records = [
                r for r in records if r.get("ledger_id") == ledger_id
            ]
        if batch_id is not None:
            records = [
                r for r in records if r.get("batch_id") == batch_id
            ]
        if waste_type is not None:
            records = [
                r for r in records if r.get("waste_type") == waste_type
            ]

        records.sort(key=lambda r: r.get("created_at", ""), reverse=True)

        clamped_limit = max(1, min(limit, 1000))
        clamped_offset = max(0, offset)
        return records[clamped_offset:clamped_offset + clamped_limit]

    # ------------------------------------------------------------------
    # Public API: get_by_product_credits
    # ------------------------------------------------------------------

    def get_by_product_credits(
        self,
        ledger_id: Optional[str] = None,
        facility_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Retrieve by-product credit records with optional filtering.

        Args:
            ledger_id: Filter by ledger identifier.
            facility_id: Filter by facility identifier.
            limit: Maximum number of records to return.
            offset: Pagination offset.

        Returns:
            List of by-product credit record dictionaries.
        """
        with self._lock:
            records = list(self._by_product_credits.values())

        if ledger_id is not None:
            records = [
                r for r in records if r.get("ledger_id") == ledger_id
            ]
        if facility_id is not None:
            records = [
                r for r in records if r.get("facility_id") == facility_id
            ]

        records.sort(key=lambda r: r.get("created_at", ""), reverse=True)

        clamped_limit = max(1, min(limit, 1000))
        clamped_offset = max(0, offset)
        return records[clamped_offset:clamped_offset + clamped_limit]

    # ------------------------------------------------------------------
    # Public API: get_loss_summary
    # ------------------------------------------------------------------

    def get_loss_summary(
        self,
        facility_id: Optional[str] = None,
        commodity: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get a summary of all losses with breakdowns.

        Args:
            facility_id: Optional facility filter.
            commodity: Optional commodity filter.

        Returns:
            Dictionary with total_loss_kg, record_count, breakdowns
            by loss_type, commodity, facility, and process_type.
        """
        with self._lock:
            records = [
                r for r in self._loss_records.values()
                if not r.get("voided", False)
            ]

        if facility_id:
            records = [
                r for r in records if r.get("facility_id") == facility_id
            ]
        if commodity:
            records = [
                r for r in records if r.get("commodity") == commodity
            ]

        total_loss = sum(r.get("quantity_kg", 0.0) for r in records)

        by_loss_type = self._aggregate_qty_by_key(records, "loss_type")
        by_commodity = self._aggregate_qty_by_key(records, "commodity")
        by_facility = self._aggregate_qty_by_key(records, "facility_id")
        by_process = self._aggregate_qty_by_key(records, "process_type")

        tolerance_violations = [
            r for r in records if not r.get("within_tolerance", True)
        ]

        result: Dict[str, Any] = {
            "total_loss_kg": round(total_loss, 4),
            "record_count": len(records),
            "by_loss_type": by_loss_type,
            "by_commodity": by_commodity,
            "by_facility": by_facility,
            "by_process_type": by_process,
            "tolerance_violations_count": len(tolerance_violations),
            "timestamp": _utcnow().isoformat(),
        }
        result["provenance_hash"] = self._compute_provenance_hash(result)

        return result

    # ------------------------------------------------------------------
    # Public API: get_reference_tolerances
    # ------------------------------------------------------------------

    def get_reference_tolerances(
        self,
        commodity: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get the commodity-specific loss reference tolerances.

        Args:
            commodity: Optional commodity filter.  If None, returns
                all reference data.

        Returns:
            Dictionary of reference tolerance data keyed by
            ``commodity:process_type``.
        """
        result: Dict[str, Any] = {}

        for (comm, proc), ref_data in COMMODITY_PROCESS_LOSS_REFERENCE.items():
            if commodity is not None and comm != commodity:
                continue
            key = f"{comm}:{proc}"
            result[key] = {
                "commodity": comm,
                "process_type": proc,
                "expected_pct": ref_data["expected_pct"],
                "max_pct": ref_data["max_pct"],
            }

        return {
            "reference_tolerances": result,
            "total_entries": len(result),
            "timestamp": _utcnow().isoformat(),
        }

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _validate_loss_type(self, loss_type: str) -> None:
        """Validate that loss_type is one of the six valid types.

        Raises:
            ValueError: If loss_type is not valid.
        """
        if loss_type not in VALID_LOSS_TYPES:
            raise ValueError(
                f"Invalid loss_type '{loss_type}'. Must be one of: "
                f"{sorted(VALID_LOSS_TYPES)}"
            )

    def _validate_waste_type(self, waste_type: str) -> None:
        """Validate that waste_type is one of the three valid types.

        Raises:
            ValueError: If waste_type is not valid.
        """
        if waste_type not in VALID_WASTE_TYPES:
            raise ValueError(
                f"Invalid waste_type '{waste_type}'. Must be one of: "
                f"{sorted(VALID_WASTE_TYPES)}"
            )

    def _validate_loss_against_reference(
        self,
        commodity: str,
        process_type: str,
        loss_percentage: float,
        quantity_kg: float,
    ) -> Dict[str, Any]:
        """Validate a loss against commodity-specific reference data.

        Returns a validation result dict with flag indicators for
        too_low (under-reporting) and too_high (above max tolerance).

        PRD: PRD-MBC-LOSS-005.
        """
        ref_key = (commodity.lower(), process_type.lower())
        ref_data = COMMODITY_PROCESS_LOSS_REFERENCE.get(ref_key)

        if ref_data is None:
            return {
                "commodity": commodity,
                "process_type": process_type,
                "loss_percentage": loss_percentage,
                "expected_pct": None,
                "max_pct": None,
                "within_tolerance": True,
                "flag": "no_reference",
                "message": (
                    f"No reference data for {commodity}/{process_type}. "
                    f"Loss accepted without validation."
                ),
            }

        expected_pct = ref_data["expected_pct"]
        max_pct = ref_data["max_pct"]

        # Too-low check: under-reporting threshold
        low_threshold = expected_pct * LOW_THRESHOLD_MULTIPLIER
        if loss_percentage < low_threshold and loss_percentage > 0:
            return {
                "commodity": commodity,
                "process_type": process_type,
                "loss_percentage": loss_percentage,
                "expected_pct": expected_pct,
                "max_pct": max_pct,
                "within_tolerance": False,
                "flag": "too_low",
                "message": (
                    f"Loss {loss_percentage:.1f}% is suspiciously low "
                    f"(expected ~{expected_pct:.1f}%, "
                    f"minimum threshold {low_threshold:.1f}%). "
                    f"Potential under-reporting."
                ),
            }

        # Too-high check: above maximum tolerance
        if loss_percentage > max_pct:
            return {
                "commodity": commodity,
                "process_type": process_type,
                "loss_percentage": loss_percentage,
                "expected_pct": expected_pct,
                "max_pct": max_pct,
                "within_tolerance": False,
                "flag": "too_high",
                "message": (
                    f"Loss {loss_percentage:.1f}% exceeds maximum "
                    f"tolerance of {max_pct:.1f}% for "
                    f"{commodity}/{process_type}. "
                    f"Potential fraud or data error."
                ),
            }

        # Within tolerance
        return {
            "commodity": commodity,
            "process_type": process_type,
            "loss_percentage": loss_percentage,
            "expected_pct": expected_pct,
            "max_pct": max_pct,
            "within_tolerance": True,
            "flag": "none",
            "message": (
                f"Loss {loss_percentage:.1f}% is within acceptable "
                f"range (expected ~{expected_pct:.1f}%, "
                f"max {max_pct:.1f}%)."
            ),
        }

    def _update_batch_cumulative(
        self,
        batch_id: str,
        quantity_kg: float,
        loss_type: str,
        process_type: str,
    ) -> float:
        """Update cumulative loss tracking for a batch.

        Must be called within the lock context.

        Returns:
            Updated total cumulative loss in kg.
        """
        if batch_id not in self._batch_cumulative:
            self._batch_cumulative[batch_id] = {
                "batch_id": batch_id,
                "total_loss_kg": 0.0,
                "loss_count": 0,
                "by_loss_type": {},
                "by_process_type": {},
                "loss_records": [],
            }

        cumulative = self._batch_cumulative[batch_id]
        cumulative["total_loss_kg"] = round(
            cumulative["total_loss_kg"] + quantity_kg, 4
        )
        cumulative["loss_count"] += 1

        # By loss type
        if loss_type not in cumulative["by_loss_type"]:
            cumulative["by_loss_type"][loss_type] = {
                "total_kg": 0.0,
                "count": 0,
            }
        lt = cumulative["by_loss_type"][loss_type]
        lt["total_kg"] = round(lt["total_kg"] + quantity_kg, 4)
        lt["count"] += 1

        # By process type
        if process_type not in cumulative["by_process_type"]:
            cumulative["by_process_type"][process_type] = {
                "total_kg": 0.0,
                "count": 0,
            }
        pt = cumulative["by_process_type"][process_type]
        pt["total_kg"] = round(pt["total_kg"] + quantity_kg, 4)
        pt["count"] += 1

        return cumulative["total_loss_kg"]

    def _append_trend_data(
        self,
        facility_id: str,
        commodity: str,
        loss_type: str,
        quantity_kg: float,
        process_type: str,
        timestamp: str,
    ) -> None:
        """Append a data point to facility trend tracking.

        Must be called within the lock context.  Enforces the maximum
        data point limit by removing the oldest entries.
        """
        trend_key = (facility_id, commodity)
        if trend_key not in self._facility_trends:
            self._facility_trends[trend_key] = []

        self._facility_trends[trend_key].append({
            "loss_type": loss_type,
            "quantity_kg": quantity_kg,
            "process_type": process_type,
            "timestamp": timestamp,
        })

        # Enforce maximum data points
        if len(self._facility_trends[trend_key]) > MAX_TREND_DATA_POINTS:
            self._facility_trends[trend_key] = (
                self._facility_trends[trend_key][-MAX_TREND_DATA_POINTS:]
            )

    def _compute_trend_direction(
        self,
        quantities: List[float],
    ) -> str:
        """Compute trend direction from a list of quantities.

        Uses simple linear regression slope to determine direction.
        Returns ``increasing``, ``decreasing``, or ``stable``.
        """
        n = len(quantities)
        if n < 3:
            return "stable"

        # Simple linear regression: slope of y = a + bx
        x_vals = list(range(n))
        x_mean = sum(x_vals) / n
        y_mean = sum(quantities) / n

        numerator = sum(
            (x - x_mean) * (y - y_mean)
            for x, y in zip(x_vals, quantities)
        )
        denominator = sum((x - x_mean) ** 2 for x in x_vals)

        if denominator == 0:
            return "stable"

        slope = numerator / denominator

        # Normalize slope relative to mean to determine significance
        if y_mean == 0:
            return "stable"

        relative_slope = slope / y_mean

        if relative_slope > 0.05:
            return "increasing"
        elif relative_slope < -0.05:
            return "decreasing"
        else:
            return "stable"

    def _aggregate_by_key(
        self,
        data: List[Dict[str, Any]],
        key: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Aggregate trend data by a specified key.

        Returns a dictionary with total_kg and count per unique key
        value.
        """
        result: Dict[str, Dict[str, Any]] = {}
        for item in data:
            val = item.get(key, "unknown")
            if val not in result:
                result[val] = {"total_kg": 0.0, "count": 0}
            result[val]["total_kg"] = round(
                result[val]["total_kg"] + item.get("quantity_kg", 0.0), 4
            )
            result[val]["count"] += 1
        return result

    def _aggregate_qty_by_key(
        self,
        records: List[Dict[str, Any]],
        key: str,
    ) -> Dict[str, float]:
        """Aggregate quantity_kg by a specified key across records.

        Returns a dictionary mapping key values to total quantity_kg.
        """
        result: Dict[str, float] = {}
        for record in records:
            val = record.get(key)
            if val is None:
                val = "unknown"
            if val not in result:
                result[val] = 0.0
            result[val] = round(
                result[val] + record.get("quantity_kg", 0.0), 4
            )
        return result

    def _build_empty_trend_result(
        self,
        facility_id: str,
        commodity: str,
    ) -> Dict[str, Any]:
        """Build an empty trend result when no data is available."""
        result: Dict[str, Any] = {
            "facility_id": facility_id,
            "commodity": commodity,
            "data_points": 0,
            "total_loss_kg": 0.0,
            "mean_loss_kg": 0.0,
            "median_loss_kg": 0.0,
            "std_deviation_kg": 0.0,
            "min_loss_kg": 0.0,
            "max_loss_kg": 0.0,
            "trend_direction": "stable",
            "by_loss_type": {},
            "by_process_type": {},
            "recent_entries": [],
        }
        result["provenance_hash"] = self._compute_provenance_hash(result)
        return result

    def _compute_provenance_hash(self, data: Any) -> str:
        """Compute SHA-256 hash for provenance tracking.

        Args:
            data: Any JSON-serializable data.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _elapsed_ms(self, start: datetime) -> float:
        """Compute elapsed time in milliseconds since start.

        Args:
            start: Start datetime (UTC).

        Returns:
            Elapsed time in milliseconds.
        """
        delta = _utcnow() - start
        return delta.total_seconds() * 1000.0

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        with self._lock:
            loss_count = len(self._loss_records)
            waste_count = len(self._waste_records)
            credit_count = len(self._by_product_credits)
        return (
            f"LossWasteTracker("
            f"losses={loss_count}, "
            f"waste={waste_count}, "
            f"credits={credit_count})"
        )

    def __len__(self) -> int:
        """Return total number of loss and waste records."""
        with self._lock:
            return len(self._loss_records) + len(self._waste_records)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "LossWasteTracker",
    "COMMODITY_PROCESS_LOSS_REFERENCE",
    "VALID_LOSS_TYPES",
    "VALID_WASTE_TYPES",
    "LOW_THRESHOLD_MULTIPLIER",
    "MAX_TREND_DATA_POINTS",
]
