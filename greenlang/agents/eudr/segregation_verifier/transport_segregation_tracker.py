# -*- coding: utf-8 -*-
"""
Transport Segregation Tracker Engine - AGENT-EUDR-010: Segregation Verifier (Feature 3)

Tracks, verifies, and manages transport-level segregation of EUDR-compliant vs
non-compliant commodity material. Registers transport vehicles with dedication
status, records and verifies cleaning events, tracks cargo history for
contamination risk assessment, validates seal integrity, checks route
segregation at intermediate stops, and calculates composite transport
segregation scores.

Zero-Hallucination Guarantees:
    - Cargo contamination risk is deterministic string matching on previous
      cargo types against a hardcoded risk table (no ML/LLM).
    - Cleaning verification compares cleaning_method + duration_minutes
      against per-transport-type minimum requirements (no ML/LLM).
    - Seal integrity verification is string equality and presence check
      (no ML/LLM).
    - Dedicated vehicle bonus is a fixed additive score (no ML/LLM).
    - Route segregation analysis checks each stop facility against a
      compliance status lookup (no ML/LLM).
    - Composite transport score uses deterministic weighted arithmetic
      (no ML/LLM).
    - SHA-256 provenance hashes on all records and verification results.
    - No ML/LLM used for any scoring, verification, or risk logic.

Performance Targets:
    - Vehicle registration: <5ms
    - Transport verification: <10ms
    - Cleaning record: <3ms
    - Cargo history lookup: <2ms
    - Route analysis (10 stops): <10ms
    - Score calculation: <5ms

Regulatory References:
    - EUDR Article 4: Due diligence obligations requiring segregation.
    - EUDR Article 10(2)(f): Verification of transport segregation measures.
    - EUDR Article 14: Record-keeping for transport evidence.
    - EUDR Article 31: Five-year data retention for audit trails.
    - ISO 22095:2020: Chain of custody -- Transport segregation model.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-010 (Feature 3: Transport Segregation Tracking)
Agent ID: GL-EUDR-SGV-010
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version for provenance tracking
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for audit provenance."""
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _generate_id() -> str:
    """Generate a unique identifier using UUID4."""
    return str(uuid.uuid4())

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Transport types with their cleaning requirement levels (0-1, higher = stricter).
TRANSPORT_TYPES: Dict[str, float] = {
    "bulk_truck": 0.80,
    "container_truck": 0.50,
    "tanker": 0.95,
    "dry_bulk_vessel": 0.85,
    "container_vessel": 0.40,
    "tanker_vessel": 0.95,
    "rail_hopper": 0.80,
    "rail_container": 0.50,
    "barge": 0.75,
    "air_freight": 0.40,
}

#: Cleaning methods with quality ratings (0.0-1.0).
CLEANING_METHODS: Dict[str, float] = {
    "power_wash": 0.80,
    "steam_clean": 0.90,
    "fumigation": 0.85,
    "flush": 0.75,
    "sweep_wash": 0.60,
    "compressed_air": 0.50,
    "tank_wash": 0.95,
}

#: Minimum cleaning durations per transport type and method (minutes).
#: transport_type -> {method -> min_duration_minutes}
CLEANING_DURATIONS: Dict[str, Dict[str, int]] = {
    "bulk_truck": {
        "power_wash": 30, "steam_clean": 45, "sweep_wash": 20,
        "compressed_air": 15, "fumigation": 120,
    },
    "container_truck": {
        "power_wash": 15, "steam_clean": 20, "sweep_wash": 10,
        "compressed_air": 10,
    },
    "tanker": {
        "tank_wash": 60, "steam_clean": 90, "flush": 45,
        "power_wash": 60,
    },
    "dry_bulk_vessel": {
        "power_wash": 120, "steam_clean": 180, "sweep_wash": 60,
        "compressed_air": 45, "fumigation": 360,
    },
    "container_vessel": {
        "power_wash": 20, "steam_clean": 30, "sweep_wash": 15,
    },
    "tanker_vessel": {
        "tank_wash": 180, "steam_clean": 240, "flush": 120,
    },
    "rail_hopper": {
        "power_wash": 45, "steam_clean": 60, "sweep_wash": 30,
        "compressed_air": 20, "fumigation": 180,
    },
    "rail_container": {
        "power_wash": 15, "steam_clean": 20, "sweep_wash": 10,
    },
    "barge": {
        "power_wash": 90, "steam_clean": 120, "sweep_wash": 45,
        "fumigation": 240,
    },
    "air_freight": {
        "power_wash": 15, "steam_clean": 20, "compressed_air": 10,
    },
}

#: Bonus score for dedicated vehicles.
DEDICATED_VEHICLE_BONUS: float = 20.0

#: Maximum number of previous cargoes to track.
MAX_CARGO_HISTORY: int = 5

#: Transport score weights for composite calculation.
TRANSPORT_SCORE_WEIGHTS: Dict[str, float] = {
    "cleaning": 0.30,
    "cargo_history": 0.25,
    "seal_integrity": 0.20,
    "dedication": 0.15,
    "route_compliance": 0.10,
}

#: Cargo types considered high-risk for contamination.
HIGH_RISK_CARGO_TYPES: Tuple[str, ...] = (
    "non_compliant",
    "mixed",
    "unknown",
    "chemicals",
    "waste",
)

#: Cargo types considered low-risk.
LOW_RISK_CARGO_TYPES: Tuple[str, ...] = (
    "compliant",
    "empty",
    "clean",
)

#: Valid vehicle statuses.
VALID_VEHICLE_STATUSES: Tuple[str, ...] = (
    "active", "inactive", "maintenance", "decommissioned",
)

# ---------------------------------------------------------------------------
# Internal Data Classes
# ---------------------------------------------------------------------------

@dataclass
class VehicleRecord:
    """Internal record for a transport vehicle.

    Attributes:
        vehicle_id: Unique identifier for this vehicle.
        vehicle_type: Type of transport vehicle.
        owner_operator_id: Identifier of the owner/operator.
        dedicated_status: Whether the vehicle is dedicated to compliant cargo.
        last_cargo_type: Type of the most recent cargo.
        last_cleaning_date: Date of most recent cleaning.
        cleaning_method: Method used for last cleaning.
        cargo_history: Rolling list of last N cargo types.
        status: Current vehicle status.
        verification_count: Number of verifications performed.
        last_verification_date: Date of most recent verification.
        metadata: Arbitrary key-value metadata.
        provenance_hash: SHA-256 provenance hash.
        created_at: Record creation timestamp (UTC).
    """

    vehicle_id: str = ""
    vehicle_type: str = ""
    owner_operator_id: str = ""
    dedicated_status: bool = False
    last_cargo_type: str = ""
    last_cleaning_date: Optional[datetime] = None
    cleaning_method: str = ""
    cargo_history: List[str] = field(default_factory=list)
    status: str = "active"
    verification_count: int = 0
    last_verification_date: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance_hash: str = ""
    created_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert vehicle record to dictionary for hashing."""
        return {
            "vehicle_id": self.vehicle_id,
            "vehicle_type": self.vehicle_type,
            "owner_operator_id": self.owner_operator_id,
            "dedicated_status": self.dedicated_status,
            "last_cargo_type": self.last_cargo_type,
            "last_cleaning_date": (
                str(self.last_cleaning_date) if self.last_cleaning_date else ""
            ),
            "cleaning_method": self.cleaning_method,
            "cargo_history": list(self.cargo_history),
            "status": self.status,
            "verification_count": self.verification_count,
            "last_verification_date": (
                str(self.last_verification_date)
                if self.last_verification_date else ""
            ),
            "metadata": dict(self.metadata),
            "created_at": str(self.created_at) if self.created_at else "",
        }

@dataclass
class TransportVerificationRecord:
    """Result of a transport segregation verification.

    Attributes:
        verification_id: Unique identifier for this verification.
        vehicle_id: Identifier of the verified vehicle.
        batch_id: Identifier of the batch being transported.
        route_origin: Origin facility/location identifier.
        route_destination: Destination facility/location identifier.
        cleaning_verified: Whether cleaning was verified.
        cleaning_score: Cleaning quality score (0-100).
        seal_number: Seal number applied to the vehicle/container.
        seal_intact: Whether the seal is verified intact.
        previous_cargoes: List of previous cargo types.
        cargo_risk_score: Cargo contamination risk score (0-100).
        dedication_score: Dedication status score (0-100).
        route_score: Route compliance score (0-100).
        score: Overall transport segregation score (0-100).
        timestamp: When the verification was performed (UTC).
        findings: List of verification findings.
        provenance_hash: SHA-256 provenance hash.
    """

    verification_id: str = ""
    vehicle_id: str = ""
    batch_id: str = ""
    route_origin: str = ""
    route_destination: str = ""
    cleaning_verified: bool = False
    cleaning_score: float = 0.0
    seal_number: str = ""
    seal_intact: bool = False
    previous_cargoes: List[str] = field(default_factory=list)
    cargo_risk_score: float = 0.0
    dedication_score: float = 0.0
    route_score: float = 100.0
    score: float = 0.0
    timestamp: Optional[datetime] = None
    findings: List[str] = field(default_factory=list)
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert verification record to dictionary."""
        return {
            "verification_id": self.verification_id,
            "vehicle_id": self.vehicle_id,
            "batch_id": self.batch_id,
            "route_origin": self.route_origin,
            "route_destination": self.route_destination,
            "cleaning_verified": self.cleaning_verified,
            "cleaning_score": self.cleaning_score,
            "seal_number": self.seal_number,
            "seal_intact": self.seal_intact,
            "previous_cargoes": list(self.previous_cargoes),
            "cargo_risk_score": self.cargo_risk_score,
            "dedication_score": self.dedication_score,
            "route_score": self.route_score,
            "score": self.score,
            "timestamp": str(self.timestamp) if self.timestamp else "",
            "findings": list(self.findings),
        }

@dataclass
class CleaningRecord:
    """Record of a vehicle cleaning event.

    Attributes:
        cleaning_id: Unique identifier for this cleaning.
        vehicle_id: Identifier of the cleaned vehicle.
        cleaning_method: Method used for cleaning.
        cleaning_date: Date and time of cleaning (UTC).
        duration_minutes: Duration of cleaning in minutes.
        verified_by: Identifier of the verification party.
        certificate_ref: Reference to cleaning certificate.
        compliant: Whether the cleaning meets minimum requirements.
        next_cleaning_due: Suggested date for next cleaning.
        findings: Any issues found during cleaning verification.
        provenance_hash: SHA-256 provenance hash.
    """

    cleaning_id: str = ""
    vehicle_id: str = ""
    cleaning_method: str = ""
    cleaning_date: Optional[datetime] = None
    duration_minutes: int = 0
    verified_by: str = ""
    certificate_ref: str = ""
    compliant: bool = False
    next_cleaning_due: Optional[datetime] = None
    findings: List[str] = field(default_factory=list)
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert cleaning record to dictionary."""
        return {
            "cleaning_id": self.cleaning_id,
            "vehicle_id": self.vehicle_id,
            "cleaning_method": self.cleaning_method,
            "cleaning_date": (
                str(self.cleaning_date) if self.cleaning_date else ""
            ),
            "duration_minutes": self.duration_minutes,
            "verified_by": self.verified_by,
            "certificate_ref": self.certificate_ref,
            "compliant": self.compliant,
            "next_cleaning_due": (
                str(self.next_cleaning_due) if self.next_cleaning_due else ""
            ),
            "findings": list(self.findings),
        }

# ---------------------------------------------------------------------------
# TransportSegregationTracker
# ---------------------------------------------------------------------------

class TransportSegregationTracker:
    """Production-grade transport segregation tracking engine for EUDR compliance.

    Registers transport vehicles, records cleaning events, tracks cargo
    history, verifies seal integrity, assesses contamination risk from
    previous cargoes, validates route compliance at intermediate stops,
    and calculates composite transport segregation scores.

    All operations are deterministic with zero LLM/ML involvement. Every
    result object carries a SHA-256 provenance hash for complete audit
    trail per EUDR Article 31 (5-year retention).

    Example::

        tracker = TransportSegregationTracker()
        vehicle = tracker.register_vehicle(
            vehicle_id="VH-001",
            vehicle_type="bulk_truck",
            owner_operator_id="OP-01",
            dedicated_status=True,
        )
        cleaning = tracker.record_cleaning(
            vehicle_id="VH-001",
            cleaning_method="power_wash",
            duration_minutes=45,
            verified_by="inspector-01",
        )
        result = tracker.verify_transport_segregation(
            vehicle_id="VH-001",
            batch_id="BATCH-001",
            route_origin="FAC-A",
            route_destination="FAC-B",
        )
        assert result.score >= 0.0

    Attributes:
        _vehicles: In-memory vehicle store keyed by vehicle_id.
        _cleanings: Vehicle -> [CleaningRecord, ...] cleaning log.
        _verifications: Vehicle -> [TransportVerificationRecord, ...] log.
        _dedicated_bonus: Score bonus for dedicated vehicles.
    """

    def __init__(self, config: Any = None) -> None:
        """Initialize the TransportSegregationTracker.

        Args:
            config: Optional configuration object. Supports attributes:
                - dedicated_vehicle_bonus_score (float): Bonus score for
                  dedicated vehicles (0-100).
                - max_previous_cargoes_tracked (int): Max cargo history depth.
                - cleaning_verification_required (bool): Whether cleaning
                  must be verified before transport.
        """
        self._dedicated_bonus: float = DEDICATED_VEHICLE_BONUS
        self._max_cargo_history: int = MAX_CARGO_HISTORY
        self._cleaning_required: bool = True

        if config is not None:
            self._dedicated_bonus = float(
                getattr(
                    config, "dedicated_vehicle_bonus_score",
                    DEDICATED_VEHICLE_BONUS,
                )
            )
            self._max_cargo_history = int(
                getattr(
                    config, "max_previous_cargoes_tracked",
                    MAX_CARGO_HISTORY,
                )
            )
            self._cleaning_required = bool(
                getattr(config, "cleaning_verification_required", True)
            )

        # In-memory stores
        self._vehicles: Dict[str, VehicleRecord] = {}
        self._cleanings: Dict[str, List[CleaningRecord]] = {}
        self._verifications: Dict[str, List[TransportVerificationRecord]] = {}

        logger.info(
            "TransportSegregationTracker initialized: "
            "dedicated_bonus=%.1f, max_cargo_history=%d, "
            "cleaning_required=%s",
            self._dedicated_bonus,
            self._max_cargo_history,
            self._cleaning_required,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vehicle_count(self) -> int:
        """Return total number of registered vehicles."""
        return len(self._vehicles)

    # ------------------------------------------------------------------
    # Public API: register_vehicle
    # ------------------------------------------------------------------

    def register_vehicle(
        self,
        vehicle_id: str,
        vehicle_type: str,
        owner_operator_id: str,
        dedicated_status: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> VehicleRecord:
        """Register a new transport vehicle.

        Creates a vehicle record with type, owner, and dedication status.
        The vehicle is immediately available for cleaning recording and
        transport verification.

        Args:
            vehicle_id: Unique identifier for the vehicle.
            vehicle_type: Type of transport (see TRANSPORT_TYPES keys).
            owner_operator_id: Identifier of the owner/operator.
            dedicated_status: Whether vehicle is dedicated to compliant cargo.
            metadata: Optional additional metadata.

        Returns:
            The newly registered VehicleRecord with provenance hash.

        Raises:
            ValueError: If vehicle_id already exists or inputs are invalid.
        """
        start_time = time.monotonic()

        self._validate_vehicle_inputs(
            vehicle_id, vehicle_type, owner_operator_id,
        )

        if vehicle_id in self._vehicles:
            raise ValueError(
                f"Vehicle '{vehicle_id}' already exists"
            )

        now = utcnow()
        vehicle = VehicleRecord(
            vehicle_id=vehicle_id,
            vehicle_type=vehicle_type,
            owner_operator_id=owner_operator_id,
            dedicated_status=dedicated_status,
            cargo_history=[],
            metadata=dict(metadata) if metadata else {},
            created_at=now,
        )
        vehicle.provenance_hash = _compute_hash(vehicle.to_dict())

        self._vehicles[vehicle_id] = vehicle
        self._cleanings[vehicle_id] = []
        self._verifications[vehicle_id] = []

        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        logger.info(
            "Registered vehicle %s [type=%s, dedicated=%s, owner=%s] in %.2fms",
            vehicle_id, vehicle_type, dedicated_status,
            owner_operator_id, elapsed_ms,
        )

        return vehicle

    # ------------------------------------------------------------------
    # Public API: get_vehicle
    # ------------------------------------------------------------------

    def get_vehicle(self, vehicle_id: str) -> Optional[VehicleRecord]:
        """Retrieve a vehicle record by identifier.

        Args:
            vehicle_id: The vehicle identifier to look up.

        Returns:
            The VehicleRecord if found, None otherwise.
        """
        return self._vehicles.get(vehicle_id)

    # ------------------------------------------------------------------
    # Public API: update_vehicle
    # ------------------------------------------------------------------

    def update_vehicle(
        self,
        vehicle_id: str,
        updates: Dict[str, Any],
    ) -> VehicleRecord:
        """Update an existing vehicle record.

        Args:
            vehicle_id: Identifier of the vehicle to update.
            updates: Dictionary of field_name -> new_value pairs.
                Allowed fields: vehicle_type, owner_operator_id,
                dedicated_status, status, metadata.

        Returns:
            Updated VehicleRecord with new provenance hash.

        Raises:
            ValueError: If vehicle not found or no valid updates.
        """
        vehicle = self._vehicles.get(vehicle_id)
        if vehicle is None:
            raise ValueError(f"Vehicle '{vehicle_id}' not found")

        allowed_fields = {
            "vehicle_type", "owner_operator_id", "dedicated_status",
            "status", "metadata",
        }
        applicable = {
            k: v for k, v in updates.items() if k in allowed_fields
        }
        if not applicable:
            raise ValueError("No valid fields to update")

        for field_name, new_value in applicable.items():
            if field_name == "vehicle_type" and new_value not in TRANSPORT_TYPES:
                raise ValueError(
                    f"vehicle_type must be one of "
                    f"{sorted(TRANSPORT_TYPES.keys())}, got '{new_value}'"
                )
            setattr(vehicle, field_name, new_value)

        vehicle.provenance_hash = _compute_hash(vehicle.to_dict())

        logger.info(
            "Updated vehicle %s: fields=%s",
            vehicle_id, sorted(applicable.keys()),
        )

        return vehicle

    # ------------------------------------------------------------------
    # Public API: verify_transport_segregation
    # ------------------------------------------------------------------

    def verify_transport_segregation(
        self,
        vehicle_id: str,
        batch_id: str,
        route_origin: str,
        route_destination: str,
        seal_number: str = "",
        route_stops: Optional[List[Dict[str, Any]]] = None,
    ) -> TransportVerificationRecord:
        """Verify transport segregation for a vehicle-batch combination.

        Performs a comprehensive deterministic verification covering
        cleaning status, previous cargo contamination risk, seal
        integrity, dedication status, and route compliance. Calculates
        a composite transport segregation score.

        Args:
            vehicle_id: Identifier of the transport vehicle.
            batch_id: Identifier of the batch being transported.
            route_origin: Origin facility/location identifier.
            route_destination: Destination facility/location identifier.
            seal_number: Optional seal number for integrity check.
            route_stops: Optional list of intermediate stop dicts with
                keys: facility_id, compliance_status.

        Returns:
            TransportVerificationRecord with scores and findings.

        Raises:
            ValueError: If vehicle not found.
        """
        start_time = time.monotonic()

        vehicle = self._vehicles.get(vehicle_id)
        if vehicle is None:
            raise ValueError(f"Vehicle '{vehicle_id}' not found")

        findings: List[str] = []

        # 1. Cleaning verification
        cleaning_result = self._verify_cleaning_status(vehicle)
        cleaning_score = cleaning_result["score"]
        cleaning_verified = cleaning_result["verified"]
        findings.extend(cleaning_result.get("findings", []))

        # 2. Cargo history risk
        cargo_result = self.assess_cargo_contamination_risk(vehicle_id)
        cargo_risk_score = cargo_result.get("risk_score", 0.0)
        # Convert risk score to quality score (higher = better)
        cargo_quality = max(0.0, 100.0 - cargo_risk_score)
        findings.extend(cargo_result.get("findings", []))

        # 3. Seal integrity
        seal_result = self.verify_seal_integrity(vehicle_id, seal_number)
        seal_score = seal_result.get("score", 0.0)
        seal_intact = seal_result.get("intact", False)
        findings.extend(seal_result.get("findings", []))

        # 4. Dedication status
        dedication_result = self.check_dedicated_status(vehicle_id)
        dedication_score = dedication_result.get("score", 0.0)
        findings.extend(dedication_result.get("findings", []))

        # 5. Route compliance
        route_score = 100.0
        if route_stops:
            route_result = self.analyze_route_segregation(
                vehicle_id, route_stops,
            )
            route_score = route_result.get("route_score", 100.0)
            findings.extend(route_result.get("findings", []))

        # Composite score
        composite = (
            cleaning_score * TRANSPORT_SCORE_WEIGHTS["cleaning"]
            + cargo_quality * TRANSPORT_SCORE_WEIGHTS["cargo_history"]
            + seal_score * TRANSPORT_SCORE_WEIGHTS["seal_integrity"]
            + dedication_score * TRANSPORT_SCORE_WEIGHTS["dedication"]
            + route_score * TRANSPORT_SCORE_WEIGHTS["route_compliance"]
        )

        # Apply dedication bonus
        if vehicle.dedicated_status:
            composite = min(100.0, composite + self._dedicated_bonus)

        composite = round(composite, 2)

        now = utcnow()
        verification = TransportVerificationRecord(
            verification_id=_generate_id(),
            vehicle_id=vehicle_id,
            batch_id=batch_id,
            route_origin=route_origin,
            route_destination=route_destination,
            cleaning_verified=cleaning_verified,
            cleaning_score=round(cleaning_score, 2),
            seal_number=seal_number,
            seal_intact=seal_intact,
            previous_cargoes=list(vehicle.cargo_history),
            cargo_risk_score=round(cargo_risk_score, 2),
            dedication_score=round(dedication_score, 2),
            route_score=round(route_score, 2),
            score=composite,
            timestamp=now,
            findings=findings,
        )
        verification.provenance_hash = _compute_hash(verification.to_dict())

        # Update vehicle state
        vehicle.verification_count += 1
        vehicle.last_verification_date = now
        vehicle.last_cargo_type = "compliant"

        # Add to cargo history (rolling window)
        vehicle.cargo_history.append("compliant")
        if len(vehicle.cargo_history) > self._max_cargo_history:
            vehicle.cargo_history = vehicle.cargo_history[
                -self._max_cargo_history:
            ]

        vehicle.provenance_hash = _compute_hash(vehicle.to_dict())

        # Store verification
        self._verifications[vehicle_id].append(verification)

        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        logger.info(
            "Verified transport for vehicle %s batch %s: score=%.1f, "
            "cleaning=%s, seal=%s, findings=%d in %.2fms",
            vehicle_id, batch_id, composite, cleaning_verified,
            seal_intact, len(findings), elapsed_ms,
        )

        return verification

    # ------------------------------------------------------------------
    # Public API: record_cleaning
    # ------------------------------------------------------------------

    def record_cleaning(
        self,
        vehicle_id: str,
        cleaning_method: str,
        duration_minutes: int,
        verified_by: str = "",
        certificate_ref: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CleaningRecord:
        """Record a cleaning event for a transport vehicle.

        Validates the cleaning method and duration against minimum
        requirements for the vehicle type. Updates the vehicle's
        cleaning status.

        Args:
            vehicle_id: Identifier of the vehicle.
            cleaning_method: Method used (see CLEANING_METHODS keys).
            duration_minutes: Duration of cleaning in minutes (> 0).
            verified_by: Identifier of the verification party.
            certificate_ref: Reference to cleaning certificate.
            metadata: Optional additional metadata.

        Returns:
            The recorded CleaningRecord with compliance status.

        Raises:
            ValueError: If vehicle not found, method invalid, or
                duration non-positive.
        """
        start_time = time.monotonic()

        vehicle = self._vehicles.get(vehicle_id)
        if vehicle is None:
            raise ValueError(f"Vehicle '{vehicle_id}' not found")

        if cleaning_method not in CLEANING_METHODS:
            raise ValueError(
                f"cleaning_method must be one of "
                f"{sorted(CLEANING_METHODS.keys())}, got '{cleaning_method}'"
            )

        if duration_minutes <= 0:
            raise ValueError(
                f"duration_minutes must be > 0, got {duration_minutes}"
            )

        # Check compliance against minimum requirements
        compliance_result = self._check_cleaning_compliance(
            vehicle.vehicle_type, cleaning_method, duration_minutes,
        )
        compliant = compliance_result["compliant"]
        findings = compliance_result.get("findings", [])

        now = utcnow()
        record = CleaningRecord(
            cleaning_id=_generate_id(),
            vehicle_id=vehicle_id,
            cleaning_method=cleaning_method,
            cleaning_date=now,
            duration_minutes=duration_minutes,
            verified_by=verified_by,
            certificate_ref=certificate_ref,
            compliant=compliant,
            next_cleaning_due=now + timedelta(days=30),
            findings=findings,
        )
        record.provenance_hash = _compute_hash(record.to_dict())

        # Update vehicle cleaning status
        vehicle.last_cleaning_date = now
        vehicle.cleaning_method = cleaning_method
        vehicle.provenance_hash = _compute_hash(vehicle.to_dict())

        # Store
        self._cleanings[vehicle_id].append(record)

        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        logger.info(
            "Recorded cleaning for vehicle %s [method=%s, "
            "duration=%dmin, compliant=%s] in %.2fms",
            vehicle_id, cleaning_method, duration_minutes,
            compliant, elapsed_ms,
        )

        return record

    # ------------------------------------------------------------------
    # Public API: check_previous_cargoes
    # ------------------------------------------------------------------

    def check_previous_cargoes(
        self,
        vehicle_id: str,
        depth: int = 5,
    ) -> List[str]:
        """Return the last N cargo types for a vehicle.

        Args:
            vehicle_id: Identifier of the vehicle.
            depth: Number of previous cargoes to return (default 5).

        Returns:
            List of cargo type strings, most recent first.

        Raises:
            ValueError: If vehicle not found.
        """
        vehicle = self._vehicles.get(vehicle_id)
        if vehicle is None:
            raise ValueError(f"Vehicle '{vehicle_id}' not found")

        history = list(vehicle.cargo_history)
        history.reverse()
        return history[:depth]

    # ------------------------------------------------------------------
    # Public API: assess_cargo_contamination_risk
    # ------------------------------------------------------------------

    def assess_cargo_contamination_risk(
        self,
        vehicle_id: str,
    ) -> Dict[str, Any]:
        """Assess contamination risk based on cargo history.

        Uses deterministic string matching against HIGH_RISK_CARGO_TYPES
        and LOW_RISK_CARGO_TYPES to compute a risk score. Each high-risk
        entry in the cargo history adds risk points.

        Args:
            vehicle_id: Identifier of the vehicle.

        Returns:
            Dictionary with keys:
                risk_score: Score from 0 (no risk) to 100 (maximum risk).
                risk_level: Deterministic level (low, medium, high, critical).
                high_risk_count: Number of high-risk entries.
                total_entries: Total cargo history entries.
                findings: List of risk findings.

        Raises:
            ValueError: If vehicle not found.
        """
        vehicle = self._vehicles.get(vehicle_id)
        if vehicle is None:
            raise ValueError(f"Vehicle '{vehicle_id}' not found")

        history = vehicle.cargo_history
        findings: List[str] = []

        if not history:
            return {
                "risk_score": 0.0,
                "risk_level": "low",
                "high_risk_count": 0,
                "total_entries": 0,
                "findings": ["INFO: No cargo history available"],
            }

        high_risk_count = sum(
            1 for cargo in history if cargo in HIGH_RISK_CARGO_TYPES
        )
        total = len(history)

        # Risk score: each high-risk entry contributes proportionally
        # Max risk at 100% high-risk history
        risk_ratio = high_risk_count / total if total > 0 else 0.0

        # Weight recent entries more heavily (most recent = highest weight)
        weighted_risk = 0.0
        for idx, cargo in enumerate(reversed(history)):
            weight = 1.0 + (idx * 0.5)  # Older entries get higher index
            if cargo in HIGH_RISK_CARGO_TYPES:
                # More recent = higher weight for risk
                recency_weight = len(history) - idx
                weighted_risk += recency_weight

        max_weighted = sum(range(1, len(history) + 1))
        risk_score = (
            (weighted_risk / max_weighted * 100.0) if max_weighted > 0 else 0.0
        )
        risk_score = min(100.0, round(risk_score, 2))

        # Classify risk level
        if risk_score >= 75.0:
            risk_level = "critical"
            findings.append(
                f"CRITICAL: {high_risk_count}/{total} recent cargoes "
                f"are high-risk"
            )
        elif risk_score >= 50.0:
            risk_level = "high"
            findings.append(
                f"WARNING: {high_risk_count}/{total} recent cargoes "
                f"are high-risk"
            )
        elif risk_score >= 25.0:
            risk_level = "medium"
            findings.append(
                f"INFO: {high_risk_count}/{total} recent cargoes "
                f"are high-risk"
            )
        else:
            risk_level = "low"

        # Check most recent cargo specifically
        if history and history[-1] in HIGH_RISK_CARGO_TYPES:
            findings.append(
                f"WARNING: Most recent cargo was '{history[-1]}' - "
                f"cleaning verification essential"
            )

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "high_risk_count": high_risk_count,
            "total_entries": total,
            "findings": findings,
        }

    # ------------------------------------------------------------------
    # Public API: verify_seal_integrity
    # ------------------------------------------------------------------

    def verify_seal_integrity(
        self,
        vehicle_id: str,
        seal_number: str,
    ) -> Dict[str, Any]:
        """Verify the integrity of a transport seal.

        Checks that a seal number is provided, is non-empty, and follows
        a basic format. Seal verification is presence and format checking
        only (deterministic, no external lookup).

        Args:
            vehicle_id: Identifier of the vehicle.
            seal_number: Seal number string.

        Returns:
            Dictionary with keys:
                intact: Whether the seal passes verification.
                score: Seal quality score (0-100).
                seal_number: The seal number.
                findings: List of findings.

        Raises:
            ValueError: If vehicle not found.
        """
        vehicle = self._vehicles.get(vehicle_id)
        if vehicle is None:
            raise ValueError(f"Vehicle '{vehicle_id}' not found")

        findings: List[str] = []

        if not seal_number or not seal_number.strip():
            return {
                "intact": False,
                "score": 0.0,
                "seal_number": "",
                "findings": [
                    "CRITICAL: No seal number provided - "
                    "transport segregation cannot be verified"
                ],
            }

        seal_number = seal_number.strip()
        score = 100.0
        intact = True

        # Basic format check (minimum length)
        if len(seal_number) < 4:
            score -= 30.0
            findings.append(
                f"WARNING: Seal number '{seal_number}' is unusually short"
            )

        # Check for common invalid patterns
        if seal_number.lower() in ("none", "n/a", "tbd", "pending", "unknown"):
            score = 0.0
            intact = False
            findings.append(
                f"CRITICAL: Seal number '{seal_number}' is a placeholder, "
                f"not a valid seal"
            )

        return {
            "intact": intact,
            "score": max(0.0, round(score, 2)),
            "seal_number": seal_number,
            "findings": findings,
        }

    # ------------------------------------------------------------------
    # Public API: check_dedicated_status
    # ------------------------------------------------------------------

    def check_dedicated_status(
        self,
        vehicle_id: str,
    ) -> Dict[str, Any]:
        """Check if a vehicle is truly dedicated to compliant cargo.

        Verifies the dedicated_status flag and cross-references against
        cargo history. A vehicle claiming dedicated status but with
        non-compliant cargo in its history is flagged.

        Args:
            vehicle_id: Identifier of the vehicle.

        Returns:
            Dictionary with keys:
                dedicated: Whether the vehicle is dedicated.
                verified: Whether dedicated status is confirmed by history.
                score: Dedication score (0-100).
                findings: List of findings.

        Raises:
            ValueError: If vehicle not found.
        """
        vehicle = self._vehicles.get(vehicle_id)
        if vehicle is None:
            raise ValueError(f"Vehicle '{vehicle_id}' not found")

        findings: List[str] = []
        dedicated = vehicle.dedicated_status

        if not dedicated:
            return {
                "dedicated": False,
                "verified": True,
                "score": 50.0,
                "findings": [
                    "INFO: Vehicle is not dedicated to compliant cargo"
                ],
            }

        # Check cargo history for non-compliant entries
        non_compliant_in_history = any(
            cargo in HIGH_RISK_CARGO_TYPES
            for cargo in vehicle.cargo_history
        )

        if non_compliant_in_history:
            findings.append(
                "WARNING: Vehicle is marked as dedicated but has "
                "non-compliant cargo in history"
            )
            return {
                "dedicated": True,
                "verified": False,
                "score": 60.0,
                "findings": findings,
            }

        return {
            "dedicated": True,
            "verified": True,
            "score": 100.0,
            "findings": [
                "INFO: Dedicated status confirmed - "
                "no non-compliant cargo in history"
            ],
        }

    # ------------------------------------------------------------------
    # Public API: calculate_transport_score
    # ------------------------------------------------------------------

    def calculate_transport_score(
        self,
        vehicle_id: str,
    ) -> float:
        """Calculate the composite transport segregation score for a vehicle.

        Uses the most recent verification score. If no verifications
        exist, performs a basic assessment based on vehicle attributes.

        Args:
            vehicle_id: Identifier of the vehicle.

        Returns:
            Transport segregation score (0.0-100.0).

        Raises:
            ValueError: If vehicle not found.
        """
        vehicle = self._vehicles.get(vehicle_id)
        if vehicle is None:
            raise ValueError(f"Vehicle '{vehicle_id}' not found")

        verifications = self._verifications.get(vehicle_id, [])
        if verifications:
            return verifications[-1].score

        # No verifications - calculate from vehicle attributes
        dedication_score = 100.0 if vehicle.dedicated_status else 50.0
        cleaning_score = 50.0 if vehicle.last_cleaning_date else 0.0

        cargo_result = self.assess_cargo_contamination_risk(vehicle_id)
        cargo_risk = cargo_result.get("risk_score", 50.0)
        cargo_quality = max(0.0, 100.0 - cargo_risk)

        composite = (
            cleaning_score * 0.35
            + cargo_quality * 0.30
            + dedication_score * 0.20
            + 50.0 * 0.15  # Default seal/route score
        )

        if vehicle.dedicated_status:
            composite = min(100.0, composite + self._dedicated_bonus)

        return round(composite, 2)

    # ------------------------------------------------------------------
    # Public API: get_vehicle_history
    # ------------------------------------------------------------------

    def get_vehicle_history(
        self,
        vehicle_id: str,
    ) -> List[Dict[str, Any]]:
        """Return the full verification history for a vehicle.

        Args:
            vehicle_id: Identifier of the vehicle.

        Returns:
            List of verification dictionaries, most recent first.

        Raises:
            ValueError: If vehicle not found.
        """
        if vehicle_id not in self._vehicles:
            raise ValueError(f"Vehicle '{vehicle_id}' not found")

        verifications = self._verifications.get(vehicle_id, [])
        result = [v.to_dict() for v in reversed(verifications)]
        return result

    # ------------------------------------------------------------------
    # Public API: get_cleaning_history
    # ------------------------------------------------------------------

    def get_cleaning_history(
        self,
        vehicle_id: str,
    ) -> List[Dict[str, Any]]:
        """Return the full cleaning history for a vehicle.

        Args:
            vehicle_id: Identifier of the vehicle.

        Returns:
            List of cleaning dictionaries, most recent first.

        Raises:
            ValueError: If vehicle not found.
        """
        if vehicle_id not in self._vehicles:
            raise ValueError(f"Vehicle '{vehicle_id}' not found")

        cleanings = self._cleanings.get(vehicle_id, [])
        result = [c.to_dict() for c in reversed(cleanings)]
        return result

    # ------------------------------------------------------------------
    # Public API: analyze_route_segregation
    # ------------------------------------------------------------------

    def analyze_route_segregation(
        self,
        vehicle_id: str,
        route_stops: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Analyze route segregation at intermediate stops.

        Checks each intermediate stop's compliance_status to determine
        whether the vehicle may be contaminated by stopping at
        non-compliant facilities.

        Args:
            vehicle_id: Identifier of the vehicle.
            route_stops: List of stop dictionaries, each with keys:
                facility_id (str), compliance_status (str).

        Returns:
            Dictionary with keys:
                route_score: Score from 0 (all non-compliant) to 100
                    (all compliant).
                non_compliant_stops: Number of non-compliant stops.
                total_stops: Total number of stops.
                stop_details: List of per-stop assessment dicts.
                findings: List of findings.

        Raises:
            ValueError: If vehicle not found.
        """
        vehicle = self._vehicles.get(vehicle_id)
        if vehicle is None:
            raise ValueError(f"Vehicle '{vehicle_id}' not found")

        if not route_stops:
            return {
                "route_score": 100.0,
                "non_compliant_stops": 0,
                "total_stops": 0,
                "stop_details": [],
                "findings": ["INFO: No intermediate stops on route"],
            }

        findings: List[str] = []
        stop_details: List[Dict[str, Any]] = []
        non_compliant_count = 0

        for stop in route_stops:
            fac_id = stop.get("facility_id", "unknown")
            status = stop.get("compliance_status", "unknown")

            detail = {
                "facility_id": fac_id,
                "compliance_status": status,
                "risk": "none",
            }

            if status == "non_compliant":
                non_compliant_count += 1
                detail["risk"] = "high"
                findings.append(
                    f"WARNING: Route includes non-compliant facility '{fac_id}'"
                )
            elif status in ("pending", "unknown"):
                detail["risk"] = "medium"
                findings.append(
                    f"INFO: Route includes facility '{fac_id}' with "
                    f"'{status}' compliance status"
                )

            stop_details.append(detail)

        total = len(route_stops)
        if total > 0:
            compliant_ratio = 1.0 - (non_compliant_count / total)
            route_score = round(compliant_ratio * 100.0, 2)
        else:
            route_score = 100.0

        return {
            "route_score": route_score,
            "non_compliant_stops": non_compliant_count,
            "total_stops": total,
            "stop_details": stop_details,
            "findings": findings,
        }

    # ------------------------------------------------------------------
    # Public API: get_summary_statistics
    # ------------------------------------------------------------------

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Return aggregate statistics for all vehicles.

        Returns:
            Dictionary with vehicle counts, verification counts, and
            cleaning counts.
        """
        by_type: Dict[str, int] = {}
        dedicated_count = 0
        total_verifications = 0
        total_cleanings = 0

        for vehicle in self._vehicles.values():
            by_type[vehicle.vehicle_type] = (
                by_type.get(vehicle.vehicle_type, 0) + 1
            )
            if vehicle.dedicated_status:
                dedicated_count += 1

        for v_list in self._verifications.values():
            total_verifications += len(v_list)

        for c_list in self._cleanings.values():
            total_cleanings += len(c_list)

        return {
            "total_vehicles": len(self._vehicles),
            "dedicated_vehicles": dedicated_count,
            "by_vehicle_type": by_type,
            "total_verifications": total_verifications,
            "total_cleanings": total_cleanings,
        }

    # ------------------------------------------------------------------
    # Public API: clear
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all vehicles, cleanings, and verifications. For testing."""
        self._vehicles.clear()
        self._cleanings.clear()
        self._verifications.clear()
        logger.info("TransportSegregationTracker cleared all data")

    # ------------------------------------------------------------------
    # Internal: _validate_vehicle_inputs
    # ------------------------------------------------------------------

    def _validate_vehicle_inputs(
        self,
        vehicle_id: str,
        vehicle_type: str,
        owner_operator_id: str,
    ) -> None:
        """Validate vehicle registration inputs.

        Raises:
            ValueError: If any input fails validation.
        """
        errors: List[str] = []

        if not vehicle_id or not vehicle_id.strip():
            errors.append("vehicle_id must not be empty")

        if vehicle_type not in TRANSPORT_TYPES:
            errors.append(
                f"vehicle_type must be one of "
                f"{sorted(TRANSPORT_TYPES.keys())}, got '{vehicle_type}'"
            )

        if not owner_operator_id or not owner_operator_id.strip():
            errors.append("owner_operator_id must not be empty")

        if errors:
            raise ValueError(
                "Vehicle registration validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    # ------------------------------------------------------------------
    # Internal: _verify_cleaning_status
    # ------------------------------------------------------------------

    def _verify_cleaning_status(
        self,
        vehicle: VehicleRecord,
    ) -> Dict[str, Any]:
        """Verify the cleaning status of a vehicle.

        Args:
            vehicle: The vehicle record to check.

        Returns:
            Dictionary with score, verified flag, and findings.
        """
        findings: List[str] = []

        if vehicle.last_cleaning_date is None:
            return {
                "score": 0.0,
                "verified": False,
                "findings": [
                    "CRITICAL: Vehicle has never been cleaned - "
                    "cleaning required before transport"
                ],
            }

        now = utcnow()
        days_since = (now - vehicle.last_cleaning_date).days

        # Check if cleaning is recent enough (within 30 days)
        if days_since > 30:
            findings.append(
                f"WARNING: Last cleaning was {days_since} days ago - "
                f"re-cleaning recommended"
            )
            score = max(0.0, 50.0 - (days_since - 30) * 2.0)
        elif days_since > 7:
            score = 70.0
            findings.append(
                f"INFO: Last cleaning was {days_since} days ago"
            )
        else:
            score = 100.0

        # Check cleaning method quality
        method_quality = CLEANING_METHODS.get(vehicle.cleaning_method, 0.5)
        transport_requirement = TRANSPORT_TYPES.get(
            vehicle.vehicle_type, 0.5,
        )

        if method_quality < transport_requirement:
            penalty = (transport_requirement - method_quality) * 50.0
            score = max(0.0, score - penalty)
            findings.append(
                f"WARNING: Cleaning method '{vehicle.cleaning_method}' "
                f"may be insufficient for {vehicle.vehicle_type}"
            )

        return {
            "score": round(score, 2),
            "verified": score >= 50.0,
            "days_since_cleaning": days_since,
            "findings": findings,
        }

    # ------------------------------------------------------------------
    # Internal: _check_cleaning_compliance
    # ------------------------------------------------------------------

    def _check_cleaning_compliance(
        self,
        vehicle_type: str,
        cleaning_method: str,
        duration_minutes: int,
    ) -> Dict[str, Any]:
        """Check if cleaning meets minimum requirements.

        Args:
            vehicle_type: Type of vehicle.
            cleaning_method: Cleaning method used.
            duration_minutes: Cleaning duration in minutes.

        Returns:
            Dictionary with compliant flag and findings.
        """
        findings: List[str] = []

        # Look up minimum duration
        type_durations = CLEANING_DURATIONS.get(vehicle_type, {})
        min_duration = type_durations.get(cleaning_method, 0)

        if min_duration > 0 and duration_minutes < min_duration:
            findings.append(
                f"WARNING: Cleaning duration {duration_minutes}min is below "
                f"minimum {min_duration}min for {vehicle_type}/{cleaning_method}"
            )
            compliant = False
        elif min_duration == 0:
            # Method not in the lookup for this vehicle type
            # Check if ANY method is listed (method may be inappropriate)
            if type_durations:
                findings.append(
                    f"INFO: Cleaning method '{cleaning_method}' is not a "
                    f"standard method for {vehicle_type}. Standard methods: "
                    f"{sorted(type_durations.keys())}"
                )
            compliant = True
        else:
            compliant = True

        # Check method quality vs transport requirement
        method_quality = CLEANING_METHODS.get(cleaning_method, 0.5)
        transport_req = TRANSPORT_TYPES.get(vehicle_type, 0.5)

        if method_quality < transport_req:
            findings.append(
                f"WARNING: Cleaning method '{cleaning_method}' quality "
                f"({method_quality:.2f}) is below requirement for "
                f"{vehicle_type} ({transport_req:.2f})"
            )
            compliant = False

        return {
            "compliant": compliant,
            "min_duration": min_duration,
            "actual_duration": duration_minutes,
            "findings": findings,
        }

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return total number of registered vehicles."""
        return self.vehicle_count

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return (
            f"TransportSegregationTracker("
            f"vehicles={self.vehicle_count}, "
            f"dedicated_bonus={self._dedicated_bonus})"
        )

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Module version
    "_MODULE_VERSION",
    # Constants
    "TRANSPORT_TYPES",
    "CLEANING_METHODS",
    "CLEANING_DURATIONS",
    "DEDICATED_VEHICLE_BONUS",
    "MAX_CARGO_HISTORY",
    "TRANSPORT_SCORE_WEIGHTS",
    "HIGH_RISK_CARGO_TYPES",
    "LOW_RISK_CARGO_TYPES",
    "VALID_VEHICLE_STATUSES",
    # Data classes
    "VehicleRecord",
    "TransportVerificationRecord",
    "CleaningRecord",
    # Engine class
    "TransportSegregationTracker",
]
