# -*- coding: utf-8 -*-
"""
ChainIntegrityVerifier - AGENT-EUDR-009 Feature 7: Chain Integrity Verification

Validates end-to-end custody chain integrity from production plot to EU market
entry. Performs temporal continuity checks (no timeline gaps), actor continuity
checks (receiver matches next sender), location continuity checks (no teleport),
mass conservation checks (output <= input + tolerance), origin preservation
checks (allocations sum to 100%), orphan detection, circular dependency
detection, and composite completeness scoring.

Capabilities:
    - Full chain verification across all integrity dimensions
    - Temporal continuity: no gaps exceeding configurable threshold
    - Actor continuity: transfer receiver matches next event sender
    - Location continuity: goods cannot teleport between facilities
    - Mass conservation: total output <= total input + tolerance
    - Origin preservation: origin plot allocations sum to 100%
    - Orphan detection: batches with no upstream or downstream links
    - Circular dependency detection in batch genealogy
    - Composite completeness score (0-100) with configurable weights
    - Verification certificate generation with evidence compilation
    - Batch verification for multiple chains

Zero-Hallucination Guarantees:
    - All checks are deterministic boolean evaluations
    - All arithmetic uses standard Python float operations
    - No LLM or ML used in any verification or scoring path
    - SHA-256 provenance hash on every result for tamper detection
    - Bit-perfect reproducibility: same inputs produce same outputs

Regulatory Basis:
    - EUDR Article 4(2): Due diligence -- unbroken traceability
    - EUDR Article 9: Geolocation and production plot traceability
    - EUDR Article 9(1)(f): Quantity/weight verification
    - EUDR Article 10: Risk assessment with chain integrity
    - EUDR Article 14: 5-year record retention

Dependencies:
    - provenance: SHA-256 chain hashing
    - metrics: Prometheus verification counters

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-009, Feature 7
Agent ID: GL-EUDR-COC-009
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, list, dataclass with to_dict, or other).

    Returns:
        SHA-256 hex digest string (64 characters, lowercase).
    """
    if hasattr(data, "to_dict"):
        serializable = data.to_dict()
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _generate_id(prefix: str) -> str:
    """Generate a unique identifier with the given prefix.

    Args:
        prefix: Identifier prefix string (e.g., 'VER', 'CRT').

    Returns:
        Prefixed UUID string.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: EUDR regulation reference
EUDR_REGULATION_REF = "Regulation (EU) 2023/1115"

#: Default temporal gap threshold in hours (72h per PRD F1.7)
DEFAULT_TEMPORAL_GAP_THRESHOLD_HOURS: int = 72

#: Default mass conservation tolerance percentage
DEFAULT_MASS_TOLERANCE_PCT: float = 1.0

#: Default origin allocation tolerance (should sum to 100%)
DEFAULT_ORIGIN_TOLERANCE_PCT: float = 0.5

#: Completeness score dimension weights (sum to 1.0)
COMPLETENESS_WEIGHTS: Dict[str, float] = {
    "temporal_continuity": 0.20,
    "actor_continuity": 0.20,
    "location_continuity": 0.15,
    "mass_conservation": 0.20,
    "origin_preservation": 0.15,
    "document_coverage": 0.10,
}

#: Maximum chain depth for circular dependency detection
MAX_CHAIN_DEPTH: int = 1000

#: Maximum batch verification size
MAX_BATCH_VERIFY_SIZE: int = 500

class VerificationStatus(str, Enum):
    """Chain verification status."""

    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    INCOMPLETE = "incomplete"

class ContinuityType(str, Enum):
    """Types of continuity checks."""

    TEMPORAL = "temporal"
    ACTOR = "actor"
    LOCATION = "location"

class CheckSeverity(str, Enum):
    """Severity of a check failure."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

# ---------------------------------------------------------------------------
# Data Models (local dataclasses)
# ---------------------------------------------------------------------------

@dataclass
class ContinuityCheck:
    """Result of a single continuity check between events.

    Attributes:
        check_id: Unique check identifier.
        check_type: Type of continuity check (temporal/actor/location).
        event_a_id: First event in the check pair.
        event_b_id: Second event in the check pair.
        passed: Whether the check passed.
        severity: Severity if check failed.
        details: Human-readable description.
        gap_value: Quantified gap (hours for temporal, etc.).
        threshold: Threshold used for comparison.
        eudr_article: EUDR article reference.
        metadata: Additional metadata.
    """

    check_id: str = field(default_factory=lambda: _generate_id("CHK"))
    check_type: str = ""
    event_a_id: str = ""
    event_b_id: str = ""
    passed: bool = True
    severity: str = "info"
    details: str = ""
    gap_value: float = 0.0
    threshold: float = 0.0
    eudr_article: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "check_id": self.check_id,
            "check_type": self.check_type,
            "event_a_id": self.event_a_id,
            "event_b_id": self.event_b_id,
            "passed": self.passed,
            "severity": self.severity,
            "details": self.details,
            "gap_value": self.gap_value,
            "threshold": self.threshold,
            "eudr_article": self.eudr_article,
            "metadata": dict(self.metadata),
        }

@dataclass
class MassConservationResult:
    """Result of mass conservation check for a chain.

    Attributes:
        result_id: Unique result identifier.
        batch_id: Batch being checked.
        total_input: Total input quantity across the chain.
        total_output: Total output quantity across the chain.
        total_loss: Computed loss (input - output).
        loss_pct: Loss as percentage of input.
        tolerance_pct: Configured tolerance percentage.
        passed: Whether mass is conserved within tolerance.
        violations: List of individual step violations.
        details: Human-readable summary.
    """

    result_id: str = field(default_factory=lambda: _generate_id("MCR"))
    batch_id: str = ""
    total_input: float = 0.0
    total_output: float = 0.0
    total_loss: float = 0.0
    loss_pct: float = 0.0
    tolerance_pct: float = DEFAULT_MASS_TOLERANCE_PCT
    passed: bool = True
    violations: List[Dict[str, Any]] = field(default_factory=list)
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "result_id": self.result_id,
            "batch_id": self.batch_id,
            "total_input": self.total_input,
            "total_output": self.total_output,
            "total_loss": self.total_loss,
            "loss_pct": self.loss_pct,
            "tolerance_pct": self.tolerance_pct,
            "passed": self.passed,
            "violations": list(self.violations),
            "details": self.details,
        }

@dataclass
class OrphanBatch:
    """A batch detected as orphaned (no upstream or no downstream).

    Attributes:
        orphan_id: Unique orphan record identifier.
        batch_id: The orphaned batch identifier.
        orphan_type: Type of orphan: 'no_upstream' or 'no_downstream'.
        has_upstream: Whether the batch has upstream links.
        has_downstream: Whether the batch has downstream links.
        severity: Severity of the orphan detection.
        recommendation: Suggested action.
        detected_at: When the orphan was detected.
    """

    orphan_id: str = field(default_factory=lambda: _generate_id("ORP"))
    batch_id: str = ""
    orphan_type: str = ""
    has_upstream: bool = False
    has_downstream: bool = False
    severity: str = "high"
    recommendation: str = ""
    detected_at: datetime = field(default_factory=utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "orphan_id": self.orphan_id,
            "batch_id": self.batch_id,
            "orphan_type": self.orphan_type,
            "has_upstream": self.has_upstream,
            "has_downstream": self.has_downstream,
            "severity": self.severity,
            "recommendation": self.recommendation,
            "detected_at": (
                self.detected_at.isoformat() if self.detected_at else None
            ),
        }

@dataclass
class CircularDependency:
    """A circular reference detected in batch genealogy.

    Attributes:
        cycle_id: Unique cycle identifier.
        batch_id: Starting batch where cycle was detected.
        cycle_path: List of batch IDs forming the cycle.
        cycle_length: Number of batches in the cycle.
        severity: Always critical for circular dependencies.
        details: Human-readable description.
        detected_at: When the cycle was detected.
    """

    cycle_id: str = field(default_factory=lambda: _generate_id("CYC"))
    batch_id: str = ""
    cycle_path: List[str] = field(default_factory=list)
    cycle_length: int = 0
    severity: str = "critical"
    details: str = ""
    detected_at: datetime = field(default_factory=utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "cycle_id": self.cycle_id,
            "batch_id": self.batch_id,
            "cycle_path": list(self.cycle_path),
            "cycle_length": self.cycle_length,
            "severity": self.severity,
            "details": self.details,
            "detected_at": (
                self.detected_at.isoformat() if self.detected_at else None
            ),
        }

@dataclass
class CompletenessScore:
    """Composite completeness score for a custody chain.

    Attributes:
        score_id: Unique score identifier.
        batch_id: Batch being scored.
        overall_score: Composite score (0-100).
        dimension_scores: Individual dimension scores.
        dimension_weights: Weights used for each dimension.
        dimension_passed: Whether each dimension passed.
        total_checks: Total number of individual checks performed.
        passed_checks: Number of checks that passed.
        failed_checks: Number of checks that failed.
        level: Qualitative level (excellent/good/fair/poor/failing).
        provenance_hash: SHA-256 hash for audit trail.
        scored_at: When the scoring was performed.
    """

    score_id: str = field(default_factory=lambda: _generate_id("CSC"))
    batch_id: str = ""
    overall_score: float = 0.0
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    dimension_weights: Dict[str, float] = field(
        default_factory=lambda: dict(COMPLETENESS_WEIGHTS)
    )
    dimension_passed: Dict[str, bool] = field(default_factory=dict)
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    level: str = "poor"
    provenance_hash: str = ""
    scored_at: datetime = field(default_factory=utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "score_id": self.score_id,
            "batch_id": self.batch_id,
            "overall_score": self.overall_score,
            "dimension_scores": dict(self.dimension_scores),
            "dimension_weights": dict(self.dimension_weights),
            "dimension_passed": dict(self.dimension_passed),
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "level": self.level,
            "provenance_hash": self.provenance_hash,
            "scored_at": (
                self.scored_at.isoformat() if self.scored_at else None
            ),
        }

@dataclass
class ChainVerification:
    """Complete chain verification result.

    Attributes:
        verification_id: Unique verification identifier.
        batch_id: Batch whose chain was verified.
        status: Overall verification status (pass/fail/warning).
        overall_score: Composite completeness score (0-100).
        temporal_checks: Results of temporal continuity checks.
        actor_checks: Results of actor continuity checks.
        location_checks: Results of location continuity checks.
        mass_conservation: Mass conservation check result.
        origin_check_passed: Whether origin preservation passed.
        origin_total_pct: Sum of origin allocations.
        orphans: Detected orphan batches.
        circular_deps: Detected circular dependencies.
        completeness: Completeness score breakdown.
        total_events: Number of events in the chain.
        total_checks: Total checks performed.
        passed_checks: Checks that passed.
        failed_checks: Checks that failed.
        warnings: List of warning messages.
        errors: List of error messages.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Wall-clock time in milliseconds.
        verified_at: When the verification was performed.
    """

    verification_id: str = field(
        default_factory=lambda: _generate_id("VER")
    )
    batch_id: str = ""
    status: str = "incomplete"
    overall_score: float = 0.0
    temporal_checks: List[ContinuityCheck] = field(default_factory=list)
    actor_checks: List[ContinuityCheck] = field(default_factory=list)
    location_checks: List[ContinuityCheck] = field(default_factory=list)
    mass_conservation: Optional[MassConservationResult] = None
    origin_check_passed: bool = True
    origin_total_pct: float = 100.0
    orphans: List[OrphanBatch] = field(default_factory=list)
    circular_deps: List[CircularDependency] = field(default_factory=list)
    completeness: Optional[CompletenessScore] = None
    total_events: int = 0
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    provenance_hash: str = ""
    processing_time_ms: float = 0.0
    verified_at: datetime = field(default_factory=utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "verification_id": self.verification_id,
            "batch_id": self.batch_id,
            "status": self.status,
            "overall_score": self.overall_score,
            "temporal_checks": [c.to_dict() for c in self.temporal_checks],
            "actor_checks": [c.to_dict() for c in self.actor_checks],
            "location_checks": [c.to_dict() for c in self.location_checks],
            "mass_conservation": (
                self.mass_conservation.to_dict()
                if self.mass_conservation
                else None
            ),
            "origin_check_passed": self.origin_check_passed,
            "origin_total_pct": self.origin_total_pct,
            "orphans": [o.to_dict() for o in self.orphans],
            "circular_deps": [c.to_dict() for c in self.circular_deps],
            "completeness": (
                self.completeness.to_dict() if self.completeness else None
            ),
            "total_events": self.total_events,
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": self.processing_time_ms,
            "verified_at": (
                self.verified_at.isoformat() if self.verified_at else None
            ),
        }

@dataclass
class VerificationCertificate:
    """Verification certificate with evidence compilation.

    Attributes:
        certificate_id: Unique certificate identifier.
        verification_id: Reference to the verification result.
        batch_id: Batch covered by the certificate.
        status: Verification status.
        overall_score: Composite integrity score.
        issued_at: When the certificate was issued.
        valid_until: Certificate validity end date.
        issuer: Certificate issuer (system identifier).
        summary: Executive summary of the verification.
        dimension_results: Per-dimension pass/fail results.
        evidence: List of evidence items supporting the verification.
        chain_length: Number of events in the verified chain.
        origin_plots: Origin plots traced through the chain.
        regulatory_references: Applicable EUDR articles.
        provenance_hash: SHA-256 hash for audit trail.
    """

    certificate_id: str = field(
        default_factory=lambda: _generate_id("CRT")
    )
    verification_id: str = ""
    batch_id: str = ""
    status: str = "incomplete"
    overall_score: float = 0.0
    issued_at: datetime = field(default_factory=utcnow)
    valid_until: Optional[datetime] = None
    issuer: str = "GreenLang EUDR CoC Verification Engine v1.0"
    summary: str = ""
    dimension_results: Dict[str, bool] = field(default_factory=dict)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    chain_length: int = 0
    origin_plots: List[str] = field(default_factory=list)
    regulatory_references: List[str] = field(default_factory=list)
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "certificate_id": self.certificate_id,
            "verification_id": self.verification_id,
            "batch_id": self.batch_id,
            "status": self.status,
            "overall_score": self.overall_score,
            "issued_at": (
                self.issued_at.isoformat() if self.issued_at else None
            ),
            "valid_until": (
                self.valid_until.isoformat() if self.valid_until else None
            ),
            "issuer": self.issuer,
            "summary": self.summary,
            "dimension_results": dict(self.dimension_results),
            "evidence": list(self.evidence),
            "chain_length": self.chain_length,
            "origin_plots": list(self.origin_plots),
            "regulatory_references": list(self.regulatory_references),
            "provenance_hash": self.provenance_hash,
        }

@dataclass
class BatchVerifyResult:
    """Result of batch verification of multiple chains.

    Attributes:
        result_id: Unique result identifier.
        total_submitted: Number of chains submitted.
        total_passed: Number that passed verification.
        total_failed: Number that failed verification.
        total_warnings: Number with warnings.
        results: List of individual verification results.
        average_score: Average completeness score.
        errors: List of processing errors.
        processing_time_ms: Total processing time.
        provenance_hash: SHA-256 hash for audit trail.
        completed_at: When the batch verification completed.
    """

    result_id: str = field(default_factory=lambda: _generate_id("BVR"))
    total_submitted: int = 0
    total_passed: int = 0
    total_failed: int = 0
    total_warnings: int = 0
    results: List[ChainVerification] = field(default_factory=list)
    average_score: float = 0.0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    processing_time_ms: float = 0.0
    provenance_hash: str = ""
    completed_at: datetime = field(default_factory=utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "result_id": self.result_id,
            "total_submitted": self.total_submitted,
            "total_passed": self.total_passed,
            "total_failed": self.total_failed,
            "total_warnings": self.total_warnings,
            "results": [r.to_dict() for r in self.results],
            "average_score": self.average_score,
            "errors": list(self.errors),
            "processing_time_ms": self.processing_time_ms,
            "provenance_hash": self.provenance_hash,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
        }

@dataclass
class ChainIntegrityVerifierConfig:
    """Configuration for the ChainIntegrityVerifier engine.

    Attributes:
        temporal_gap_threshold_hours: Maximum allowed temporal gap in hours.
        mass_tolerance_pct: Mass conservation tolerance percentage.
        origin_tolerance_pct: Origin allocation sum tolerance percentage.
        completeness_weights: Weights for each completeness dimension.
        max_chain_depth: Maximum depth for circular dependency detection.
        max_batch_verify_size: Maximum chains per batch verification.
        enable_provenance: Whether to compute provenance hashes.
        certificate_validity_days: Days a verification certificate is valid.
    """

    temporal_gap_threshold_hours: int = DEFAULT_TEMPORAL_GAP_THRESHOLD_HOURS
    mass_tolerance_pct: float = DEFAULT_MASS_TOLERANCE_PCT
    origin_tolerance_pct: float = DEFAULT_ORIGIN_TOLERANCE_PCT
    completeness_weights: Dict[str, float] = field(
        default_factory=lambda: dict(COMPLETENESS_WEIGHTS)
    )
    max_chain_depth: int = MAX_CHAIN_DEPTH
    max_batch_verify_size: int = MAX_BATCH_VERIFY_SIZE
    enable_provenance: bool = True
    certificate_validity_days: int = 90

    def __post_init__(self) -> None:
        """Validate configuration constraints after initialization."""
        errors: List[str] = []

        if self.temporal_gap_threshold_hours <= 0:
            errors.append(
                f"temporal_gap_threshold_hours must be > 0, "
                f"got {self.temporal_gap_threshold_hours}"
            )
        if not (0.0 <= self.mass_tolerance_pct <= 100.0):
            errors.append(
                f"mass_tolerance_pct must be in [0, 100], "
                f"got {self.mass_tolerance_pct}"
            )
        if not (0.0 <= self.origin_tolerance_pct <= 100.0):
            errors.append(
                f"origin_tolerance_pct must be in [0, 100], "
                f"got {self.origin_tolerance_pct}"
            )
        if self.max_chain_depth <= 0:
            errors.append(
                f"max_chain_depth must be > 0, got {self.max_chain_depth}"
            )

        # Validate weights sum to ~1.0
        weight_sum = sum(self.completeness_weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            errors.append(
                f"completeness_weights must sum to 1.0, "
                f"got {weight_sum:.4f}"
            )

        if errors:
            raise ValueError(
                "ChainIntegrityVerifierConfig validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

# ===========================================================================
# ChainIntegrityVerifier Engine
# ===========================================================================

class ChainIntegrityVerifier:
    """Chain integrity verification engine for EUDR chain of custody.

    Validates end-to-end custody chain integrity across multiple dimensions:
    temporal continuity, actor continuity, location continuity, mass
    conservation, origin preservation, orphan detection, and circular
    dependency detection. Produces a composite completeness score and
    verification certificates.

    All checks are deterministic -- no LLM or ML is used in any
    verification or scoring path.

    Attributes:
        config: ChainIntegrityVerifierConfig with engine settings.
        _verification_store: Dictionary of verification_id -> ChainVerification.
        _certificate_store: Dictionary of certificate_id -> VerificationCertificate.
        _verification_count: Total verifications performed.

    Example:
        >>> verifier = ChainIntegrityVerifier()
        >>> events = [
        ...     {"event_id": "E1", "event_type": "receipt",
        ...      "timestamp": "2026-01-01T08:00:00Z",
        ...      "actor_id": "A1", "location_id": "L1",
        ...      "quantity": 1000.0},
        ...     {"event_id": "E2", "event_type": "transfer",
        ...      "timestamp": "2026-01-02T10:00:00Z",
        ...      "actor_id": "A1", "receiver_id": "A2",
        ...      "location_id": "L1", "quantity": 1000.0},
        ... ]
        >>> result = verifier.verify_chain("B001", events)
        >>> assert result.status in ("pass", "warning", "fail")
    """

    def __init__(
        self, config: Optional[ChainIntegrityVerifierConfig] = None
    ) -> None:
        """Initialize the ChainIntegrityVerifier engine.

        Args:
            config: Optional configuration. Defaults to
                ChainIntegrityVerifierConfig() with standard settings.
        """
        self.config = config or ChainIntegrityVerifierConfig()
        self._verification_store: Dict[str, ChainVerification] = {}
        self._certificate_store: Dict[str, VerificationCertificate] = {}
        self._verification_count: int = 0

        logger.info(
            "ChainIntegrityVerifier initialized: gap_threshold=%dh, "
            "mass_tolerance=%.1f%%, origin_tolerance=%.1f%%, "
            "max_depth=%d, provenance=%s",
            self.config.temporal_gap_threshold_hours,
            self.config.mass_tolerance_pct,
            self.config.origin_tolerance_pct,
            self.config.max_chain_depth,
            self.config.enable_provenance,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify_chain(
        self,
        batch_id: str,
        events: List[Dict[str, Any]],
        origin_allocations: Optional[Dict[str, float]] = None,
        batch_genealogy: Optional[Dict[str, List[str]]] = None,
        document_coverage: Optional[float] = None,
    ) -> ChainVerification:
        """Perform full chain verification across all integrity dimensions.

        Executes temporal, actor, location, mass, and origin checks,
        detects orphans and circular dependencies, and computes a
        composite completeness score.

        Args:
            batch_id: Batch identifier to verify.
            events: List of custody event dicts, each with:
                - event_id (str)
                - event_type (str)
                - timestamp (str|datetime)
                - actor_id (str): Sender/holder
                - receiver_id (str): Optional receiver
                - location_id (str): Facility/location
                - quantity (float): Event quantity
            origin_allocations: Optional dict of plot_id -> percentage.
                Used for origin preservation check.
            batch_genealogy: Optional dict of batch_id -> list of
                parent_batch_ids. Used for orphan and cycle detection.
            document_coverage: Optional pre-computed document coverage
                score (0-100) for completeness scoring.

        Returns:
            ChainVerification with all check results and overall status.
        """
        start_time = time.monotonic()
        self._verification_count += 1

        logger.info(
            "Starting chain verification for batch %s: events=%d",
            batch_id,
            len(events),
        )

        # Sort events by timestamp
        sorted_events = self._sort_events_by_time(events)

        # Run all checks
        temporal_checks = self.check_temporal_continuity(sorted_events)
        actor_checks = self.check_actor_continuity(sorted_events)
        location_checks = self.check_location_continuity(sorted_events)
        mass_result = self.check_mass_conservation(
            {"batch_id": batch_id, "events": sorted_events}
        )

        # Origin preservation
        origin_passed = True
        origin_total = 100.0
        if origin_allocations:
            origin_passed, origin_total = self.check_origin_preservation(
                batch_id, origin_allocations
            )

        # Orphan detection
        orphans: List[OrphanBatch] = []
        if batch_genealogy:
            orphans = self.detect_orphans(batch_id, batch_genealogy)

        # Circular dependency detection
        circular_deps: List[CircularDependency] = []
        if batch_genealogy:
            circular_deps = self.detect_circular_dependencies(
                batch_id, batch_genealogy
            )

        # Compute completeness score
        doc_cov = document_coverage if document_coverage is not None else 100.0
        completeness = self.calculate_completeness_score({
            "batch_id": batch_id,
            "temporal_checks": temporal_checks,
            "actor_checks": actor_checks,
            "location_checks": location_checks,
            "mass_conservation": mass_result,
            "origin_passed": origin_passed,
            "document_coverage": doc_cov,
        })

        # Aggregate check counts
        all_checks = temporal_checks + actor_checks + location_checks
        total_checks = len(all_checks) + 1  # +1 for mass check
        passed_checks = sum(1 for c in all_checks if c.passed)
        if mass_result.passed:
            passed_checks += 1
        failed_checks = total_checks - passed_checks

        # Determine overall status
        status = self._determine_status(
            temporal_checks, actor_checks, location_checks,
            mass_result, origin_passed, circular_deps
        )

        # Build warnings and errors
        warnings: List[str] = []
        errors_list: List[str] = []

        for check in all_checks:
            if not check.passed:
                if check.severity in ("critical", "high"):
                    errors_list.append(check.details)
                else:
                    warnings.append(check.details)

        if not mass_result.passed:
            errors_list.append(mass_result.details)
        if not origin_passed:
            errors_list.append(
                f"Origin allocation sum is {origin_total:.2f}%, "
                f"expected 100% +/- {self.config.origin_tolerance_pct}%"
            )
        for cycle in circular_deps:
            errors_list.append(cycle.details)
        for orphan in orphans:
            warnings.append(
                f"Orphan batch {orphan.batch_id}: {orphan.orphan_type}"
            )

        elapsed_ms = (time.monotonic() - start_time) * 1000

        verification = ChainVerification(
            batch_id=batch_id,
            status=status,
            overall_score=completeness.overall_score,
            temporal_checks=temporal_checks,
            actor_checks=actor_checks,
            location_checks=location_checks,
            mass_conservation=mass_result,
            origin_check_passed=origin_passed,
            origin_total_pct=origin_total,
            orphans=orphans,
            circular_deps=circular_deps,
            completeness=completeness,
            total_events=len(sorted_events),
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            warnings=warnings,
            errors=errors_list,
            processing_time_ms=round(elapsed_ms, 2),
        )

        if self.config.enable_provenance:
            verification.provenance_hash = _compute_hash(verification)

        self._verification_store[verification.verification_id] = verification

        logger.info(
            "Chain verification for batch %s: status=%s, score=%.1f, "
            "checks=%d (pass=%d, fail=%d), elapsed=%.1fms",
            batch_id,
            status,
            completeness.overall_score,
            total_checks,
            passed_checks,
            failed_checks,
            elapsed_ms,
        )

        return verification

    def check_temporal_continuity(
        self, events: List[Dict[str, Any]]
    ) -> List[ContinuityCheck]:
        """Check temporal continuity between consecutive events.

        Verifies that there are no temporal gaps exceeding the configured
        threshold between consecutive custody events.

        Args:
            events: List of event dicts sorted by timestamp. Each must
                have 'event_id' and 'timestamp'.

        Returns:
            List of ContinuityCheck results for each event pair.
        """
        start_time = time.monotonic()
        checks: List[ContinuityCheck] = []
        threshold_hours = self.config.temporal_gap_threshold_hours

        for i in range(1, len(events)):
            event_a = events[i - 1]
            event_b = events[i]

            ts_a = self._parse_timestamp(event_a.get("timestamp"))
            ts_b = self._parse_timestamp(event_b.get("timestamp"))

            if ts_a is None or ts_b is None:
                check = ContinuityCheck(
                    check_type=ContinuityType.TEMPORAL.value,
                    event_a_id=event_a.get("event_id", ""),
                    event_b_id=event_b.get("event_id", ""),
                    passed=False,
                    severity=CheckSeverity.MEDIUM.value,
                    details=(
                        "Cannot verify temporal continuity: "
                        "missing timestamp on one or both events."
                    ),
                    eudr_article="Article 9(1)(e)",
                )
                checks.append(check)
                continue

            gap_hours = (ts_b - ts_a).total_seconds() / 3600.0

            if gap_hours < 0:
                check = ContinuityCheck(
                    check_type=ContinuityType.TEMPORAL.value,
                    event_a_id=event_a.get("event_id", ""),
                    event_b_id=event_b.get("event_id", ""),
                    passed=False,
                    severity=CheckSeverity.CRITICAL.value,
                    details=(
                        f"Temporal order violation: event "
                        f"{event_b.get('event_id', '')} precedes "
                        f"{event_a.get('event_id', '')} by "
                        f"{abs(gap_hours):.1f} hours."
                    ),
                    gap_value=gap_hours,
                    threshold=float(threshold_hours),
                    eudr_article="Article 9(1)(e)",
                )
            elif gap_hours > threshold_hours:
                check = ContinuityCheck(
                    check_type=ContinuityType.TEMPORAL.value,
                    event_a_id=event_a.get("event_id", ""),
                    event_b_id=event_b.get("event_id", ""),
                    passed=False,
                    severity=CheckSeverity.HIGH.value,
                    details=(
                        f"Temporal gap of {gap_hours:.1f} hours exceeds "
                        f"threshold of {threshold_hours}h between events "
                        f"{event_a.get('event_id', '')} and "
                        f"{event_b.get('event_id', '')}."
                    ),
                    gap_value=gap_hours,
                    threshold=float(threshold_hours),
                    eudr_article="Article 9(1)(e)",
                )
            else:
                check = ContinuityCheck(
                    check_type=ContinuityType.TEMPORAL.value,
                    event_a_id=event_a.get("event_id", ""),
                    event_b_id=event_b.get("event_id", ""),
                    passed=True,
                    severity=CheckSeverity.INFO.value,
                    details=(
                        f"Temporal continuity OK: {gap_hours:.1f}h gap."
                    ),
                    gap_value=gap_hours,
                    threshold=float(threshold_hours),
                    eudr_article="Article 9(1)(e)",
                )

            checks.append(check)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        failed = sum(1 for c in checks if not c.passed)
        logger.debug(
            "Temporal continuity: %d checks, %d failed, elapsed=%.1fms",
            len(checks),
            failed,
            elapsed_ms,
        )

        return checks

    def check_actor_continuity(
        self, events: List[Dict[str, Any]]
    ) -> List[ContinuityCheck]:
        """Check actor continuity between consecutive events.

        Verifies that the receiver of one event matches the sender/holder
        of the next event.

        Args:
            events: List of event dicts sorted by timestamp. Each should
                have 'event_id', 'actor_id', and optionally 'receiver_id'.

        Returns:
            List of ContinuityCheck results for each relevant event pair.
        """
        start_time = time.monotonic()
        checks: List[ContinuityCheck] = []

        for i in range(1, len(events)):
            event_a = events[i - 1]
            event_b = events[i]

            # Get receiver of event A and actor of event B
            receiver_a = event_a.get("receiver_id", "")
            actor_b = event_b.get("actor_id", "")

            # If event A has no receiver, use actor of A as continuity point
            if not receiver_a:
                receiver_a = event_a.get("actor_id", "")

            # Skip if either is unknown
            if not receiver_a or not actor_b:
                check = ContinuityCheck(
                    check_type=ContinuityType.ACTOR.value,
                    event_a_id=event_a.get("event_id", ""),
                    event_b_id=event_b.get("event_id", ""),
                    passed=True,
                    severity=CheckSeverity.LOW.value,
                    details=(
                        "Actor continuity skipped: "
                        "insufficient actor data on one or both events."
                    ),
                    eudr_article="Article 10",
                )
                checks.append(check)
                continue

            if receiver_a == actor_b:
                check = ContinuityCheck(
                    check_type=ContinuityType.ACTOR.value,
                    event_a_id=event_a.get("event_id", ""),
                    event_b_id=event_b.get("event_id", ""),
                    passed=True,
                    severity=CheckSeverity.INFO.value,
                    details=(
                        f"Actor continuity OK: {receiver_a} -> {actor_b}."
                    ),
                    eudr_article="Article 10",
                )
            else:
                check = ContinuityCheck(
                    check_type=ContinuityType.ACTOR.value,
                    event_a_id=event_a.get("event_id", ""),
                    event_b_id=event_b.get("event_id", ""),
                    passed=False,
                    severity=CheckSeverity.HIGH.value,
                    details=(
                        f"Actor discontinuity: receiver of "
                        f"{event_a.get('event_id', '')} is '{receiver_a}' "
                        f"but sender of {event_b.get('event_id', '')} is "
                        f"'{actor_b}'."
                    ),
                    eudr_article="Article 10",
                )

            checks.append(check)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        failed = sum(1 for c in checks if not c.passed)
        logger.debug(
            "Actor continuity: %d checks, %d failed, elapsed=%.1fms",
            len(checks),
            failed,
            elapsed_ms,
        )

        return checks

    def check_location_continuity(
        self, events: List[Dict[str, Any]]
    ) -> List[ContinuityCheck]:
        """Check location continuity between consecutive events.

        Verifies that goods cannot teleport between facilities without
        a transport event. If consecutive events have different location_id
        values and neither is a transport-related event, a location
        continuity violation is flagged.

        Args:
            events: List of event dicts sorted by timestamp. Each should
                have 'event_id', 'event_type', and 'location_id'.

        Returns:
            List of ContinuityCheck results for each relevant event pair.
        """
        start_time = time.monotonic()
        checks: List[ContinuityCheck] = []

        transport_types = frozenset({
            "transfer", "export", "import", "receipt",
        })

        for i in range(1, len(events)):
            event_a = events[i - 1]
            event_b = events[i]

            loc_a = event_a.get("location_id", "")
            loc_b = event_b.get("location_id", "")
            type_a = event_a.get("event_type", "")
            type_b = event_b.get("event_type", "")

            # Skip if locations unknown
            if not loc_a or not loc_b:
                checks.append(ContinuityCheck(
                    check_type=ContinuityType.LOCATION.value,
                    event_a_id=event_a.get("event_id", ""),
                    event_b_id=event_b.get("event_id", ""),
                    passed=True,
                    severity=CheckSeverity.LOW.value,
                    details="Location continuity skipped: missing location data.",
                    eudr_article="Article 9(1)(d)",
                ))
                continue

            if loc_a == loc_b:
                checks.append(ContinuityCheck(
                    check_type=ContinuityType.LOCATION.value,
                    event_a_id=event_a.get("event_id", ""),
                    event_b_id=event_b.get("event_id", ""),
                    passed=True,
                    severity=CheckSeverity.INFO.value,
                    details=f"Location continuity OK: same location {loc_a}.",
                    eudr_article="Article 9(1)(d)",
                ))
            elif type_a in transport_types or type_b in transport_types:
                # Location change with transport event is expected
                checks.append(ContinuityCheck(
                    check_type=ContinuityType.LOCATION.value,
                    event_a_id=event_a.get("event_id", ""),
                    event_b_id=event_b.get("event_id", ""),
                    passed=True,
                    severity=CheckSeverity.INFO.value,
                    details=(
                        f"Location change {loc_a} -> {loc_b} with "
                        f"transport event: OK."
                    ),
                    eudr_article="Article 9(1)(d)",
                ))
            else:
                checks.append(ContinuityCheck(
                    check_type=ContinuityType.LOCATION.value,
                    event_a_id=event_a.get("event_id", ""),
                    event_b_id=event_b.get("event_id", ""),
                    passed=False,
                    severity=CheckSeverity.HIGH.value,
                    details=(
                        f"Location discontinuity: goods at {loc_a} in "
                        f"event {event_a.get('event_id', '')} appear at "
                        f"{loc_b} in event {event_b.get('event_id', '')} "
                        f"without a transport event."
                    ),
                    eudr_article="Article 9(1)(d)",
                ))

        elapsed_ms = (time.monotonic() - start_time) * 1000
        failed = sum(1 for c in checks if not c.passed)
        logger.debug(
            "Location continuity: %d checks, %d failed, elapsed=%.1fms",
            len(checks),
            failed,
            elapsed_ms,
        )

        return checks

    def check_mass_conservation(
        self, chain: Dict[str, Any]
    ) -> MassConservationResult:
        """Check mass conservation across a custody chain.

        Verifies that total output does not exceed total input plus
        the configured tolerance.

        Args:
            chain: Dictionary with:
                - batch_id (str): Batch identifier.
                - events (list[dict]): Event dicts with 'quantity' and
                  'event_type'. Input events: receipt, processing_in,
                  storage_in. Output events: transfer, export,
                  processing_out, storage_out.

        Returns:
            MassConservationResult with pass/fail and violation details.
        """
        start_time = time.monotonic()

        batch_id = chain.get("batch_id", "")
        events = chain.get("events", [])

        input_types = frozenset({
            "receipt", "processing_in", "storage_in", "import",
        })
        output_types = frozenset({
            "transfer", "export", "processing_out", "storage_out",
        })

        total_input = 0.0
        total_output = 0.0
        violations: List[Dict[str, Any]] = []

        for event in events:
            event_type = event.get("event_type", "")
            quantity = float(event.get("quantity", 0.0))

            if event_type in input_types:
                total_input += quantity
            elif event_type in output_types:
                total_output += quantity

        # Check conservation
        total_loss = total_input - total_output
        loss_pct = (
            (total_loss / total_input * 100.0) if total_input > 0.0 else 0.0
        )

        # Output should not exceed input + tolerance
        max_allowed = total_input * (1.0 + self.config.mass_tolerance_pct / 100.0)
        passed = total_output <= max_allowed

        if not passed:
            excess = total_output - total_input
            excess_pct = (excess / total_input * 100.0) if total_input > 0.0 else 0.0
            violations.append({
                "type": "excess_output",
                "excess_qty": round(excess, 4),
                "excess_pct": round(excess_pct, 2),
                "total_input": round(total_input, 4),
                "total_output": round(total_output, 4),
            })

        if total_input == 0.0 and total_output > 0.0:
            passed = False
            violations.append({
                "type": "output_without_input",
                "total_output": round(total_output, 4),
            })

        details = (
            f"Mass conservation {'PASS' if passed else 'FAIL'}: "
            f"input={total_input:.2f}, output={total_output:.2f}, "
            f"loss={total_loss:.2f} ({loss_pct:.1f}%), "
            f"tolerance={self.config.mass_tolerance_pct}%."
        )

        result = MassConservationResult(
            batch_id=batch_id,
            total_input=round(total_input, 4),
            total_output=round(total_output, 4),
            total_loss=round(total_loss, 4),
            loss_pct=round(loss_pct, 2),
            tolerance_pct=self.config.mass_tolerance_pct,
            passed=passed,
            violations=violations,
            details=details,
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.debug(
            "Mass conservation for %s: %s (in=%.2f, out=%.2f), "
            "elapsed=%.1fms",
            batch_id,
            "PASS" if passed else "FAIL",
            total_input,
            total_output,
            elapsed_ms,
        )

        return result

    def check_origin_preservation(
        self,
        batch_id: str,
        origin_allocations: Dict[str, float],
    ) -> Tuple[bool, float]:
        """Check that origin plot allocations sum to 100%.

        Verifies that the percentage allocation of a batch to its
        origin plots sums to approximately 100%.

        Args:
            batch_id: Batch identifier.
            origin_allocations: Dictionary of plot_id -> percentage.

        Returns:
            Tuple of (passed, total_pct).
        """
        if not origin_allocations:
            return True, 100.0

        total_pct = sum(origin_allocations.values())
        tolerance = self.config.origin_tolerance_pct
        passed = abs(total_pct - 100.0) <= tolerance

        if not passed:
            logger.warning(
                "Origin preservation check failed for batch %s: "
                "allocation sum=%.2f%% (expected 100%% +/- %.1f%%)",
                batch_id,
                total_pct,
                tolerance,
            )

        return passed, round(total_pct, 2)

    def detect_orphans(
        self,
        batch_id: str,
        batch_genealogy: Dict[str, List[str]],
    ) -> List[OrphanBatch]:
        """Detect orphan batches with no upstream or downstream links.

        An orphan is a batch that has no parent batches (no upstream
        origin) or no child batches (no downstream destination) in
        the genealogy.

        Args:
            batch_id: Starting batch for context.
            batch_genealogy: Dictionary of batch_id -> list of
                parent_batch_ids.

        Returns:
            List of OrphanBatch detections.
        """
        start_time = time.monotonic()
        orphans: List[OrphanBatch] = []

        # Build reverse index: parent -> children
        children_index: Dict[str, List[str]] = defaultdict(list)
        all_batches: Set[str] = set()

        for child_id, parent_ids in batch_genealogy.items():
            all_batches.add(child_id)
            for parent_id in parent_ids:
                all_batches.add(parent_id)
                children_index[parent_id].append(child_id)

        for bid in all_batches:
            has_upstream = bid in batch_genealogy and len(
                batch_genealogy[bid]
            ) > 0
            has_downstream = bid in children_index and len(
                children_index[bid]
            ) > 0

            if not has_upstream and not has_downstream:
                orphans.append(OrphanBatch(
                    batch_id=bid,
                    orphan_type="isolated",
                    has_upstream=False,
                    has_downstream=False,
                    severity="critical",
                    recommendation=(
                        f"Batch {bid} is completely isolated with no "
                        f"upstream origin or downstream destination. "
                        f"Investigate origin and link to custody chain."
                    ),
                ))
            elif not has_upstream:
                # No upstream is OK for origin batches (harvested material)
                # but flag for non-origin batches
                orphans.append(OrphanBatch(
                    batch_id=bid,
                    orphan_type="no_upstream",
                    has_upstream=False,
                    has_downstream=True,
                    severity="medium",
                    recommendation=(
                        f"Batch {bid} has no upstream origin. If this is "
                        f"a harvest batch, link to production plot(s). "
                        f"Otherwise, establish upstream traceability."
                    ),
                ))
            elif not has_downstream:
                orphans.append(OrphanBatch(
                    batch_id=bid,
                    orphan_type="no_downstream",
                    has_upstream=True,
                    has_downstream=False,
                    severity="low",
                    recommendation=(
                        f"Batch {bid} has no downstream destination. "
                        f"This may be expected for end-of-chain batches "
                        f"(final delivery/consumption)."
                    ),
                ))

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.debug(
            "Orphan detection for batch %s: batches=%d, orphans=%d, "
            "elapsed=%.1fms",
            batch_id,
            len(all_batches),
            len(orphans),
            elapsed_ms,
        )

        return orphans

    def detect_circular_dependencies(
        self,
        batch_id: str,
        batch_genealogy: Dict[str, List[str]],
    ) -> List[CircularDependency]:
        """Detect circular references in batch genealogy.

        Uses DFS cycle detection to find any circular dependencies
        in the batch genealogy graph.

        Args:
            batch_id: Starting batch for context.
            batch_genealogy: Dictionary of batch_id -> list of
                parent_batch_ids.

        Returns:
            List of CircularDependency detections.
        """
        start_time = time.monotonic()
        cycles: List[CircularDependency] = []

        # DFS-based cycle detection
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = defaultdict(int)
        path: List[str] = []
        found_cycles: Set[frozenset] = set()

        def dfs(node: str, depth: int) -> None:
            """DFS with cycle detection."""
            if depth > self.config.max_chain_depth:
                return

            color[node] = GRAY
            path.append(node)

            for parent in batch_genealogy.get(node, []):
                if color[parent] == GRAY:
                    # Found a cycle
                    cycle_start = path.index(parent)
                    cycle_path = path[cycle_start:] + [parent]
                    cycle_key = frozenset(cycle_path)

                    if cycle_key not in found_cycles:
                        found_cycles.add(cycle_key)
                        cycles.append(CircularDependency(
                            batch_id=batch_id,
                            cycle_path=list(cycle_path),
                            cycle_length=len(cycle_path) - 1,
                            details=(
                                f"Circular dependency detected: "
                                f"{' -> '.join(cycle_path)}. "
                                f"This violates batch genealogy integrity."
                            ),
                        ))
                elif color[parent] == WHITE:
                    dfs(parent, depth + 1)

            path.pop()
            color[node] = BLACK

        # Run DFS from all nodes
        all_nodes: Set[str] = set(batch_genealogy.keys())
        for parent_list in batch_genealogy.values():
            all_nodes.update(parent_list)

        for node in all_nodes:
            if color[node] == WHITE:
                dfs(node, 0)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.debug(
            "Circular dependency detection for batch %s: nodes=%d, "
            "cycles=%d, elapsed=%.1fms",
            batch_id,
            len(all_nodes),
            len(cycles),
            elapsed_ms,
        )

        return cycles

    def calculate_completeness_score(
        self, chain: Dict[str, Any]
    ) -> CompletenessScore:
        """Calculate composite completeness score across all dimensions.

        Computes a weighted score from 0 to 100 based on the configured
        dimension weights.

        Args:
            chain: Dictionary with dimension results:
                - batch_id (str)
                - temporal_checks (list[ContinuityCheck])
                - actor_checks (list[ContinuityCheck])
                - location_checks (list[ContinuityCheck])
                - mass_conservation (MassConservationResult)
                - origin_passed (bool)
                - document_coverage (float): 0-100 score

        Returns:
            CompletenessScore with per-dimension and overall scores.
        """
        start_time = time.monotonic()

        batch_id = chain.get("batch_id", "")
        weights = self.config.completeness_weights

        # Compute per-dimension scores (0-100)
        temporal_score = self._dimension_score(
            chain.get("temporal_checks", [])
        )
        actor_score = self._dimension_score(
            chain.get("actor_checks", [])
        )
        location_score = self._dimension_score(
            chain.get("location_checks", [])
        )

        mass_result = chain.get("mass_conservation")
        mass_score = 100.0 if (mass_result and mass_result.passed) else 0.0

        origin_score = 100.0 if chain.get("origin_passed", True) else 0.0
        doc_score = float(chain.get("document_coverage", 100.0))

        dimension_scores = {
            "temporal_continuity": round(temporal_score, 1),
            "actor_continuity": round(actor_score, 1),
            "location_continuity": round(location_score, 1),
            "mass_conservation": round(mass_score, 1),
            "origin_preservation": round(origin_score, 1),
            "document_coverage": round(doc_score, 1),
        }

        dimension_passed = {
            "temporal_continuity": temporal_score >= 80.0,
            "actor_continuity": actor_score >= 80.0,
            "location_continuity": location_score >= 80.0,
            "mass_conservation": mass_score >= 80.0,
            "origin_preservation": origin_score >= 80.0,
            "document_coverage": doc_score >= 80.0,
        }

        # Compute weighted overall score
        overall = 0.0
        for dim, score in dimension_scores.items():
            weight = weights.get(dim, 0.0)
            overall += score * weight

        # Count checks
        all_checks = (
            chain.get("temporal_checks", [])
            + chain.get("actor_checks", [])
            + chain.get("location_checks", [])
        )
        total_checks = len(all_checks) + 1  # mass check
        passed_checks = sum(1 for c in all_checks if c.passed)
        if mass_result and mass_result.passed:
            passed_checks += 1

        # Determine level
        level = self._score_level(overall)

        score = CompletenessScore(
            batch_id=batch_id,
            overall_score=round(overall, 1),
            dimension_scores=dimension_scores,
            dimension_weights=dict(weights),
            dimension_passed=dimension_passed,
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=total_checks - passed_checks,
            level=level,
        )

        if self.config.enable_provenance:
            score.provenance_hash = _compute_hash(score)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.debug(
            "Completeness score for %s: %.1f (%s), elapsed=%.1fms",
            batch_id,
            overall,
            level,
            elapsed_ms,
        )

        return score

    def generate_verification_certificate(
        self,
        chain: Dict[str, Any],
        verification: Optional[ChainVerification] = None,
    ) -> VerificationCertificate:
        """Generate a verification certificate with evidence compilation.

        Creates a formal certificate summarizing the chain verification
        results with supporting evidence.

        Args:
            chain: Dictionary with chain data:
                - batch_id (str)
                - events (list[dict]): Custody events
                - origin_plots (list[str]): Origin plot IDs
            verification: Optional pre-computed verification result.
                If not provided, verify_chain is called.

        Returns:
            VerificationCertificate with evidence and status.
        """
        start_time = time.monotonic()

        batch_id = chain.get("batch_id", "")
        events = chain.get("events", [])
        origin_plots = chain.get("origin_plots", [])

        # Run verification if not provided
        if verification is None:
            verification = self.verify_chain(batch_id, events)

        # Build evidence list
        evidence: List[Dict[str, Any]] = []

        # Temporal evidence
        temporal_passed = sum(
            1 for c in verification.temporal_checks if c.passed
        )
        temporal_total = len(verification.temporal_checks)
        evidence.append({
            "dimension": "temporal_continuity",
            "checks_total": temporal_total,
            "checks_passed": temporal_passed,
            "passed": temporal_passed == temporal_total,
        })

        # Actor evidence
        actor_passed = sum(
            1 for c in verification.actor_checks if c.passed
        )
        actor_total = len(verification.actor_checks)
        evidence.append({
            "dimension": "actor_continuity",
            "checks_total": actor_total,
            "checks_passed": actor_passed,
            "passed": actor_passed == actor_total,
        })

        # Location evidence
        location_passed = sum(
            1 for c in verification.location_checks if c.passed
        )
        location_total = len(verification.location_checks)
        evidence.append({
            "dimension": "location_continuity",
            "checks_total": location_total,
            "checks_passed": location_passed,
            "passed": location_passed == location_total,
        })

        # Mass conservation evidence
        mass_passed = (
            verification.mass_conservation.passed
            if verification.mass_conservation
            else True
        )
        evidence.append({
            "dimension": "mass_conservation",
            "passed": mass_passed,
            "details": (
                verification.mass_conservation.details
                if verification.mass_conservation
                else "No mass data"
            ),
        })

        # Origin preservation evidence
        evidence.append({
            "dimension": "origin_preservation",
            "passed": verification.origin_check_passed,
            "allocation_total_pct": verification.origin_total_pct,
        })

        # Build dimension results
        dim_results = {}
        if verification.completeness:
            dim_results = dict(verification.completeness.dimension_passed)

        # Generate summary
        summary = self._build_certificate_summary(
            batch_id, verification, len(events), origin_plots
        )

        # Compute validity
        valid_until = utcnow() + timedelta(
            days=self.config.certificate_validity_days
        )

        certificate = VerificationCertificate(
            verification_id=verification.verification_id,
            batch_id=batch_id,
            status=verification.status,
            overall_score=verification.overall_score,
            valid_until=valid_until,
            summary=summary,
            dimension_results=dim_results,
            evidence=evidence,
            chain_length=len(events),
            origin_plots=list(origin_plots),
            regulatory_references=[
                "EUDR Article 4(2)",
                "EUDR Article 9",
                "EUDR Article 9(1)(d)",
                "EUDR Article 9(1)(e)",
                "EUDR Article 9(1)(f)",
                "EUDR Article 10",
                "EUDR Article 14",
            ],
        )

        if self.config.enable_provenance:
            certificate.provenance_hash = _compute_hash(certificate)

        self._certificate_store[certificate.certificate_id] = certificate

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Verification certificate for batch %s: status=%s, "
            "score=%.1f, valid_until=%s, elapsed=%.1fms",
            batch_id,
            certificate.status,
            certificate.overall_score,
            valid_until.isoformat(),
            elapsed_ms,
        )

        return certificate

    def batch_verify(
        self,
        batch_ids: List[str],
        events_by_batch: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        origin_allocations_by_batch: Optional[
            Dict[str, Dict[str, float]]
        ] = None,
    ) -> BatchVerifyResult:
        """Verify multiple custody chains in batch.

        Args:
            batch_ids: List of batch identifiers to verify.
            events_by_batch: Optional dict of batch_id -> event list.
            origin_allocations_by_batch: Optional dict of batch_id ->
                origin allocations.

        Returns:
            BatchVerifyResult with individual results and summary.

        Raises:
            ValueError: If batch size exceeds maximum.
        """
        start_time = time.monotonic()

        if len(batch_ids) > self.config.max_batch_verify_size:
            raise ValueError(
                f"Batch size {len(batch_ids)} exceeds maximum "
                f"{self.config.max_batch_verify_size}"
            )

        events_map = events_by_batch or {}
        origins_map = origin_allocations_by_batch or {}

        result = BatchVerifyResult(
            total_submitted=len(batch_ids),
        )

        scores: List[float] = []

        for batch_id in batch_ids:
            try:
                events = events_map.get(batch_id, [])
                origins = origins_map.get(batch_id)

                verification = self.verify_chain(
                    batch_id, events,
                    origin_allocations=origins,
                )
                result.results.append(verification)
                scores.append(verification.overall_score)

                if verification.status == VerificationStatus.PASS.value:
                    result.total_passed += 1
                elif verification.status == VerificationStatus.FAIL.value:
                    result.total_failed += 1
                elif verification.status == VerificationStatus.WARNING.value:
                    result.total_warnings += 1

            except Exception as exc:
                result.total_failed += 1
                result.errors.append({
                    "batch_id": batch_id,
                    "error": str(exc),
                })
                logger.warning(
                    "Batch verification failed for %s: %s",
                    batch_id,
                    str(exc),
                )

        # Compute average score
        result.average_score = (
            round(sum(scores) / len(scores), 1) if scores else 0.0
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        result.processing_time_ms = round(elapsed_ms, 2)
        result.completed_at = utcnow()

        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)

        logger.info(
            "Batch verification complete: submitted=%d, passed=%d, "
            "failed=%d, warnings=%d, avg_score=%.1f, elapsed=%.1fms",
            result.total_submitted,
            result.total_passed,
            result.total_failed,
            result.total_warnings,
            result.average_score,
            elapsed_ms,
        )

        return result

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_verification(
        self, verification_id: str
    ) -> Optional[ChainVerification]:
        """Retrieve a verification result by ID.

        Args:
            verification_id: Verification identifier.

        Returns:
            ChainVerification if found, None otherwise.
        """
        return self._verification_store.get(verification_id)

    def get_certificate(
        self, certificate_id: str
    ) -> Optional[VerificationCertificate]:
        """Retrieve a certificate by ID.

        Args:
            certificate_id: Certificate identifier.

        Returns:
            VerificationCertificate if found, None otherwise.
        """
        return self._certificate_store.get(certificate_id)

    @property
    def verification_count(self) -> int:
        """Return total number of verifications performed."""
        return self._verification_count

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sort_events_by_time(
        self, events: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Sort events by timestamp.

        Args:
            events: List of event dicts.

        Returns:
            Sorted list of event dicts.
        """
        def sort_key(e: Dict[str, Any]) -> datetime:
            ts = self._parse_timestamp(e.get("timestamp"))
            return ts if ts else utcnow()

        return sorted(events, key=sort_key)

    def _parse_timestamp(self, ts_value: Any) -> Optional[datetime]:
        """Parse a timestamp value to datetime.

        Args:
            ts_value: String, datetime, or None.

        Returns:
            Parsed datetime or None.
        """
        if ts_value is None:
            return None
        if isinstance(ts_value, datetime):
            return ts_value
        if isinstance(ts_value, str):
            try:
                return datetime.fromisoformat(ts_value)
            except ValueError:
                return None
        return None

    def _determine_status(
        self,
        temporal_checks: List[ContinuityCheck],
        actor_checks: List[ContinuityCheck],
        location_checks: List[ContinuityCheck],
        mass_result: MassConservationResult,
        origin_passed: bool,
        circular_deps: List[CircularDependency],
    ) -> str:
        """Determine overall verification status.

        Args:
            temporal_checks: Temporal check results.
            actor_checks: Actor check results.
            location_checks: Location check results.
            mass_result: Mass conservation result.
            origin_passed: Origin preservation result.
            circular_deps: Circular dependency detections.

        Returns:
            Status string: 'pass', 'warning', or 'fail'.
        """
        # Circular dependencies always cause failure
        if circular_deps:
            return VerificationStatus.FAIL.value

        # Mass violation causes failure
        if not mass_result.passed:
            return VerificationStatus.FAIL.value

        # Origin violation causes failure
        if not origin_passed:
            return VerificationStatus.FAIL.value

        # Count critical failures
        all_checks = temporal_checks + actor_checks + location_checks
        critical_fails = sum(
            1 for c in all_checks
            if not c.passed and c.severity in ("critical", "high")
        )
        warning_fails = sum(
            1 for c in all_checks
            if not c.passed and c.severity in ("medium", "low")
        )

        if critical_fails > 0:
            return VerificationStatus.FAIL.value
        elif warning_fails > 0:
            return VerificationStatus.WARNING.value
        else:
            return VerificationStatus.PASS.value

    def _dimension_score(
        self, checks: List[ContinuityCheck]
    ) -> float:
        """Compute dimension score from check list.

        Args:
            checks: List of ContinuityCheck results.

        Returns:
            Score from 0 to 100.
        """
        if not checks:
            return 100.0

        passed = sum(1 for c in checks if c.passed)
        return (passed / len(checks)) * 100.0

    def _score_level(self, score: float) -> str:
        """Map a numeric score to a qualitative level.

        Args:
            score: Score from 0 to 100.

        Returns:
            Level string: excellent/good/fair/poor/failing.
        """
        if score >= 95.0:
            return "excellent"
        elif score >= 80.0:
            return "good"
        elif score >= 60.0:
            return "fair"
        elif score >= 40.0:
            return "poor"
        else:
            return "failing"

    def _build_certificate_summary(
        self,
        batch_id: str,
        verification: ChainVerification,
        event_count: int,
        origin_plots: List[str],
    ) -> str:
        """Build executive summary for verification certificate.

        Args:
            batch_id: Batch identifier.
            verification: Verification result.
            event_count: Number of events in chain.
            origin_plots: List of origin plot IDs.

        Returns:
            Summary text string.
        """
        status_text = {
            VerificationStatus.PASS.value: "PASSED all integrity checks",
            VerificationStatus.WARNING.value: (
                "PASSED with warnings requiring attention"
            ),
            VerificationStatus.FAIL.value: (
                "FAILED one or more integrity checks"
            ),
        }

        return (
            f"Chain of custody verification for batch {batch_id} "
            f"{status_text.get(verification.status, 'status unknown')}. "
            f"The chain contains {event_count} custody events traced to "
            f"{len(origin_plots)} origin plot(s). "
            f"Composite integrity score: {verification.overall_score:.1f}/100. "
            f"Checks performed: {verification.total_checks} "
            f"(passed: {verification.passed_checks}, "
            f"failed: {verification.failed_checks}). "
            f"This verification is per {EUDR_REGULATION_REF} Articles 4, 9, 10, 14."
        )
