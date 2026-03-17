# -*- coding: utf-8 -*-
"""
EvidenceChainEngine - PACK-018 EU Green Claims Prep Engine 2
=============================================================

Manages evidence collection, chain-of-custody tracking, and document
validation per EU Green Claims Directive Article 10.

The EU Green Claims Directive (Proposal COM/2023/166) requires traders
to maintain robust evidence chains that demonstrate the substantiation
of their environmental claims.  Article 10 establishes requirements for
the verification and documentation of claims, including:

Article 10 Requirements:
    - Para 1: Environmental claims shall be verified by an independent
      and accredited verifier before being communicated to consumers.
    - Para 2: Verification shall be based on the substantiation
      requirements of Article 3 and the communication requirements
      of Article 5.
    - Para 3: The verifier shall issue a certificate of conformity
      confirming that the claim has been substantiated and communicated
      in accordance with the Directive.
    - Para 4: Member States shall designate bodies accredited in
      accordance with Regulation (EC) No 765/2008 to act as verifiers.

Chain-of-Custody Principles:
    - Every document in the evidence chain must have a verifiable hash
    - Documents must be traceable from source to claim
    - Validity periods must be tracked and monitored
    - Accreditation references must be maintained for verifiers
    - The chain must be complete (no missing links)

Regulatory References:
    - EU Green Claims Directive Proposal COM/2023/166, Article 10
    - Regulation (EC) No 765/2008 (Accreditation and Market Surveillance)
    - ISO/IEC 17025 (Laboratory Competence)
    - ISO/IEC 17065 (Product Certification Bodies)
    - ISO 14025 (Environmental Product Declarations)

Zero-Hallucination:
    - Chain strength uses deterministic weighted scoring
    - Document validity uses date comparison arithmetic
    - Completeness checks use set-based coverage calculation
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-018 EU Green Claims Prep
Status:  Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


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
    """Convert value to Decimal safely.

    Args:
        value: Numeric value (int, float, str, or Decimal).

    Returns:
        Decimal representation.
    """
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _safe_divide(
    numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0")
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _round_val(value: Decimal, places: int = 3) -> Decimal:
    """Round a Decimal value to the specified number of decimal places.

    Uses ROUND_HALF_UP for regulatory consistency.

    Args:
        value: Decimal value to round.
        places: Number of decimal places (default 3).

    Returns:
        Rounded Decimal value.
    """
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(
        Decimal("0.001"), rounding=ROUND_HALF_UP
    ))


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EvidenceStatus(str, Enum):
    """Status of an evidence document in the chain.

    Tracks the lifecycle of each evidence document from initial
    submission through validation, expiry, or rejection.
    """
    PENDING = "pending"
    COLLECTED = "collected"
    VALIDATED = "validated"
    EXPIRED = "expired"
    REJECTED = "rejected"


class EvidenceType(str, Enum):
    """Types of evidence documents in the chain-of-custody.

    Per Article 3 and Article 10, evidence must be based on widely
    recognised scientific methods and verified by accredited bodies.
    These categories cover the full spectrum of acceptable evidence.
    """
    CERTIFICATION = "certification"
    LCA_STUDY = "lca_study"
    TEST_REPORT = "test_report"
    AUDIT_REPORT = "audit_report"
    MEASUREMENT = "measurement"
    THIRD_PARTY_VERIFICATION = "third_party_verification"
    SUPPLIER_DECLARATION = "supplier_declaration"
    LABORATORY_RESULT = "laboratory_result"
    MONITORING_DATA = "monitoring_data"
    OFFSET_REGISTRY = "offset_registry"


class ChainVerificationStatus(str, Enum):
    """Verification status of the entire evidence chain.

    Indicates whether the chain meets the verification requirements
    of Article 10 for communicating the environmental claim.
    """
    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    UNVERIFIED = "unverified"
    VERIFICATION_FAILED = "verification_failed"


class DocumentTier(str, Enum):
    """Reliability tier of a document in the evidence chain.

    Documents are classified by their reliability based on the
    source, methodology, and verification level.
    """
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# Minimum required document types for a complete evidence chain.
# Per Article 3(2) and Article 10, a robust chain should include
# at least one document from each of these core categories.
CORE_EVIDENCE_TYPES: List[str] = [
    EvidenceType.LCA_STUDY.value,
    EvidenceType.THIRD_PARTY_VERIFICATION.value,
]

# Document type reliability weights for chain strength calculation.
# Higher weights indicate more reliable evidence sources.
EVIDENCE_TYPE_WEIGHTS: Dict[str, Decimal] = {
    EvidenceType.CERTIFICATION.value: Decimal("0.95"),
    EvidenceType.LCA_STUDY.value: Decimal("0.90"),
    EvidenceType.THIRD_PARTY_VERIFICATION.value: Decimal("0.90"),
    EvidenceType.AUDIT_REPORT.value: Decimal("0.85"),
    EvidenceType.LABORATORY_RESULT.value: Decimal("0.85"),
    EvidenceType.TEST_REPORT.value: Decimal("0.80"),
    EvidenceType.MONITORING_DATA.value: Decimal("0.75"),
    EvidenceType.MEASUREMENT.value: Decimal("0.70"),
    EvidenceType.OFFSET_REGISTRY.value: Decimal("0.65"),
    EvidenceType.SUPPLIER_DECLARATION.value: Decimal("0.50"),
}

# Chain strength scoring dimension weights.
CHAIN_STRENGTH_WEIGHTS: Dict[str, Decimal] = {
    "document_quality": Decimal("30"),
    "chain_completeness": Decimal("25"),
    "temporal_validity": Decimal("20"),
    "verification_depth": Decimal("15"),
    "traceability": Decimal("10"),
}

# Validity warning threshold in days (documents expiring within this
# window are flagged as approaching expiry).
VALIDITY_WARNING_DAYS: int = 90


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class DocumentRecord(BaseModel):
    """A document in the evidence chain per Article 10.

    Represents a single document that forms part of the chain-of-custody
    for an environmental claim, including its metadata, validity period,
    and integrity hash.
    """
    doc_id: str = Field(
        default_factory=_new_uuid,
        description="Unique document identifier",
    )
    title: str = Field(
        ...,
        description="Document title or name",
        max_length=500,
    )
    evidence_type: EvidenceType = Field(
        ...,
        description="Type of evidence this document provides",
    )
    source_url: str = Field(
        default="",
        description="URL or reference to the original document",
        max_length=2000,
    )
    issue_date: Optional[str] = Field(
        default=None,
        description="Date the document was issued (ISO 8601 date string)",
        max_length=10,
    )
    expiry_date: Optional[str] = Field(
        default=None,
        description="Date the document expires (ISO 8601 date string)",
        max_length=10,
    )
    issuing_body: str = Field(
        default="",
        description="Organisation that issued the document",
        max_length=500,
    )
    accreditation_reference: str = Field(
        default="",
        description="Accreditation number or reference of the issuing body",
        max_length=200,
    )
    sha256_hash: str = Field(
        default="",
        description="SHA-256 hash of the document content for integrity verification",
    )
    status: EvidenceStatus = Field(
        default=EvidenceStatus.PENDING,
        description="Current status of this document",
    )
    tier: DocumentTier = Field(
        default=DocumentTier.SECONDARY,
        description="Reliability tier of the document",
    )
    is_third_party: bool = Field(
        default=False,
        description="Whether the document was produced by an independent third party",
    )
    methodology: str = Field(
        default="",
        description="Methodology or standard used to produce the document",
        max_length=500,
    )
    scope_description: str = Field(
        default="",
        description="Scope of the evidence covered by this document",
        max_length=2000,
    )

    @field_validator("title")
    @classmethod
    def validate_title_not_empty(cls, v: str) -> str:
        """Ensure document title is not empty."""
        if not v.strip():
            raise ValueError("Document title must not be empty")
        return v


class EvidenceChain(BaseModel):
    """A chain of evidence documents supporting an environmental claim.

    Represents the complete chain-of-custody linking source documents
    to the environmental claim per Article 10 requirements.
    """
    chain_id: str = Field(
        default_factory=_new_uuid,
        description="Unique chain identifier",
    )
    claim_id: str = Field(
        ...,
        description="ID of the environmental claim this chain supports",
    )
    documents: List[DocumentRecord] = Field(
        default_factory=list,
        description="Ordered list of documents in the chain",
    )
    chain_complete: bool = Field(
        default=False,
        description="Whether the chain includes all required document types",
    )
    verification_status: ChainVerificationStatus = Field(
        default=ChainVerificationStatus.UNVERIFIED,
        description="Overall verification status of the chain",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Chain creation timestamp (UTC)",
    )
    last_validated_at: Optional[datetime] = Field(
        default=None,
        description="Last validation timestamp (UTC)",
    )
    strength_score: Decimal = Field(
        default=Decimal("0.00"),
        description="Overall chain strength score (0-100)",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of the entire chain",
    )


class ChainValidationResult(BaseModel):
    """Result of an evidence chain validation.

    Contains detailed results from validating the completeness,
    validity, and integrity of an evidence chain.
    """
    result_id: str = Field(
        default_factory=_new_uuid,
        description="Unique result identifier",
    )
    chain_id: str = Field(
        default="",
        description="ID of the validated chain",
    )
    claim_id: str = Field(
        default="",
        description="ID of the associated claim",
    )
    is_complete: bool = Field(
        default=False,
        description="Whether the chain has all required document types",
    )
    total_documents: int = Field(
        default=0,
        description="Total number of documents in the chain",
    )
    valid_documents: int = Field(
        default=0,
        description="Number of currently valid documents",
    )
    expired_documents: int = Field(
        default=0,
        description="Number of expired documents",
    )
    expiring_soon_documents: int = Field(
        default=0,
        description="Number of documents expiring within warning threshold",
    )
    rejected_documents: int = Field(
        default=0,
        description="Number of rejected documents",
    )
    missing_types: List[str] = Field(
        default_factory=list,
        description="Evidence types required but not present",
    )
    integrity_verified: bool = Field(
        default=False,
        description="Whether all document hashes are present and valid",
    )
    issues: List[str] = Field(
        default_factory=list,
        description="Issues found during validation",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for improving the chain",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION,
        description="Engine version",
    )
    calculated_at: datetime = Field(
        default_factory=_utcnow,
        description="Validation timestamp (UTC)",
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Processing time in milliseconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of the validation result",
    )


class DocumentValidityResult(BaseModel):
    """Result of checking document validity across a set of documents.

    Reports on temporal validity, integrity, and status of documents.
    """
    result_id: str = Field(
        default_factory=_new_uuid,
        description="Unique result identifier",
    )
    total_documents: int = Field(
        default=0,
        description="Total documents checked",
    )
    valid_count: int = Field(
        default=0,
        description="Number of currently valid documents",
    )
    expired_count: int = Field(
        default=0,
        description="Number of expired documents",
    )
    expiring_soon_count: int = Field(
        default=0,
        description="Documents expiring within warning threshold",
    )
    no_expiry_count: int = Field(
        default=0,
        description="Documents with no expiry date set",
    )
    missing_hash_count: int = Field(
        default=0,
        description="Documents without integrity hash",
    )
    document_details: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-document validity details",
    )
    overall_validity_pct: Decimal = Field(
        default=Decimal("0.00"),
        description="Percentage of documents that are currently valid",
    )
    issues: List[str] = Field(
        default_factory=list,
        description="Issues found during validity checks",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION,
        description="Engine version",
    )
    calculated_at: datetime = Field(
        default_factory=_utcnow,
        description="Check timestamp (UTC)",
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Processing time in milliseconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of the result",
    )


class ChainStrengthResult(BaseModel):
    """Result of chain strength calculation.

    Provides a detailed breakdown of the evidence chain strength
    across multiple quality dimensions.
    """
    result_id: str = Field(
        default_factory=_new_uuid,
        description="Unique result identifier",
    )
    chain_id: str = Field(
        default="",
        description="ID of the assessed chain",
    )
    overall_strength: Decimal = Field(
        default=Decimal("0.00"),
        description="Overall chain strength score (0-100)",
    )
    dimension_scores: Dict[str, str] = Field(
        default_factory=dict,
        description="Scores per strength dimension",
    )
    strength_label: str = Field(
        default="",
        description="Human-readable strength label (Strong, Moderate, Weak)",
    )
    weakest_dimension: str = Field(
        default="",
        description="Name of the weakest scoring dimension",
    )
    strongest_dimension: str = Field(
        default="",
        description="Name of the strongest scoring dimension",
    )
    document_type_coverage: Dict[str, bool] = Field(
        default_factory=dict,
        description="Coverage of each evidence type in the chain",
    )
    third_party_ratio: Decimal = Field(
        default=Decimal("0.00"),
        description="Ratio of third-party documents (0.00-1.00)",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for strengthening the chain",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION,
        description="Engine version",
    )
    calculated_at: datetime = Field(
        default_factory=_utcnow,
        description="Calculation timestamp (UTC)",
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Processing time in milliseconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of the result",
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class EvidenceChainEngine:
    """Evidence chain management engine per EU Green Claims Directive Art. 10.

    Provides deterministic, zero-hallucination evidence chain management:

    - Build evidence chains from document collections
    - Validate chain completeness against claim requirements
    - Check document validity (expiry, integrity, status)
    - Calculate chain strength across multiple dimensions

    All calculations use Decimal arithmetic with ROUND_HALF_UP rounding.
    Every result includes a SHA-256 provenance hash for audit trail.

    Usage::

        engine = EvidenceChainEngine()
        documents = [
            DocumentRecord(
                title="LCA Study Report",
                evidence_type=EvidenceType.LCA_STUDY,
                issuing_body="Accredited Lab",
                is_third_party=True,
                issue_date="2025-01-15",
                expiry_date="2028-01-15",
            ),
        ]
        chain_result = engine.build_evidence_chain("claim-123", documents)
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self) -> None:
        """Initialise EvidenceChainEngine."""
        self._chains: Dict[str, EvidenceChain] = {}
        logger.info(
            "EvidenceChainEngine v%s initialised", self.engine_version
        )

    # ------------------------------------------------------------------ #
    # Build Evidence Chain                                                  #
    # ------------------------------------------------------------------ #

    def build_evidence_chain(
        self,
        claim_id: str,
        documents: List[DocumentRecord],
    ) -> Dict[str, Any]:
        """Build an evidence chain from a set of documents.

        Creates a new evidence chain linking the provided documents
        to the specified claim.  Automatically assesses completeness,
        assigns verification status, and computes chain strength.

        Args:
            claim_id: ID of the environmental claim.
            documents: List of documents to include in the chain.

        Returns:
            Dict with keys: chain (EvidenceChain),
            validation (ChainValidationResult), provenance_hash (str).
        """
        t0 = time.perf_counter()

        # Assign hashes to documents that lack them
        processed_docs: List[DocumentRecord] = []
        for doc in documents:
            if not doc.sha256_hash:
                doc.sha256_hash = _compute_hash(doc)
            processed_docs.append(doc)

        # Determine chain completeness
        provided_types = {d.evidence_type.value for d in processed_docs}
        missing_types = [
            ct for ct in CORE_EVIDENCE_TYPES
            if ct not in provided_types
        ]
        chain_complete = len(missing_types) == 0

        # Determine verification status
        verification_status = self._determine_verification_status(
            processed_docs
        )

        # Build chain object
        chain = EvidenceChain(
            claim_id=claim_id,
            documents=processed_docs,
            chain_complete=chain_complete,
            verification_status=verification_status,
        )

        # Calculate strength score
        strength_result = self._calculate_chain_strength_internal(chain)
        chain.strength_score = strength_result["overall_strength"]

        # Compute chain provenance hash
        chain.provenance_hash = _compute_hash(chain)

        # Store the chain
        self._chains[chain.chain_id] = chain

        # Build validation result
        validation = self._validate_chain_internal(chain)

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        logger.info(
            "Built evidence chain '%s' for claim '%s': %d documents, "
            "complete=%s, status=%s, strength=%s in %.3f ms",
            chain.chain_id,
            claim_id,
            len(processed_docs),
            chain_complete,
            verification_status.value,
            chain.strength_score,
            elapsed_ms,
        )

        return {
            "chain": chain,
            "validation": validation,
            "provenance_hash": chain.provenance_hash,
        }

    # ------------------------------------------------------------------ #
    # Validate Chain Completeness                                           #
    # ------------------------------------------------------------------ #

    def validate_chain_completeness(
        self,
        chain: EvidenceChain,
    ) -> Dict[str, Any]:
        """Validate whether an evidence chain is complete.

        Checks that all required evidence types are present, documents
        are valid, and the chain meets Article 10 requirements.

        Args:
            chain: The evidence chain to validate.

        Returns:
            Dict with keys: result (ChainValidationResult),
            provenance_hash (str).
        """
        t0 = time.perf_counter()

        result = self._validate_chain_internal(chain)
        result.processing_time_ms = _round3(
            (time.perf_counter() - t0) * 1000.0
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Validated chain '%s': complete=%s, %d valid/%d total, "
            "%d issues in %.3f ms",
            chain.chain_id,
            result.is_complete,
            result.valid_documents,
            result.total_documents,
            len(result.issues),
            result.processing_time_ms,
        )

        return {
            "result": result,
            "provenance_hash": result.provenance_hash,
        }

    # ------------------------------------------------------------------ #
    # Check Document Validity                                               #
    # ------------------------------------------------------------------ #

    def check_document_validity(
        self,
        documents: List[DocumentRecord],
    ) -> Dict[str, Any]:
        """Check the validity of a set of documents.

        Evaluates each document for temporal validity (issue/expiry dates),
        integrity (SHA-256 hash presence), and status.

        Args:
            documents: List of documents to check.

        Returns:
            Dict with keys: result (DocumentValidityResult),
            provenance_hash (str).
        """
        t0 = time.perf_counter()

        today = date.today()
        today_str = today.isoformat()

        valid_count = 0
        expired_count = 0
        expiring_soon_count = 0
        no_expiry_count = 0
        missing_hash_count = 0
        document_details: List[Dict[str, Any]] = []
        issues: List[str] = []

        for doc in documents:
            detail: Dict[str, Any] = {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "evidence_type": doc.evidence_type.value,
                "status": doc.status.value,
            }

            # Check hash integrity
            if not doc.sha256_hash:
                missing_hash_count += 1
                detail["has_hash"] = False
                issues.append(
                    f"Document '{doc.title}' (ID: {doc.doc_id}) is missing "
                    f"an integrity hash"
                )
            else:
                detail["has_hash"] = True

            # Check temporal validity
            if doc.expiry_date:
                detail["expiry_date"] = doc.expiry_date
                if doc.expiry_date < today_str:
                    expired_count += 1
                    detail["validity"] = "expired"
                    issues.append(
                        f"Document '{doc.title}' (ID: {doc.doc_id}) expired "
                        f"on {doc.expiry_date}"
                    )
                else:
                    # Check if expiring soon
                    try:
                        expiry = date.fromisoformat(doc.expiry_date)
                        days_remaining = (expiry - today).days
                        detail["days_remaining"] = days_remaining
                        if days_remaining <= VALIDITY_WARNING_DAYS:
                            expiring_soon_count += 1
                            detail["validity"] = "expiring_soon"
                            issues.append(
                                f"Document '{doc.title}' (ID: {doc.doc_id}) "
                                f"expires in {days_remaining} days"
                            )
                        else:
                            valid_count += 1
                            detail["validity"] = "valid"
                    except ValueError:
                        detail["validity"] = "invalid_date_format"
                        issues.append(
                            f"Document '{doc.title}' (ID: {doc.doc_id}) has "
                            f"an invalid expiry date format"
                        )
            else:
                no_expiry_count += 1
                detail["validity"] = "no_expiry_set"
                # Documents without expiry are considered valid but flagged
                valid_count += 1

            # Check status
            if doc.status == EvidenceStatus.REJECTED:
                detail["validity"] = "rejected"
                issues.append(
                    f"Document '{doc.title}' (ID: {doc.doc_id}) has been rejected"
                )
            elif doc.status == EvidenceStatus.EXPIRED:
                if detail.get("validity") != "expired":
                    expired_count += 1
                    detail["validity"] = "expired"

            document_details.append(detail)

        total = len(documents)
        overall_validity_pct = _safe_divide(
            _decimal(valid_count) * Decimal("100"),
            _decimal(total) if total > 0 else Decimal("1"),
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = DocumentValidityResult(
            total_documents=total,
            valid_count=valid_count,
            expired_count=expired_count,
            expiring_soon_count=expiring_soon_count,
            no_expiry_count=no_expiry_count,
            missing_hash_count=missing_hash_count,
            document_details=document_details,
            overall_validity_pct=_round_val(overall_validity_pct, 2),
            issues=issues,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Checked %d documents: %d valid, %d expired, %d expiring soon, "
            "%d missing hash in %.3f ms",
            total,
            valid_count,
            expired_count,
            expiring_soon_count,
            missing_hash_count,
            elapsed_ms,
        )

        return {
            "result": result,
            "provenance_hash": result.provenance_hash,
        }

    # ------------------------------------------------------------------ #
    # Calculate Chain Strength                                              #
    # ------------------------------------------------------------------ #

    def calculate_chain_strength(
        self,
        chain: EvidenceChain,
    ) -> Dict[str, Any]:
        """Calculate the overall strength of an evidence chain.

        Evaluates the chain across five dimensions:
        1. Document Quality (30%): Weighted average of document type scores
        2. Chain Completeness (25%): Coverage of required evidence types
        3. Temporal Validity (20%): Proportion of currently valid documents
        4. Verification Depth (15%): Third-party and accreditation coverage
        5. Traceability (10%): Hash integrity and source references

        Args:
            chain: The evidence chain to assess.

        Returns:
            Dict with keys: result (ChainStrengthResult),
            provenance_hash (str).
        """
        t0 = time.perf_counter()

        internal = self._calculate_chain_strength_internal(chain)

        # Determine strength label
        overall = internal["overall_strength"]
        if overall >= Decimal("75"):
            strength_label = "Strong"
        elif overall >= Decimal("50"):
            strength_label = "Moderate"
        elif overall >= Decimal("25"):
            strength_label = "Weak"
        else:
            strength_label = "Very Weak"

        # Identify weakest and strongest dimensions
        dim_scores = internal["dimension_scores"]
        weakest = min(dim_scores, key=dim_scores.get) if dim_scores else ""
        strongest = max(dim_scores, key=dim_scores.get) if dim_scores else ""

        # Document type coverage
        doc_type_coverage: Dict[str, bool] = {}
        provided_types = {
            d.evidence_type.value for d in chain.documents
        }
        for et in EvidenceType:
            doc_type_coverage[et.value] = et.value in provided_types

        # Third-party ratio
        total_docs = len(chain.documents)
        tp_count = sum(1 for d in chain.documents if d.is_third_party)
        tp_ratio = _safe_divide(
            _decimal(tp_count),
            _decimal(total_docs) if total_docs > 0 else Decimal("1"),
        )

        # Recommendations
        recommendations = self._generate_strength_recommendations(
            chain, dim_scores, overall
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = ChainStrengthResult(
            chain_id=chain.chain_id,
            overall_strength=_round_val(overall, 2),
            dimension_scores={
                k: str(_round_val(v, 2)) for k, v in dim_scores.items()
            },
            strength_label=strength_label,
            weakest_dimension=weakest,
            strongest_dimension=strongest,
            document_type_coverage=doc_type_coverage,
            third_party_ratio=_round_val(tp_ratio, 2),
            recommendations=recommendations,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Calculated chain strength for '%s': %s (%s), "
            "weakest=%s, strongest=%s in %.3f ms",
            chain.chain_id,
            result.overall_strength,
            strength_label,
            weakest,
            strongest,
            elapsed_ms,
        )

        return {
            "result": result,
            "provenance_hash": result.provenance_hash,
        }

    # ------------------------------------------------------------------ #
    # Private Methods                                                       #
    # ------------------------------------------------------------------ #

    def _determine_verification_status(
        self, documents: List[DocumentRecord]
    ) -> ChainVerificationStatus:
        """Determine the overall verification status from documents.

        Args:
            documents: List of documents in the chain.

        Returns:
            ChainVerificationStatus enum value.
        """
        if not documents:
            return ChainVerificationStatus.UNVERIFIED

        has_tpv = any(
            d.evidence_type == EvidenceType.THIRD_PARTY_VERIFICATION
            and d.status == EvidenceStatus.VALIDATED
            for d in documents
        )

        validated_count = sum(
            1 for d in documents if d.status == EvidenceStatus.VALIDATED
        )
        total = len(documents)

        rejected_count = sum(
            1 for d in documents if d.status == EvidenceStatus.REJECTED
        )

        if rejected_count > 0:
            return ChainVerificationStatus.VERIFICATION_FAILED

        if has_tpv and validated_count == total:
            return ChainVerificationStatus.VERIFIED

        if validated_count > 0:
            return ChainVerificationStatus.PARTIALLY_VERIFIED

        return ChainVerificationStatus.UNVERIFIED

    def _validate_chain_internal(
        self, chain: EvidenceChain
    ) -> ChainValidationResult:
        """Internal chain validation logic.

        Args:
            chain: The evidence chain to validate.

        Returns:
            ChainValidationResult with validation details.
        """
        today_str = date.today().isoformat()
        docs = chain.documents

        total = len(docs)
        valid_count = 0
        expired_count = 0
        expiring_soon_count = 0
        rejected_count = 0
        issues: List[str] = []
        recommendations: List[str] = []

        for doc in docs:
            if doc.status == EvidenceStatus.REJECTED:
                rejected_count += 1
                issues.append(
                    f"Document '{doc.title}' has been rejected"
                )
                continue

            if doc.expiry_date and doc.expiry_date < today_str:
                expired_count += 1
                issues.append(
                    f"Document '{doc.title}' expired on {doc.expiry_date}"
                )
            elif doc.expiry_date:
                try:
                    expiry = date.fromisoformat(doc.expiry_date)
                    days_left = (expiry - date.today()).days
                    if days_left <= VALIDITY_WARNING_DAYS:
                        expiring_soon_count += 1
                    else:
                        valid_count += 1
                except ValueError:
                    valid_count += 1
            else:
                valid_count += 1

        # Check for required types
        provided_types = {d.evidence_type.value for d in docs}
        missing_types = [
            ct for ct in CORE_EVIDENCE_TYPES
            if ct not in provided_types
        ]

        if missing_types:
            issues.append(
                f"Chain is missing required evidence types: "
                f"{', '.join(missing_types)}"
            )
            recommendations.append(
                f"Obtain documents of type: {', '.join(missing_types)}"
            )

        is_complete = len(missing_types) == 0 and total > 0

        # Check integrity
        docs_without_hash = [
            d for d in docs if not d.sha256_hash
        ]
        integrity_verified = len(docs_without_hash) == 0 and total > 0
        if docs_without_hash:
            issues.append(
                f"{len(docs_without_hash)} document(s) are missing "
                f"integrity hashes"
            )
            recommendations.append(
                "Compute and assign SHA-256 hashes to all documents "
                "for integrity verification"
            )

        if expired_count > 0:
            recommendations.append(
                f"Renew {expired_count} expired document(s) to maintain "
                f"chain validity"
            )

        if expiring_soon_count > 0:
            recommendations.append(
                f"Plan renewal of {expiring_soon_count} document(s) "
                f"expiring within {VALIDITY_WARNING_DAYS} days"
            )

        if not any(d.is_third_party for d in docs) and total > 0:
            recommendations.append(
                "Include at least one third-party verified document "
                "to strengthen the evidence chain"
            )

        return ChainValidationResult(
            chain_id=chain.chain_id,
            claim_id=chain.claim_id,
            is_complete=is_complete,
            total_documents=total,
            valid_documents=valid_count,
            expired_documents=expired_count,
            expiring_soon_documents=expiring_soon_count,
            rejected_documents=rejected_count,
            missing_types=missing_types,
            integrity_verified=integrity_verified,
            issues=issues,
            recommendations=recommendations,
        )

    def _calculate_chain_strength_internal(
        self, chain: EvidenceChain
    ) -> Dict[str, Any]:
        """Internal chain strength calculation.

        Args:
            chain: The evidence chain to assess.

        Returns:
            Dict with overall_strength (Decimal) and dimension_scores (dict).
        """
        docs = chain.documents
        if not docs:
            return {
                "overall_strength": Decimal("0"),
                "dimension_scores": {
                    k: Decimal("0") for k in CHAIN_STRENGTH_WEIGHTS
                },
            }

        # 1. Document Quality (0-100)
        doc_quality = self._score_document_quality(docs)

        # 2. Chain Completeness (0-100)
        chain_completeness = self._score_chain_completeness(docs)

        # 3. Temporal Validity (0-100)
        temporal_validity = self._score_temporal_validity(docs)

        # 4. Verification Depth (0-100)
        verification_depth = self._score_verification_depth(docs)

        # 5. Traceability (0-100)
        traceability = self._score_traceability(docs)

        dimension_scores: Dict[str, Decimal] = {
            "document_quality": doc_quality,
            "chain_completeness": chain_completeness,
            "temporal_validity": temporal_validity,
            "verification_depth": verification_depth,
            "traceability": traceability,
        }

        # Weighted total
        overall = Decimal("0")
        for dim, weight in CHAIN_STRENGTH_WEIGHTS.items():
            overall += dimension_scores[dim] * weight / Decimal("100")

        return {
            "overall_strength": _round_val(overall, 2),
            "dimension_scores": dimension_scores,
        }

    def _score_document_quality(
        self, docs: List[DocumentRecord]
    ) -> Decimal:
        """Score the average document quality based on type weights.

        Args:
            docs: List of documents.

        Returns:
            Document quality score (0-100).
        """
        if not docs:
            return Decimal("0")

        total_weight = Decimal("0")
        for doc in docs:
            weight = EVIDENCE_TYPE_WEIGHTS.get(
                doc.evidence_type.value, Decimal("0.50")
            )
            # Boost for validated documents
            if doc.status == EvidenceStatus.VALIDATED:
                weight = min(weight + Decimal("0.05"), Decimal("1.00"))
            # Penalty for rejected documents
            elif doc.status == EvidenceStatus.REJECTED:
                weight = Decimal("0")
            total_weight += weight

        avg_weight = _safe_divide(
            total_weight, _decimal(len(docs))
        )
        return _round_val(avg_weight * Decimal("100"), 2)

    def _score_chain_completeness(
        self, docs: List[DocumentRecord]
    ) -> Decimal:
        """Score the chain completeness based on evidence type coverage.

        Args:
            docs: List of documents.

        Returns:
            Chain completeness score (0-100).
        """
        provided = {d.evidence_type.value for d in docs}
        all_types = {et.value for et in EvidenceType}

        # Core types coverage (60 points)
        core_covered = sum(
            1 for ct in CORE_EVIDENCE_TYPES if ct in provided
        )
        core_total = len(CORE_EVIDENCE_TYPES)
        core_score = _safe_divide(
            _decimal(core_covered) * Decimal("60"),
            _decimal(core_total) if core_total > 0 else Decimal("1"),
        )

        # Additional types coverage (40 points)
        non_core = all_types - set(CORE_EVIDENCE_TYPES)
        extra_covered = sum(1 for et in non_core if et in provided)
        extra_total = len(non_core)
        extra_score = _safe_divide(
            _decimal(extra_covered) * Decimal("40"),
            _decimal(extra_total) if extra_total > 0 else Decimal("1"),
        )

        return min(
            _round_val(core_score + extra_score, 2), Decimal("100")
        )

    def _score_temporal_validity(
        self, docs: List[DocumentRecord]
    ) -> Decimal:
        """Score temporal validity of documents.

        Args:
            docs: List of documents.

        Returns:
            Temporal validity score (0-100).
        """
        if not docs:
            return Decimal("0")

        today_str = date.today().isoformat()
        valid_count = 0

        for doc in docs:
            if doc.status == EvidenceStatus.REJECTED:
                continue
            if doc.expiry_date:
                if doc.expiry_date >= today_str:
                    valid_count += 1
            else:
                # No expiry set = assumed valid
                valid_count += 1

        ratio = _safe_divide(
            _decimal(valid_count), _decimal(len(docs))
        )
        return _round_val(ratio * Decimal("100"), 2)

    def _score_verification_depth(
        self, docs: List[DocumentRecord]
    ) -> Decimal:
        """Score the depth of independent verification.

        Args:
            docs: List of documents.

        Returns:
            Verification depth score (0-100).
        """
        if not docs:
            return Decimal("0")

        score = Decimal("0")

        # Third-party ratio (up to 40 points)
        tp_count = sum(1 for d in docs if d.is_third_party)
        tp_ratio = _safe_divide(
            _decimal(tp_count), _decimal(len(docs))
        )
        score += _round_val(tp_ratio * Decimal("40"), 2)

        # Has accreditation references (up to 30 points)
        accredited_count = sum(
            1 for d in docs if d.accreditation_reference.strip()
        )
        accred_ratio = _safe_divide(
            _decimal(accredited_count), _decimal(len(docs))
        )
        score += _round_val(accred_ratio * Decimal("30"), 2)

        # Has third-party verification document (up to 20 points)
        has_tpv = any(
            d.evidence_type == EvidenceType.THIRD_PARTY_VERIFICATION
            for d in docs
        )
        if has_tpv:
            score += Decimal("20")

        # Has audit report (up to 10 points)
        has_audit = any(
            d.evidence_type == EvidenceType.AUDIT_REPORT for d in docs
        )
        if has_audit:
            score += Decimal("10")

        return min(score, Decimal("100"))

    def _score_traceability(
        self, docs: List[DocumentRecord]
    ) -> Decimal:
        """Score the traceability of the evidence chain.

        Args:
            docs: List of documents.

        Returns:
            Traceability score (0-100).
        """
        if not docs:
            return Decimal("0")

        score = Decimal("0")

        # Hash coverage (up to 40 points)
        hashed_count = sum(1 for d in docs if d.sha256_hash)
        hash_ratio = _safe_divide(
            _decimal(hashed_count), _decimal(len(docs))
        )
        score += _round_val(hash_ratio * Decimal("40"), 2)

        # Source URL coverage (up to 25 points)
        sourced_count = sum(1 for d in docs if d.source_url.strip())
        source_ratio = _safe_divide(
            _decimal(sourced_count), _decimal(len(docs))
        )
        score += _round_val(source_ratio * Decimal("25"), 2)

        # Issuing body coverage (up to 20 points)
        issuer_count = sum(1 for d in docs if d.issuing_body.strip())
        issuer_ratio = _safe_divide(
            _decimal(issuer_count), _decimal(len(docs))
        )
        score += _round_val(issuer_ratio * Decimal("20"), 2)

        # Scope description coverage (up to 15 points)
        scope_count = sum(1 for d in docs if d.scope_description.strip())
        scope_ratio = _safe_divide(
            _decimal(scope_count), _decimal(len(docs))
        )
        score += _round_val(scope_ratio * Decimal("15"), 2)

        return min(score, Decimal("100"))

    def _generate_strength_recommendations(
        self,
        chain: EvidenceChain,
        dim_scores: Dict[str, Decimal],
        overall: Decimal,
    ) -> List[str]:
        """Generate recommendations to improve chain strength.

        Args:
            chain: The evidence chain.
            dim_scores: Dimension scores.
            overall: Overall strength score.

        Returns:
            List of recommendation strings.
        """
        recommendations: List[str] = []

        if dim_scores.get(
            "document_quality", Decimal("0")
        ) < Decimal("60"):
            recommendations.append(
                "Upgrade evidence sources to higher-tier document types "
                "(certifications, LCA studies, third-party verifications)"
            )

        if dim_scores.get(
            "chain_completeness", Decimal("0")
        ) < Decimal("60"):
            provided = {
                d.evidence_type.value for d in chain.documents
            }
            missing_core = [
                ct for ct in CORE_EVIDENCE_TYPES
                if ct not in provided
            ]
            if missing_core:
                recommendations.append(
                    f"Add missing core evidence types: "
                    f"{', '.join(missing_core)}"
                )

        if dim_scores.get(
            "temporal_validity", Decimal("0")
        ) < Decimal("70"):
            recommendations.append(
                "Renew expired documents and ensure all documents "
                "have current validity dates"
            )

        if dim_scores.get(
            "verification_depth", Decimal("0")
        ) < Decimal("60"):
            recommendations.append(
                "Increase independent verification by obtaining "
                "third-party assessments and accredited audit reports"
            )

        if dim_scores.get(
            "traceability", Decimal("0")
        ) < Decimal("60"):
            recommendations.append(
                "Improve traceability by adding SHA-256 hashes, "
                "source URLs, and scope descriptions to all documents"
            )

        if not recommendations and overall < Decimal("100"):
            recommendations.append(
                "Chain strength is adequate; continue monitoring "
                "document validity and updating evidence as needed"
            )

        return recommendations
