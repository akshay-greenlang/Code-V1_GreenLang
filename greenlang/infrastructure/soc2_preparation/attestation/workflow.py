# -*- coding: utf-8 -*-
"""
Attestation Workflow Module - SEC-009 Phase 7

Implements the attestation state machine and lifecycle management for SOC 2
management attestations. Supports the full workflow from draft creation through
signature collection to final activation.

State Machine:
    DRAFT -> REVIEW -> PENDING_SIGNATURE -> SIGNED -> ACTIVE

Classes:
    - AttestationStatus: Enumeration of attestation lifecycle states
    - Attestation: Core attestation data model
    - AttestationCreate: Input model for creating new attestations
    - AttestationWorkflow: Main workflow orchestration class

Example:
    >>> workflow = AttestationWorkflow(config)
    >>> attestation = await workflow.create_attestation(
    ...     attestation_type="soc2_readiness_attestation",
    ...     document_name="Q4 2026 SOC 2 Readiness",
    ... )
    >>> await workflow.submit_for_review(attestation.attestation_id)
    >>> await workflow.request_signatures(
    ...     attestation.attestation_id,
    ...     signers=[ceo_id, ciso_id],
    ... )

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-009 SOC 2 Type II Preparation
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class AttestationStatus(str, Enum):
    """Lifecycle states for management attestations.

    State transitions follow the workflow:
        DRAFT -> REVIEW -> PENDING_SIGNATURE -> SIGNED -> ACTIVE

    EXPIRED and REVOKED are terminal states that can be reached from
    ACTIVE or SIGNED states.
    """

    DRAFT = "draft"
    """Initial state. Attestation is being prepared but not ready for review."""

    REVIEW = "review"
    """Attestation is under internal review before signature collection."""

    PENDING_SIGNATURE = "pending_signature"
    """Attestation has been sent to signers and awaiting signatures."""

    SIGNED = "signed"
    """All required signatures have been collected."""

    ACTIVE = "active"
    """Attestation is finalized and active for the audit period."""

    EXPIRED = "expired"
    """Attestation has passed its validity period."""

    REVOKED = "revoked"
    """Attestation was manually revoked before expiration."""


class AttestationType(str, Enum):
    """Supported attestation document types for SOC 2 audits."""

    SOC2_READINESS_ATTESTATION = "soc2_readiness_attestation"
    """CEO/CISO sign pre-audit to attest readiness for SOC 2 examination."""

    MANAGEMENT_ASSERTION_LETTER = "management_assertion_letter"
    """CEO/CFO sign with report to assert control effectiveness."""

    CONTROL_OWNER_ATTESTATION = "control_owner_attestation"
    """Control owners sign quarterly to confirm control operation."""

    SUBSERVICE_ORGANIZATION_LIST = "subservice_organization_list"
    """List of carved-out services with management acknowledgment."""

    COMPLEMENTARY_USER_ENTITY_CONTROLS = "complementary_user_entity_controls"
    """Customer responsibilities that complement service organization controls."""


# ---------------------------------------------------------------------------
# Valid State Transitions
# ---------------------------------------------------------------------------


_VALID_TRANSITIONS: Dict[AttestationStatus, List[AttestationStatus]] = {
    AttestationStatus.DRAFT: [AttestationStatus.REVIEW],
    AttestationStatus.REVIEW: [
        AttestationStatus.DRAFT,
        AttestationStatus.PENDING_SIGNATURE,
    ],
    AttestationStatus.PENDING_SIGNATURE: [
        AttestationStatus.REVIEW,
        AttestationStatus.SIGNED,
    ],
    AttestationStatus.SIGNED: [
        AttestationStatus.ACTIVE,
        AttestationStatus.REVOKED,
    ],
    AttestationStatus.ACTIVE: [
        AttestationStatus.EXPIRED,
        AttestationStatus.REVOKED,
    ],
    AttestationStatus.EXPIRED: [],
    AttestationStatus.REVOKED: [],
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class SignerInfo(BaseModel):
    """Information about a required signer for an attestation."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    signer_id: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Unique identifier for the signer (user ID).",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Display name of the signer.",
    )
    title: str = Field(
        default="",
        max_length=256,
        description="Job title of the signer (e.g., 'Chief Executive Officer').",
    )
    email: str = Field(
        ...,
        max_length=256,
        description="Email address for signature requests.",
    )
    signed_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when this signer signed (UTC).",
    )
    signature_hash: Optional[str] = Field(
        default=None,
        max_length=128,
        description="SHA-256 hash of the signature data.",
    )


class AttestationCreate(BaseModel):
    """Input model for creating a new attestation."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    attestation_type: AttestationType = Field(
        ...,
        description="Type of attestation document to create.",
    )
    document_name: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Human-readable name for this attestation document.",
    )
    description: str = Field(
        default="",
        max_length=4096,
        description="Detailed description of the attestation purpose.",
    )
    audit_period_start: Optional[datetime] = Field(
        default=None,
        description="Start of the audit period this attestation covers (UTC).",
    )
    audit_period_end: Optional[datetime] = Field(
        default=None,
        description="End of the audit period this attestation covers (UTC).",
    )
    expires_at: Optional[datetime] = Field(
        default=None,
        description="Expiration date for the attestation (UTC).",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the attestation.",
    )
    created_by: str = Field(
        default="",
        max_length=256,
        description="User ID of the person creating the attestation.",
    )


class Attestation(BaseModel):
    """Core attestation data model with full lifecycle tracking.

    Represents a management attestation document with its current status,
    required signers, signature data, and complete audit trail.

    Attributes:
        attestation_id: Unique identifier (UUID).
        attestation_type: Type of attestation document.
        document_name: Human-readable document name.
        description: Detailed description of purpose.
        status: Current lifecycle status.
        document_content: Generated document content (markdown/HTML).
        document_hash: SHA-256 hash of document content for integrity.
        signers: List of required signers with signature status.
        audit_period_start: Start of covered audit period.
        audit_period_end: End of covered audit period.
        expires_at: Attestation expiration date.
        created_at: Creation timestamp.
        updated_at: Last modification timestamp.
        created_by: Creator user ID.
        envelope_id: External signature service envelope ID.
        signed_document_url: URL to download the signed document.
        metadata: Additional metadata.
        version: Optimistic concurrency version.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        json_schema_extra={
            "examples": [
                {
                    "attestation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                    "attestation_type": "soc2_readiness_attestation",
                    "document_name": "Q4 2026 SOC 2 Readiness Attestation",
                    "status": "draft",
                }
            ]
        },
    )

    attestation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the attestation.",
    )
    attestation_type: AttestationType = Field(
        ...,
        description="Type of attestation document.",
    )
    document_name: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Human-readable document name.",
    )
    description: str = Field(
        default="",
        max_length=4096,
        description="Detailed description of the attestation purpose.",
    )
    status: AttestationStatus = Field(
        default=AttestationStatus.DRAFT,
        description="Current lifecycle status.",
    )
    document_content: str = Field(
        default="",
        description="Generated document content (markdown/HTML).",
    )
    document_hash: str = Field(
        default="",
        max_length=128,
        description="SHA-256 hash of document content for integrity verification.",
    )
    signers: List[SignerInfo] = Field(
        default_factory=list,
        description="List of required signers with signature status.",
    )
    audit_period_start: Optional[datetime] = Field(
        default=None,
        description="Start of the audit period this attestation covers (UTC).",
    )
    audit_period_end: Optional[datetime] = Field(
        default=None,
        description="End of the audit period this attestation covers (UTC).",
    )
    expires_at: Optional[datetime] = Field(
        default=None,
        description="Expiration date for the attestation (UTC).",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp (UTC).",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last modification timestamp (UTC).",
    )
    created_by: str = Field(
        default="",
        max_length=256,
        description="User ID of the creator.",
    )
    envelope_id: Optional[str] = Field(
        default=None,
        max_length=256,
        description="External signature service envelope ID (DocuSign/Adobe Sign).",
    )
    signed_document_url: Optional[str] = Field(
        default=None,
        max_length=2048,
        description="URL to download the signed document.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the attestation.",
    )
    version: int = Field(
        default=1,
        ge=1,
        description="Optimistic concurrency version number.",
    )

    @field_validator("created_at", "updated_at", "expires_at", mode="before")
    @classmethod
    def ensure_utc_datetime(cls, v: Optional[datetime]) -> Optional[datetime]:
        """Ensure datetime values are timezone-aware UTC."""
        if v is None:
            return v
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            return v.astimezone(timezone.utc)
        return v

    @property
    def is_fully_signed(self) -> bool:
        """Check if all required signers have signed."""
        if not self.signers:
            return False
        return all(s.signed_at is not None for s in self.signers)

    @property
    def pending_signers(self) -> List[SignerInfo]:
        """Return list of signers who have not yet signed."""
        return [s for s in self.signers if s.signed_at is None]

    @property
    def is_expired(self) -> bool:
        """Check if the attestation has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at


# ---------------------------------------------------------------------------
# Attestation Workflow
# ---------------------------------------------------------------------------


class AttestationWorkflow:
    """Orchestrates the management attestation lifecycle.

    Manages attestation creation, state transitions, signature collection,
    and finalization. Integrates with external signature services and
    maintains complete audit trails for compliance.

    Attributes:
        config: SOC2Config instance for configuration.
        _attestations: In-memory attestation storage (replaced by DB in production).

    Example:
        >>> workflow = AttestationWorkflow(config)
        >>> attestation = await workflow.create_attestation(
        ...     attestation_type="soc2_readiness_attestation",
        ...     document_name="Q4 2026 SOC 2 Readiness",
        ... )
        >>> await workflow.submit_for_review(attestation.attestation_id)
    """

    def __init__(self, config: Any = None) -> None:
        """Initialize AttestationWorkflow.

        Args:
            config: Optional SOC2Config instance. If None, uses default config.
        """
        self.config = config
        self._attestations: Dict[str, Attestation] = {}
        logger.info("AttestationWorkflow initialized")

    async def create_attestation(
        self,
        attestation_type: str,
        document_name: str,
        description: str = "",
        audit_period_start: Optional[datetime] = None,
        audit_period_end: Optional[datetime] = None,
        expires_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
        created_by: str = "",
    ) -> Attestation:
        """Create a new attestation document in DRAFT status.

        Args:
            attestation_type: Type of attestation (e.g., 'soc2_readiness_attestation').
            document_name: Human-readable name for the document.
            description: Detailed description of the attestation purpose.
            audit_period_start: Start of covered audit period (UTC).
            audit_period_end: End of covered audit period (UTC).
            expires_at: Expiration date for the attestation (UTC).
            metadata: Additional metadata dictionary.
            created_by: User ID of the creator.

        Returns:
            The created Attestation in DRAFT status.

        Raises:
            ValueError: If attestation_type is invalid.
        """
        start_time = datetime.now(timezone.utc)

        # Validate attestation type
        try:
            att_type = AttestationType(attestation_type)
        except ValueError as e:
            valid_types = [t.value for t in AttestationType]
            raise ValueError(
                f"Invalid attestation_type '{attestation_type}'. "
                f"Valid types: {valid_types}"
            ) from e

        attestation = Attestation(
            attestation_type=att_type,
            document_name=document_name,
            description=description,
            status=AttestationStatus.DRAFT,
            audit_period_start=audit_period_start,
            audit_period_end=audit_period_end,
            expires_at=expires_at,
            metadata=metadata or {},
            created_by=created_by,
        )

        self._attestations[attestation.attestation_id] = attestation

        elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        logger.info(
            "Attestation created: id=%s, type=%s, name='%s', elapsed=%.2fms",
            attestation.attestation_id,
            attestation.attestation_type.value,
            attestation.document_name,
            elapsed_ms,
        )

        return attestation

    async def submit_for_review(self, attestation_id: str) -> None:
        """Submit an attestation for internal review.

        Transitions the attestation from DRAFT to REVIEW status.

        Args:
            attestation_id: UUID of the attestation to submit.

        Raises:
            ValueError: If attestation not found or invalid state transition.
        """
        attestation = await self._get_attestation(attestation_id)
        await self._transition_status(
            attestation,
            AttestationStatus.REVIEW,
            "Submitted for review",
        )

        logger.info(
            "Attestation submitted for review: id=%s, name='%s'",
            attestation_id,
            attestation.document_name,
        )

    async def request_signatures(
        self,
        attestation_id: str,
        signers: List[Dict[str, str]],
    ) -> None:
        """Request signatures from the specified signers.

        Transitions from REVIEW to PENDING_SIGNATURE and records the
        required signers.

        Args:
            attestation_id: UUID of the attestation.
            signers: List of signer dictionaries with keys:
                - signer_id: Unique user identifier
                - name: Display name
                - title: Job title
                - email: Email address

        Raises:
            ValueError: If attestation not found, invalid state, or no signers.
        """
        if not signers:
            raise ValueError("At least one signer is required.")

        attestation = await self._get_attestation(attestation_id)

        # Create SignerInfo objects
        signer_infos = [
            SignerInfo(
                signer_id=s.get("signer_id", str(uuid.uuid4())),
                name=s.get("name", "Unknown"),
                title=s.get("title", ""),
                email=s.get("email", ""),
            )
            for s in signers
        ]

        # Update attestation
        attestation.signers = signer_infos

        await self._transition_status(
            attestation,
            AttestationStatus.PENDING_SIGNATURE,
            f"Signature requested from {len(signers)} signer(s)",
        )

        logger.info(
            "Signatures requested: id=%s, signers=%d, names=%s",
            attestation_id,
            len(signers),
            [s.name for s in signer_infos],
        )

    async def record_signature(
        self,
        attestation_id: str,
        signer_id: str,
        signature_data: bytes,
    ) -> None:
        """Record a signature from a signer.

        Updates the signer's record with the signature timestamp and hash.
        If all signers have signed, transitions to SIGNED status.

        Args:
            attestation_id: UUID of the attestation.
            signer_id: UUID of the signer.
            signature_data: Raw signature data bytes.

        Raises:
            ValueError: If attestation not found, signer not found, or invalid state.
        """
        attestation = await self._get_attestation(attestation_id)

        if attestation.status != AttestationStatus.PENDING_SIGNATURE:
            raise ValueError(
                f"Cannot record signature in status '{attestation.status.value}'. "
                f"Expected 'pending_signature'."
            )

        # Find the signer
        signer = next((s for s in attestation.signers if s.signer_id == signer_id), None)
        if signer is None:
            raise ValueError(f"Signer '{signer_id}' not found in attestation.")

        if signer.signed_at is not None:
            raise ValueError(f"Signer '{signer_id}' has already signed.")

        # Record signature
        signature_hash = hashlib.sha256(signature_data).hexdigest()
        signer.signed_at = datetime.now(timezone.utc)
        signer.signature_hash = signature_hash
        attestation.updated_at = datetime.now(timezone.utc)

        logger.info(
            "Signature recorded: attestation=%s, signer=%s, hash=%s",
            attestation_id,
            signer_id,
            signature_hash[:16] + "...",
        )

        # Check if all signers have signed
        if attestation.is_fully_signed:
            await self._transition_status(
                attestation,
                AttestationStatus.SIGNED,
                "All signatures collected",
            )
            logger.info(
                "All signatures collected: attestation=%s",
                attestation_id,
            )

    async def finalize_attestation(self, attestation_id: str) -> None:
        """Finalize a signed attestation to make it active.

        Transitions from SIGNED to ACTIVE status, making the attestation
        effective for the audit period.

        Args:
            attestation_id: UUID of the attestation to finalize.

        Raises:
            ValueError: If attestation not found or invalid state transition.
        """
        attestation = await self._get_attestation(attestation_id)
        await self._transition_status(
            attestation,
            AttestationStatus.ACTIVE,
            "Attestation finalized and activated",
        )

        logger.info(
            "Attestation finalized: id=%s, name='%s'",
            attestation_id,
            attestation.document_name,
        )

    async def get_pending_signatures(self, user_id: str) -> List[Attestation]:
        """Get all attestations pending signature from a specific user.

        Args:
            user_id: UUID of the user to check for pending signatures.

        Returns:
            List of Attestation objects where the user has a pending signature.
        """
        pending = []
        for attestation in self._attestations.values():
            if attestation.status != AttestationStatus.PENDING_SIGNATURE:
                continue
            for signer in attestation.signers:
                if signer.signer_id == user_id and signer.signed_at is None:
                    pending.append(attestation)
                    break

        logger.debug(
            "Pending signatures for user=%s: count=%d",
            user_id,
            len(pending),
        )
        return pending

    async def get_attestation(self, attestation_id: str) -> Optional[Attestation]:
        """Get an attestation by ID.

        Args:
            attestation_id: UUID of the attestation.

        Returns:
            The Attestation if found, None otherwise.
        """
        return self._attestations.get(attestation_id)

    async def list_attestations(
        self,
        status: Optional[AttestationStatus] = None,
        attestation_type: Optional[AttestationType] = None,
    ) -> List[Attestation]:
        """List attestations with optional filtering.

        Args:
            status: Filter by attestation status.
            attestation_type: Filter by attestation type.

        Returns:
            List of matching Attestation objects.
        """
        results = list(self._attestations.values())

        if status is not None:
            results = [a for a in results if a.status == status]

        if attestation_type is not None:
            results = [a for a in results if a.attestation_type == attestation_type]

        # Sort by created_at descending
        results.sort(key=lambda a: a.created_at, reverse=True)
        return results

    async def revoke_attestation(
        self,
        attestation_id: str,
        reason: str = "",
    ) -> None:
        """Revoke an active or signed attestation.

        Args:
            attestation_id: UUID of the attestation to revoke.
            reason: Reason for revocation.

        Raises:
            ValueError: If attestation not found or cannot be revoked.
        """
        attestation = await self._get_attestation(attestation_id)
        await self._transition_status(
            attestation,
            AttestationStatus.REVOKED,
            f"Revoked: {reason}" if reason else "Revoked",
        )

        logger.warning(
            "Attestation revoked: id=%s, reason='%s'",
            attestation_id,
            reason,
        )

    async def set_document_content(
        self,
        attestation_id: str,
        content: str,
    ) -> None:
        """Set the document content for an attestation.

        Updates the document_content and computes the document_hash.

        Args:
            attestation_id: UUID of the attestation.
            content: Document content (markdown/HTML).

        Raises:
            ValueError: If attestation not found.
        """
        attestation = await self._get_attestation(attestation_id)
        attestation.document_content = content
        attestation.document_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        attestation.updated_at = datetime.now(timezone.utc)

        logger.info(
            "Document content set: attestation=%s, hash=%s, length=%d",
            attestation_id,
            attestation.document_hash[:16] + "...",
            len(content),
        )

    # -----------------------------------------------------------------------
    # Private Methods
    # -----------------------------------------------------------------------

    async def _get_attestation(self, attestation_id: str) -> Attestation:
        """Get an attestation or raise ValueError if not found."""
        attestation = self._attestations.get(attestation_id)
        if attestation is None:
            raise ValueError(f"Attestation '{attestation_id}' not found.")
        return attestation

    async def _transition_status(
        self,
        attestation: Attestation,
        new_status: AttestationStatus,
        reason: str = "",
    ) -> None:
        """Transition attestation to a new status with validation.

        Args:
            attestation: The attestation to transition.
            new_status: Target status.
            reason: Reason for the transition (for audit log).

        Raises:
            ValueError: If the transition is not valid.
        """
        current_status = attestation.status
        valid_targets = _VALID_TRANSITIONS.get(current_status, [])

        if new_status not in valid_targets:
            raise ValueError(
                f"Invalid status transition: '{current_status.value}' -> "
                f"'{new_status.value}'. Valid transitions from '{current_status.value}': "
                f"{[s.value for s in valid_targets]}"
            )

        old_status = attestation.status
        attestation.status = new_status
        attestation.updated_at = datetime.now(timezone.utc)
        attestation.version += 1

        logger.info(
            "Attestation status transition: id=%s, %s -> %s, reason='%s'",
            attestation.attestation_id,
            old_status.value,
            new_status.value,
            reason,
        )


__all__ = [
    "AttestationStatus",
    "AttestationType",
    "SignerInfo",
    "AttestationCreate",
    "Attestation",
    "AttestationWorkflow",
]
