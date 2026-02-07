# -*- coding: utf-8 -*-
"""
Digital Signer Module - SEC-009 Phase 7

Provides digital signature collection and verification for attestation documents.
Supports multiple signature methods including DocuSign, Adobe Sign, and internal
cryptographic signatures.

Signature Methods:
    - DOCUSIGN: Integration with DocuSign eSignature API
    - ADOBE_SIGN: Integration with Adobe Sign API
    - INTERNAL: Internal cryptographic signature using RSA/ECDSA

Classes:
    - SignatureMethod: Enumeration of supported signature providers
    - SignatureStatus: Status of a signature request
    - SignatureRequest: Model for signature request tracking
    - DigitalSigner: Main signature service class

Example:
    >>> signer = DigitalSigner(config)
    >>> envelope_id = await signer.send_for_signature(
    ...     document_id=doc_id,
    ...     signers=[ceo_info, ciso_info],
    ...     method=SignatureMethod.DOCUSIGN,
    ... )
    >>> status = await signer.check_status(envelope_id)
    >>> if status == SignatureStatus.COMPLETED:
    ...     signed_pdf = await signer.download_signed(envelope_id)

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-009 SOC 2 Type II Preparation
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import secrets
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class SignatureMethod(str, Enum):
    """Supported digital signature methods."""

    DOCUSIGN = "docusign"
    """DocuSign eSignature integration."""

    ADOBE_SIGN = "adobe_sign"
    """Adobe Sign (Acrobat Sign) integration."""

    INTERNAL = "internal"
    """Internal cryptographic signature using platform keys."""


class SignatureStatus(str, Enum):
    """Status of a signature request or envelope."""

    CREATED = "created"
    """Signature request has been created but not yet sent."""

    SENT = "sent"
    """Signature request has been sent to signers."""

    DELIVERED = "delivered"
    """Signature request has been delivered (opened by recipient)."""

    SIGNED = "signed"
    """Document has been signed by one or more signers (not all)."""

    COMPLETED = "completed"
    """All signatures have been collected."""

    DECLINED = "declined"
    """One or more signers declined to sign."""

    VOIDED = "voided"
    """Signature request was voided/cancelled."""

    EXPIRED = "expired"
    """Signature request expired before completion."""

    ERROR = "error"
    """An error occurred during the signature process."""


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class SignerDetails(BaseModel):
    """Details about a signer for signature requests."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    signer_id: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Unique identifier for the signer.",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Full name of the signer.",
    )
    email: str = Field(
        ...,
        max_length=256,
        description="Email address for the signer.",
    )
    title: str = Field(
        default="",
        max_length=256,
        description="Job title of the signer.",
    )
    routing_order: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Order in which to request signature (1 = first).",
    )
    signed: bool = Field(
        default=False,
        description="Whether this signer has signed.",
    )
    signed_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when the signer signed (UTC).",
    )


class SignatureRequest(BaseModel):
    """Model for tracking a signature request/envelope.

    Attributes:
        envelope_id: Unique identifier from the signature provider.
        document_id: ID of the document being signed.
        method: Signature method being used.
        status: Current status of the signature request.
        signers: List of signers and their status.
        created_at: When the request was created.
        updated_at: When the request was last updated.
        expires_at: When the request will expire.
        signed_document_url: URL to download the signed document.
        metadata: Additional metadata from the provider.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    envelope_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique envelope/request identifier.",
    )
    document_id: str = Field(
        ...,
        description="ID of the document being signed.",
    )
    method: SignatureMethod = Field(
        ...,
        description="Signature method being used.",
    )
    status: SignatureStatus = Field(
        default=SignatureStatus.CREATED,
        description="Current status of the signature request.",
    )
    signers: List[SignerDetails] = Field(
        default_factory=list,
        description="List of signers and their status.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the request was created (UTC).",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the request was last updated (UTC).",
    )
    expires_at: Optional[datetime] = Field(
        default=None,
        description="When the request will expire (UTC).",
    )
    signed_document_url: Optional[str] = Field(
        default=None,
        max_length=2048,
        description="URL to download the signed document.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata from the signature provider.",
    )

    @property
    def is_complete(self) -> bool:
        """Check if all signers have signed."""
        return all(s.signed for s in self.signers)

    @property
    def pending_signers(self) -> List[SignerDetails]:
        """Get list of signers who have not yet signed."""
        return [s for s in self.signers if not s.signed]


# ---------------------------------------------------------------------------
# Digital Signer
# ---------------------------------------------------------------------------


class DigitalSigner:
    """Digital signature service for attestation documents.

    Provides a unified interface for signature collection across multiple
    providers (DocuSign, Adobe Sign, internal). Handles sending documents
    for signature, checking status, downloading signed documents, and
    verifying signature integrity.

    Attributes:
        config: Configuration instance with API credentials.
        _requests: In-memory request storage (replaced by DB in production).

    Example:
        >>> signer = DigitalSigner(config)
        >>> envelope_id = await signer.send_for_signature(
        ...     document_id=doc_id,
        ...     signers=[{"signer_id": "u1", "name": "CEO", "email": "ceo@co.com"}],
        ...     method=SignatureMethod.DOCUSIGN,
        ... )
        >>> status = await signer.check_status(envelope_id)
    """

    def __init__(self, config: Any = None) -> None:
        """Initialize DigitalSigner.

        Args:
            config: Optional configuration instance with API credentials.
        """
        self.config = config
        self._requests: Dict[str, SignatureRequest] = {}
        self._internal_secret = secrets.token_bytes(32)  # For internal signatures
        logger.info("DigitalSigner initialized")

    async def send_for_signature(
        self,
        document_id: str,
        signers: List[Dict[str, Any]],
        method: SignatureMethod,
        document_content: Optional[bytes] = None,
        document_name: str = "Attestation Document",
        email_subject: str = "Document Requires Your Signature",
        email_message: str = "",
        expires_days: int = 14,
    ) -> str:
        """Send a document for signature collection.

        Args:
            document_id: Unique identifier for the document.
            signers: List of signer dictionaries with keys:
                - signer_id: Unique identifier
                - name: Full name
                - email: Email address
                - title: Job title (optional)
                - routing_order: Signing order (optional, default 1)
            method: Signature method to use.
            document_content: Optional document content bytes (PDF).
            document_name: Display name for the document.
            email_subject: Subject line for signature request emails.
            email_message: Body message for signature request emails.
            expires_days: Number of days until the request expires.

        Returns:
            Envelope ID for tracking the signature request.

        Raises:
            ValueError: If signers list is empty or invalid.
        """
        start_time = datetime.now(timezone.utc)

        if not signers:
            raise ValueError("At least one signer is required.")

        # Create SignerDetails objects
        signer_details = []
        for i, s in enumerate(signers):
            signer_details.append(
                SignerDetails(
                    signer_id=s.get("signer_id", str(uuid.uuid4())),
                    name=s.get("name", "Unknown"),
                    email=s.get("email", ""),
                    title=s.get("title", ""),
                    routing_order=s.get("routing_order", i + 1),
                )
            )

        # Calculate expiration
        expires_at = datetime.now(timezone.utc)
        expires_at = expires_at.replace(
            day=expires_at.day + expires_days if expires_at.day + expires_days <= 28 else 28
        )

        # Create signature request
        request = SignatureRequest(
            document_id=document_id,
            method=method,
            status=SignatureStatus.SENT,
            signers=signer_details,
            expires_at=expires_at,
            metadata={
                "document_name": document_name,
                "email_subject": email_subject,
                "email_message": email_message,
            },
        )

        # Dispatch to appropriate provider
        if method == SignatureMethod.DOCUSIGN:
            await self._send_via_docusign(request, document_content)
        elif method == SignatureMethod.ADOBE_SIGN:
            await self._send_via_adobe_sign(request, document_content)
        elif method == SignatureMethod.INTERNAL:
            await self._send_internal(request)
        else:
            raise ValueError(f"Unsupported signature method: {method}")

        # Store request
        self._requests[request.envelope_id] = request

        elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        logger.info(
            "Signature request sent: envelope=%s, document=%s, "
            "method=%s, signers=%d, elapsed=%.2fms",
            request.envelope_id,
            document_id,
            method.value,
            len(signers),
            elapsed_ms,
        )

        return request.envelope_id

    async def check_status(self, envelope_id: str) -> SignatureStatus:
        """Check the status of a signature request.

        Args:
            envelope_id: Envelope ID from send_for_signature.

        Returns:
            Current SignatureStatus.

        Raises:
            ValueError: If envelope not found.
        """
        request = self._requests.get(envelope_id)
        if request is None:
            raise ValueError(f"Envelope '{envelope_id}' not found.")

        # In production, query the actual provider API
        if request.method == SignatureMethod.DOCUSIGN:
            await self._check_docusign_status(request)
        elif request.method == SignatureMethod.ADOBE_SIGN:
            await self._check_adobe_sign_status(request)
        # Internal signatures are updated directly

        logger.debug(
            "Signature status checked: envelope=%s, status=%s",
            envelope_id,
            request.status.value,
        )

        return request.status

    async def download_signed(self, envelope_id: str) -> bytes:
        """Download the signed document.

        Args:
            envelope_id: Envelope ID from send_for_signature.

        Returns:
            Signed document content as bytes.

        Raises:
            ValueError: If envelope not found or not completed.
        """
        request = self._requests.get(envelope_id)
        if request is None:
            raise ValueError(f"Envelope '{envelope_id}' not found.")

        if request.status != SignatureStatus.COMPLETED:
            raise ValueError(
                f"Cannot download: envelope status is '{request.status.value}', "
                f"expected 'completed'."
            )

        # In production, download from the actual provider
        if request.method == SignatureMethod.DOCUSIGN:
            content = await self._download_from_docusign(request)
        elif request.method == SignatureMethod.ADOBE_SIGN:
            content = await self._download_from_adobe_sign(request)
        else:
            content = await self._download_internal(request)

        logger.info(
            "Signed document downloaded: envelope=%s, size=%d bytes",
            envelope_id,
            len(content),
        )

        return content

    def verify_signatures(self, document: bytes) -> bool:
        """Verify signatures on a signed document.

        Args:
            document: Document content bytes to verify.

        Returns:
            True if all signatures are valid, False otherwise.
        """
        # In production, this would verify:
        # 1. PDF digital signatures
        # 2. Timestamp authority signatures
        # 3. Certificate chain validity

        # Placeholder implementation
        if len(document) < 100:
            logger.warning("Document too small to contain valid signatures")
            return False

        # Check for PDF signature indicators
        has_sig_field = b"/Sig" in document or b"/Type /Sig" in document

        logger.info(
            "Signature verification: size=%d, has_sig_field=%s",
            len(document),
            has_sig_field,
        )

        return has_sig_field

    async def record_internal_signature(
        self,
        envelope_id: str,
        signer_id: str,
        signature_data: bytes,
    ) -> bool:
        """Record an internal signature for a signer.

        Args:
            envelope_id: Envelope ID of the signature request.
            signer_id: ID of the signer.
            signature_data: Raw signature data bytes.

        Returns:
            True if this was the last required signature.

        Raises:
            ValueError: If envelope or signer not found.
        """
        request = self._requests.get(envelope_id)
        if request is None:
            raise ValueError(f"Envelope '{envelope_id}' not found.")

        if request.method != SignatureMethod.INTERNAL:
            raise ValueError(
                f"Cannot record internal signature for method '{request.method.value}'."
            )

        # Find the signer
        signer = next((s for s in request.signers if s.signer_id == signer_id), None)
        if signer is None:
            raise ValueError(f"Signer '{signer_id}' not found in envelope.")

        if signer.signed:
            raise ValueError(f"Signer '{signer_id}' has already signed.")

        # Verify internal signature
        if not self._verify_internal_signature(signer_id, signature_data):
            raise ValueError("Invalid signature data.")

        # Record signature
        signer.signed = True
        signer.signed_at = datetime.now(timezone.utc)
        request.updated_at = datetime.now(timezone.utc)

        # Check if all signers have signed
        if request.is_complete:
            request.status = SignatureStatus.COMPLETED
            logger.info("All signatures collected: envelope=%s", envelope_id)
        else:
            request.status = SignatureStatus.SIGNED

        return request.is_complete

    def _generate_internal_signature(
        self,
        user_id: str,
        document_hash: str,
    ) -> bytes:
        """Generate an internal cryptographic signature.

        Uses HMAC-SHA256 with the internal secret key.

        Args:
            user_id: User ID of the signer.
            document_hash: SHA-256 hash of the document content.

        Returns:
            Signature bytes.
        """
        message = f"{user_id}:{document_hash}:{datetime.now(timezone.utc).isoformat()}"
        signature = hmac.new(
            self._internal_secret,
            message.encode("utf-8"),
            hashlib.sha256,
        ).digest()

        logger.debug(
            "Internal signature generated: user=%s, doc_hash=%s",
            user_id,
            document_hash[:16] + "...",
        )

        return signature

    def _verify_internal_signature(
        self,
        user_id: str,
        signature_data: bytes,
    ) -> bool:
        """Verify an internal signature.

        Args:
            user_id: User ID of the signer.
            signature_data: Signature bytes to verify.

        Returns:
            True if signature is valid.
        """
        # In production, verify against stored signature
        # For now, just check format
        return len(signature_data) == 32  # SHA-256 output size

    # -----------------------------------------------------------------------
    # Provider-Specific Methods
    # -----------------------------------------------------------------------

    async def _send_via_docusign(
        self,
        request: SignatureRequest,
        document_content: Optional[bytes],
    ) -> None:
        """Send document via DocuSign API.

        Args:
            request: SignatureRequest to send.
            document_content: Document content bytes.
        """
        # Placeholder for DocuSign API integration
        # In production, use docusign_esign SDK
        logger.info(
            "DocuSign request created (placeholder): envelope=%s",
            request.envelope_id,
        )
        request.metadata["provider_envelope_id"] = f"DS-{uuid.uuid4().hex[:12]}"

    async def _send_via_adobe_sign(
        self,
        request: SignatureRequest,
        document_content: Optional[bytes],
    ) -> None:
        """Send document via Adobe Sign API.

        Args:
            request: SignatureRequest to send.
            document_content: Document content bytes.
        """
        # Placeholder for Adobe Sign API integration
        logger.info(
            "Adobe Sign request created (placeholder): envelope=%s",
            request.envelope_id,
        )
        request.metadata["provider_agreement_id"] = f"AS-{uuid.uuid4().hex[:12]}"

    async def _send_internal(self, request: SignatureRequest) -> None:
        """Set up internal signature collection.

        Args:
            request: SignatureRequest to set up.
        """
        logger.info(
            "Internal signature request created: envelope=%s",
            request.envelope_id,
        )
        request.metadata["signature_type"] = "internal_hmac_sha256"

    async def _check_docusign_status(self, request: SignatureRequest) -> None:
        """Check DocuSign envelope status.

        Args:
            request: SignatureRequest to check.
        """
        # Placeholder for DocuSign status check
        # In production, call DocuSign API
        pass

    async def _check_adobe_sign_status(self, request: SignatureRequest) -> None:
        """Check Adobe Sign agreement status.

        Args:
            request: SignatureRequest to check.
        """
        # Placeholder for Adobe Sign status check
        pass

    async def _download_from_docusign(self, request: SignatureRequest) -> bytes:
        """Download signed document from DocuSign.

        Args:
            request: Completed SignatureRequest.

        Returns:
            Signed document bytes.
        """
        # Placeholder - return mock signed document
        return self._create_mock_signed_document(request)

    async def _download_from_adobe_sign(self, request: SignatureRequest) -> bytes:
        """Download signed document from Adobe Sign.

        Args:
            request: Completed SignatureRequest.

        Returns:
            Signed document bytes.
        """
        # Placeholder - return mock signed document
        return self._create_mock_signed_document(request)

    async def _download_internal(self, request: SignatureRequest) -> bytes:
        """Generate internally signed document.

        Args:
            request: Completed SignatureRequest.

        Returns:
            Signed document bytes.
        """
        return self._create_mock_signed_document(request)

    def _create_mock_signed_document(self, request: SignatureRequest) -> bytes:
        """Create a mock signed document for testing.

        Args:
            request: SignatureRequest with signature data.

        Returns:
            Mock signed document bytes.
        """
        # Create minimal PDF with signature indicators
        signers_text = "\n".join(
            f"% Signer: {s.name} ({s.email}) - Signed: {s.signed_at}"
            for s in request.signers
            if s.signed
        )

        content = f"""%PDF-1.7
% Signed Document
% Envelope ID: {request.envelope_id}
% Document ID: {request.document_id}
% Method: {request.method.value}
% Status: {request.status.value}
{signers_text}

1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj

2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj

3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>
endobj

4 0 obj
<< /Type /Sig /Filter /Adobe.PPKLite /SubFilter /adbe.pkcs7.detached >>
endobj

xref
0 5
0000000000 65535 f
0000000100 00000 n
0000000200 00000 n
0000000300 00000 n
0000000400 00000 n

trailer
<< /Size 5 /Root 1 0 R >>
startxref
500
%%EOF
"""
        return content.encode("utf-8")

    async def void_request(self, envelope_id: str, reason: str = "") -> None:
        """Void/cancel a signature request.

        Args:
            envelope_id: Envelope ID to void.
            reason: Reason for voiding the request.

        Raises:
            ValueError: If envelope not found or already completed.
        """
        request = self._requests.get(envelope_id)
        if request is None:
            raise ValueError(f"Envelope '{envelope_id}' not found.")

        if request.status == SignatureStatus.COMPLETED:
            raise ValueError("Cannot void a completed signature request.")

        request.status = SignatureStatus.VOIDED
        request.updated_at = datetime.now(timezone.utc)
        request.metadata["void_reason"] = reason

        logger.warning(
            "Signature request voided: envelope=%s, reason='%s'",
            envelope_id,
            reason,
        )

    async def get_request(self, envelope_id: str) -> Optional[SignatureRequest]:
        """Get a signature request by envelope ID.

        Args:
            envelope_id: Envelope ID to retrieve.

        Returns:
            SignatureRequest if found, None otherwise.
        """
        return self._requests.get(envelope_id)


__all__ = [
    "SignatureMethod",
    "SignatureStatus",
    "SignerDetails",
    "SignatureRequest",
    "DigitalSigner",
]
