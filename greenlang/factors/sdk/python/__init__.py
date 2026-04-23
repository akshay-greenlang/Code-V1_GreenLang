# -*- coding: utf-8 -*-
"""GreenLang Factors — Production Python SDK.

Full-featured, typed HTTP client wrapping the Factors REST API
(``/api/v1/factors`` and ``/api/v1/editions``).

Highlights:
    * Sync + Async clients (``FactorsClient`` + ``AsyncFactorsClient``)
    * JWT + API Key authentication, optional HMAC request signing (Pro+)
    * Automatic 429 / 5xx retry with exponential backoff (tenacity)
    * ETag-aware response cache (If-None-Match)
    * Typed Pydantic v2 request + response models (on ``GreenLangBase``)
    * Cursor + offset paginators
    * HMAC-SHA256 webhook signature verifier
    * CLI: ``greenlang-factors search|resolve|explain|get-factor|list-editions``

Example::

    from greenlang.factors.sdk.python import FactorsClient, APIKeyAuth

    client = FactorsClient(
        base_url="https://api.greenlang.io",
        auth=APIKeyAuth(api_key="gl_fac_..."),
    )
    hits = client.search("diesel US Scope 1", limit=5)
    for f in hits.factors:
        print(f.factor_id, f.co2e_per_unit)

See :mod:`greenlang.factors.sdk.python.client` for full API surface.
"""
from __future__ import annotations

from .auth import APIKeyAuth, AuthProvider, HMACAuth, JWTAuth
from .client import AsyncFactorsClient, FactorsClient
from .errors import (
    AuthError,
    EditionMismatchError,
    EditionPinError,
    EntitlementError,
    FactorCannotResolveSafelyError,
    FactorNotFoundError,
    FactorsAPIError,
    LicenseError,
    LicensingGapError,
    RateLimitError,
    TierError,
    ValidationError,
)
from .client import (
    CertificatePinError,
    CertPinnedHTTPAdapter,
    GREENLANG_CA_PEM,
)
from .verify import ReceiptVerificationError, verify_receipt
from .webhooks import (
    WebhookVerificationError,
    compute_signature,
    verify_webhook,
    verify_webhook_bytes,
    verify_webhook_strict,
)
from .models import (
    ActivitySchema,
    AuditBundle,
    BatchJobHandle,
    ChosenFactor,
    CoverageReport,
    DeprecationStatus,
    Edition,
    Factor,
    FactorDiff,
    FactorMatch,
    GasBreakdown,
    Jurisdiction,
    LicensingEnvelope,
    MethodPack,
    Override,
    QualityEnvelope,
    QualityScore,
    ResolutionRequest,
    ResolvedFactor,
    SearchResponse,
    SignedReceipt,
    Source,
    SourceDescriptor,
    Uncertainty,
    UncertaintyEnvelope,
)

__version__ = "1.2.0"
"""Public version string for the GreenLang Factors Python SDK.

Bumped to 1.2.0 for Wave 2 / Wave 2a / Wave 2.5 envelope support:
 - Signed receipt key renames (signed_receipt / alg / payload_hash).
 - New typed envelope models (ChosenFactor, SourceDescriptor, quality
   composite FQS 0-100, UncertaintyEnvelope, LicensingEnvelope,
   DeprecationStatus, SignedReceipt).
 - ``audit_text`` + ``audit_text_draft`` on :class:`ResolvedFactor`.
 - :class:`FactorCannotResolveSafelyError` for resolver 422 refusals.
"""

__all__ = [
    # Client
    "FactorsClient",
    "AsyncFactorsClient",
    # Auth
    "AuthProvider",
    "APIKeyAuth",
    "JWTAuth",
    "HMACAuth",
    # Webhooks
    "WebhookVerificationError",
    "compute_signature",
    "verify_webhook",
    "verify_webhook_bytes",
    "verify_webhook_strict",
    # Models
    "Factor",
    "Edition",
    "Source",
    "MethodPack",
    "ResolvedFactor",
    "ResolutionRequest",
    "QualityScore",
    "GasBreakdown",
    "Uncertainty",
    "Jurisdiction",
    "ActivitySchema",
    "SearchResponse",
    "FactorMatch",
    "FactorDiff",
    "AuditBundle",
    "Override",
    "CoverageReport",
    "BatchJobHandle",
    # Wave 2 envelopes
    "ChosenFactor",
    "SourceDescriptor",
    "QualityEnvelope",
    "UncertaintyEnvelope",
    "LicensingEnvelope",
    "DeprecationStatus",
    "SignedReceipt",
    # Errors
    "FactorsAPIError",
    "RateLimitError",
    "TierError",
    "FactorNotFoundError",
    "FactorCannotResolveSafelyError",
    "LicenseError",
    "LicensingGapError",
    "EntitlementError",
    "ValidationError",
    "AuthError",
    "EditionPinError",
    "EditionMismatchError",
    # Receipt verification
    "ReceiptVerificationError",
    "verify_receipt",
    # Pinning
    "CertificatePinError",
    "CertPinnedHTTPAdapter",
    "GREENLANG_CA_PEM",
    # Metadata
    "__version__",
]
