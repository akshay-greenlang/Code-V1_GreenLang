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
    MethodPackCoverage,
    MethodPackCoverageReport,
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

__version__ = "1.3.0"
"""Public version string for the GreenLang Factors Python SDK.

Bumped to 1.3.0 (2026-04-24) for Wave 5 contract disambiguations.
See ``RELEASE_NOTES_v1.3.0.md`` for the full changelog. Summary:

 - Uncertainty unit disambiguation: ``uncertainty`` is ABSOLUTE (native
   unit), ``uncertainty_percent`` is RELATIVE (0-100). Both fields
   surfaced on :class:`Uncertainty` and :class:`UncertaintyEnvelope`.
 - Deprecation-status canonicalization: the wire may carry a bare string
   or a dict; the SDK always exposes a typed :class:`DeprecationStatus`
   via :meth:`DeprecationStatus.from_any`. Canonical keys are
   ``status`` / ``successor_id`` / ``reason`` / ``deprecated_at``.
 - Coverage endpoint unification: new :class:`MethodPackCoverageReport`
   (``{packs:[...], overall:{...}}``) returned from
   :meth:`FactorsClient.method_pack_coverage`, regardless of whether a
   ``?pack=<slug>`` filter was applied. The Wave 4-G legacy shape is
   inflated transparently.

All three changes are SDK-side normalizations — the wire protocol is
backward compatible and older clients continue to work unchanged.
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
    "MethodPackCoverage",
    "MethodPackCoverageReport",
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
