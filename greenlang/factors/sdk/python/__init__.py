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
    ProfileGatedError,
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
    AlphaFactor,
    AlphaPack,
    AlphaSource,
    AuditBundle,
    BatchJobHandle,
    ChosenFactor,
    Citation,
    CoverageReport,
    DeprecationStatus,
    Edition,
    Extraction,
    Factor,
    FactorDiff,
    FactorMatch,
    GasBreakdown,
    HealthResponse,
    Jurisdiction,
    LicensingEnvelope,
    ListFactorsResponse,
    MethodPack,
    MethodPackCoverage,
    MethodPackCoverageReport,
    Override,
    QualityEnvelope,
    QualityScore,
    ResolutionRequest,
    ResolvedFactor,
    Review,
    SearchResponse,
    SignedReceipt,
    Source,
    SourceDescriptor,
    Uncertainty,
    UncertaintyEnvelope,
)

__version__ = "0.1.0"
"""Public version string for the GreenLang Factors Python SDK.

Renumbered to 0.1.0 (2026-04-25) per CTO doc §19.1 v0.1 Alpha contract.
The 1.3.0 line was forward-development released too aggressively; the
distribution is now collapsed to a clean 0.x alpha line. SDK is marked
``Development Status :: 3 - Alpha``; breaking changes are expected
until v1.0 GA.

The alpha surface ships ONLY the five read-only GETs declared in
CTO doc §19.1:

  * :meth:`FactorsClient.health` -> ``GET /v1/healthz``
  * :meth:`FactorsClient.list_factors` -> ``GET /v1/factors``
  * :meth:`FactorsClient.get_factor` -> ``GET /v1/factors/{urn}``
  * :meth:`FactorsClient.list_sources` -> ``GET /v1/sources``
  * :meth:`FactorsClient.list_packs` -> ``GET /v1/packs``

Forward-development methods (resolve, explain, batch, edition pinning,
signed-receipt verification) remain on the client class but are gated
behind ``release_profile.feature_enabled(...)``; under
``GL_FACTORS_RELEASE_PROFILE=alpha-v0.1`` they raise
``ProfileGatedError``. They re-enable under ``beta-v0.5`` and higher.

See ``RELEASE_NOTES_v0.1.0.md`` for the full alpha contract.
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
    # v0.1 Alpha — URN-primary models
    "AlphaFactor",
    "AlphaPack",
    "AlphaSource",
    "Citation",
    "Extraction",
    "HealthResponse",
    "ListFactorsResponse",
    "Review",
    "ProfileGatedError",
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
