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
    FactorNotFoundError,
    FactorsAPIError,
    LicenseError,
    RateLimitError,
    TierError,
    ValidationError,
)
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
    CoverageReport,
    Edition,
    Factor,
    FactorDiff,
    FactorMatch,
    GasBreakdown,
    Jurisdiction,
    MethodPack,
    Override,
    QualityScore,
    ResolutionRequest,
    ResolvedFactor,
    SearchResponse,
    Source,
    Uncertainty,
)

__version__ = "1.0.0"

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
    # Errors
    "FactorsAPIError",
    "RateLimitError",
    "TierError",
    "FactorNotFoundError",
    "LicenseError",
    "ValidationError",
    "AuthError",
    # Metadata
    "__version__",
]
