# -*- coding: utf-8 -*-
"""
GreenLang Factors API — FastAPI application factory.

Mounts every router and middleware required for the FY27 launch:

  * /v1/health                 — liveness + active edition id
  * /v1/resolve                — POST, full 7-step cascade with explain
  * /v1/factors/{id}           — GET, factor record + explain
  * /v1/factors/{id}/explain   — GET, explain payload only
  * /v1/coverage               — GET, three-label counts per family
  * /v1/quality/fqs            — GET, composite FQS distribution per family
  * /v1/editions/{id}          — GET, signed edition manifest
  * /v1/admin/*                — operator console backends (admin tier)
  * /v1/billing/*              — Stripe-backed checkout & webhook (optional)
  * /v1/oem/*                  — OEM partner / sub-tenant flows (optional)
  * /metrics                   — Prometheus exposition
  * /openapi.json, /docs, /redoc

Middleware order (outer → inner):

    EditionPinMiddleware
    AuthMeteringMiddleware
    RateLimitMiddleware
    LicensingGuardMiddleware
    SignedReceiptsMiddleware

Run with:

    gunicorn -k uvicorn.workers.UvicornWorker greenlang.factors.factors_app:app

Or in tests:

    from fastapi.testclient import TestClient
    from greenlang.factors.factors_app import create_factors_app
    app = create_factors_app()
    client = TestClient(app)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _redact_dsn(dsn: str) -> str:
    """Hide credentials in a DSN before logging it."""
    if "@" not in dsn or "://" not in dsn:
        return dsn
    scheme, rest = dsn.split("://", 1)
    if "@" in rest:
        _, host = rest.rsplit("@", 1)
        return f"{scheme}://***@{host}"
    return dsn


def _assert_signing_keys_loaded_for_prod() -> None:
    """
    DEP2 startup guard: in staging/production, a missing Ed25519 private key
    is a hard boot failure. Signed receipts are non-negotiable there — we
    never silently fall back to HMAC like we do in dev.

    Raises:
        RuntimeError: if APP_ENV is staging/production and the Ed25519
            private key env var is unset or empty.
    """
    app_env = (os.getenv("APP_ENV") or os.getenv("GL_ENV") or os.getenv("ENVIRONMENT") or "").lower()
    if app_env not in {"staging", "production", "prod"}:
        return
    priv = os.getenv("GL_FACTORS_ED25519_PRIVATE_KEY") or os.getenv("SIGNING_KEY_ED25519_PRIV")
    if not priv or not priv.strip():
        raise RuntimeError(
            "GL_FACTORS_ED25519_PRIVATE_KEY (or SIGNING_KEY_ED25519_PRIV) is not set "
            f"but APP_ENV={app_env!r}. Signed receipts are mandatory in staging and "
            "production. Verify ExternalSecret `factors-api-secrets` synced from Vault "
            "path kv/factors/<env>/signing (see deployment/runbooks/factors_incidents.md "
            "#signed-receipt-verification-failure). Do NOT disable signing to work around this."
        )


def create_factors_app(
    *,
    service: Optional[Any] = None,
    enable_admin: bool = True,
    enable_billing: Optional[bool] = None,
    enable_oem: Optional[bool] = None,
    enable_metrics: bool = True,
):
    """Factory returning a fully-mounted FastAPI app for Factors.

    All dependencies are injected; call sites pass ``service`` for tests
    and let the default ``FactorCatalogService.from_environment()`` boot
    the production catalog otherwise.
    """
    # DEP2: fail fast at boot if we're in staging/prod without signing keys.
    # Placed before any FastAPI import so the traceback is small and crisp.
    _assert_signing_keys_loaded_for_prod()

    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(
        title="GreenLang Factors API",
        version="1.0.0",
        description=(
            "Canonical climate reference layer. Resolve emissions factors with "
            "full provenance, version pinning, and signed receipts. Every "
            "factor response includes an `explain` block by default; pass "
            "`?compact=true` to opt out. See https://developers.greenlang.ai."
        ),
        openapi_url="/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # --- service singleton (catalog + repo) --------------------------------
    if service is None:
        try:
            from greenlang.factors.service import FactorCatalogService
            service = FactorCatalogService.from_environment()
        except Exception as exc:  # noqa: BLE001
            logger.warning("FactorCatalogService.from_environment() failed: %s", exc)
            service = None
    app.state.factors_service = service

    # --- middleware stack (applied bottom-up by FastAPI) -------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("GL_FACTORS_CORS_ORIGINS", "*").split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    try:
        from greenlang.factors.middleware.signed_receipts import SignedReceiptsMiddleware
        app.add_middleware(SignedReceiptsMiddleware)
    except Exception as exc:  # noqa: BLE001
        logger.warning("SignedReceiptsMiddleware not mounted: %s", exc)

    try:
        from greenlang.factors.middleware.licensing_guard import LicensingGuardMiddleware
        app.add_middleware(LicensingGuardMiddleware)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LicensingGuardMiddleware not mounted: %s", exc)

    try:
        from greenlang.factors.middleware.rate_limiter import RateLimitMiddleware
        app.add_middleware(RateLimitMiddleware)
    except Exception as exc:  # noqa: BLE001
        logger.warning("RateLimitMiddleware not mounted: %s", exc)

    try:
        from greenlang.factors.middleware.auth_metering import AuthMeteringMiddleware
        app.add_middleware(AuthMeteringMiddleware)
    except Exception as exc:  # noqa: BLE001
        logger.warning("AuthMeteringMiddleware not mounted: %s", exc)

    try:
        from greenlang.factors.middleware.edition_pin import EditionPinMiddleware
        app.add_middleware(EditionPinMiddleware)
    except Exception as exc:  # noqa: BLE001
        logger.warning("EditionPinMiddleware not mounted: %s", exc)

    # --- release-profile feature gate (WS11-T1) ----------------------------
    # Determine which surfaces are exposed for the active profile. In ``dev``
    # every feature is on so existing tests keep passing; in ``alpha-v0.1``
    # only the five always-on endpoints survive.
    from greenlang.factors.release_profile import (
        current_profile,
        feature_enabled,
        filter_app_routes,
    )
    _profile = current_profile()
    logger.info("Factors release profile: %s", _profile.value)

    # --- routers -----------------------------------------------------------
    # Alpha v0.1 short-circuit (CTO doc §19.1): under release_profile=alpha-v0.1
    # the public surface is EXACTLY the 5 read-only GETs in api_v0_1_alpha_routes.
    # We skip mounting api_v1_router (which carries resolve/explain/batch/edition
    # routes that are out of scope for alpha) and instead mount only the alpha
    # router plus the /api/v1 -> 410 Gone catch-all.
    if _profile.value == "alpha-v0.1":
        try:
            from greenlang.factors.api_v0_1_alpha_routes import (
                router as alpha_v0_1_router,
                deprecated_router as alpha_v0_1_deprecated_router,
            )
            app.include_router(alpha_v0_1_router)
            app.include_router(alpha_v0_1_deprecated_router)
            logger.info(
                "Mounted api_v0_1_alpha_routes (%d public GETs); skipped api_v1_router under alpha profile",
                5,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to mount api_v0_1_alpha_routes: %s", exc)
            raise

        # Wave D / TaskCreate #31: bind a real :class:`AlphaFactorRepository`
        # so v0.1-shape records round-trip without ``_coerce_v0_1`` lossy
        # re-shaping. Env-var precedence: GL_FACTORS_ALPHA_REPO_DSN wins;
        # else fall back to ``sqlite:///<GL_FACTORS_SQLITE_PATH>`` if that
        # path is set; else ``sqlite:///:memory:``.
        try:
            from greenlang.factors.repositories import AlphaFactorRepository
            dsn = os.getenv("GL_FACTORS_ALPHA_REPO_DSN", "").strip()
            if not dsn:
                sqlite_path = os.getenv("GL_FACTORS_SQLITE_PATH", "").strip()
                dsn = (
                    f"sqlite:///{sqlite_path}" if sqlite_path
                    else "sqlite:///:memory:"
                )
            app.state.alpha_factor_repo = AlphaFactorRepository(dsn)
            logger.info("Wired AlphaFactorRepository (dsn=%s)", _redact_dsn(dsn))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "AlphaFactorRepository not wired: %s. Alpha router will fall "
                "back to the legacy factors_service path.",
                exc,
            )
            app.state.alpha_factor_repo = None
    else:
        try:
            from greenlang.factors.api_v1_routes import api_v1_router
            app.include_router(api_v1_router)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to mount api_v1_router: %s", exc)
            raise

    # W4-G / MP14: method-pack coverage endpoint (separate router to avoid
    # colliding with W4-C's api_v1_routes ownership).
    if feature_enabled("method_packs") or feature_enabled("explain_endpoint"):
        try:
            from greenlang.factors.method_packs.coverage_routes import (
                method_packs_router,
            )
            app.include_router(method_packs_router)
            logger.info("Mounted method_packs_router (MP14 coverage endpoint)")
        except Exception as exc:  # noqa: BLE001
            logger.warning("method_packs_router not mounted: %s", exc)
    else:
        logger.info("method_packs_router gated by release_profile=%s", _profile.value)

    if enable_admin and feature_enabled("admin_console"):
        try:
            from greenlang.factors.api_v1_routes import admin_router
            app.include_router(admin_router)
        except Exception as exc:  # noqa: BLE001
            logger.warning("admin_router not mounted: %s", exc)
    elif enable_admin:
        logger.info("admin_router gated by release_profile=%s", _profile.value)

    # W4-C / API15: GraphQL surface. Soft-mount so the app still boots
    # when ``strawberry-graphql`` isn't installed — calls to /v1/graphql
    # then return 503 with an install hint.
    if feature_enabled("graphql"):
        try:
            from greenlang.factors.graphql import graphql_router
            app.include_router(graphql_router)
            logger.info("Mounted graphql_router")
        except Exception as exc:  # noqa: BLE001
            logger.info("graphql_router not mounted (optional): %s", exc)
    else:
        logger.info("graphql_router gated by release_profile=%s", _profile.value)

    # Billing router (Agent 7 owns greenlang/factors/billing/api.py).
    if enable_billing in (None, True) and feature_enabled("billing"):
        try:
            from greenlang.factors.billing.api import billing_router  # type: ignore
            app.include_router(billing_router)
            logger.info("Mounted billing_router")
        except Exception as exc:  # noqa: BLE001
            if enable_billing:
                logger.error("billing_router required but not available: %s", exc)
            else:
                logger.info("billing_router not available; skipping")

        # TODO-marker (Agent W4-E / C5): self-serve billing routes for the
        # FY27 Pricing Page. Mounts POST /v1/billing/checkout/session,
        # POST /v1/billing/portal/session, and GET /v1/billing/subscription
        # alongside the existing billing_router.
        try:
            from greenlang.factors.billing.api_routes import (  # type: ignore
                router as billing_self_serve_router,
            )
            app.include_router(billing_self_serve_router)
            logger.info("Mounted billing self-serve api_routes router (W4-E)")
        except Exception as exc:  # noqa: BLE001
            logger.warning("billing self-serve api_routes not mounted: %s", exc)
    elif enable_billing in (None, True):
        logger.info("billing_router gated by release_profile=%s", _profile.value)

    # OEM router (Agent 8 owns greenlang/factors/onboarding/api.py).
    if enable_oem in (None, True) and feature_enabled("oem"):
        try:
            from greenlang.factors.onboarding.api import oem_router  # type: ignore
            app.include_router(oem_router)
            logger.info("Mounted oem_router")
        except Exception as exc:  # noqa: BLE001
            if enable_oem:
                logger.error("oem_router required but not available: %s", exc)
            else:
                logger.info("oem_router not available; skipping")
    elif enable_oem in (None, True):
        logger.info("oem_router gated by release_profile=%s", _profile.value)

    # Prometheus /metrics
    if enable_metrics:
        try:
            from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
            from starlette.responses import Response as _PromResponse

            @app.get("/metrics", include_in_schema=False)
            def metrics():
                return _PromResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Prometheus /metrics not mounted: %s", exc)

    # Final pass: cull routes inside api_v1_router that belong to
    # disabled features (resolve, batch, explain, coverage, fqs, edition,
    # signed-receipt status). No-op when every feature is enabled (dev/GA).
    try:
        filter_app_routes(app)
    except Exception as exc:  # noqa: BLE001
        logger.warning("release_profile route filter failed: %s", exc)

    return app


# Module-level singleton so `gunicorn ... greenlang.factors.factors_app:app`
# works without explicit factory invocation.
try:
    app = create_factors_app()
except Exception as _boot_exc:  # noqa: BLE001
    # In dev / test, missing optional deps must not prevent imports.
    # In staging / production, the signing-keys assertion MUST propagate —
    # silently swallowing a boot failure there would unsign the audit trail.
    _boot_env = (os.getenv("APP_ENV") or os.getenv("GL_ENV") or os.getenv("ENVIRONMENT") or "").lower()
    if _boot_env in {"staging", "production", "prod"}:
        raise
    app = None  # type: ignore[assignment]
