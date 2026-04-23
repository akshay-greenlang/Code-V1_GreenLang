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

    # --- routers -----------------------------------------------------------
    try:
        from greenlang.factors.api_v1_routes import api_v1_router
        app.include_router(api_v1_router)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to mount api_v1_router: %s", exc)
        raise

    if enable_admin:
        try:
            from greenlang.factors.api_v1_routes import admin_router
            app.include_router(admin_router)
        except Exception as exc:  # noqa: BLE001
            logger.warning("admin_router not mounted: %s", exc)

    # Billing router (Agent 7 owns greenlang/factors/billing/api.py).
    if enable_billing in (None, True):
        try:
            from greenlang.factors.billing.api import billing_router  # type: ignore
            app.include_router(billing_router)
            logger.info("Mounted billing_router")
        except Exception as exc:  # noqa: BLE001
            if enable_billing:
                logger.error("billing_router required but not available: %s", exc)
            else:
                logger.info("billing_router not available; skipping")

    # OEM router (Agent 8 owns greenlang/factors/onboarding/api.py).
    if enable_oem in (None, True):
        try:
            from greenlang.factors.onboarding.api import oem_router  # type: ignore
            app.include_router(oem_router)
            logger.info("Mounted oem_router")
        except Exception as exc:  # noqa: BLE001
            if enable_oem:
                logger.error("oem_router required but not available: %s", exc)
            else:
                logger.info("oem_router not available; skipping")

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

    return app


# Module-level singleton so `gunicorn ... greenlang.factors.factors_app:app`
# works without explicit factory invocation.
try:
    app = create_factors_app()
except Exception:  # noqa: BLE001
    # In dev / test, missing optional deps must not prevent imports.
    app = None  # type: ignore[assignment]
