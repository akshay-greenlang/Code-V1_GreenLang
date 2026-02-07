# -*- coding: utf-8 -*-
"""
Auth Setup - JWT Authentication Service (SEC-001)

Central module for configuring JWT authentication on any FastAPI application.
Provides ``configure_auth(app)`` which:

1. Registers ``AuthenticationMiddleware`` from ``greenlang.auth.middleware``
   so every request gets an ``AuthContext`` injected into ``request.state.auth``.
2. Includes the auth service API routers (auth, user, admin).
3. Calls ``protect_router()`` on every router in the app to inject
   ``AuthDependency`` and ``PermissionDependency`` as FastAPI dependencies.

Usage:
    >>> from fastapi import FastAPI
    >>> from greenlang.infrastructure.auth_service.auth_setup import configure_auth
    >>>
    >>> app = FastAPI()
    >>> # ... include your routers ...
    >>> configure_auth(app)

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Set

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI
    from fastapi.routing import APIRoute

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# Audit service imports (SEC-005)
try:
    from greenlang.infrastructure.audit_service.api import audit_router
    from greenlang.infrastructure.audit_service.middleware import AuditMiddleware

    AUDIT_SERVICE_AVAILABLE = True
except ImportError:
    audit_router = None
    AuditMiddleware = None
    AUDIT_SERVICE_AVAILABLE = False

# Secrets service imports (SEC-006)
try:
    from greenlang.infrastructure.secrets_service.api import secrets_router

    SECRETS_SERVICE_AVAILABLE = True
except ImportError:
    secrets_router = None
    SECRETS_SERVICE_AVAILABLE = False

# Security scanning service imports (SEC-007)
try:
    from greenlang.infrastructure.security_scanning.api import security_router

    SECURITY_SCANNING_AVAILABLE = True
except ImportError:
    security_router = None
    SECURITY_SCANNING_AVAILABLE = False

# SOC 2 preparation service imports (SEC-009)
try:
    from greenlang.infrastructure.soc2_preparation.api import soc2_router

    SOC2_PREPARATION_AVAILABLE = True
except ImportError:
    soc2_router = None
    SOC2_PREPARATION_AVAILABLE = False

# Security Operations service imports (SEC-010)
try:
    from greenlang.infrastructure.incident_response.api import incident_router

    INCIDENT_RESPONSE_AVAILABLE = True
except ImportError:
    incident_router = None
    INCIDENT_RESPONSE_AVAILABLE = False

try:
    from greenlang.infrastructure.threat_modeling.api import threat_router

    THREAT_MODELING_AVAILABLE = True
except ImportError:
    threat_router = None
    THREAT_MODELING_AVAILABLE = False

try:
    from greenlang.infrastructure.waf_management.api import waf_router

    WAF_MANAGEMENT_AVAILABLE = True
except ImportError:
    waf_router = None
    WAF_MANAGEMENT_AVAILABLE = False

try:
    from greenlang.infrastructure.vulnerability_disclosure.api import vdp_router

    VDP_AVAILABLE = True
except ImportError:
    vdp_router = None
    VDP_AVAILABLE = False

try:
    from greenlang.infrastructure.compliance_automation.api import compliance_router

    COMPLIANCE_AUTOMATION_AVAILABLE = True
except ImportError:
    compliance_router = None
    COMPLIANCE_AUTOMATION_AVAILABLE = False

try:
    from greenlang.infrastructure.security_training.api import training_router

    SECURITY_TRAINING_AVAILABLE = True
except ImportError:
    training_router = None
    SECURITY_TRAINING_AVAILABLE = False

# PII service imports (SEC-011)
try:
    from greenlang.infrastructure.pii_service.api import pii_router

    PII_SERVICE_AVAILABLE = True
except ImportError:
    pii_router = None
    PII_SERVICE_AVAILABLE = False


def configure_auth(
    app: "FastAPI",
    *,
    jwt_handler: Any = None,
    api_key_manager: Any = None,
    exclude_paths: Optional[Set[str]] = None,
    include_auth_routes: bool = True,
    protect_existing_routes: bool = True,
    enrich_with_rbac: bool = True,
) -> None:
    """Configure JWT authentication on a FastAPI application.

    This is the **single entry-point** for wiring SEC-001 into any GreenLang
    FastAPI service.  It performs three steps:

    1. Register ``AuthenticationMiddleware`` (sets ``request.state.auth``).
    2. Include the auth-service routers (``/auth/*``, ``/auth/admin/*``).
    3. Walk every existing route and inject ``AuthDependency`` +
       ``PermissionDependency`` via ``protect_router()``.

    Args:
        app: The FastAPI application instance.
        jwt_handler: Optional ``JWTHandler`` for the middleware. When *None*,
            the middleware still runs but relies on downstream ``AuthDependency``
            to validate tokens.
        api_key_manager: Optional ``APIKeyManager`` for API-key auth.
        exclude_paths: Additional paths to exclude from authentication
            (merged with the default public-path set).
        include_auth_routes: When *True* (default), mount the auth, user,
            and admin routers on the app.
        protect_existing_routes: When *True* (default), call
            ``protect_router()`` on every router already mounted on the app.
        enrich_with_rbac: When *True* (default), register a FastAPI
            dependency that loads the authenticated user's RBAC roles and
            permissions from the database (via ``RBACCache`` /
            ``AssignmentService``) into the ``AuthContext`` on every
            request.  Requires SEC-002 RBAC modules to be installed.

    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> configure_auth(app, jwt_handler=my_jwt_handler)
    """
    if not FASTAPI_AVAILABLE:
        logger.warning("FastAPI not available; skipping auth configuration")
        return

    # ------------------------------------------------------------------
    # Step 1: Register AuthenticationMiddleware
    # ------------------------------------------------------------------
    _register_middleware(
        app,
        jwt_handler=jwt_handler,
        api_key_manager=api_key_manager,
        exclude_paths=exclude_paths,
    )

    # ------------------------------------------------------------------
    # Step 2: Include auth service routers
    # ------------------------------------------------------------------
    if include_auth_routes:
        _include_auth_routers(app)

    # ------------------------------------------------------------------
    # Step 2.5: Register RBAC context enrichment (SEC-002)
    # ------------------------------------------------------------------
    if enrich_with_rbac:
        _register_rbac_enrichment(app)

    # ------------------------------------------------------------------
    # Step 3: Protect existing routes
    # ------------------------------------------------------------------
    if protect_existing_routes:
        _protect_all_routes(app)

    logger.info("SEC-001 auth configuration complete for app '%s'", app.title)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _register_middleware(
    app: "FastAPI",
    *,
    jwt_handler: Any = None,
    api_key_manager: Any = None,
    exclude_paths: Optional[Set[str]] = None,
) -> None:
    """Register ``AuthenticationMiddleware`` on the FastAPI app."""
    from greenlang.infrastructure.auth_service.route_protector import PUBLIC_PATHS

    # Merge default public paths with any user-supplied exclusions
    effective_excludes: List[str] = sorted(
        PUBLIC_PATHS | (exclude_paths or set())
    )

    try:
        from greenlang.auth.middleware import AuthenticationMiddleware

        app.add_middleware(
            AuthenticationMiddleware,
            jwt_handler=jwt_handler,
            api_key_manager=api_key_manager,
            exclude_paths=effective_excludes,
            require_auth=False,  # enforcement handled by AuthDependency
        )
        logger.info(
            "AuthenticationMiddleware registered with %d excluded paths",
            len(effective_excludes),
        )
    except ImportError:
        logger.warning(
            "greenlang.auth.middleware not available; "
            "AuthenticationMiddleware NOT registered. "
            "AuthDependency will handle token validation directly."
        )

    # Register AuditMiddleware after AuthenticationMiddleware (SEC-005)
    # Note: FastAPI middleware order is LIFO, so this will run AFTER auth
    if AUDIT_SERVICE_AVAILABLE and AuditMiddleware is not None:
        try:
            app.add_middleware(AuditMiddleware)
            logger.info("AuditMiddleware registered (SEC-005)")
        except Exception as exc:
            logger.warning(
                "Failed to register AuditMiddleware: %s", exc
            )


def _include_auth_routers(app: "FastAPI") -> None:
    """Mount the auth-service API routers on the application."""
    try:
        from greenlang.infrastructure.auth_service.api import (
            auth_router,
            user_router,
        )

        app.include_router(auth_router)
        app.include_router(user_router)
        logger.info("Auth and user routers included")
    except ImportError:
        logger.warning("Auth service API routers not available")

    try:
        from greenlang.infrastructure.auth_service.api import admin_router

        app.include_router(admin_router)
        logger.info("Admin router included")
    except ImportError:
        logger.debug("Admin router not available; skipping")

    try:
        from greenlang.infrastructure.rbac_service.api import rbac_router

        app.include_router(rbac_router)
        logger.info("RBAC router included")
    except ImportError:
        logger.debug("RBAC router not available; skipping")

    try:
        from greenlang.infrastructure.encryption_service.api import encryption_router

        app.include_router(encryption_router)
        logger.info("Encryption router included")
    except ImportError:
        logger.debug("Encryption router not available; skipping")

    # Include audit service router (SEC-005)
    if AUDIT_SERVICE_AVAILABLE and audit_router is not None:
        app.include_router(
            audit_router,
            prefix="/api/v1/audit",
            tags=["audit"],
        )
        logger.info("Audit router included (SEC-005)")
    else:
        logger.debug("Audit router not available; skipping")

    # Include secrets service router (SEC-006)
    if SECRETS_SERVICE_AVAILABLE and secrets_router is not None:
        app.include_router(
            secrets_router,
            prefix="/api/v1/secrets",
            tags=["secrets"],
        )
        logger.info("Secrets router included (SEC-006)")
    else:
        logger.debug("Secrets router not available; skipping")

    # Include security scanning service router (SEC-007)
    if SECURITY_SCANNING_AVAILABLE and security_router is not None:
        app.include_router(
            security_router,
            prefix="/api/v1/security",
            tags=["security"],
        )
        logger.info("Security scanning router included (SEC-007)")
    else:
        logger.debug("Security scanning router not available; skipping")

    # Include SOC 2 preparation service router (SEC-009)
    if SOC2_PREPARATION_AVAILABLE and soc2_router is not None:
        app.include_router(soc2_router)
        logger.info("SOC 2 preparation router included (SEC-009)")
    else:
        logger.debug("SOC 2 preparation router not available; skipping")

    # Include Security Operations service routers (SEC-010)
    if INCIDENT_RESPONSE_AVAILABLE and incident_router is not None:
        app.include_router(
            incident_router,
            prefix="/api/v1/secops",
            tags=["secops", "incident-response"],
        )
        logger.info("Incident response router included (SEC-010)")
    else:
        logger.debug("Incident response router not available; skipping")

    if THREAT_MODELING_AVAILABLE and threat_router is not None:
        app.include_router(
            threat_router,
            prefix="/api/v1/secops",
            tags=["secops", "threat-modeling"],
        )
        logger.info("Threat modeling router included (SEC-010)")
    else:
        logger.debug("Threat modeling router not available; skipping")

    if WAF_MANAGEMENT_AVAILABLE and waf_router is not None:
        app.include_router(
            waf_router,
            prefix="/api/v1/secops",
            tags=["secops", "waf"],
        )
        logger.info("WAF management router included (SEC-010)")
    else:
        logger.debug("WAF management router not available; skipping")

    if VDP_AVAILABLE and vdp_router is not None:
        app.include_router(
            vdp_router,
            prefix="/api/v1/secops",
            tags=["secops", "vdp"],
        )
        logger.info("Vulnerability disclosure router included (SEC-010)")
    else:
        logger.debug("Vulnerability disclosure router not available; skipping")

    if COMPLIANCE_AUTOMATION_AVAILABLE and compliance_router is not None:
        app.include_router(
            compliance_router,
            prefix="/api/v1/secops",
            tags=["secops", "compliance"],
        )
        logger.info("Compliance automation router included (SEC-010)")
    else:
        logger.debug("Compliance automation router not available; skipping")

    if SECURITY_TRAINING_AVAILABLE and training_router is not None:
        app.include_router(
            training_router,
            prefix="/api/v1/secops",
            tags=["secops", "training"],
        )
        logger.info("Security training router included (SEC-010)")
    else:
        logger.debug("Security training router not available; skipping")

    # Include PII service router (SEC-011)
    if PII_SERVICE_AVAILABLE and pii_router is not None:
        app.include_router(pii_router)
        logger.info("PII service router included (SEC-011)")
    else:
        logger.debug("PII service router not available; skipping")


def _protect_all_routes(app: "FastAPI") -> None:
    """Walk every APIRouter on the app and apply auth protection."""
    from greenlang.infrastructure.auth_service.route_protector import (
        AuthDependency,
        PermissionDependency,
        _is_public_path,
        _lookup_permission_for_route,
        _normalise_path,
        PERMISSION_MAP,
    )

    auth_dep = AuthDependency()
    protected = 0
    skipped = 0

    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue

        route_path = _normalise_path(route.path)

        if _is_public_path(route_path):
            skipped += 1
            continue

        # Inject AuthDependency (skip if already present)
        has_auth = any(
            isinstance(getattr(d, "dependency", None), AuthDependency)
            for d in route.dependencies
        )
        if not has_auth:
            from fastapi import Depends

            route.dependencies.append(Depends(auth_dep))

        # Inject PermissionDependency for each method
        for method in route.methods or {"GET"}:
            perm = _lookup_permission_for_route(
                method, route_path, PERMISSION_MAP
            )
            if perm is not None:
                has_perm = any(
                    isinstance(getattr(d, "dependency", None), PermissionDependency)
                    and getattr(d.dependency, "_permission", None) == perm
                    for d in route.dependencies
                )
                if not has_perm:
                    from fastapi import Depends as _Depends

                    route.dependencies.append(
                        _Depends(PermissionDependency(perm))
                    )

        protected += 1

    logger.info(
        "App routes protected: %d secured, %d skipped (public)",
        protected,
        skipped,
    )


# ---------------------------------------------------------------------------
# RBAC enrichment (SEC-002)
# ---------------------------------------------------------------------------


class _RBACEnrichmentDependency:
    """FastAPI dependency that loads user roles/permissions from DB into AuthContext.

    When registered as an app-level dependency, this runs on every request
    after ``AuthenticationMiddleware`` has populated ``request.state.auth``.
    If the ``AuthContext`` already carries permissions (e.g. embedded in JWT
    claims), this is a no-op.  Otherwise it queries ``RBACCache`` (L1/L2)
    and falls back to ``AssignmentService`` (L3 / PostgreSQL).

    The dependency is intentionally lenient: if the RBAC subsystem is
    unreachable the request continues with whatever permissions the JWT
    already contained.  A warning is logged so operators can investigate.
    """

    async def __call__(self, request: Any) -> None:
        """Enrich the current AuthContext with DB-backed RBAC data.

        Args:
            request: The current Starlette/FastAPI ``Request``.
        """
        auth = getattr(getattr(request, "state", None), "auth", None)
        if auth is None:
            return  # not authenticated -- nothing to enrich
        if getattr(auth, "user_id", None) is None:
            return  # anonymous / incomplete context

        # Skip enrichment when permissions are already populated
        # (e.g. from JWT claims or a previous middleware pass).
        if getattr(auth, "permissions", None):
            return

        try:
            tenant_id = getattr(auth, "tenant_id", None)
            user_id = auth.user_id

            # Attempt cache-first resolution
            cache = getattr(
                getattr(request.app, "state", None),
                "rbac_cache",
                None,
            )
            if cache is not None:
                perms = await cache.get_permissions(tenant_id, user_id)
                if perms is not None:
                    auth.permissions = perms
                    roles = await cache.get_roles(tenant_id, user_id)
                    if roles is not None:
                        auth.roles = roles
                    return

            # Fall back to direct DB lookup via AssignmentService
            assignment_svc = getattr(
                getattr(request.app, "state", None),
                "rbac_assignment_service",
                None,
            )
            if assignment_svc is not None:
                perms = await assignment_svc.get_user_permissions(
                    user_id, tenant_id
                )
                auth.permissions = perms

                roles = await assignment_svc.get_user_roles(
                    user_id, tenant_id
                )
                auth.roles = [r.name for r in roles] if roles else []

                # Populate cache for subsequent requests
                if cache is not None:
                    await cache.set_permissions(tenant_id, user_id, perms)
                    await cache.set_roles(
                        tenant_id, user_id, auth.roles
                    )

        except Exception as exc:
            logger.warning(
                "RBAC enrichment failed for user=%s tenant=%s: %s",
                getattr(auth, "user_id", "?"),
                getattr(auth, "tenant_id", "?"),
                exc,
            )


def _register_rbac_enrichment(app: "FastAPI") -> None:
    """Register the RBAC enrichment dependency on the application.

    Adds ``_RBACEnrichmentDependency`` as an app-level FastAPI dependency
    so that every request automatically loads the caller's RBAC roles and
    permissions from the database (with caching) into the ``AuthContext``.

    This is a soft integration: if the RBAC modules are not installed the
    function silently returns so that SEC-001 continues to work standalone.

    Args:
        app: The FastAPI application instance.
    """
    try:
        from fastapi import Depends  # noqa: F811

        enrichment = _RBACEnrichmentDependency()
        app.router.dependencies.append(Depends(enrichment))
        logger.info(
            "RBAC enrichment dependency registered (SEC-002)"
        )
    except Exception as exc:
        logger.debug(
            "RBAC enrichment registration skipped: %s", exc
        )
