# -*- coding: utf-8 -*-
"""
GL-EUDR-APP Router Registration - Unified Route Assembly

Aggregates all GL-EUDR-APP API routers (core platform + AGENT-EUDR-001 supply
chain mapper) into a single ``register_all_routers`` function that mounts
them on a FastAPI application instance.

Core Platform Routes (8 routers):
    supplier_routes    -> /api/v1/suppliers
    plot_routes        -> /api/v1/plots
    dds_routes         -> /api/v1/dds
    document_routes    -> /api/v1/documents
    pipeline_routes    -> /api/v1/pipeline
    risk_routes        -> /api/v1/risk
    dashboard_routes   -> /api/v1/dashboard
    settings_routes    -> /api/v1/settings

Supply Chain Mapper Routes (AGENT-EUDR-001, 25+ endpoints):
    graph_routes       -> /v1/supply-chain/v1/eudr-scm/graphs
    mapping_routes     -> /v1/supply-chain/v1/eudr-scm/mapping
    traceability_routes-> /v1/supply-chain/v1/eudr-scm/traceability
    risk_routes        -> /v1/supply-chain/v1/eudr-scm/risk
    gap_routes         -> /v1/supply-chain/v1/eudr-scm/gaps
    visualization_routes -> /v1/supply-chain/v1/eudr-scm/visualization
    onboarding_routes  -> /v1/supply-chain/v1/eudr-scm/onboarding

Auth & RBAC:
    All routes are protected by SEC-001 JWT authentication and SEC-002 RBAC
    permission checks. The auth middleware applies to all routes registered
    through this module.

Example:
    >>> from fastapi import FastAPI
    >>> from services.api.routers import register_all_routers
    >>> app = FastAPI()
    >>> register_all_routers(app)

Author: GreenLang Platform Team
Date: March 2026
Application: GL-EUDR-APP v1.0 + AGENT-EUDR-001 Integration
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from services.api import (
    dashboard_router,
    dds_router,
    document_router,
    pipeline_router,
    plot_router,
    risk_router,
    settings_router,
    supplier_router,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core platform router list
# ---------------------------------------------------------------------------

_CORE_ROUTERS = [
    supplier_router,
    plot_router,
    dds_router,
    document_router,
    pipeline_router,
    risk_router,
    dashboard_router,
    settings_router,
]


def _register_core_routers(app: Any) -> int:
    """Register the 8 core GL-EUDR-APP platform routers.

    Args:
        app: FastAPI application instance.

    Returns:
        Number of routers registered.
    """
    for router in _CORE_ROUTERS:
        app.include_router(router)

    logger.info(
        "Registered %d core GL-EUDR-APP routers: "
        "suppliers, plots, dds, documents, pipeline, risk, dashboard, settings",
        len(_CORE_ROUTERS),
    )
    return len(_CORE_ROUTERS)


def _register_supply_chain_mapper_router(
    app: Any,
    prefix: str = "/v1/supply-chain",
) -> bool:
    """Register the AGENT-EUDR-001 Supply Chain Mapper router.

    Imports the aggregated router from
    ``greenlang.agents.eudr.supply_chain_mapper.api.router`` and mounts it
    under the specified prefix. The supply chain mapper router already
    includes its own ``/v1/eudr-scm`` sub-prefix, so the full path becomes
    ``{prefix}/v1/eudr-scm/...``.

    Auth middleware from SEC-001 and SEC-002 is applied through the
    individual route dependencies defined in
    ``greenlang.agents.eudr.supply_chain_mapper.api.dependencies``.

    Args:
        app: FastAPI application instance.
        prefix: URL prefix for supply chain mapper routes.

    Returns:
        True if registration succeeded, False if the module is unavailable.
    """
    try:
        from greenlang.agents.eudr.supply_chain_mapper.api.router import (
            router as scm_router,
        )

        app.include_router(
            scm_router,
            prefix=prefix,
            tags=["EUDR Supply Chain Mapper"],
        )

        logger.info(
            "Registered AGENT-EUDR-001 Supply Chain Mapper router: "
            "prefix=%s, sub_prefix=/v1/eudr-scm, endpoints=25+",
            prefix,
        )
        return True

    except ImportError as exc:
        logger.warning(
            "AGENT-EUDR-001 Supply Chain Mapper router not available: %s. "
            "Supply chain mapping endpoints will not be accessible. "
            "Ensure greenlang.agents.eudr.supply_chain_mapper is installed.",
            exc,
        )
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def register_all_routers(
    app: Any,
    scm_prefix: Optional[str] = None,
    scm_enabled: bool = True,
) -> dict:
    """Register all GL-EUDR-APP routers on a FastAPI application.

    Mounts both the core platform routers and (optionally) the
    AGENT-EUDR-001 Supply Chain Mapper router. Returns a summary
    dictionary for logging and health checks.

    Args:
        app: FastAPI application instance.
        scm_prefix: URL prefix for supply chain mapper routes. Defaults
            to ``/v1/supply-chain`` if not specified.
        scm_enabled: Whether to register supply chain mapper routes.
            Controlled by ``EUDRAppConfig.scm_enabled``.

    Returns:
        Dictionary with registration summary:
            - core_routers: Number of core routers registered.
            - scm_registered: Whether the SCM router was registered.
            - total_routers: Total router count.

    Example:
        >>> from fastapi import FastAPI
        >>> from services.api.routers import register_all_routers
        >>> app = FastAPI()
        >>> summary = register_all_routers(app)
        >>> assert summary["core_routers"] == 8
    """
    prefix = scm_prefix or "/v1/supply-chain"
    core_count = _register_core_routers(app)

    scm_registered = False
    if scm_enabled:
        scm_registered = _register_supply_chain_mapper_router(app, prefix)
    else:
        logger.info(
            "AGENT-EUDR-001 Supply Chain Mapper is disabled via config "
            "(scm_enabled=False)"
        )

    total = core_count + (1 if scm_registered else 0)

    summary = {
        "core_routers": core_count,
        "scm_registered": scm_registered,
        "scm_prefix": prefix if scm_registered else None,
        "total_routers": total,
    }

    logger.info(
        "Router registration complete: core=%d, scm=%s, total=%d",
        core_count,
        "registered" if scm_registered else "skipped",
        total,
    )
    return summary


__all__ = [
    "register_all_routers",
]
