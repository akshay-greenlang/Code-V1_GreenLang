# -*- coding: utf-8 -*-
"""
API routes for GreenLang.

This module provides REST API endpoints for various features including
emission factors, editions, calculations, dashboards, workflows, agents,
and analytics.

Organization:
- factors: Emission factor query endpoints
- editions: Factor catalog edition management
- calculations: Emission calculation endpoints
- health: Health check and monitoring endpoints
- dashboards: Dashboard and analytics endpoints
- marketplace: Marketplace endpoints
- billing: Stripe Checkout creation (FY27 Pricing Page)

Each submodule import is wrapped in a defensive try/except so a broken
sibling (legacy import path that hasn't been migrated yet) cannot block
import of healthy routers. Failed imports are logged and the symbol is
set to ``None`` so downstream callers can detect the absence and react.
"""

import logging as _logging

_logger = _logging.getLogger(__name__)


def _safe_import(module_path: str, attr: str = "router"):
    """Best-effort import — return None and log on failure."""
    try:
        module = __import__(module_path, fromlist=[attr])
        return getattr(module, attr)
    except Exception as exc:  # noqa: BLE001 -- never let one broken sibling block all
        _logger.warning("Could not import %s.%s: %s", module_path, attr, exc)
        return None


factors_router = _safe_import("greenlang.integration.api.routes.factors")
editions_router = _safe_import("greenlang.integration.api.routes.editions")
calculations_router = _safe_import("greenlang.integration.api.routes.calculations")
health_router = _safe_import("greenlang.integration.api.routes.health")
dashboards_router = _safe_import("greenlang.integration.api.routes.dashboards")
marketplace_router = _safe_import("greenlang.integration.api.routes.marketplace")
billing_router = _safe_import("greenlang.integration.api.routes.billing")

__all__ = [
    "factors_router",
    "editions_router",
    "calculations_router",
    "health_router",
    "dashboards_router",
    "marketplace_router",
    "billing_router",
]
