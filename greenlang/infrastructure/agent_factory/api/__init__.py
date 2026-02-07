# -*- coding: utf-8 -*-
"""
Agent Factory API Routes - INFRA-010 Phase 4

FastAPI routers for managing agents, lifecycle, execution queue,
the Agent Hub package registry, and async operations.

Public API:
    - factory_router: Core CRUD and execution endpoints for agents.
    - lifecycle_router: Deploy, rollback, drain, retire, health.
    - queue_router: Task queue status, retry, cancel, DLQ.
    - hub_router: Package search, publish, download.
    - operations_router: Async 202 pattern for long-running operations.
    - FactoryContextMiddleware: Tenant and correlation-ID injection.
    - CostTrackingMiddleware: Per-request cost tracking.

Example:
    >>> from fastapi import FastAPI
    >>> from greenlang.infrastructure.agent_factory.api import (
    ...     factory_router, lifecycle_router, queue_router, hub_router,
    ...     operations_router,
    ... )
    >>> app = FastAPI()
    >>> app.include_router(factory_router)
    >>> app.include_router(lifecycle_router)
    >>> app.include_router(queue_router)
    >>> app.include_router(hub_router)
    >>> app.include_router(operations_router)

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from greenlang.infrastructure.agent_factory.api.factory_routes import (
    router as factory_router,
)
from greenlang.infrastructure.agent_factory.api.lifecycle_routes import (
    router as lifecycle_router,
)
from greenlang.infrastructure.agent_factory.api.queue_routes import (
    router as queue_router,
)
from greenlang.infrastructure.agent_factory.api.hub_routes import (
    router as hub_router,
)
from greenlang.infrastructure.agent_factory.api.operations_routes import (
    router as operations_router,
)
from greenlang.infrastructure.agent_factory.api.middleware import (
    CostTrackingMiddleware,
    FactoryContextMiddleware,
)

__all__ = [
    "CostTrackingMiddleware",
    "FactoryContextMiddleware",
    "factory_router",
    "hub_router",
    "lifecycle_router",
    "operations_router",
    "queue_router",
]
