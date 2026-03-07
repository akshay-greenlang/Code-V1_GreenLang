# -*- coding: utf-8 -*-
"""
API Routes for GL-EUDR-APP v1.0

EU Deforestation Regulation (EUDR) Compliance Platform REST API.
Implements FastAPI routers for supplier management, plot tracking,
due diligence statements, document handling, pipeline orchestration,
risk assessment, dashboard metrics, application settings, and
supply chain mapping (AGENT-EUDR-001).

Built on GreenLang standard patterns: Pydantic validation, structured
error responses, pagination, and audit-ready logging.
"""

from .supplier_routes import router as supplier_router
from .plot_routes import router as plot_router
from .dds_routes import router as dds_router
from .document_routes import router as document_router
from .pipeline_routes import router as pipeline_router
from .risk_routes import router as risk_router
from .dashboard_routes import router as dashboard_router
from .settings_routes import router as settings_router

__all__ = [
    "supplier_router",
    "plot_router",
    "dds_router",
    "document_router",
    "pipeline_router",
    "risk_router",
    "dashboard_router",
    "settings_router",
]
