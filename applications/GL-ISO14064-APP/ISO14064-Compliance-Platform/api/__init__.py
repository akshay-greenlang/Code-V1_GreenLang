"""
GL-ISO14064-APP REST API Package

ISO 14064-1:2018 Compliance Platform for organizational-level quantification
and reporting of GHG emissions and removals.  Provides 60+ endpoints across
12 routers covering the full ISO 14064-1 lifecycle: organizations, boundaries,
inventories, quantification, removals, significance assessment, verification,
reporting, management plans, crosswalk, and dashboard.

Routes:
    - organization_routes: Organization and entity CRUD
    - boundary_routes: Organizational and operational boundary management
    - inventory_routes: Inventory lifecycle and status transitions
    - quantification_routes: Emission source management and calculation
    - removals_routes: GHG removal source management
    - significance_routes: Significance assessment for indirect categories
    - verification_routes: ISO 14064-3 verification workflow
    - reports_routes: Report generation and data export
    - management_routes: Management plan and action CRUD
    - crosswalk_routes: ISO 14064-1 / GHG Protocol crosswalk
    - dashboard_routes: Dashboard metrics and KPIs
"""

from api.organization_routes import router as organization_router
from api.boundary_routes import router as boundary_router
from api.inventory_routes import router as inventory_router
from api.quantification_routes import router as quantification_router
from api.removals_routes import router as removals_router
from api.significance_routes import router as significance_router
from api.verification_routes import router as verification_router
from api.reports_routes import router as reports_router
from api.management_routes import router as management_router
from api.crosswalk_routes import router as crosswalk_router
from api.dashboard_routes import router as dashboard_router
from api.quality_routes import router as quality_router
from api.settings_routes import router as settings_router

__all__ = [
    "organization_router",
    "boundary_router",
    "inventory_router",
    "quantification_router",
    "removals_router",
    "significance_router",
    "verification_router",
    "reports_router",
    "management_router",
    "crosswalk_router",
    "dashboard_router",
    "quality_router",
    "settings_router",
]
