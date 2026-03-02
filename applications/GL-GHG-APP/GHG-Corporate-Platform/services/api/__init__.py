"""
GL-GHG-APP REST API Package

GHG Protocol Corporate Accounting and Reporting Standard platform.
Provides 57 endpoints across 9 routers for complete GHG inventory
management, emission calculation, reporting, verification, and target tracking.

Routes:
    - inventory_routes: Organization, entity, boundary, and inventory management
    - scope1_routes: Scope 1 direct emission calculations and breakdowns
    - scope2_routes: Scope 2 indirect emissions with dual reporting
    - scope3_routes: Scope 3 value chain emissions (15 categories)
    - reporting_routes: Report generation, disclosures, and export
    - dashboard_routes: KPIs, trends, and visualization data
    - verification_routes: Third-party verification workflow
    - target_routes: Reduction targets and SBTi alignment
    - settings_routes: Platform configuration and defaults
"""

from services.api.inventory_routes import router as inventory_router
from services.api.scope1_routes import router as scope1_router
from services.api.scope2_routes import router as scope2_router
from services.api.scope3_routes import router as scope3_router
from services.api.reporting_routes import router as reporting_router
from services.api.dashboard_routes import router as dashboard_router
from services.api.verification_routes import router as verification_router
from services.api.target_routes import router as target_router
from services.api.settings_routes import router as settings_router

__all__ = [
    "inventory_router",
    "scope1_router",
    "scope2_router",
    "scope3_router",
    "reporting_router",
    "dashboard_router",
    "verification_router",
    "target_router",
    "settings_router",
]
