"""
GL-SBTi-APP REST API Package

SBTi (Science Based Targets initiative) Target Setting & Validation Platform
for managing science-based emissions reduction targets across all scopes,
including near-term, long-term, and net-zero commitments.  Supports ACA, SDA,
and FLAG pathways, Scope 3 screening, financial institution portfolios,
five-year review cycles, recalculation triggers, temperature scoring,
cross-framework alignment, and comprehensive gap analysis.

Provides 90+ endpoints across 16 routers covering the full SBTi target
lifecycle: target CRUD, pathway calculation, validation & readiness, Scope 3
screening, FLAG assessment, sector pathways, progress tracking, temperature
scoring, recalculation management, five-year reviews, financial institution
portfolios, framework alignment, reporting & export, executive dashboard,
gap analysis, and platform settings.

Routes:
    - target_routes: Target CRUD, status, submission, coverage checks
    - pathway_routes: ACA, SDA, intensity, supplier engagement, FLAG pathways
    - validation_routes: Full validation, criteria checks, readiness, net-zero
    - scope3_routes: 40% trigger, category breakdown, hotspot, coverage
    - flag_routes: 20% FLAG trigger, commodity/sector pathways, removals
    - sector_routes: Sector detection, pathway calculation, blending, benchmarks
    - progress_routes: Annual progress, variance, projections, dashboard
    - temperature_routes: Company & portfolio temperature scores, peer ranking
    - recalculation_routes: 5% threshold check, M&A impact, audit trail
    - review_routes: Five-year review cycle, deadlines, readiness, outcomes
    - fi_routes: FI portfolios, WACI, PCAF, financed emissions, FINZ
    - framework_routes: Cross-framework alignment, mapping, unified reports
    - reporting_routes: Submission forms, progress reports, export (PDF/Excel)
    - dashboard_routes: Readiness score, target summary, pathway overlay
    - gap_routes: Data gaps, ambition gaps, action plans, benchmarks
    - settings_routes: Org settings, sector config, MRV connection, criteria
"""

from .target_routes import router as target_router
from .pathway_routes import router as pathway_router
from .validation_routes import router as validation_router
from .scope3_routes import router as scope3_router
from .flag_routes import router as flag_router
from .sector_routes import router as sector_router
from .progress_routes import router as progress_router
from .temperature_routes import router as temperature_router
from .recalculation_routes import router as recalculation_router
from .review_routes import router as review_router
from .fi_routes import router as fi_router
from .framework_routes import router as framework_router
from .reporting_routes import router as reporting_router
from .dashboard_routes import router as dashboard_router
from .gap_routes import router as gap_router
from .settings_routes import router as settings_router

__all__ = [
    "target_router",
    "pathway_router",
    "validation_router",
    "scope3_router",
    "flag_router",
    "sector_router",
    "progress_router",
    "temperature_router",
    "recalculation_router",
    "review_router",
    "fi_router",
    "framework_router",
    "reporting_router",
    "dashboard_router",
    "gap_router",
    "settings_router",
]
