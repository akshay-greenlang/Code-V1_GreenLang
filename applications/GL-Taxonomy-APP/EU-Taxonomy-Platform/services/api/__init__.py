"""
GL-Taxonomy-APP REST API Package

EU Taxonomy Alignment & Green Investment Ratio Platform for managing
regulatory-compliant taxonomy assessments, eligibility screening,
substantial contribution testing, DNSH assessments, minimum safeguards
verification, KPI calculations (Turnover/CapEx/OpEx), Green Asset Ratio
(GAR/BTAR) computation for financial institutions, and comprehensive
Article 8 / EBA Pillar III reporting.

Provides 120+ endpoints across 15 routers covering the full EU Taxonomy
alignment lifecycle: activity catalog, eligibility screening, substantial
contribution assessment, DNSH assessment, minimum safeguards, KPI
calculation, GAR/BTAR, full alignment workflow, Article 8/EBA reporting,
portfolio management, executive dashboard, data quality, regulatory
tracking, gap analysis, and platform settings.

Routes:
    - activity_routes: Activity catalog CRUD, NACE lookup, sector/objective browse
    - screening_routes: Eligibility screening, batch, de minimis, sector breakdown
    - sc_routes: Substantial contribution assessment, TSC criteria, evidence
    - dnsh_routes: DNSH assessment (climate/water/CE/pollution/biodiversity)
    - safeguards_routes: Minimum safeguards (HR/anti-corruption/tax/competition)
    - kpi_routes: Turnover/CapEx/OpEx KPI calculation, CapEx plan, dashboard
    - gar_routes: GAR stock/flow, BTAR, EBA template, mortgage check
    - alignment_routes: Full 4-step alignment, portfolio alignment, funnel
    - reporting_routes: Article 8, EBA reports, XBRL, PDF/Excel export
    - portfolio_routes: Portfolio CRUD, holdings, CSV/Excel upload
    - dashboard_routes: Executive dashboard, KPI cards, sector/trend charts
    - data_quality_routes: DQ assessment, dimension scores, evidence, trends
    - regulatory_routes: Delegated acts, TSC updates, Omnibus impact
    - gap_routes: Gap analysis, DNSH/safeguard/data gaps, action plans
    - settings_routes: Org settings, reporting periods, thresholds, MRV mapping
"""

from .activity_routes import router as activity_router
from .screening_routes import router as screening_router
from .sc_routes import router as sc_router
from .dnsh_routes import router as dnsh_router
from .safeguards_routes import router as safeguards_router
from .kpi_routes import router as kpi_router
from .gar_routes import router as gar_router
from .alignment_routes import router as alignment_router
from .reporting_routes import router as reporting_router
from .portfolio_routes import router as portfolio_router
from .dashboard_routes import router as dashboard_router
from .data_quality_routes import router as data_quality_router
from .regulatory_routes import router as regulatory_router
from .gap_routes import router as gap_router
from .settings_routes import router as settings_router

__all__ = [
    "activity_router",
    "screening_router",
    "sc_router",
    "dnsh_router",
    "safeguards_router",
    "kpi_router",
    "gar_router",
    "alignment_router",
    "reporting_router",
    "portfolio_router",
    "dashboard_router",
    "data_quality_router",
    "regulatory_router",
    "gap_router",
    "settings_router",
]
