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

# Duplicate Detection Agent imports (AGENT-DATA-011)
try:
    from greenlang.duplicate_detector.api.router import router as dedup_router

    DUPLICATE_DETECTOR_AVAILABLE = True
except ImportError:
    dedup_router = None
    DUPLICATE_DETECTOR_AVAILABLE = False

# Missing Value Imputer Agent imports (AGENT-DATA-012)
try:
    from greenlang.missing_value_imputer.api.router import router as imputer_router

    MISSING_VALUE_IMPUTER_AVAILABLE = True
except ImportError:
    imputer_router = None
    MISSING_VALUE_IMPUTER_AVAILABLE = False

# Outlier Detection Agent imports (AGENT-DATA-013)
try:
    from greenlang.outlier_detector.api.router import router as outlier_router

    OUTLIER_DETECTOR_AVAILABLE = True
except ImportError:
    outlier_router = None
    OUTLIER_DETECTOR_AVAILABLE = False

# Time Series Gap Filler Agent imports (AGENT-DATA-014)
try:
    from greenlang.time_series_gap_filler.setup import get_router as get_gap_filler_router

    _gap_filler_router = get_gap_filler_router()
    GAP_FILLER_AVAILABLE = True
except ImportError:
    _gap_filler_router = None
    GAP_FILLER_AVAILABLE = False

# Cross-Source Reconciliation Agent imports (AGENT-DATA-015)
try:
    from greenlang.cross_source_reconciliation.setup import get_router as get_reconciliation_router

    _reconciliation_router = get_reconciliation_router()
    RECONCILIATION_AVAILABLE = True
except ImportError:
    _reconciliation_router = None
    RECONCILIATION_AVAILABLE = False

# Data Freshness Monitor Agent imports (AGENT-DATA-016)
try:
    from greenlang.data_freshness_monitor.setup import get_router as get_freshness_router

    _freshness_router = get_freshness_router()
    FRESHNESS_MONITOR_AVAILABLE = True
except ImportError:
    _freshness_router = None
    FRESHNESS_MONITOR_AVAILABLE = False

# Schema Migration Agent imports (AGENT-DATA-017)
try:
    from greenlang.schema_migration.setup import get_router as get_sm_router

    _sm_router = get_sm_router()
    SCHEMA_MIGRATION_AVAILABLE = True
except ImportError:
    _sm_router = None
    SCHEMA_MIGRATION_AVAILABLE = False

# Data Lineage Tracker Agent imports (AGENT-DATA-018)
try:
    from greenlang.data_lineage_tracker.setup import get_router as get_dlt_router

    DATA_LINEAGE_AVAILABLE = True
    _dlt_router = get_dlt_router()
except ImportError:
    DATA_LINEAGE_AVAILABLE = False
    _dlt_router = None

# Validation Rule Engine imports (AGENT-DATA-019)
try:
    from greenlang.validation_rule_engine.setup import get_router as get_vre_router
    VALIDATION_RULE_ENGINE_AVAILABLE = True
    _vre_router = get_vre_router()
except ImportError:
    VALIDATION_RULE_ENGINE_AVAILABLE = False
    _vre_router = None

# Climate Hazard Connector imports (AGENT-DATA-020)
try:
    from greenlang.climate_hazard.setup import get_router as get_chc_router
    CLIMATE_HAZARD_AVAILABLE = True
    _chc_router = get_chc_router()
except ImportError:
    CLIMATE_HAZARD_AVAILABLE = False
    _chc_router = None

# Stationary Combustion imports (AGENT-MRV-001)
try:
    from greenlang.stationary_combustion.setup import get_router as get_sc_router
    STATIONARY_COMBUSTION_AVAILABLE = True
    _sc_router = get_sc_router()
except ImportError:
    STATIONARY_COMBUSTION_AVAILABLE = False
    _sc_router = None

# Refrigerants & F-Gas imports (AGENT-MRV-002)
try:
    from greenlang.refrigerants_fgas.setup import get_router as get_rf_router
    REFRIGERANTS_FGAS_AVAILABLE = True
    _rf_router = get_rf_router()
except ImportError:
    REFRIGERANTS_FGAS_AVAILABLE = False
    _rf_router = None

# Mobile Combustion imports (AGENT-MRV-003)
try:
    from greenlang.mobile_combustion.setup import get_router as get_mc_router
    MOBILE_COMBUSTION_AVAILABLE = True
    _mc_router = get_mc_router()
except ImportError:
    MOBILE_COMBUSTION_AVAILABLE = False
    _mc_router = None

# Process Emissions imports (AGENT-MRV-004)
try:
    from greenlang.process_emissions.setup import get_router as get_pe_router
    PROCESS_EMISSIONS_AVAILABLE = True
    _pe_router = get_pe_router()
except ImportError:
    PROCESS_EMISSIONS_AVAILABLE = False
    _pe_router = None

# Fugitive Emissions imports (AGENT-MRV-005)
try:
    from greenlang.fugitive_emissions.setup import get_router as get_fue_router
    FUGITIVE_EMISSIONS_AVAILABLE = True
    _fue_router = get_fue_router()
except ImportError:
    FUGITIVE_EMISSIONS_AVAILABLE = False
    _fue_router = None

# Land Use Emissions imports (AGENT-MRV-006)
try:
    from greenlang.land_use_emissions.setup import get_router as get_lu_router
    LAND_USE_EMISSIONS_AVAILABLE = True
    _lu_router = get_lu_router()
except ImportError:
    LAND_USE_EMISSIONS_AVAILABLE = False
    _lu_router = None

# Waste Treatment Emissions imports (AGENT-MRV-007)
try:
    from greenlang.waste_treatment_emissions.setup import get_router as get_wt_router
    WASTE_TREATMENT_EMISSIONS_AVAILABLE = True
    _wt_router = get_wt_router()
except ImportError:
    WASTE_TREATMENT_EMISSIONS_AVAILABLE = False
    _wt_router = None


# Agricultural Emissions imports (AGENT-MRV-008)
try:
    from greenlang.agricultural_emissions.setup import get_router as get_ag_router
    AGRICULTURAL_EMISSIONS_AVAILABLE = True
    _ag_router = get_ag_router()
except ImportError:
    AGRICULTURAL_EMISSIONS_AVAILABLE = False
    _ag_router = None

# Scope 2 Location-Based Emissions imports (AGENT-MRV-009)
try:
    from greenlang.scope2_location.api.router import router as _s2l_router
    SCOPE2_LOCATION_AVAILABLE = True
except ImportError:
    SCOPE2_LOCATION_AVAILABLE = False
    _s2l_router = None

# Scope 2 Market-Based Emissions imports (AGENT-MRV-010)
try:
    from greenlang.scope2_market.api.router import router as _s2m_router
    SCOPE2_MARKET_AVAILABLE = True
except ImportError:
    SCOPE2_MARKET_AVAILABLE = False
    _s2m_router = None

# Steam/Heat Purchase imports (AGENT-MRV-011)
try:
    from greenlang.steam_heat_purchase.api.router import router as _shp_router
    STEAM_HEAT_PURCHASE_AVAILABLE = True
except ImportError:
    STEAM_HEAT_PURCHASE_AVAILABLE = False
    _shp_router = None

# Cooling Purchase imports (AGENT-MRV-012)
try:
    from greenlang.cooling_purchase.api.router import router as _cp_router
    COOLING_PURCHASE_AVAILABLE = True
except ImportError:
    COOLING_PURCHASE_AVAILABLE = False
    _cp_router = None

# Dual Reporting Reconciliation imports (AGENT-MRV-013)
try:
    from greenlang.dual_reporting_reconciliation.api.router import router as _drr_router
    DUAL_REPORTING_AVAILABLE = True
except ImportError:
    DUAL_REPORTING_AVAILABLE = False
    _drr_router = None

# Purchased Goods & Services imports (AGENT-MRV-014)
try:
    from greenlang.purchased_goods_services.setup import get_router as get_pgs_router
    PURCHASED_GOODS_AVAILABLE = True
    _pgs_router = get_pgs_router()
except ImportError:
    PURCHASED_GOODS_AVAILABLE = False
    _pgs_router = None

# Capital Goods imports (AGENT-MRV-015)
try:
    from greenlang.capital_goods.setup import get_router as get_cg_router
    CAPITAL_GOODS_AVAILABLE = True
    _cg_router = get_cg_router()
except ImportError:
    CAPITAL_GOODS_AVAILABLE = False
    _cg_router = None

# Fuel & Energy Activities imports (AGENT-MRV-016)
try:
    from greenlang.fuel_energy_activities.setup import get_router as get_fea_router
    FUEL_ENERGY_ACTIVITIES_AVAILABLE = True
    _fea_router = get_fea_router()
except ImportError:
    FUEL_ENERGY_ACTIVITIES_AVAILABLE = False
    _fea_router = None

# Upstream Transportation & Distribution imports (AGENT-MRV-017)
try:
    from greenlang.upstream_transportation.setup import get_router as get_uto_router
    UPSTREAM_TRANSPORTATION_AVAILABLE = True
    _uto_router = get_uto_router()
except ImportError:
    UPSTREAM_TRANSPORTATION_AVAILABLE = False
    _uto_router = None

# Waste Generated imports (AGENT-MRV-018)
try:
    from greenlang.waste_generated.setup import get_router as get_wg_router
    WASTE_GENERATED_AVAILABLE = True
    _wg_router = get_wg_router()
except ImportError:
    WASTE_GENERATED_AVAILABLE = False
    _wg_router = None

# Business Travel imports (AGENT-MRV-019)
try:
    from greenlang.business_travel.setup import get_router as get_bt_router
    BUSINESS_TRAVEL_AVAILABLE = True
    _bt_router = get_bt_router()
except ImportError:
    BUSINESS_TRAVEL_AVAILABLE = False
    _bt_router = None

# Employee Commuting imports (AGENT-MRV-020)
try:
    from greenlang.employee_commuting.setup import get_router as get_ec_router
    EMPLOYEE_COMMUTING_AVAILABLE = True
    _ec_router = get_ec_router()
except ImportError:
    EMPLOYEE_COMMUTING_AVAILABLE = False
    _ec_router = None

# Upstream Leased Assets imports (AGENT-MRV-021)
try:
    from greenlang.upstream_leased_assets.setup import get_router as get_ula_router
    UPSTREAM_LEASED_AVAILABLE = True
    _ula_router = get_ula_router()
except ImportError:
    UPSTREAM_LEASED_AVAILABLE = False
    _ula_router = None

# Downstream Transportation imports (AGENT-MRV-022)
try:
    from greenlang.downstream_transportation.setup import get_router as get_dto_router
    DOWNSTREAM_TRANSPORT_AVAILABLE = True
    _dto_router = get_dto_router()
except ImportError:
    DOWNSTREAM_TRANSPORT_AVAILABLE = False
    _dto_router = None

# Processing of Sold Products imports (AGENT-MRV-023)
try:
    from greenlang.processing_sold_products.setup import get_router as get_psp_router
    PROCESSING_SOLD_PRODUCTS_AVAILABLE = True
    _psp_router = get_psp_router()
except ImportError:
    PROCESSING_SOLD_PRODUCTS_AVAILABLE = False
    _psp_router = None

# Use of Sold Products imports (AGENT-MRV-024)
try:
    from greenlang.use_of_sold_products.setup import get_router as get_usp_router
    USE_OF_SOLD_PRODUCTS_AVAILABLE = True
    _usp_router = get_usp_router()
except ImportError:
    USE_OF_SOLD_PRODUCTS_AVAILABLE = False
    _usp_router = None

# End-of-Life Treatment imports (AGENT-MRV-025)
try:
    from greenlang.end_of_life_treatment.setup import get_router as get_eol_router
    END_OF_LIFE_TREATMENT_AVAILABLE = True
    _eol_router = get_eol_router()
except ImportError:
    END_OF_LIFE_TREATMENT_AVAILABLE = False
    _eol_router = None

# Downstream Leased Assets imports (AGENT-MRV-026)
try:
    from greenlang.downstream_leased_assets.setup import get_router as get_dla_router
    DOWNSTREAM_LEASED_ASSETS_AVAILABLE = True
    _dla_router = get_dla_router()
except ImportError:
    DOWNSTREAM_LEASED_ASSETS_AVAILABLE = False
    _dla_router = None

# Franchises imports (AGENT-MRV-027)
try:
    from greenlang.franchises.setup import get_router as get_frn_router
    FRANCHISES_AVAILABLE = True
    _frn_router = get_frn_router()
except ImportError:
    FRANCHISES_AVAILABLE = False
    _frn_router = None

# Investments imports (AGENT-MRV-028)
try:
    from greenlang.investments.setup import get_router as get_inv_router
    INVESTMENTS_AVAILABLE = True
    _inv_router = get_inv_router()
except ImportError:
    INVESTMENTS_AVAILABLE = False
    _inv_router = None

# Scope 3 Category Mapper Agent imports (AGENT-MRV-029)
try:
    from greenlang.scope3_category_mapper.setup import get_router as get_scm_router
    SCOPE3_CATEGORY_MAPPER_AVAILABLE = True
    _scm_router = get_scm_router()
except ImportError:
    SCOPE3_CATEGORY_MAPPER_AVAILABLE = False
    _scm_router = None

# Audit Trail & Lineage Agent imports (AGENT-MRV-030)
try:
    from greenlang.audit_trail_lineage.setup import get_router as get_atl_router
    AUDIT_TRAIL_LINEAGE_AVAILABLE = True
    _atl_router = get_atl_router()
except ImportError:
    AUDIT_TRAIL_LINEAGE_AVAILABLE = False
    _atl_router = None

# ISO 14064 Compliance Platform imports (GL-ISO14064-APP)
try:
    from applications.gl_iso14064_app.api.router import router as _iso14064_router
    ISO14064_APP_AVAILABLE = True
except ImportError:
    ISO14064_APP_AVAILABLE = False
    _iso14064_router = None

# CDP Disclosure Platform imports (GL-CDP-APP) - APP-007
try:
    from applications.GL_CDP_APP.CDP_Disclosure_Platform.services.setup import create_cdp_app
    CDP_APP_AVAILABLE = True
except ImportError:
    CDP_APP_AVAILABLE = False
    create_cdp_app = None

# TCFD Climate Disclosure Platform imports (GL-TCFD-APP) - APP-008
try:
    from applications.GL_TCFD_APP.TCFD_Disclosure_Platform.services.setup import get_router as get_tcfd_router
    _tcfd_router = get_tcfd_router()
    TCFD_APP_AVAILABLE = True
except ImportError:
    TCFD_APP_AVAILABLE = False
    _tcfd_router = None

# SBTi Target Validation Platform imports (GL-SBTi-APP) - APP-009
try:
    from applications.GL_SBTi_APP.SBTi_Target_Platform.services.setup import get_router as get_sbti_router
    _sbti_router = get_sbti_router()
    SBTI_APP_AVAILABLE = True
except ImportError:
    SBTI_APP_AVAILABLE = False
    _sbti_router = None

# EU Taxonomy Alignment Platform imports (GL-Taxonomy-APP) - APP-010
try:
    from applications.GL_Taxonomy_APP.EU_Taxonomy_Platform.services.setup import get_router as get_taxonomy_router
    _taxonomy_router = get_taxonomy_router()
    TAXONOMY_APP_AVAILABLE = True
except ImportError:
    TAXONOMY_APP_AVAILABLE = False
    _taxonomy_router = None


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

    # Include Duplicate Detection Agent router (AGENT-DATA-011)
    if DUPLICATE_DETECTOR_AVAILABLE and dedup_router is not None:
        app.include_router(dedup_router)
        logger.info("Duplicate Detection Agent router included (AGENT-DATA-011)")
    else:
        logger.debug("Duplicate Detection Agent router not available; skipping")

    # Include Missing Value Imputer Agent router (AGENT-DATA-012)
    if MISSING_VALUE_IMPUTER_AVAILABLE and imputer_router is not None:
        app.include_router(imputer_router)
        logger.info("Missing Value Imputer Agent router included (AGENT-DATA-012)")
    else:
        logger.debug("Missing Value Imputer Agent router not available; skipping")

    # Include Outlier Detection Agent router (AGENT-DATA-013)
    if OUTLIER_DETECTOR_AVAILABLE and outlier_router is not None:
        app.include_router(outlier_router)
        logger.info("Outlier Detection Agent router included (AGENT-DATA-013)")
    else:
        logger.debug("Outlier Detection Agent router not available; skipping")

    # Include Time Series Gap Filler Agent router (AGENT-DATA-014)
    if GAP_FILLER_AVAILABLE and _gap_filler_router is not None:
        app.include_router(_gap_filler_router)
        logger.info("Time Series Gap Filler Agent router included (AGENT-DATA-014)")
    else:
        logger.debug("Time Series Gap Filler Agent router not available; skipping")

    # Include Cross-Source Reconciliation Agent router (AGENT-DATA-015)
    if RECONCILIATION_AVAILABLE and _reconciliation_router is not None:
        app.include_router(_reconciliation_router)
        logger.info("Cross-Source Reconciliation Agent router included (AGENT-DATA-015)")
    else:
        logger.debug("Cross-Source Reconciliation Agent router not available; skipping")

    # Include Data Freshness Monitor Agent router (AGENT-DATA-016)
    if FRESHNESS_MONITOR_AVAILABLE and _freshness_router is not None:
        app.include_router(_freshness_router)
        logger.info("Data Freshness Monitor Agent router included (AGENT-DATA-016)")
    else:
        logger.debug("Data Freshness Monitor Agent router not available; skipping")

    # Include Schema Migration Agent router (AGENT-DATA-017)
    if SCHEMA_MIGRATION_AVAILABLE and _sm_router is not None:
        app.include_router(_sm_router)
        logger.info("Schema Migration Agent router included (AGENT-DATA-017)")
    else:
        logger.debug("Schema Migration Agent router not available; skipping")

    # Include Data Lineage Tracker Agent router (AGENT-DATA-018)
    if DATA_LINEAGE_AVAILABLE and _dlt_router is not None:
        app.include_router(_dlt_router)
        logger.info("Data Lineage Tracker Agent router included (AGENT-DATA-018)")
    else:
        logger.debug("Data Lineage Tracker Agent router not available; skipping")

    # Include Validation Rule Engine router (AGENT-DATA-019)
    if VALIDATION_RULE_ENGINE_AVAILABLE and _vre_router is not None:
        app.include_router(_vre_router)
        logger.info("Validation Rule Engine router included (AGENT-DATA-019)")
    else:
        logger.debug("Validation Rule Engine router not available; skipping")

    # Climate Hazard Connector router (AGENT-DATA-020)
    if CLIMATE_HAZARD_AVAILABLE and _chc_router is not None:
        app.include_router(_chc_router)
        logger.info("Climate Hazard Connector router included (AGENT-DATA-020)")
    else:
        logger.debug("Climate Hazard Connector router not available; skipping")

    # Stationary Combustion router (AGENT-MRV-001)
    if STATIONARY_COMBUSTION_AVAILABLE and _sc_router is not None:
        app.include_router(_sc_router)
        logger.info("Stationary Combustion router included (AGENT-MRV-001)")
    else:
        logger.debug("Stationary Combustion router not available; skipping")

    # Refrigerants & F-Gas router (AGENT-MRV-002)
    if REFRIGERANTS_FGAS_AVAILABLE and _rf_router is not None:
        app.include_router(_rf_router)
        logger.info("Refrigerants & F-Gas router included (AGENT-MRV-002)")
    else:
        logger.debug("Refrigerants & F-Gas router not available; skipping")

    # Mobile Combustion router (AGENT-MRV-003)
    if MOBILE_COMBUSTION_AVAILABLE and _mc_router is not None:
        app.include_router(_mc_router)
        logger.info("Mobile Combustion router included (AGENT-MRV-003)")
    else:
        logger.debug("Mobile Combustion router not available; skipping")

    # Process Emissions router (AGENT-MRV-004)
    if PROCESS_EMISSIONS_AVAILABLE and _pe_router is not None:
        app.include_router(_pe_router)
        logger.info("Process Emissions router included (AGENT-MRV-004)")
    else:
        logger.debug("Process Emissions router not available; skipping")

    # Fugitive Emissions router (AGENT-MRV-005)
    if FUGITIVE_EMISSIONS_AVAILABLE and _fue_router is not None:
        app.include_router(_fue_router)
        logger.info("Fugitive Emissions router included (AGENT-MRV-005)")
    else:
        logger.debug("Fugitive Emissions router not available; skipping")

    # Land Use Emissions router (AGENT-MRV-006)
    if LAND_USE_EMISSIONS_AVAILABLE and _lu_router is not None:
        app.include_router(_lu_router)
        logger.info("Land Use Emissions router included (AGENT-MRV-006)")
    else:
        logger.debug("Land Use Emissions router not available; skipping")

    # Waste Treatment Emissions router (AGENT-MRV-007)
    if WASTE_TREATMENT_EMISSIONS_AVAILABLE and _wt_router is not None:
        app.include_router(_wt_router)
        logger.info("Waste Treatment Emissions router included (AGENT-MRV-007)")
    else:
        logger.debug("Waste Treatment Emissions router not available; skipping")

    # Agricultural Emissions router (AGENT-MRV-008)
    if AGRICULTURAL_EMISSIONS_AVAILABLE and _ag_router is not None:
        app.include_router(_ag_router)
        logger.info("Agricultural Emissions router included (AGENT-MRV-008)")
    else:
        logger.debug("Agricultural Emissions router not available; skipping")

    # Scope 2 Location-Based Emissions router (AGENT-MRV-009)
    if SCOPE2_LOCATION_AVAILABLE and _s2l_router is not None:
        app.include_router(_s2l_router)
        logger.info("Scope 2 Location-Based router included (AGENT-MRV-009)")
    else:
        logger.debug("Scope 2 Location-Based router not available; skipping")

    # Scope 2 Market-Based Emissions router (AGENT-MRV-010)
    if SCOPE2_MARKET_AVAILABLE and _s2m_router is not None:
        app.include_router(_s2m_router)
        logger.info("Scope 2 Market-Based router included (AGENT-MRV-010)")
    else:
        logger.debug("Scope 2 Market-Based router not available; skipping")

    # Steam/Heat Purchase router (AGENT-MRV-011)
    if STEAM_HEAT_PURCHASE_AVAILABLE and _shp_router is not None:
        app.include_router(_shp_router)
        logger.info("Steam/Heat Purchase router included (AGENT-MRV-011)")
    else:
        logger.debug("Steam/Heat Purchase router not available; skipping")

    # Cooling Purchase router (AGENT-MRV-012)
    if COOLING_PURCHASE_AVAILABLE and _cp_router is not None:
        app.include_router(_cp_router)
        logger.info("Cooling Purchase router included (AGENT-MRV-012)")
    else:
        logger.debug("Cooling Purchase router not available; skipping")

    # Dual Reporting Reconciliation router (AGENT-MRV-013)
    if DUAL_REPORTING_AVAILABLE and _drr_router is not None:
        app.include_router(_drr_router)
        logger.info("Dual Reporting Reconciliation router included (AGENT-MRV-013)")
    else:
        logger.debug("Dual Reporting Reconciliation router not available; skipping")

    # Purchased Goods & Services router (AGENT-MRV-014)
    if PURCHASED_GOODS_AVAILABLE and _pgs_router is not None:
        app.include_router(_pgs_router)
        logger.info("Purchased Goods & Services router included (AGENT-MRV-014)")
    else:
        logger.debug("Purchased Goods & Services router not available; skipping")

    # Capital Goods router (AGENT-MRV-015)
    if CAPITAL_GOODS_AVAILABLE and _cg_router is not None:
        app.include_router(_cg_router)
        logger.info("Capital Goods router included (AGENT-MRV-015)")
    else:
        logger.debug("Capital Goods router not available; skipping")

    # Fuel & Energy Activities router (AGENT-MRV-016)
    if FUEL_ENERGY_ACTIVITIES_AVAILABLE and _fea_router is not None:
        app.include_router(_fea_router)
        logger.info("Fuel & Energy Activities router included (AGENT-MRV-016)")
    else:
        logger.debug("Fuel & Energy Activities router not available; skipping")

    # Upstream Transportation & Distribution router (AGENT-MRV-017)
    if UPSTREAM_TRANSPORTATION_AVAILABLE and _uto_router is not None:
        app.include_router(_uto_router)
        logger.info("Upstream Transportation & Distribution router included (AGENT-MRV-017)")
    else:
        logger.debug("Upstream Transportation & Distribution router not available; skipping")

    # Waste Generated router (AGENT-MRV-018)
    if WASTE_GENERATED_AVAILABLE and _wg_router is not None:
        app.include_router(_wg_router)
        logger.info("Waste Generated router included (AGENT-MRV-018)")
    else:
        logger.debug("Waste Generated router not available; skipping")

    # Business Travel router (AGENT-MRV-019)
    if BUSINESS_TRAVEL_AVAILABLE and _bt_router is not None:
        app.include_router(_bt_router)
        logger.info("Business Travel router included (AGENT-MRV-019)")
    else:
        logger.debug("Business Travel router not available; skipping")

    # Employee Commuting router (AGENT-MRV-020)
    if EMPLOYEE_COMMUTING_AVAILABLE and _ec_router is not None:
        app.include_router(_ec_router)
        logger.info("Employee Commuting router included (AGENT-MRV-020)")
    else:
        logger.debug("Employee Commuting router not available; skipping")

    # Upstream Leased Assets router (AGENT-MRV-021)
    if UPSTREAM_LEASED_AVAILABLE and _ula_router is not None:
        app.include_router(_ula_router)
        logger.info("Upstream Leased Assets router included (AGENT-MRV-021)")
    else:
        logger.debug("Upstream Leased Assets router not available; skipping")

    # Downstream Transportation router (AGENT-MRV-022)
    if DOWNSTREAM_TRANSPORT_AVAILABLE and _dto_router is not None:
        app.include_router(_dto_router)
        logger.info("Downstream Transportation router included (AGENT-MRV-022)")
    else:
        logger.debug("Downstream Transportation router not available; skipping")

    # Processing of Sold Products router (AGENT-MRV-023)
    if PROCESSING_SOLD_PRODUCTS_AVAILABLE and _psp_router is not None:
        app.include_router(_psp_router)
        logger.info("Processing of Sold Products router included (AGENT-MRV-023)")
    else:
        logger.debug("Processing of Sold Products router not available; skipping")

    # Use of Sold Products router (AGENT-MRV-024)
    if USE_OF_SOLD_PRODUCTS_AVAILABLE and _usp_router is not None:
        app.include_router(_usp_router)
        logger.info("Use of Sold Products router included (AGENT-MRV-024)")
    else:
        logger.debug("Use of Sold Products router not available; skipping")

    # End-of-Life Treatment router (AGENT-MRV-025)
    if END_OF_LIFE_TREATMENT_AVAILABLE and _eol_router is not None:
        app.include_router(_eol_router)
        logger.info("End-of-Life Treatment router included (AGENT-MRV-025)")
    else:
        logger.debug("End-of-Life Treatment router not available; skipping")

    # Downstream Leased Assets router (AGENT-MRV-026)
    if DOWNSTREAM_LEASED_ASSETS_AVAILABLE and _dla_router is not None:
        app.include_router(_dla_router)
        logger.info("Downstream Leased Assets router included (AGENT-MRV-026)")
    else:
        logger.debug("Downstream Leased Assets router not available; skipping")

    # Franchises router (AGENT-MRV-027)
    if FRANCHISES_AVAILABLE and _frn_router is not None:
        app.include_router(_frn_router)
        logger.info("Franchises router included (AGENT-MRV-027)")
    else:
        logger.debug("Franchises router not available; skipping")

    # Investments router (AGENT-MRV-028)
    if INVESTMENTS_AVAILABLE and _inv_router is not None:
        app.include_router(_inv_router)
        logger.info("Investments router included (AGENT-MRV-028)")
    else:
        logger.debug("Investments router not available; skipping")

    # Scope 3 Category Mapper router (AGENT-MRV-029)
    if SCOPE3_CATEGORY_MAPPER_AVAILABLE and _scm_router is not None:
        app.include_router(_scm_router)
        logger.info("Scope 3 Category Mapper router included (AGENT-MRV-029)")
    else:
        logger.debug("Scope 3 Category Mapper router not available; skipping")

    # Audit Trail & Lineage router (AGENT-MRV-030)
    if AUDIT_TRAIL_LINEAGE_AVAILABLE and _atl_router is not None:
        app.include_router(_atl_router)
        logger.info("Audit Trail & Lineage router included (AGENT-MRV-030)")
    else:
        logger.debug("Audit Trail & Lineage router not available; skipping")

    # ISO 14064 Compliance Platform router (GL-ISO14064-APP)
    if ISO14064_APP_AVAILABLE and _iso14064_router is not None:
        app.include_router(
            _iso14064_router,
            prefix="/api/v1/iso14064",
            tags=["iso14064"],
        )
        logger.info("ISO 14064 Compliance Platform router included (GL-ISO14064-APP)")
    else:
        logger.debug("ISO 14064 Compliance Platform router not available; skipping")

    # CDP Disclosure Platform router (GL-CDP-APP) - APP-007
    if CDP_APP_AVAILABLE and create_cdp_app is not None:
        try:
            from applications.GL_CDP_APP.CDP_Disclosure_Platform.services.api import (
                questionnaire_router,
                response_router,
                scoring_router,
                gap_analysis_router,
                benchmarking_router,
                supply_chain_router,
                transition_plan_router,
                reporting_router,
                dashboard_router,
                settings_router,
            )
            for _cdp_rtr in [
                questionnaire_router, response_router, scoring_router,
                gap_analysis_router, benchmarking_router, supply_chain_router,
                transition_plan_router, reporting_router, dashboard_router,
                settings_router,
            ]:
                if _cdp_rtr is not None:
                    app.include_router(_cdp_rtr)
            logger.info("CDP Disclosure Platform routers included (GL-CDP-APP)")
        except ImportError:
            logger.debug("CDP Disclosure Platform routers import failed; skipping")
    else:
        logger.debug("CDP Disclosure Platform not available; skipping")

    # TCFD Climate Disclosure Platform router (GL-TCFD-APP) - APP-008
    if TCFD_APP_AVAILABLE and _tcfd_router is not None:
        app.include_router(
            _tcfd_router,
            prefix="/api/v1/tcfd",
            tags=["tcfd"],
        )
        logger.info("TCFD Climate Disclosure Platform router included (GL-TCFD-APP)")
    else:
        logger.debug("TCFD Climate Disclosure Platform not available; skipping")

    # SBTi Target Validation Platform router (GL-SBTi-APP) - APP-009
    if SBTI_APP_AVAILABLE and _sbti_router is not None:
        app.include_router(
            _sbti_router,
            prefix="/api/v1/sbti",
            tags=["sbti"],
        )
        logger.info("SBTi Target Validation Platform router included (GL-SBTi-APP)")
    else:
        logger.debug("SBTi Target Validation Platform not available; skipping")

    # EU Taxonomy Alignment Platform router (GL-Taxonomy-APP) - APP-010
    if TAXONOMY_APP_AVAILABLE and _taxonomy_router is not None:
        app.include_router(
            _taxonomy_router,
            prefix="/api/v1/taxonomy",
            tags=["taxonomy"],
        )
        logger.info("EU Taxonomy Alignment Platform router included (GL-Taxonomy-APP)")
    else:
        logger.debug("EU Taxonomy Alignment Platform not available; skipping")


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
