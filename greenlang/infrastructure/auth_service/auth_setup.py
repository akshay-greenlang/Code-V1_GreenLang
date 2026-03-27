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
    from greenlang.agents.data.duplicate_detector.api.router import router as dedup_router

    DUPLICATE_DETECTOR_AVAILABLE = True
except ImportError:
    dedup_router = None
    DUPLICATE_DETECTOR_AVAILABLE = False

# Missing Value Imputer Agent imports (AGENT-DATA-012)
try:
    from greenlang.agents.data.missing_value_imputer.api.router import router as imputer_router

    MISSING_VALUE_IMPUTER_AVAILABLE = True
except ImportError:
    imputer_router = None
    MISSING_VALUE_IMPUTER_AVAILABLE = False

# Outlier Detection Agent imports (AGENT-DATA-013)
try:
    from greenlang.agents.data.outlier_detector.api.router import router as outlier_router

    OUTLIER_DETECTOR_AVAILABLE = True
except ImportError:
    outlier_router = None
    OUTLIER_DETECTOR_AVAILABLE = False

# Time Series Gap Filler Agent imports (AGENT-DATA-014)
try:
    from greenlang.agents.data.time_series_gap_filler.setup import get_router as get_gap_filler_router

    _gap_filler_router = get_gap_filler_router()
    GAP_FILLER_AVAILABLE = True
except ImportError:
    _gap_filler_router = None
    GAP_FILLER_AVAILABLE = False

# Cross-Source Reconciliation Agent imports (AGENT-DATA-015)
try:
    from greenlang.agents.data.cross_source_reconciliation.setup import get_router as get_reconciliation_router

    _reconciliation_router = get_reconciliation_router()
    RECONCILIATION_AVAILABLE = True
except ImportError:
    _reconciliation_router = None
    RECONCILIATION_AVAILABLE = False

# Data Freshness Monitor Agent imports (AGENT-DATA-016)
try:
    from greenlang.agents.data.data_freshness_monitor.setup import get_router as get_freshness_router

    _freshness_router = get_freshness_router()
    FRESHNESS_MONITOR_AVAILABLE = True
except ImportError:
    _freshness_router = None
    FRESHNESS_MONITOR_AVAILABLE = False

# Schema Migration Agent imports (AGENT-DATA-017)
try:
    from greenlang.agents.data.schema_migration.setup import get_router as get_sm_router

    _sm_router = get_sm_router()
    SCHEMA_MIGRATION_AVAILABLE = True
except ImportError:
    _sm_router = None
    SCHEMA_MIGRATION_AVAILABLE = False

# Data Lineage Tracker Agent imports (AGENT-DATA-018)
try:
    from greenlang.agents.data.data_lineage_tracker.setup import get_router as get_dlt_router

    DATA_LINEAGE_AVAILABLE = True
    _dlt_router = get_dlt_router()
except ImportError:
    DATA_LINEAGE_AVAILABLE = False
    _dlt_router = None

# Validation Rule Engine imports (AGENT-DATA-019)
try:
    from greenlang.agents.data.validation_rule_engine.setup import get_router as get_vre_router
    VALIDATION_RULE_ENGINE_AVAILABLE = True
    _vre_router = get_vre_router()
except ImportError:
    VALIDATION_RULE_ENGINE_AVAILABLE = False
    _vre_router = None

# Climate Hazard Connector imports (AGENT-DATA-020)
try:
    from greenlang.agents.data.climate_hazard.setup import get_router as get_chc_router
    CLIMATE_HAZARD_AVAILABLE = True
    _chc_router = get_chc_router()
except ImportError:
    CLIMATE_HAZARD_AVAILABLE = False
    _chc_router = None

# Stationary Combustion imports (AGENT-MRV-001)
try:
    from greenlang.agents.mrv.stationary_combustion.setup import get_router as get_sc_router
    STATIONARY_COMBUSTION_AVAILABLE = True
    _sc_router = get_sc_router()
except ImportError:
    STATIONARY_COMBUSTION_AVAILABLE = False
    _sc_router = None

# Refrigerants & F-Gas imports (AGENT-MRV-002)
try:
    from greenlang.agents.mrv.refrigerants_fgas.setup import get_router as get_rf_router
    REFRIGERANTS_FGAS_AVAILABLE = True
    _rf_router = get_rf_router()
except ImportError:
    REFRIGERANTS_FGAS_AVAILABLE = False
    _rf_router = None

# Mobile Combustion imports (AGENT-MRV-003)
try:
    from greenlang.agents.mrv.mobile_combustion.setup import get_router as get_mc_router
    MOBILE_COMBUSTION_AVAILABLE = True
    _mc_router = get_mc_router()
except ImportError:
    MOBILE_COMBUSTION_AVAILABLE = False
    _mc_router = None

# Process Emissions imports (AGENT-MRV-004)
try:
    from greenlang.agents.mrv.process_emissions.setup import get_router as get_pe_router
    PROCESS_EMISSIONS_AVAILABLE = True
    _pe_router = get_pe_router()
except ImportError:
    PROCESS_EMISSIONS_AVAILABLE = False
    _pe_router = None

# Fugitive Emissions imports (AGENT-MRV-005)
try:
    from greenlang.agents.mrv.fugitive_emissions.setup import get_router as get_fue_router
    FUGITIVE_EMISSIONS_AVAILABLE = True
    _fue_router = get_fue_router()
except ImportError:
    FUGITIVE_EMISSIONS_AVAILABLE = False
    _fue_router = None

# Land Use Emissions imports (AGENT-MRV-006)
try:
    from greenlang.agents.mrv.land_use_emissions.setup import get_router as get_lu_router
    LAND_USE_EMISSIONS_AVAILABLE = True
    _lu_router = get_lu_router()
except ImportError:
    LAND_USE_EMISSIONS_AVAILABLE = False
    _lu_router = None

# Waste Treatment Emissions imports (AGENT-MRV-007)
try:
    from greenlang.agents.mrv.waste_treatment_emissions.setup import get_router as get_wt_router
    WASTE_TREATMENT_EMISSIONS_AVAILABLE = True
    _wt_router = get_wt_router()
except ImportError:
    WASTE_TREATMENT_EMISSIONS_AVAILABLE = False
    _wt_router = None


# Agricultural Emissions imports (AGENT-MRV-008)
try:
    from greenlang.agents.mrv.agricultural_emissions.setup import get_router as get_ag_router
    AGRICULTURAL_EMISSIONS_AVAILABLE = True
    _ag_router = get_ag_router()
except ImportError:
    AGRICULTURAL_EMISSIONS_AVAILABLE = False
    _ag_router = None

# Scope 2 Location-Based Emissions imports (AGENT-MRV-009)
try:
    from greenlang.agents.mrv.scope2_location.api.router import router as _s2l_router
    SCOPE2_LOCATION_AVAILABLE = True
except ImportError:
    SCOPE2_LOCATION_AVAILABLE = False
    _s2l_router = None

# Scope 2 Market-Based Emissions imports (AGENT-MRV-010)
try:
    from greenlang.agents.mrv.scope2_market.api.router import router as _s2m_router
    SCOPE2_MARKET_AVAILABLE = True
except ImportError:
    SCOPE2_MARKET_AVAILABLE = False
    _s2m_router = None

# Steam/Heat Purchase imports (AGENT-MRV-011)
try:
    from greenlang.agents.mrv.steam_heat_purchase.api.router import router as _shp_router
    STEAM_HEAT_PURCHASE_AVAILABLE = True
except ImportError:
    STEAM_HEAT_PURCHASE_AVAILABLE = False
    _shp_router = None

# Cooling Purchase imports (AGENT-MRV-012)
try:
    from greenlang.agents.mrv.cooling_purchase.api.router import router as _cp_router
    COOLING_PURCHASE_AVAILABLE = True
except ImportError:
    COOLING_PURCHASE_AVAILABLE = False
    _cp_router = None

# Dual Reporting Reconciliation imports (AGENT-MRV-013)
try:
    from greenlang.agents.mrv.dual_reporting_reconciliation.api.router import router as _drr_router
    DUAL_REPORTING_AVAILABLE = True
except ImportError:
    DUAL_REPORTING_AVAILABLE = False
    _drr_router = None

# Purchased Goods & Services imports (AGENT-MRV-014)
try:
    from greenlang.agents.mrv.purchased_goods_services.setup import get_router as get_pgs_router
    PURCHASED_GOODS_AVAILABLE = True
    _pgs_router = get_pgs_router()
except ImportError:
    PURCHASED_GOODS_AVAILABLE = False
    _pgs_router = None

# Capital Goods imports (AGENT-MRV-015)
try:
    from greenlang.agents.mrv.capital_goods.setup import get_router as get_cg_router
    CAPITAL_GOODS_AVAILABLE = True
    _cg_router = get_cg_router()
except ImportError:
    CAPITAL_GOODS_AVAILABLE = False
    _cg_router = None

# Fuel & Energy Activities imports (AGENT-MRV-016)
try:
    from greenlang.agents.mrv.fuel_energy_activities.setup import get_router as get_fea_router
    FUEL_ENERGY_ACTIVITIES_AVAILABLE = True
    _fea_router = get_fea_router()
except ImportError:
    FUEL_ENERGY_ACTIVITIES_AVAILABLE = False
    _fea_router = None

# Upstream Transportation & Distribution imports (AGENT-MRV-017)
try:
    from greenlang.agents.mrv.upstream_transportation.setup import get_router as get_uto_router
    UPSTREAM_TRANSPORTATION_AVAILABLE = True
    _uto_router = get_uto_router()
except ImportError:
    UPSTREAM_TRANSPORTATION_AVAILABLE = False
    _uto_router = None

# Waste Generated imports (AGENT-MRV-018)
try:
    from greenlang.agents.mrv.waste_generated.setup import get_router as get_wg_router
    WASTE_GENERATED_AVAILABLE = True
    _wg_router = get_wg_router()
except ImportError:
    WASTE_GENERATED_AVAILABLE = False
    _wg_router = None

# Business Travel imports (AGENT-MRV-019)
try:
    from greenlang.agents.mrv.business_travel.setup import get_router as get_bt_router
    BUSINESS_TRAVEL_AVAILABLE = True
    _bt_router = get_bt_router()
except ImportError:
    BUSINESS_TRAVEL_AVAILABLE = False
    _bt_router = None

# Employee Commuting imports (AGENT-MRV-020)
try:
    from greenlang.agents.mrv.employee_commuting.setup import get_router as get_ec_router
    EMPLOYEE_COMMUTING_AVAILABLE = True
    _ec_router = get_ec_router()
except ImportError:
    EMPLOYEE_COMMUTING_AVAILABLE = False
    _ec_router = None

# Upstream Leased Assets imports (AGENT-MRV-021)
try:
    from greenlang.agents.mrv.upstream_leased_assets.setup import get_router as get_ula_router
    UPSTREAM_LEASED_AVAILABLE = True
    _ula_router = get_ula_router()
except ImportError:
    UPSTREAM_LEASED_AVAILABLE = False
    _ula_router = None

# Downstream Transportation imports (AGENT-MRV-022)
try:
    from greenlang.agents.mrv.downstream_transportation.setup import get_router as get_dto_router
    DOWNSTREAM_TRANSPORT_AVAILABLE = True
    _dto_router = get_dto_router()
except ImportError:
    DOWNSTREAM_TRANSPORT_AVAILABLE = False
    _dto_router = None

# Processing of Sold Products imports (AGENT-MRV-023)
try:
    from greenlang.agents.mrv.processing_sold_products.setup import get_router as get_psp_router
    PROCESSING_SOLD_PRODUCTS_AVAILABLE = True
    _psp_router = get_psp_router()
except ImportError:
    PROCESSING_SOLD_PRODUCTS_AVAILABLE = False
    _psp_router = None

# Use of Sold Products imports (AGENT-MRV-024)
try:
    from greenlang.agents.mrv.use_of_sold_products.setup import get_router as get_usp_router
    USE_OF_SOLD_PRODUCTS_AVAILABLE = True
    _usp_router = get_usp_router()
except ImportError:
    USE_OF_SOLD_PRODUCTS_AVAILABLE = False
    _usp_router = None

# End-of-Life Treatment imports (AGENT-MRV-025)
try:
    from greenlang.agents.mrv.end_of_life_treatment.setup import get_router as get_eol_router
    END_OF_LIFE_TREATMENT_AVAILABLE = True
    _eol_router = get_eol_router()
except ImportError:
    END_OF_LIFE_TREATMENT_AVAILABLE = False
    _eol_router = None

# Downstream Leased Assets imports (AGENT-MRV-026)
try:
    from greenlang.agents.mrv.downstream_leased_assets.setup import get_router as get_dla_router
    DOWNSTREAM_LEASED_ASSETS_AVAILABLE = True
    _dla_router = get_dla_router()
except ImportError:
    DOWNSTREAM_LEASED_ASSETS_AVAILABLE = False
    _dla_router = None

# Franchises imports (AGENT-MRV-027)
try:
    from greenlang.agents.mrv.franchises.setup import get_router as get_frn_router
    FRANCHISES_AVAILABLE = True
    _frn_router = get_frn_router()
except ImportError:
    FRANCHISES_AVAILABLE = False
    _frn_router = None

# Investments imports (AGENT-MRV-028)
try:
    from greenlang.agents.mrv.investments.setup import get_router as get_inv_router
    INVESTMENTS_AVAILABLE = True
    _inv_router = get_inv_router()
except ImportError:
    INVESTMENTS_AVAILABLE = False
    _inv_router = None

# Scope 3 Category Mapper Agent imports (AGENT-MRV-029)
try:
    from greenlang.agents.mrv.scope3_category_mapper.setup import get_router as get_scm_router
    SCOPE3_CATEGORY_MAPPER_AVAILABLE = True
    _scm_router = get_scm_router()
except ImportError:
    SCOPE3_CATEGORY_MAPPER_AVAILABLE = False
    _scm_router = None

# Audit Trail & Lineage Agent imports (AGENT-MRV-030)
try:
    from greenlang.agents.mrv.audit_trail_lineage.setup import get_router as get_atl_router
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

# EUDR Supply Chain Mapper imports (AGENT-EUDR-001)
try:
    from greenlang.agents.eudr.supply_chain_mapper.api.router import get_router as get_eudr_scm_router
    _eudr_scm_router = get_eudr_scm_router()
    EUDR_SUPPLY_CHAIN_MAPPER_AVAILABLE = True
except ImportError:
    EUDR_SUPPLY_CHAIN_MAPPER_AVAILABLE = False
    _eudr_scm_router = None

# EUDR Geolocation Verification imports (AGENT-EUDR-002)
try:
    from greenlang.agents.eudr.geolocation_verification.api.router import get_router as get_eudr_geo_router
    _eudr_geo_router = get_eudr_geo_router()
    EUDR_GEOLOCATION_VERIFICATION_AVAILABLE = True
except ImportError:
    EUDR_GEOLOCATION_VERIFICATION_AVAILABLE = False
    _eudr_geo_router = None

# EUDR Satellite Monitoring imports (AGENT-EUDR-003)
try:
    from greenlang.agents.eudr.satellite_monitoring.api.router import get_router as get_eudr_sat_router
    _eudr_sat_router = get_eudr_sat_router()
    EUDR_SATELLITE_MONITORING_AVAILABLE = True
except ImportError:
    EUDR_SATELLITE_MONITORING_AVAILABLE = False
    _eudr_sat_router = None

# EUDR Forest Cover Analysis imports (AGENT-EUDR-004)
try:
    from greenlang.agents.eudr.forest_cover_analysis.api.router import get_router as get_eudr_fca_router
    _eudr_fca_router = get_eudr_fca_router()
    EUDR_FOREST_COVER_ANALYSIS_AVAILABLE = True
except ImportError:
    EUDR_FOREST_COVER_ANALYSIS_AVAILABLE = False
    _eudr_fca_router = None

# EUDR Land Use Change Detector imports (AGENT-EUDR-005)
try:
    from greenlang.agents.eudr.land_use_change.api.router import get_router as get_eudr_luc_router
    _eudr_luc_router = get_eudr_luc_router()
    EUDR_LAND_USE_CHANGE_AVAILABLE = True
except ImportError:
    EUDR_LAND_USE_CHANGE_AVAILABLE = False
    _eudr_luc_router = None

# EUDR Plot Boundary Manager imports (AGENT-EUDR-006)
try:
    from greenlang.agents.eudr.plot_boundary.api.router import get_router as get_eudr_pbm_router
    _eudr_pbm_router = get_eudr_pbm_router()
    EUDR_PLOT_BOUNDARY_AVAILABLE = True
except ImportError:
    EUDR_PLOT_BOUNDARY_AVAILABLE = False
    _eudr_pbm_router = None

# EUDR GPS Coordinate Validator imports (AGENT-EUDR-007)
try:
    from greenlang.agents.eudr.gps_coordinate_validator.api.router import get_router as get_eudr_gcv_router
    _eudr_gcv_router = get_eudr_gcv_router()
    EUDR_GPS_COORDINATE_AVAILABLE = True
except ImportError:
    EUDR_GPS_COORDINATE_AVAILABLE = False
    _eudr_gcv_router = None

# EUDR Multi-Tier Supplier Tracker imports (AGENT-EUDR-008)
try:
    from greenlang.agents.eudr.multi_tier_supplier.api.router import get_router as get_eudr_mst_router
    _eudr_mst_router = get_eudr_mst_router()
    EUDR_MULTI_TIER_SUPPLIER_AVAILABLE = True
except ImportError:
    EUDR_MULTI_TIER_SUPPLIER_AVAILABLE = False
    _eudr_mst_router = None

# EUDR Chain of Custody imports (AGENT-EUDR-009)
try:
    from greenlang.agents.eudr.chain_of_custody.api.router import get_router as get_eudr_coc_router
    _eudr_coc_router = get_eudr_coc_router()
    EUDR_CHAIN_OF_CUSTODY_AVAILABLE = True
except ImportError:
    EUDR_CHAIN_OF_CUSTODY_AVAILABLE = False
    _eudr_coc_router = None

# AGENT-EUDR-010: Segregation Verifier
try:
    from greenlang.agents.eudr.segregation_verifier.api.router import get_router as get_eudr_sgv_router
    _eudr_sgv_router = get_eudr_sgv_router()
    EUDR_SEGREGATION_VERIFIER_AVAILABLE = True
except ImportError:
    EUDR_SEGREGATION_VERIFIER_AVAILABLE = False
    _eudr_sgv_router = None

# AGENT-EUDR-011: Mass Balance Calculator
try:
    from greenlang.agents.eudr.mass_balance_calculator.api.router import get_router as get_eudr_mbc_router
    _eudr_mbc_router = get_eudr_mbc_router()
    EUDR_MASS_BALANCE_CALCULATOR_AVAILABLE = True
except ImportError:
    EUDR_MASS_BALANCE_CALCULATOR_AVAILABLE = False
    _eudr_mbc_router = None

# AGENT-EUDR-012: Document Authentication
try:
    from greenlang.agents.eudr.document_authentication.api.router import get_router as get_eudr_dav_router
    _eudr_dav_router = get_eudr_dav_router()
    EUDR_DOCUMENT_AUTHENTICATION_AVAILABLE = True
except ImportError:
    EUDR_DOCUMENT_AUTHENTICATION_AVAILABLE = False
    _eudr_dav_router = None

# AGENT-EUDR-013: Blockchain Integration
try:
    from greenlang.agents.eudr.blockchain_integration.api.router import get_router as get_eudr_bci_router
    _eudr_bci_router = get_eudr_bci_router()
    EUDR_BLOCKCHAIN_INTEGRATION_AVAILABLE = True
except ImportError:
    EUDR_BLOCKCHAIN_INTEGRATION_AVAILABLE = False
    _eudr_bci_router = None

# AGENT-EUDR-014: QR Code Generator
try:
    from greenlang.agents.eudr.qr_code_generator.api.router import get_router as get_eudr_qrg_router
    _eudr_qrg_router = get_eudr_qrg_router()
    EUDR_QR_CODE_GENERATOR_AVAILABLE = True
except ImportError:
    EUDR_QR_CODE_GENERATOR_AVAILABLE = False
    _eudr_qrg_router = None

# AGENT-EUDR-015: Mobile Data Collector
try:
    from greenlang.agents.eudr.mobile_data_collector.api.router import get_router as get_eudr_mdc_router
    _eudr_mdc_router = get_eudr_mdc_router()
    EUDR_MOBILE_DATA_COLLECTOR_AVAILABLE = True
except ImportError:
    EUDR_MOBILE_DATA_COLLECTOR_AVAILABLE = False
    _eudr_mdc_router = None

# AGENT-EUDR-016: Country Risk Evaluator
try:
    from greenlang.agents.eudr.country_risk_evaluator.api.router import get_router as get_eudr_cre_router
    _eudr_cre_router = get_eudr_cre_router()
    EUDR_COUNTRY_RISK_EVALUATOR_AVAILABLE = True
except ImportError:
    EUDR_COUNTRY_RISK_EVALUATOR_AVAILABLE = False
    _eudr_cre_router = None

# AGENT-EUDR-017: Supplier Risk Scorer
try:
    from greenlang.agents.eudr.supplier_risk_scorer.api.router import get_router as get_eudr_srs_router
    _eudr_srs_router = get_eudr_srs_router()
    EUDR_SUPPLIER_RISK_SCORER_AVAILABLE = True
except ImportError:
    EUDR_SUPPLIER_RISK_SCORER_AVAILABLE = False
    _eudr_srs_router = None

# AGENT-EUDR-018: Commodity Risk Analyzer
try:
    from greenlang.agents.eudr.commodity_risk_analyzer.api.router import get_router as get_eudr_cra_router
    _eudr_cra_router = get_eudr_cra_router()
    EUDR_COMMODITY_RISK_ANALYZER_AVAILABLE = True
except ImportError:
    EUDR_COMMODITY_RISK_ANALYZER_AVAILABLE = False
    _eudr_cra_router = None

# AGENT-EUDR-019: Corruption Index Monitor
try:
    from greenlang.agents.eudr.corruption_index_monitor.api.router import get_router as get_eudr_cim_router
    _eudr_cim_router = get_eudr_cim_router()
    EUDR_CORRUPTION_INDEX_MONITOR_AVAILABLE = True
except ImportError:
    EUDR_CORRUPTION_INDEX_MONITOR_AVAILABLE = False
    _eudr_cim_router = None

# AGENT-EUDR-020: Deforestation Alert System
try:
    from greenlang.agents.eudr.deforestation_alert_system.api.router import get_router as get_eudr_das_router
    _eudr_das_router = get_eudr_das_router()
    EUDR_DEFORESTATION_ALERT_SYSTEM_AVAILABLE = True
except ImportError:
    EUDR_DEFORESTATION_ALERT_SYSTEM_AVAILABLE = False
    _eudr_das_router = None

# AGENT-EUDR-021: Indigenous Rights Checker
try:
    from greenlang.agents.eudr.indigenous_rights_checker.api.router import get_router as get_eudr_irc_router
    _eudr_irc_router = get_eudr_irc_router()
    EUDR_INDIGENOUS_RIGHTS_CHECKER_AVAILABLE = True
except ImportError:
    EUDR_INDIGENOUS_RIGHTS_CHECKER_AVAILABLE = False
    _eudr_irc_router = None

# AGENT-EUDR-022: Protected Area Validator
try:
    from greenlang.agents.eudr.protected_area_validator.api.router import get_router as get_eudr_pav_router
    _eudr_pav_router = get_eudr_pav_router()
    EUDR_PROTECTED_AREA_VALIDATOR_AVAILABLE = True
except ImportError:
    EUDR_PROTECTED_AREA_VALIDATOR_AVAILABLE = False
    _eudr_pav_router = None

# AGENT-EUDR-023: Legal Compliance Verifier
try:
    from greenlang.agents.eudr.legal_compliance_verifier.api.router import get_router as get_eudr_lcv_router
    _eudr_lcv_router = get_eudr_lcv_router()
    EUDR_LEGAL_COMPLIANCE_VERIFIER_AVAILABLE = True
except ImportError:
    EUDR_LEGAL_COMPLIANCE_VERIFIER_AVAILABLE = False
    _eudr_lcv_router = None

# AGENT-EUDR-024: Third-Party Audit Manager
try:
    from greenlang.agents.eudr.third_party_audit_manager.api.router import get_router as get_eudr_tam_router
    _eudr_tam_router = get_eudr_tam_router()
    EUDR_THIRD_PARTY_AUDIT_MANAGER_AVAILABLE = True
except ImportError:
    EUDR_THIRD_PARTY_AUDIT_MANAGER_AVAILABLE = False
    _eudr_tam_router = None

# AGENT-EUDR-025: Risk Mitigation Advisor
try:
    from greenlang.agents.eudr.risk_mitigation_advisor.api.router import get_router as get_eudr_rma_router
    _eudr_rma_router = get_eudr_rma_router()
    EUDR_RISK_MITIGATION_ADVISOR_AVAILABLE = True
except ImportError:
    EUDR_RISK_MITIGATION_ADVISOR_AVAILABLE = False
    _eudr_rma_router = None

# AGENT-EUDR-027: Information Gathering Agent
try:
    from greenlang.agents.eudr.information_gathering.api import get_router as get_eudr_iga_router
    _eudr_iga_router = get_eudr_iga_router()
    EUDR_INFORMATION_GATHERING_AVAILABLE = True
except ImportError:
    EUDR_INFORMATION_GATHERING_AVAILABLE = False
    _eudr_iga_router = None

# AGENT-EUDR-028: Risk Assessment Engine
try:
    from greenlang.agents.eudr.risk_assessment_engine.api import get_router as get_eudr_rae_router
    _eudr_rae_router = get_eudr_rae_router()
    EUDR_RISK_ASSESSMENT_ENGINE_AVAILABLE = True
except ImportError:
    EUDR_RISK_ASSESSMENT_ENGINE_AVAILABLE = False
    _eudr_rae_router = None

# AGENT-EUDR-029: Mitigation Measure Designer
try:
    from greenlang.agents.eudr.mitigation_measure_designer import api as eudr_mmd_api
    _EUDR_MMD_AVAILABLE = True
except ImportError:
    _EUDR_MMD_AVAILABLE = False
    logger.debug("EUDR-029 Mitigation Measure Designer not available")

# AGENT-EUDR-030: Documentation Generator
try:
    from greenlang.agents.eudr.documentation_generator import api as eudr_dgn_api
    _EUDR_DGN_AVAILABLE = True
except ImportError:
    _EUDR_DGN_AVAILABLE = False
    logger.debug("EUDR-030 Documentation Generator not available")

# AGENT-EUDR-031: Stakeholder Engagement Tool
try:
    from greenlang.agents.eudr.stakeholder_engagement import api as eudr_set_api
    _EUDR_SET_AVAILABLE = True
except ImportError:
    _EUDR_SET_AVAILABLE = False
    logger.debug("EUDR-031 Stakeholder Engagement Tool not available")

# AGENT-EUDR-032: Grievance Mechanism Manager
try:
    from greenlang.agents.eudr.grievance_mechanism_manager import api as eudr_gmm_api
    _EUDR_GMM_AVAILABLE = True
except ImportError:
    _EUDR_GMM_AVAILABLE = False
    logger.debug("EUDR-032 Grievance Mechanism Manager not available")

# AGENT-EUDR-033: Continuous Monitoring Agent
try:
    from greenlang.agents.eudr.continuous_monitoring import api as eudr_cm_api
    _EUDR_CM_AVAILABLE = True
except ImportError:
    _EUDR_CM_AVAILABLE = False
    logger.debug("EUDR-033 Continuous Monitoring Agent not available")

# AGENT-EUDR-034: Annual Review Scheduler
try:
    from greenlang.agents.eudr.annual_review_scheduler import api as eudr_ars_api
    _EUDR_ARS_AVAILABLE = True
except ImportError:
    _EUDR_ARS_AVAILABLE = False
    logger.debug("EUDR-034 Annual Review Scheduler not available")

# AGENT-EUDR-035: Improvement Plan Creator
try:
    from greenlang.agents.eudr.improvement_plan_creator import api as eudr_ipc_api
    _EUDR_IPC_AVAILABLE = True
except ImportError:
    _EUDR_IPC_AVAILABLE = False
    logger.debug("EUDR-035 Improvement Plan Creator not available")

# AGENT-EUDR-036: EU Information System Interface
try:
    from greenlang.agents.eudr.eu_information_system_interface import api as eudr_euis_api
    _EUDR_EUIS_AVAILABLE = True
except ImportError:
    _EUDR_EUIS_AVAILABLE = False
    logger.debug("EUDR-036 EU Information System Interface not available")

# AGENT-EUDR-037: Due Diligence Statement Creator
try:
    from greenlang.agents.eudr.due_diligence_statement_creator import api as eudr_ddsc_api
    _EUDR_DDSC_AVAILABLE = True
except ImportError:
    _EUDR_DDSC_AVAILABLE = False
    logger.debug("EUDR-037 Due Diligence Statement Creator not available")

# AGENT-EUDR-038: Reference Number Generator
try:
    from greenlang.agents.eudr.reference_number_generator import api as eudr_rng_api
    _EUDR_RNG_AVAILABLE = True
except ImportError:
    _EUDR_RNG_AVAILABLE = False
    logger.debug("EUDR-038 Reference Number Generator not available")

# AGENT-EUDR-039: Customs Declaration Support
try:
    from greenlang.agents.eudr.customs_declaration_support import api as eudr_cds_api
    _EUDR_CDS_AVAILABLE = True
except ImportError:
    _EUDR_CDS_AVAILABLE = False
    logger.debug("EUDR-039 Customs Declaration Support not available")

# AGENT-EUDR-040: Authority Communication Manager
try:
    from greenlang.agents.eudr.authority_communication_manager import api as eudr_acm_api
    _EUDR_ACM_AVAILABLE = True
except ImportError:
    _EUDR_ACM_AVAILABLE = False
    logger.debug("EUDR-040 Authority Communication Manager not available")


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

    # EUDR Supply Chain Mapper router (AGENT-EUDR-001)
    if EUDR_SUPPLY_CHAIN_MAPPER_AVAILABLE and _eudr_scm_router is not None:
        app.include_router(
            _eudr_scm_router,
            prefix="/api",
            tags=["eudr-supply-chain-mapper"],
        )
        logger.info("EUDR Supply Chain Mapper router included (AGENT-EUDR-001)")
    else:
        logger.debug("EUDR Supply Chain Mapper router not available; skipping")

    # EUDR Geolocation Verification router (AGENT-EUDR-002)
    if EUDR_GEOLOCATION_VERIFICATION_AVAILABLE and _eudr_geo_router is not None:
        app.include_router(
            _eudr_geo_router,
            prefix="/api",
            tags=["eudr-geolocation-verification"],
        )
        logger.info("EUDR Geolocation Verification router included (AGENT-EUDR-002)")
    else:
        logger.debug("EUDR Geolocation Verification router not available; skipping")

    # EUDR Satellite Monitoring router (AGENT-EUDR-003)
    if EUDR_SATELLITE_MONITORING_AVAILABLE and _eudr_sat_router is not None:
        app.include_router(
            _eudr_sat_router,
            prefix="/api",
            tags=["eudr-satellite-monitoring"],
        )
        logger.info("EUDR Satellite Monitoring router included (AGENT-EUDR-003)")
    else:
        logger.debug("EUDR Satellite Monitoring router not available; skipping")

    # EUDR Forest Cover Analysis router (AGENT-EUDR-004)
    if EUDR_FOREST_COVER_ANALYSIS_AVAILABLE and _eudr_fca_router is not None:
        app.include_router(
            _eudr_fca_router,
            prefix="/api",
            tags=["eudr-forest-cover-analysis"],
        )
        logger.info("EUDR Forest Cover Analysis router included (AGENT-EUDR-004)")
    else:
        logger.debug("EUDR Forest Cover Analysis router not available; skipping")

    # EUDR Land Use Change Detector router (AGENT-EUDR-005)
    if EUDR_LAND_USE_CHANGE_AVAILABLE and _eudr_luc_router is not None:
        app.include_router(
            _eudr_luc_router,
            prefix="/api",
            tags=["eudr-land-use-change"],
        )
        logger.info("EUDR Land Use Change Detector router included (AGENT-EUDR-005)")
    else:
        logger.debug("EUDR Land Use Change Detector router not available; skipping")

    # EUDR Plot Boundary Manager router (AGENT-EUDR-006)
    if EUDR_PLOT_BOUNDARY_AVAILABLE and _eudr_pbm_router is not None:
        app.include_router(
            _eudr_pbm_router,
            prefix="/api",
            tags=["eudr-plot-boundary"],
        )
        logger.info("EUDR Plot Boundary Manager router included (AGENT-EUDR-006)")
    else:
        logger.debug("EUDR Plot Boundary Manager router not available; skipping")

    # EUDR GPS Coordinate Validator router (AGENT-EUDR-007)
    if EUDR_GPS_COORDINATE_AVAILABLE and _eudr_gcv_router is not None:
        app.include_router(
            _eudr_gcv_router,
            prefix="/api",
            tags=["eudr-gps-coordinate-validator"],
        )
        logger.info("EUDR GPS Coordinate Validator router included (AGENT-EUDR-007)")
    else:
        logger.debug("EUDR GPS Coordinate Validator router not available; skipping")

    # EUDR Multi-Tier Supplier Tracker router (AGENT-EUDR-008)
    if EUDR_MULTI_TIER_SUPPLIER_AVAILABLE and _eudr_mst_router is not None:
        app.include_router(
            _eudr_mst_router,
            prefix="/api",
            tags=["eudr-multi-tier-supplier-tracker"],
        )
        logger.info("EUDR Multi-Tier Supplier Tracker router included (AGENT-EUDR-008)")
    else:
        logger.debug("EUDR Multi-Tier Supplier Tracker router not available; skipping")

    # EUDR Chain of Custody router (AGENT-EUDR-009)
    if EUDR_CHAIN_OF_CUSTODY_AVAILABLE and _eudr_coc_router is not None:
        app.include_router(
            _eudr_coc_router,
            prefix="/api",
            tags=["eudr-chain-of-custody"],
        )
        logger.info("EUDR Chain of Custody router included (AGENT-EUDR-009)")
    else:
        logger.debug("EUDR Chain of Custody router not available; skipping")

    # EUDR Segregation Verifier router (AGENT-EUDR-010)
    if EUDR_SEGREGATION_VERIFIER_AVAILABLE and _eudr_sgv_router is not None:
        app.include_router(
            _eudr_sgv_router,
            prefix="/api",
            tags=["eudr-segregation-verifier"],
        )
        logger.info("EUDR Segregation Verifier router included (AGENT-EUDR-010)")
    else:
        logger.debug("EUDR Segregation Verifier router not available; skipping")

    # EUDR Mass Balance Calculator router (AGENT-EUDR-011)
    if EUDR_MASS_BALANCE_CALCULATOR_AVAILABLE and _eudr_mbc_router is not None:
        app.include_router(
            _eudr_mbc_router,
            prefix="/api",
            tags=["eudr-mass-balance-calculator"],
        )
        logger.info("EUDR Mass Balance Calculator router included (AGENT-EUDR-011)")
    else:
        logger.debug("EUDR Mass Balance Calculator router not available; skipping")

    # EUDR Document Authentication router (AGENT-EUDR-012)
    if EUDR_DOCUMENT_AUTHENTICATION_AVAILABLE and _eudr_dav_router is not None:
        app.include_router(
            _eudr_dav_router,
            prefix="/api",
            tags=["eudr-document-authentication"],
        )
        logger.info("EUDR Document Authentication router included (AGENT-EUDR-012)")
    else:
        logger.debug("EUDR Document Authentication router not available; skipping")

    # EUDR Blockchain Integration router (AGENT-EUDR-013)
    if EUDR_BLOCKCHAIN_INTEGRATION_AVAILABLE and _eudr_bci_router is not None:
        app.include_router(
            _eudr_bci_router,
            prefix="/api",
            tags=["eudr-blockchain-integration"],
        )
        logger.info("EUDR Blockchain Integration router included (AGENT-EUDR-013)")
    else:
        logger.debug("EUDR Blockchain Integration router not available; skipping")

    # EUDR QR Code Generator router (AGENT-EUDR-014)
    if EUDR_QR_CODE_GENERATOR_AVAILABLE and _eudr_qrg_router is not None:
        app.include_router(
            _eudr_qrg_router,
            prefix="/api",
            tags=["eudr-qr-code-generator"],
        )
        logger.info("EUDR QR Code Generator router included (AGENT-EUDR-014)")
    else:
        logger.debug("EUDR QR Code Generator router not available; skipping")

    # EUDR Mobile Data Collector router (AGENT-EUDR-015)
    if EUDR_MOBILE_DATA_COLLECTOR_AVAILABLE and _eudr_mdc_router is not None:
        app.include_router(
            _eudr_mdc_router,
            prefix="/api",
            tags=["eudr-mobile-data-collector"],
        )
        logger.info("EUDR Mobile Data Collector router included (AGENT-EUDR-015)")
    else:
        logger.debug("EUDR Mobile Data Collector router not available; skipping")

    # EUDR Country Risk Evaluator router (AGENT-EUDR-016)
    if EUDR_COUNTRY_RISK_EVALUATOR_AVAILABLE and _eudr_cre_router is not None:
        app.include_router(
            _eudr_cre_router,
            prefix="/api",
            tags=["eudr-country-risk-evaluator"],
        )
        logger.info("EUDR Country Risk Evaluator router included (AGENT-EUDR-016)")
    else:
        logger.debug("EUDR Country Risk Evaluator router not available; skipping")

    # EUDR Supplier Risk Scorer router (AGENT-EUDR-017)
    if EUDR_SUPPLIER_RISK_SCORER_AVAILABLE and _eudr_srs_router is not None:
        app.include_router(
            _eudr_srs_router,
            prefix="/api",
            tags=["eudr-supplier-risk-scorer"],
        )
        logger.info("EUDR Supplier Risk Scorer router included (AGENT-EUDR-017)")
    else:
        logger.debug("EUDR Supplier Risk Scorer router not available; skipping")

    # EUDR Commodity Risk Analyzer router (AGENT-EUDR-018)
    if EUDR_COMMODITY_RISK_ANALYZER_AVAILABLE and _eudr_cra_router is not None:
        app.include_router(_eudr_cra_router, prefix="/api", tags=["eudr-commodity-risk-analyzer"])
        logger.info("EUDR Commodity Risk Analyzer router included (AGENT-EUDR-018)")
    else:
        logger.debug("EUDR Commodity Risk Analyzer router not available; skipping")

    # EUDR Corruption Index Monitor router (AGENT-EUDR-019)
    if EUDR_CORRUPTION_INDEX_MONITOR_AVAILABLE and _eudr_cim_router is not None:
        app.include_router(_eudr_cim_router, prefix="/api", tags=["eudr-corruption-index-monitor"])
        logger.info("EUDR Corruption Index Monitor router included (AGENT-EUDR-019)")
    else:
        logger.debug("EUDR Corruption Index Monitor router not available; skipping")

    # EUDR Deforestation Alert System router (AGENT-EUDR-020)
    if EUDR_DEFORESTATION_ALERT_SYSTEM_AVAILABLE and _eudr_das_router is not None:
        app.include_router(_eudr_das_router, prefix="/api", tags=["eudr-deforestation-alert-system"])
        logger.info("EUDR Deforestation Alert System router included (AGENT-EUDR-020)")
    else:
        logger.debug("EUDR Deforestation Alert System router not available; skipping")

    # EUDR Indigenous Rights Checker router (AGENT-EUDR-021)
    if EUDR_INDIGENOUS_RIGHTS_CHECKER_AVAILABLE and _eudr_irc_router is not None:
        app.include_router(
            _eudr_irc_router,
            prefix="/api",
            tags=["eudr-indigenous-rights-checker"],
        )
        logger.info("EUDR Indigenous Rights Checker router included (AGENT-EUDR-021)")
    else:
        logger.debug("EUDR Indigenous Rights Checker router not available; skipping")

    # EUDR Protected Area Validator router (AGENT-EUDR-022)
    if EUDR_PROTECTED_AREA_VALIDATOR_AVAILABLE and _eudr_pav_router is not None:
        app.include_router(
            _eudr_pav_router,
            prefix="/api",
            tags=["eudr-protected-area-validator"],
        )
        logger.info("EUDR Protected Area Validator router included (AGENT-EUDR-022)")
    else:
        logger.debug("EUDR Protected Area Validator router not available; skipping")

    # EUDR Legal Compliance Verifier router (AGENT-EUDR-023)
    if EUDR_LEGAL_COMPLIANCE_VERIFIER_AVAILABLE and _eudr_lcv_router is not None:
        app.include_router(
            _eudr_lcv_router,
            prefix="/api",
            tags=["eudr-legal-compliance-verifier"],
        )
        logger.info("EUDR Legal Compliance Verifier router included (AGENT-EUDR-023)")
    else:
        logger.debug("EUDR Legal Compliance Verifier router not available; skipping")

    # EUDR Third-Party Audit Manager router (AGENT-EUDR-024)
    if EUDR_THIRD_PARTY_AUDIT_MANAGER_AVAILABLE and _eudr_tam_router is not None:
        app.include_router(
            _eudr_tam_router,
            prefix="/api",
            tags=["eudr-third-party-audit-manager"],
        )
        logger.info("EUDR Third-Party Audit Manager router included (AGENT-EUDR-024)")
    else:
        logger.debug("EUDR Third-Party Audit Manager router not available; skipping")

    # EUDR Risk Mitigation Advisor router (AGENT-EUDR-025)
    if EUDR_RISK_MITIGATION_ADVISOR_AVAILABLE and _eudr_rma_router is not None:
        app.include_router(
            _eudr_rma_router,
            prefix="/api",
            tags=["eudr-risk-mitigation-advisor"],
        )
        logger.info("EUDR Risk Mitigation Advisor router included (AGENT-EUDR-025)")
    else:
        logger.debug("EUDR Risk Mitigation Advisor router not available; skipping")

    # EUDR Information Gathering Agent router (AGENT-EUDR-027)
    if EUDR_INFORMATION_GATHERING_AVAILABLE and _eudr_iga_router is not None:
        app.include_router(
            _eudr_iga_router,
            prefix="/api",
            tags=["eudr-information-gathering"],
        )
        logger.info("EUDR Information Gathering Agent router included (AGENT-EUDR-027)")
    else:
        logger.debug("EUDR Information Gathering Agent router not available; skipping")

    # EUDR Risk Assessment Engine router (AGENT-EUDR-028)
    if EUDR_RISK_ASSESSMENT_ENGINE_AVAILABLE and _eudr_rae_router is not None:
        app.include_router(
            _eudr_rae_router,
            prefix="/api",
            tags=["eudr-risk-assessment-engine"],
        )
        logger.info("EUDR Risk Assessment Engine router included (AGENT-EUDR-028)")
    else:
        logger.debug("EUDR Risk Assessment Engine router not available; skipping")

    # AGENT-EUDR-029: Mitigation Measure Designer
    if _EUDR_MMD_AVAILABLE:
        try:
            app.include_router(eudr_mmd_api.get_router())
            logger.info("EUDR-029 Mitigation Measure Designer router registered")
        except Exception as e:
            logger.warning(f"Failed to register EUDR-029 router: {e}")

    # AGENT-EUDR-030: Documentation Generator
    if _EUDR_DGN_AVAILABLE:
        try:
            app.include_router(eudr_dgn_api.get_router())
            logger.info("EUDR-030 Documentation Generator router registered")
        except Exception as e:
            logger.warning(f"Failed to register EUDR-030 router: {e}")

    # AGENT-EUDR-031: Stakeholder Engagement Tool
    if _EUDR_SET_AVAILABLE:
        try:
            app.include_router(eudr_set_api.get_router())
            logger.info("EUDR-031 Stakeholder Engagement Tool router registered")
        except Exception as e:
            logger.warning(f"Failed to register EUDR-031 router: {e}")

    # AGENT-EUDR-032: Grievance Mechanism Manager
    if _EUDR_GMM_AVAILABLE:
        try:
            app.include_router(eudr_gmm_api.get_router())
            logger.info("EUDR-032 Grievance Mechanism Manager router registered")
        except Exception as e:
            logger.warning(f"Failed to register EUDR-032 router: {e}")

    # AGENT-EUDR-033: Continuous Monitoring Agent
    if _EUDR_CM_AVAILABLE:
        try:
            app.include_router(eudr_cm_api.get_router())
            logger.info("EUDR-033 Continuous Monitoring Agent router registered")
        except Exception as e:
            logger.warning(f"Failed to register EUDR-033 router: {e}")

    # AGENT-EUDR-034: Annual Review Scheduler
    if _EUDR_ARS_AVAILABLE:
        try:
            app.include_router(eudr_ars_api.get_router())
            logger.info("EUDR-034 Annual Review Scheduler router registered")
        except Exception as e:
            logger.warning(f"Failed to register EUDR-034 router: {e}")

    # AGENT-EUDR-035: Improvement Plan Creator
    if _EUDR_IPC_AVAILABLE:
        try:
            app.include_router(eudr_ipc_api.get_router())
            logger.info("EUDR-035 Improvement Plan Creator router registered")
        except Exception as e:
            logger.warning(f"Failed to register EUDR-035 router: {e}")

    # AGENT-EUDR-036: EU Information System Interface
    if _EUDR_EUIS_AVAILABLE:
        try:
            app.include_router(eudr_euis_api.get_router())
            logger.info("EUDR-036 EU Information System Interface router registered")
        except Exception as e:
            logger.warning(f"Failed to register EUDR-036 router: {e}")

    # AGENT-EUDR-037: Due Diligence Statement Creator
    if _EUDR_DDSC_AVAILABLE:
        try:
            app.include_router(eudr_ddsc_api.get_router())
            logger.info("EUDR-037 Due Diligence Statement Creator router registered")
        except Exception as e:
            logger.warning(f"Failed to register EUDR-037 router: {e}")

    # AGENT-EUDR-038: Reference Number Generator
    if _EUDR_RNG_AVAILABLE:
        try:
            app.include_router(eudr_rng_api.get_router())
            logger.info("EUDR-038 Reference Number Generator router registered")
        except Exception as e:
            logger.warning(f"Failed to register EUDR-038 router: {e}")

    # AGENT-EUDR-039: Customs Declaration Support
    if _EUDR_CDS_AVAILABLE:
        try:
            app.include_router(eudr_cds_api.get_router())
            logger.info("EUDR-039 Customs Declaration Support router registered")
        except Exception as e:
            logger.warning(f"Failed to register EUDR-039 router: {e}")

    # AGENT-EUDR-040: Authority Communication Manager
    if _EUDR_ACM_AVAILABLE:
        try:
            app.include_router(eudr_acm_api.get_router())
            logger.info("EUDR-040 Authority Communication Manager router registered")
        except Exception as e:
            logger.warning(f"Failed to register EUDR-040 router: {e}")


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
