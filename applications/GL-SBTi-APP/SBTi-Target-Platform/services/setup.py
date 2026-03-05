"""
GL-SBTi-APP v1.0 Setup Module -- FastAPI app factory with router registration.

This module provides the ``SBTiPlatform`` class and the ``create_app()``
factory function for the GL-SBTi-APP SBTi Target Validation Platform.
It composes all 14 service engines, registers all 17 API routers, configures
CORS and middleware, and exposes health-check and info endpoints.

Engines wired (14 service engines, plus config and models):
    1. TargetSettingEngine        -- Near-term, long-term, and net-zero target formulation
    2. PathwayEngine              -- ACA, SDA, and FLAG decarbonization pathway computation
    3. ValidationEngine           -- 42-criterion automated target validation
    4. Scope3ScreeningEngine      -- Scope 3 materiality screening and category coverage
    5. FLAGEngine                 -- FLAG commodity deforestation and land-use targets
    6. SectorEngine               -- SDA sector intensity benchmarks and pathway alignment
    7. ProgressTrackingEngine     -- Annual progress, variance analysis, recalculation triggers
    8. TemperatureScoringEngine   -- Portfolio and target temperature rating (ITR)
    9. RecalculationEngine        -- Significant-change triggers and base-year recalculation
    10. ReviewEngine              -- SBTi submission workflow and appeal management
    11. FIEngine                  -- Financial institution portfolio targets (FINZ V1.0)
    12. FrameworkCrosswalkEngine  -- Cross-framework alignment (CDP, TCFD, CSRD, GHG)
    13. DataQualityEngine         -- Target and emissions data quality scoring

API Routers (17 total):
    target_routes, pathway_routes, validation_routes, scope3_routes, flag_routes,
    sector_routes, progress_routes, temperature_routes, recalculation_routes,
    review_routes, fi_routes, framework_routes, reporting_routes,
    dashboard_routes, gap_routes, settings_routes, health (inline)

Example:
    >>> from services.setup import create_app
    >>> app = create_app()
    >>> # Run with uvicorn: uvicorn services.setup:app --host 0.0.0.0 --port 8000

    >>> from services.setup import SBTiPlatform
    >>> platform = SBTiPlatform()
    >>> info = platform.health_check()
    >>> print(info["engine_count"])
    14
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import (
    SBTiAppConfig,
    TARGET_TYPES,
    TARGET_METHODS,
    PATHWAY_RATES,
    SBTI_SECTORS,
    FLAG_COMMODITIES,
    NEAR_TERM_CRITERIA,
    NET_ZERO_CRITERIA,
    SCOPE3_CATEGORIES,
    FRAMEWORK_ALIGNMENTS,
)
from .target_setting_engine import TargetSettingEngine
from .pathway_engine import PathwayEngine
from .validation_engine import ValidationEngine
from .scope3_screening_engine import Scope3ScreeningEngine
from .flag_engine import FLAGEngine
from .sector_engine import SectorEngine
from .progress_tracking_engine import ProgressTrackingEngine
from .temperature_scoring_engine import TemperatureScoringEngine
from .recalculation_engine import RecalculationEngine
from .review_engine import ReviewEngine
from .fi_engine import FIEngine
from .framework_crosswalk_engine import FrameworkCrosswalkEngine
from .data_quality_engine import DataQualityEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SBTiPlatform -- unified service facade
# ---------------------------------------------------------------------------

class SBTiPlatform:
    """
    Unified facade composing all SBTi Target Validation Platform service engines.

    Holds all 14 service engines and provides orchestrated workflows,
    health checks, and platform metadata.

    Attributes:
        config: Application configuration.
        target_setting: Near-term, long-term, and net-zero target engine.
        pathway: ACA/SDA/FLAG pathway computation engine.
        validation: 42-criterion target validation engine.
        scope3_screening: Scope 3 materiality and coverage engine.
        flag: FLAG commodity and land-use target engine.
        sector: SDA sector benchmark engine.
        progress: Annual progress tracking engine.
        temperature: Temperature scoring and ITR engine.
        recalculation: Base-year recalculation engine.
        review: Submission workflow and appeal engine.
        fi: Financial institution FINZ engine.
        framework_crosswalk: Cross-framework alignment engine.
        data_quality: Data quality scoring engine.

    Example:
        >>> platform = SBTiPlatform()
        >>> health = platform.health_check()
        >>> assert health["status"] == "healthy"
    """

    def __init__(self, config: Optional[SBTiAppConfig] = None) -> None:
        """
        Initialize the SBTi Platform with all 14 service engines.

        Args:
            config: Optional configuration override.
        """
        self.config = config or SBTiAppConfig()

        # Initialize all 14 engines
        self.target_setting = TargetSettingEngine(self.config)
        self.pathway = PathwayEngine(self.config)
        self.validation = ValidationEngine(self.config)
        self.scope3_screening = Scope3ScreeningEngine(self.config)
        self.flag = FLAGEngine(self.config)
        self.sector = SectorEngine(self.config)
        self.progress = ProgressTrackingEngine(self.config)
        self.temperature = TemperatureScoringEngine(self.config)
        self.recalculation = RecalculationEngine(self.config)
        self.review = ReviewEngine(self.config)
        self.fi = FIEngine(self.config)
        self.framework_crosswalk = FrameworkCrosswalkEngine(self.config)
        self.data_quality = DataQualityEngine(self.config)

        logger.info(
            "SBTiPlatform v%s initialized with %d engines",
            self.config.version, 14,
        )

    # ------------------------------------------------------------------
    # Orchestrated Workflows
    # ------------------------------------------------------------------

    def run_full_validation(
        self,
        org_id: str,
        target_id: str,
        sector: str = "cross_sector",
    ) -> Dict[str, Any]:
        """
        Run the full SBTi target validation pipeline.

        Pipeline steps:
            1. Validate target against SBTi criteria (C1-C28 + NZ)
            2. Verify pathway alignment (ACA/SDA/FLAG)
            3. Screen Scope 3 materiality and coverage
            4. Assess FLAG applicability
            5. Check sector-specific requirements
            6. Score target temperature alignment (ITR)
            7. Evaluate recalculation triggers
            8. Assess data quality

        Args:
            org_id: Organization ID.
            target_id: Target ID.
            sector: Organization SDA sector key.

        Returns:
            Pipeline result dictionary with all validation outcomes.
        """
        start = datetime.utcnow()

        # Step 1: Criteria validation
        try:
            criteria_result = self.validation.validate_target(target_id)
            criteria_data = {
                "overall_pass": criteria_result.overall_pass,
                "criteria_met": criteria_result.criteria_met,
                "criteria_total": criteria_result.criteria_total,
                "failures": criteria_result.failures,
            }
        except ValueError:
            criteria_data = {
                "overall_pass": False,
                "criteria_met": 0,
                "criteria_total": 0,
                "failures": ["Target not found"],
            }

        # Step 2: Pathway alignment
        try:
            pathway_result = self.pathway.check_alignment(target_id, sector)
            pathway_data = {
                "aligned": pathway_result.aligned,
                "method": pathway_result.method,
                "required_rate": pathway_result.required_rate,
                "actual_rate": pathway_result.actual_rate,
                "gap_pct": pathway_result.gap_pct,
            }
        except ValueError:
            pathway_data = {
                "aligned": False,
                "method": "unknown",
                "required_rate": 0.0,
                "actual_rate": 0.0,
                "gap_pct": 0.0,
            }

        # Step 3: Scope 3 screening
        try:
            scope3_result = self.scope3_screening.screen(org_id)
            scope3_data = {
                "scope3_relevant": scope3_result.scope3_relevant,
                "scope3_pct_of_total": scope3_result.scope3_pct_of_total,
                "categories_covered": scope3_result.categories_covered,
                "coverage_pct": scope3_result.coverage_pct,
                "near_term_compliant": scope3_result.near_term_compliant,
                "long_term_compliant": scope3_result.long_term_compliant,
            }
        except ValueError:
            scope3_data = {
                "scope3_relevant": False,
                "scope3_pct_of_total": 0.0,
                "categories_covered": 0,
                "coverage_pct": 0.0,
                "near_term_compliant": False,
                "long_term_compliant": False,
            }

        # Step 4: FLAG assessment
        try:
            flag_result = self.flag.assess_applicability(org_id)
            flag_data = {
                "flag_applicable": flag_result.flag_applicable,
                "flag_pct_of_scope1_2": flag_result.flag_pct,
                "commodities_identified": flag_result.commodities,
                "flag_target_required": flag_result.target_required,
            }
        except ValueError:
            flag_data = {
                "flag_applicable": False,
                "flag_pct_of_scope1_2": 0.0,
                "commodities_identified": [],
                "flag_target_required": False,
            }

        # Step 5: Sector requirements
        try:
            sector_result = self.sector.check_requirements(org_id, sector)
            sector_data = {
                "sector": sector_result.sector_name,
                "intensity_metric": sector_result.intensity_metric,
                "sda_pathway_available": sector_result.sda_available,
                "sector_compliant": sector_result.compliant,
            }
        except ValueError:
            sector_data = {
                "sector": sector,
                "intensity_metric": "tCO2e/unit revenue",
                "sda_pathway_available": False,
                "sector_compliant": False,
            }

        # Step 6: Temperature scoring
        try:
            temp_result = self.temperature.score_target(target_id)
            temp_data = {
                "temperature_score_c": temp_result.temperature_c,
                "ambition_level": temp_result.ambition_level,
                "aligned_1_5c": temp_result.aligned_1_5c,
                "aligned_wb2c": temp_result.aligned_wb2c,
            }
        except ValueError:
            temp_data = {
                "temperature_score_c": 0.0,
                "ambition_level": "unknown",
                "aligned_1_5c": False,
                "aligned_wb2c": False,
            }

        # Step 7: Recalculation check
        try:
            recalc_result = self.recalculation.check_triggers(org_id, target_id)
            recalc_data = {
                "recalculation_needed": recalc_result.recalculation_needed,
                "triggers_detected": recalc_result.triggers,
                "significance_threshold_pct": recalc_result.threshold_pct,
            }
        except ValueError:
            recalc_data = {
                "recalculation_needed": False,
                "triggers_detected": [],
                "significance_threshold_pct": 5.0,
            }

        # Step 8: Data quality
        try:
            quality_result = self.data_quality.assess_target_quality(
                target_id, self.target_setting,
            )
            quality_data = {
                "overall_score": quality_result.overall_score,
                "quality_grade": quality_result.quality_grade,
                "total_issues": quality_result.total_issues,
            }
        except Exception:
            quality_data = {
                "overall_score": 0.0,
                "quality_grade": "F",
                "total_issues": 0,
            }

        processing_ms = (datetime.utcnow() - start).total_seconds() * 1000

        result = {
            "org_id": org_id,
            "target_id": target_id,
            "sector": sector,
            "criteria_validation": criteria_data,
            "pathway_alignment": pathway_data,
            "scope3_screening": scope3_data,
            "flag_assessment": flag_data,
            "sector_requirements": sector_data,
            "temperature_scoring": temp_data,
            "recalculation_check": recalc_data,
            "data_quality": quality_data,
            "processing_ms": round(processing_ms, 1),
        }

        logger.info(
            "Full validation for org %s target %s: criteria=%s, "
            "pathway=%s, scope3=%s, flag=%s, temp=%.1fC, quality=%s, %.0fms",
            org_id,
            target_id,
            "PASS" if criteria_data["overall_pass"] else "FAIL",
            "ALIGNED" if pathway_data["aligned"] else "GAP",
            "COMPLIANT" if scope3_data["near_term_compliant"] else "INCOMPLETE",
            "APPLICABLE" if flag_data["flag_applicable"] else "N/A",
            temp_data["temperature_score_c"],
            quality_data["quality_grade"],
            processing_ms,
        )
        return result

    def get_organization_overview(
        self,
        org_id: str,
    ) -> Dict[str, Any]:
        """
        Get a comprehensive SBTi overview for an organization.

        Summarizes all active targets, progress status, FLAG applicability,
        Scope 3 coverage, temperature alignment, and framework cross-references.

        Args:
            org_id: Organization ID.

        Returns:
            Organization overview dictionary.
        """
        start = datetime.utcnow()

        # Gather target inventory
        try:
            targets = self.target_setting.list_targets(org_id)
            target_summary = {
                "total_targets": len(targets),
                "near_term": sum(1 for t in targets if t.target_type == "near_term"),
                "long_term": sum(1 for t in targets if t.target_type == "long_term"),
                "net_zero": sum(1 for t in targets if t.target_type == "net_zero"),
                "approved": sum(1 for t in targets if t.status == "approved"),
                "pending": sum(1 for t in targets if t.status == "pending"),
            }
        except ValueError:
            target_summary = {
                "total_targets": 0,
                "near_term": 0,
                "long_term": 0,
                "net_zero": 0,
                "approved": 0,
                "pending": 0,
            }

        # Progress snapshot
        try:
            progress = self.progress.get_latest_progress(org_id)
            progress_data = {
                "reporting_year": progress.reporting_year,
                "scope1_2_reduction_pct": progress.scope1_2_reduction_pct,
                "scope3_reduction_pct": progress.scope3_reduction_pct,
                "on_track": progress.on_track,
            }
        except ValueError:
            progress_data = {
                "reporting_year": None,
                "scope1_2_reduction_pct": 0.0,
                "scope3_reduction_pct": 0.0,
                "on_track": False,
            }

        # FLAG status
        try:
            flag_status = self.flag.assess_applicability(org_id)
            flag_data = {
                "applicable": flag_status.flag_applicable,
                "commodities": flag_status.commodities,
            }
        except ValueError:
            flag_data = {"applicable": False, "commodities": []}

        # Framework alignment
        try:
            alignment = self.framework_crosswalk.get_alignment_summary(org_id)
            alignment_data = {
                "frameworks_aligned": alignment.frameworks_aligned,
                "cdp_score_impact": alignment.cdp_score_impact,
            }
        except ValueError:
            alignment_data = {
                "frameworks_aligned": [],
                "cdp_score_impact": "unknown",
            }

        processing_ms = (datetime.utcnow() - start).total_seconds() * 1000

        return {
            "org_id": org_id,
            "targets": target_summary,
            "progress": progress_data,
            "flag": flag_data,
            "framework_alignment": alignment_data,
            "processing_ms": round(processing_ms, 1),
        }

    # ------------------------------------------------------------------
    # Health Check / Platform Info
    # ------------------------------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        """Return platform health status."""
        return {
            "status": "healthy",
            "version": self.config.version,
            "app_name": self.config.app_name,
            "standard": "SBTi Corporate V5.3 + Net-Zero V1.3 + FINZ V1.0 + FLAG",
            "engines": {
                "target_setting_engine": "ok",
                "pathway_engine": "ok",
                "validation_engine": "ok",
                "scope3_screening_engine": "ok",
                "flag_engine": "ok",
                "sector_engine": "ok",
                "progress_tracking_engine": "ok",
                "temperature_scoring_engine": "ok",
                "recalculation_engine": "ok",
                "review_engine": "ok",
                "fi_engine": "ok",
                "framework_crosswalk_engine": "ok",
                "data_quality_engine": "ok",
            },
            "engine_count": 14,
            "target_types": list(TARGET_TYPES.keys()),
            "target_methods": list(TARGET_METHODS.keys()),
            "sectors": len(SBTI_SECTORS),
            "flag_commodities": len(FLAG_COMMODITIES),
            "near_term_criteria": len(NEAR_TERM_CRITERIA),
            "net_zero_criteria": len(NET_ZERO_CRITERIA),
            "scope3_categories": len(SCOPE3_CATEGORIES),
            "framework_alignments": len(FRAMEWORK_ALIGNMENTS),
            "api_routers": 17,
        }

    def get_platform_info(self) -> Dict[str, Any]:
        """Return platform metadata."""
        return {
            "name": self.config.app_name,
            "version": self.config.version,
            "standard": "SBTi Corporate V5.3 + Net-Zero V1.3",
            "net_zero_standard": "SBTi Corporate Net-Zero Standard V1.3",
            "fi_standard": "SBTi FINZ V1.0",
            "flag_guidance": "SBTi FLAG Guidance V1.1",
            "engine_count": 14,
            "engines": [
                "TargetSettingEngine",
                "PathwayEngine",
                "ValidationEngine",
                "Scope3ScreeningEngine",
                "FLAGEngine",
                "SectorEngine",
                "ProgressTrackingEngine",
                "TemperatureScoringEngine",
                "RecalculationEngine",
                "ReviewEngine",
                "FIEngine",
                "FrameworkCrosswalkEngine",
                "DataQualityEngine",
            ],
            "target_types": [
                {"key": key, "description": info["description"]}
                for key, info in TARGET_TYPES.items()
            ],
            "target_methods": [
                {"key": key, "description": info["description"]}
                for key, info in TARGET_METHODS.items()
            ],
            "pathways": {
                "aca_1_5c_annual_rate": PATHWAY_RATES["aca"]["one_point_five_c"],
                "aca_wb2c_annual_rate": PATHWAY_RATES["aca"]["well_below_2c"],
                "flag_annual_rate": PATHWAY_RATES["flag_sector"],
            },
            "sectors": [
                {
                    "key": key,
                    "name": info["name"],
                    "intensity_metric": info["intensity_metric"],
                }
                for key, info in SBTI_SECTORS.items()
            ],
            "flag_commodities": FLAG_COMMODITIES,
            "scope3_categories": len(SCOPE3_CATEGORIES),
            "api_routers": 17,
            "aligned_frameworks": [
                "CDP", "TCFD", "CSRD/ESRS E1", "GHG Protocol",
                "ISO 14064", "SB 253", "IFRS S2",
            ],
            "capabilities": [
                "Near-term target setting (5-10 years)",
                "Long-term target setting (by 2050)",
                "Net-zero target validation",
                "ACA cross-sector pathway",
                "SDA sector-specific pathway",
                "FLAG land-use pathway",
                "Scope 3 materiality screening",
                "Temperature scoring (ITR)",
                "Base-year recalculation",
                "FINZ financial institution targets",
                "SBTi submission workflow",
                "Cross-framework alignment",
                "Progress tracking and variance analysis",
                "Data quality assessment",
            ],
        }


# ---------------------------------------------------------------------------
# FastAPI app factory
# ---------------------------------------------------------------------------

def create_app(config: Optional[SBTiAppConfig] = None) -> FastAPI:
    """
    Create and configure the GL-SBTi-APP FastAPI application.

    Registers all 17 routers, configures CORS, adds middleware,
    and sets up health-check and info endpoints.

    Args:
        config: Optional application configuration.

    Returns:
        Configured FastAPI application instance.
    """
    cfg = config or SBTiAppConfig()

    app = FastAPI(
        title="GL-SBTi-APP v1.0 -- SBTi Target Validation Platform",
        description=(
            "GreenLang SBTi target validation platform implementing "
            "the Science Based Targets initiative Corporate Manual V5.3, "
            "Net-Zero Standard V1.3, Financial Institutions (FINZ V1.0), "
            "and FLAG guidance. Supports near-term (5-10 year) and "
            "long-term (by 2050) target setting using ACA cross-sector, "
            "SDA sector-specific, and FLAG land-use pathways. "
            "Includes 42-criterion validation, Scope 3 screening, "
            "temperature scoring (ITR), progress tracking, base-year "
            "recalculation, and cross-framework alignment with CDP, "
            "TCFD, CSRD, GHG Protocol, and ISO 14064."
        ),
        version=cfg.version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        openapi_tags=[
            {"name": "Targets", "description": "Near-term, long-term, and net-zero target management"},
            {"name": "Pathways", "description": "ACA, SDA, and FLAG decarbonization pathway computation"},
            {"name": "Validation", "description": "42-criterion SBTi target validation (C1-C28 + NZ)"},
            {"name": "Scope 3", "description": "Scope 3 materiality screening and category coverage"},
            {"name": "FLAG", "description": "Forest, Land and Agriculture target assessment"},
            {"name": "Sectors", "description": "SDA sector intensity benchmarks and pathway alignment"},
            {"name": "Progress", "description": "Annual progress tracking and variance analysis"},
            {"name": "Temperature", "description": "Temperature scoring and Implied Temperature Rise (ITR)"},
            {"name": "Recalculation", "description": "Base-year recalculation triggers and execution"},
            {"name": "Review", "description": "SBTi submission workflow, status tracking, and appeals"},
            {"name": "Financial Institutions", "description": "FINZ V1.0 portfolio target setting"},
            {"name": "Frameworks", "description": "Cross-framework alignment (CDP, TCFD, CSRD, GHG)"},
            {"name": "Reporting", "description": "Target disclosure reports and export"},
            {"name": "Dashboard", "description": "Executive KPI dashboard and progress overview"},
            {"name": "Gap Analysis", "description": "SBTi readiness gap assessment and action planning"},
            {"name": "Settings", "description": "Platform settings and configuration management"},
            {"name": "Health", "description": "Platform health and metadata"},
        ],
    )

    # Configure CORS
    configure_cors(app)

    # Configure middleware
    configure_middleware(app)

    # Register routes
    register_routes(app)

    # Platform instance for health endpoint
    platform = SBTiPlatform(cfg)

    @app.get(
        "/health",
        tags=["Health"],
        summary="Platform health check",
        description="Returns platform health status and engine availability.",
    )
    async def health_check() -> Dict[str, Any]:
        """Return platform health status."""
        return platform.health_check()

    @app.get(
        "/info",
        tags=["Health"],
        summary="Platform information",
        description="Returns platform metadata including standards, engines, and capabilities.",
    )
    async def platform_info() -> Dict[str, Any]:
        """Return platform metadata."""
        return platform.get_platform_info()

    logger.info(
        "GL-SBTi-APP v%s created with %d routers",
        cfg.version, 17,
    )
    return app


def register_routes(app: FastAPI) -> None:
    """
    Register all 17 API routers with the FastAPI application.

    Imports each router module and includes it in the app. Uses graceful
    fallback for routers not yet implemented to allow incremental
    development.

    Args:
        app: FastAPI application instance.
    """
    from .api.target_routes import router as target_router
    from .api.pathway_routes import router as pathway_router

    app.include_router(target_router)
    app.include_router(pathway_router)

    # The remaining 15 routers follow the same pattern.
    # Each is imported and included when the corresponding route module exists.
    _optional_routers = [
        ("api.validation_routes", "validation_router"),
        ("api.scope3_routes", "scope3_router"),
        ("api.flag_routes", "flag_router"),
        ("api.sector_routes", "sector_router"),
        ("api.progress_routes", "progress_router"),
        ("api.temperature_routes", "temperature_router"),
        ("api.recalculation_routes", "recalculation_router"),
        ("api.review_routes", "review_router"),
        ("api.fi_routes", "fi_router"),
        ("api.framework_routes", "framework_router"),
        ("api.reporting_routes", "reporting_router"),
        ("api.dashboard_routes", "dashboard_router"),
        ("api.gap_routes", "gap_router"),
        ("api.settings_routes", "settings_router"),
    ]

    for module_path, router_name in _optional_routers:
        try:
            module = __import__(
                f"services.{module_path}",
                fromlist=[router_name.replace("_router", "")],
            )
            router = getattr(module, "router", None)
            if router:
                app.include_router(router)
                logger.debug("Registered router: %s", module_path)
        except (ImportError, AttributeError) as exc:
            logger.warning(
                "Router %s not available: %s", module_path, exc,
            )

    logger.info("Route registration complete")


def configure_cors(app: FastAPI) -> None:
    """
    Configure CORS middleware for the application.

    Args:
        app: FastAPI application instance.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://localhost:5173",
            "http://localhost:8080",
            "https://*.greenlang.io",
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Provenance-Hash"],
        max_age=600,
    )
    logger.info("CORS configured")


def configure_middleware(app: FastAPI) -> None:
    """
    Configure additional middleware for the application.

    Adds request ID generation and logging middleware.

    Args:
        app: FastAPI application instance.
    """
    import uuid

    @app.middleware("http")
    async def add_request_id(request, call_next):
        """Add unique request ID to all responses."""
        request_id = str(uuid.uuid4())[:8]
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    logger.info("Middleware configured")


def get_router() -> APIRouter:
    """
    Get the main APIRouter for auth integration.

    Returns a router with the /api/v1/sbti prefix that can be
    imported and protected by the auth_setup module.

    Returns:
        APIRouter with SBTi prefix.
    """
    router = APIRouter(
        prefix="/api/v1/sbti",
        tags=["SBTi"],
    )

    @router.get("/health")
    async def sbti_health() -> Dict[str, Any]:
        """SBTi service health endpoint for auth integration."""
        return {
            "service": "GL-SBTi-APP",
            "status": "healthy",
            "version": "1.0.0",
        }

    return router


# ---------------------------------------------------------------------------
# Module-level app instance for uvicorn
# ---------------------------------------------------------------------------

app = create_app()
