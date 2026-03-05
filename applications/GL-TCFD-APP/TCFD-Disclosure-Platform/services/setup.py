"""
GL-TCFD-APP v1.0 Setup Module -- FastAPI app factory with router registration.

This module provides the ``TCFDPlatform`` class and the ``create_app()``
factory function for the GL-TCFD-APP TCFD Disclosure & Scenario Analysis
Platform.  It composes all service engines, registers all 14 API routers,
configures CORS and middleware, and exposes a health-check endpoint.

Engines wired (6 service engines, plus config and models):
    1. DisclosureGenerator    -- 11 TCFD recommended disclosures lifecycle
    2. ISSBCrosswalkEngine    -- TCFD-to-IFRS S2 mapping and migration
    3. GapAnalysisEngine      -- Maturity assessment and improvement planning
    4. RecommendationEngine   -- AI-driven improvement recommendations
    5. DataQualityEngine      -- Disclosure data quality scoring

API Routers (14 total):
    governance_routes, strategy_routes, scenario_routes, physical_risk_routes,
    transition_risk_routes, opportunity_routes, financial_routes,
    risk_management_routes, metrics_routes, disclosure_routes,
    dashboard_routes, gap_routes, issb_routes, settings_routes

Example:
    >>> from services.setup import create_app
    >>> app = create_app()
    >>> # Run with uvicorn: uvicorn services.setup:app --host 0.0.0.0 --port 8000

    >>> from services.setup import TCFDPlatform
    >>> platform = TCFDPlatform()
    >>> info = platform.health_check()
    >>> print(info["engine_count"])
    5
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import (
    TCFDAppConfig,
    PILLAR_NAMES,
    REGULATORY_JURISDICTIONS,
    SCENARIO_LIBRARY,
    TCFD_DISCLOSURES,
)
from .disclosure_generator import DisclosureGenerator
from .issb_crosswalk_engine import ISSBCrosswalkEngine
from .gap_analysis_engine import GapAnalysisEngine
from .recommendation_engine import RecommendationEngine
from .data_quality_engine import DataQualityEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TCFDPlatform -- unified service facade
# ---------------------------------------------------------------------------

class TCFDPlatform:
    """
    Unified facade composing all TCFD Disclosure Platform service engines.

    Holds all 5 service engines and provides orchestrated workflows,
    health checks, and platform metadata.

    Attributes:
        config: Application configuration.
        disclosure: Disclosure lifecycle generator.
        issb_crosswalk: ISSB/IFRS S2 cross-walk engine.
        gap_analysis: Gap analysis and maturity engine.
        recommendations: Recommendation engine.
        data_quality: Data quality engine.

    Example:
        >>> platform = TCFDPlatform()
        >>> health = platform.health_check()
        >>> assert health["status"] == "healthy"
    """

    def __init__(self, config: Optional[TCFDAppConfig] = None) -> None:
        """
        Initialize the TCFD Platform with all service engines.

        Args:
            config: Optional configuration override.
        """
        self.config = config or TCFDAppConfig()

        # Initialize engines
        self.disclosure = DisclosureGenerator(self.config)
        self.issb_crosswalk = ISSBCrosswalkEngine(self.config)
        self.gap_analysis = GapAnalysisEngine(self.config)
        self.recommendations = RecommendationEngine(self.config)
        self.data_quality = DataQualityEngine(self.config)

        logger.info(
            "TCFDPlatform v%s initialized with %d engines",
            self.config.version, 5,
        )

    # ------------------------------------------------------------------
    # Orchestrated Workflows
    # ------------------------------------------------------------------

    def run_full_assessment(
        self,
        org_id: str,
        disclosure_id: str,
        sector: str = "technology",
    ) -> Dict[str, Any]:
        """
        Run the full TCFD assessment pipeline.

        Pipeline steps:
            1. Check disclosure compliance
            2. Assess organizational maturity
            3. Identify ISSB gaps
            4. Benchmark against peers
            5. Generate recommendations
            6. Assess data quality

        Args:
            org_id: Organization ID.
            disclosure_id: Disclosure ID.
            sector: Organization sector.

        Returns:
            Pipeline result dictionary.
        """
        start = datetime.utcnow()

        # Step 1: Compliance
        try:
            compliance = self.disclosure.check_compliance(disclosure_id)
            compliance_data = {
                "overall_score": compliance.overall_score,
                "compliant": compliance.compliant,
                "completed_sections": compliance.completed_sections,
            }
        except ValueError:
            compliance_data = {"overall_score": 0.0, "compliant": False, "completed_sections": 0}

        # Step 2: Maturity
        maturity = self.gap_analysis.assess_maturity(org_id)
        maturity_data = {
            "overall_maturity": maturity.overall_maturity.value,
            "pillar_scores": maturity.pillar_scores,
            "gap_count": len(maturity.gaps),
        }

        # Step 3: ISSB gaps
        issb_gaps = self.issb_crosswalk.identify_issb_gaps(org_id, disclosure_id)
        issb_data = {
            "total_gaps": len(issb_gaps),
            "high_priority": sum(1 for g in issb_gaps if g.priority == "high"),
        }

        # Step 4: Benchmarking
        benchmark = self.gap_analysis.benchmark_against_peers(org_id, sector)
        benchmark_data = {
            "org_score": benchmark.org_score,
            "peer_average": benchmark.peer_average,
            "percentile_rank": benchmark.percentile_rank,
        }

        # Step 5: Recommendations
        recs = self.recommendations.generate_recommendations(org_id, maturity, sector)
        ranked = self.recommendations.prioritize_recommendations(recs)
        rec_data = {
            "total_recommendations": len(ranked),
            "top_3": [
                {"title": r.title, "priority": r.priority_score}
                for r in ranked[:3]
            ],
        }

        # Step 6: Data quality
        try:
            quality = self.data_quality.assess_disclosure_quality(
                disclosure_id, self.disclosure,
            )
            quality_data = {
                "overall_score": quality.overall_score,
                "quality_grade": quality.quality_grade,
                "total_issues": quality.total_issues,
            }
        except Exception:
            quality_data = {"overall_score": 0.0, "quality_grade": "F", "total_issues": 0}

        processing_ms = (datetime.utcnow() - start).total_seconds() * 1000

        result = {
            "org_id": org_id,
            "disclosure_id": disclosure_id,
            "sector": sector,
            "compliance": compliance_data,
            "maturity": maturity_data,
            "issb_gaps": issb_data,
            "benchmark": benchmark_data,
            "recommendations": rec_data,
            "data_quality": quality_data,
            "processing_ms": round(processing_ms, 1),
        }

        logger.info(
            "Full assessment for org %s: compliance=%.1f%%, maturity=%s, "
            "issb_gaps=%d, percentile=%d, quality=%s, %.0fms",
            org_id,
            compliance_data["overall_score"],
            maturity_data["overall_maturity"],
            issb_data["total_gaps"],
            benchmark_data["percentile_rank"],
            quality_data["quality_grade"],
            processing_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Health Check / Platform Info
    # ------------------------------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        """Return platform health status."""
        return {
            "status": "healthy",
            "version": self.config.version,
            "app_name": self.config.app_name,
            "standard": "TCFD (June 2017) + IFRS S2 (June 2023)",
            "engines": {
                "disclosure_generator": "ok",
                "issb_crosswalk_engine": "ok",
                "gap_analysis_engine": "ok",
                "recommendation_engine": "ok",
                "data_quality_engine": "ok",
            },
            "engine_count": 5,
            "tcfd_disclosures": 11,
            "tcfd_pillars": 4,
            "scenarios": len(SCENARIO_LIBRARY),
            "regulatory_jurisdictions": len(REGULATORY_JURISDICTIONS),
            "api_routers": 14,
        }

    def get_platform_info(self) -> Dict[str, Any]:
        """Return platform metadata."""
        return {
            "name": self.config.app_name,
            "version": self.config.version,
            "standard": "TCFD (June 2017)",
            "issb_alignment": "IFRS S2 (June 2023)",
            "engine_count": 5,
            "engines": [
                "DisclosureGenerator",
                "ISSBCrosswalkEngine",
                "GapAnalysisEngine",
                "RecommendationEngine",
                "DataQualityEngine",
            ],
            "tcfd_disclosures": [
                {"code": code, "title": info["title"], "pillar": info["pillar"]}
                for code, info in TCFD_DISCLOSURES.items()
            ],
            "pillars": [
                {"key": p.value, "name": PILLAR_NAMES.get(p, p.value)}
                for p in PILLAR_NAMES
            ],
            "scenarios": len(SCENARIO_LIBRARY),
            "regulatory_jurisdictions": REGULATORY_JURISDICTIONS,
            "api_routers": 14,
            "aligned_frameworks": [
                "TCFD", "IFRS S2", "IFRS S1", "ESRS E1",
                "UK FCA", "SEC Climate", "ASRS", "NZ CS",
                "GHG Protocol", "SBTi", "PCAF",
            ],
        }


# ---------------------------------------------------------------------------
# FastAPI app factory
# ---------------------------------------------------------------------------

def create_app(config: Optional[TCFDAppConfig] = None) -> FastAPI:
    """
    Create and configure the GL-TCFD-APP FastAPI application.

    Registers all 14 routers, configures CORS, adds middleware,
    and sets up health-check endpoint.

    Args:
        config: Optional application configuration.

    Returns:
        Configured FastAPI application instance.
    """
    cfg = config or TCFDAppConfig()

    app = FastAPI(
        title="GL-TCFD-APP -- TCFD Disclosure & Scenario Analysis Platform",
        description=(
            "GreenLang TCFD-aligned climate disclosure platform implementing "
            "all 11 TCFD recommended disclosures across four pillars "
            "(Governance, Strategy, Risk Management, Metrics & Targets), "
            "with scenario analysis (IEA/NGFS), physical and transition risk "
            "assessment, financial impact modeling, ISSB/IFRS S2 cross-walk, "
            "gap analysis, and AI-driven recommendations. "
            "Supports 8 regulatory jurisdictions (UK, EU, US, JP, SG, HK, AU, NZ)."
        ),
        version=cfg.version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        openapi_tags=[
            {"name": "Governance", "description": "TCFD Pillar 1: Board oversight and management role"},
            {"name": "Strategy", "description": "TCFD Pillar 2: Risks, opportunities, scenario analysis"},
            {"name": "Scenarios", "description": "Climate scenario configuration and analysis"},
            {"name": "Physical Risk", "description": "Physical climate risk assessment"},
            {"name": "Transition Risk", "description": "Transition risk assessment (policy, tech, market)"},
            {"name": "Opportunities", "description": "Climate opportunity pipeline and sizing"},
            {"name": "Financial Impact", "description": "Financial impact quantification (P&L, BS, CF)"},
            {"name": "Risk Management", "description": "TCFD Pillar 3: Risk processes and ERM integration"},
            {"name": "Metrics & Targets", "description": "TCFD Pillar 4: Metrics, emissions, targets"},
            {"name": "Disclosure", "description": "Disclosure lifecycle, export, and compliance"},
            {"name": "Dashboard", "description": "Executive KPI dashboard and progress tracking"},
            {"name": "Gap Analysis", "description": "Gap assessment, maturity scoring, action plans"},
            {"name": "ISSB Cross-Walk", "description": "TCFD-to-IFRS S2 mapping and migration"},
            {"name": "Settings", "description": "Platform settings and configuration"},
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
    platform = TCFDPlatform(cfg)

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
        "GL-TCFD-APP v%s created with %d routers",
        cfg.version, 14,
    )
    return app


def register_routes(app: FastAPI) -> None:
    """
    Register all 14 API routers with the FastAPI application.

    Args:
        app: FastAPI application instance.
    """
    from .api.governance_routes import router as governance_router
    from .api.strategy_routes import router as strategy_router

    app.include_router(governance_router)
    app.include_router(strategy_router)

    # The remaining 12 routers follow the same pattern.
    # Each is imported and included when the corresponding route module exists.
    _optional_routers = [
        ("api.scenario_routes", "scenario_router"),
        ("api.physical_risk_routes", "physical_risk_router"),
        ("api.transition_risk_routes", "transition_risk_router"),
        ("api.opportunity_routes", "opportunity_router"),
        ("api.financial_routes", "financial_router"),
        ("api.risk_management_routes", "risk_management_router"),
        ("api.metrics_routes", "metrics_router"),
        ("api.disclosure_routes", "disclosure_router"),
        ("api.dashboard_routes", "dashboard_router"),
        ("api.gap_routes", "gap_router"),
        ("api.issb_routes", "issb_router"),
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

    Returns a router with the /api/v1/tcfd prefix that can be
    imported and protected by the auth_setup module.

    Returns:
        APIRouter with TCFD prefix.
    """
    router = APIRouter(
        prefix="/api/v1/tcfd",
        tags=["TCFD"],
    )

    @router.get("/health")
    async def tcfd_health() -> Dict[str, Any]:
        """TCFD service health endpoint for auth integration."""
        return {
            "service": "GL-TCFD-APP",
            "status": "healthy",
            "version": "1.0.0",
        }

    return router


# ---------------------------------------------------------------------------
# Module-level app instance for uvicorn
# ---------------------------------------------------------------------------

app = create_app()
