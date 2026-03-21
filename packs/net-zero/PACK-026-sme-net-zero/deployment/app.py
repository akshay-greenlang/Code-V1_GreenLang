#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PACK-026: SME Net Zero Pack - FastAPI Application Entrypoint
=============================================================

Production FastAPI application for the SME Net Zero Pack.
Provides REST API endpoints for all 8 engines, 6 workflows, and 8 templates.

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-026 SME Net Zero Pack
Status: Production Ready
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel

# Add pack to Python path
PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACK_DIR)

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "pack026_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)
REQUEST_DURATION = Histogram(
    "pack026_http_request_duration_seconds",
    "HTTP request duration",
    ["method", "endpoint"],
)
BASELINE_CALCULATIONS = Counter(
    "pack026_baseline_calculations_total",
    "Total baseline calculations",
    ["data_tier"],
)
QUICK_WINS_GENERATED = Counter(
    "pack026_quick_wins_generated_total",
    "Total quick wins generated",
)
GRANTS_MATCHED = Counter(
    "pack026_grants_matched_total",
    "Total grants matched",
    ["region"],
)

# Import pack components
try:
    from config import PackConfig, load_preset
    from engines import (
        SMEBaselineEngine,
        SimplifiedTargetEngine,
        QuickWinsEngine,
        Scope3EstimatorEngine,
        ActionPrioritizationEngine,
        CostBenefitEngine,
        GrantFinderEngine,
        CertificationReadinessEngine,
    )
    from workflows import (
        ExpressOnboardingWorkflow,
        StandardSetupWorkflow,
        GrantApplicationWorkflow,
        QuarterlyReviewWorkflow,
        QuickWinsImplementationWorkflow,
        CertificationPathwayWorkflow,
    )
    from templates import TemplateRegistry
    from integrations import (
        PackOrchestrator,
        HealthCheck,
    )

    logger.info("✓ All PACK-026 components imported successfully")
except ImportError as e:
    logger.error(f"✗ Failed to import PACK-026 components: {e}")
    raise


# Application state
class AppState:
    """Application state container."""

    def __init__(self):
        self.config: PackConfig = None
        self.health_check: HealthCheck = None
        self.template_registry: TemplateRegistry = None
        self.start_time: datetime = None


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("🚀 Starting PACK-026 SME Net Zero Pack...")
    app_state.start_time = datetime.utcnow()

    # Load configuration
    preset = os.getenv("PACK_PRESET", "small_business")
    app_state.config = load_preset(preset)
    logger.info(f"✓ Loaded preset: {preset}")

    # Initialize health check
    app_state.health_check = HealthCheck(config=app_state.config)
    logger.info("✓ Health check initialized")

    # Initialize template registry
    app_state.template_registry = TemplateRegistry()
    logger.info(f"✓ Template registry initialized ({app_state.template_registry.template_count} templates)")

    logger.info("✅ PACK-026 startup complete")

    yield

    # Shutdown
    logger.info("🛑 Shutting down PACK-026 SME Net Zero Pack...")
    logger.info("✅ PACK-026 shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="PACK-026: SME Net Zero Pack",
    description="Comprehensive net zero solution for Small and Medium Enterprises",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests and track metrics."""
    start_time = datetime.utcnow()

    response = await call_next(request)

    duration = (datetime.utcnow() - start_time).total_seconds()
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code,
    ).inc()
    REQUEST_DURATION.labels(
        method=request.method,
        endpoint=request.url.path,
    ).observe(duration)

    logger.info(
        f"{request.method} {request.url.path} - {response.status_code} - {duration:.3f}s"
    )

    return response


# Health endpoints
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    try:
        health_status = await app_state.health_check.check_all()
        return {
            "status": "healthy" if health_status["overall_health"] == "healthy" else "degraded",
            "pack_id": "PACK-026",
            "pack_name": "SME Net Zero Pack",
            "version": "1.0.0",
            "uptime_seconds": (datetime.utcnow() - app_state.start_time).total_seconds(),
            "timestamp": datetime.utcnow().isoformat(),
            "components": health_status,
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "error": str(e)},
        )


@app.get("/ready", tags=["Health"])
async def readiness_check():
    """Readiness check endpoint."""
    try:
        health_status = await app_state.health_check.check_all()
        if health_status["overall_health"] == "healthy":
            return {"status": "ready"}
        else:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "not_ready", "details": health_status},
            )
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "error": str(e)},
        )


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint."""
    from fastapi.responses import PlainTextResponse

    return PlainTextResponse(generate_latest())


# Engine endpoints
@app.post("/engines/baseline", tags=["Engines"])
async def calculate_baseline(request: Dict[str, Any]):
    """Calculate SME emissions baseline (Bronze/Silver/Gold tier)."""
    try:
        engine = SMEBaselineEngine()
        result = await engine.calculate(request)

        # Track metrics
        BASELINE_CALCULATIONS.labels(data_tier=request.get("data_tier", "BRONZE")).inc()

        return result
    except Exception as e:
        logger.error(f"Baseline calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/engines/target", tags=["Engines"])
async def set_target(request: Dict[str, Any]):
    """Set simplified SBTi-aligned net zero target."""
    try:
        engine = SimplifiedTargetEngine()
        result = await engine.calculate(request)
        return result
    except Exception as e:
        logger.error(f"Target setting failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/engines/quick-wins", tags=["Engines"])
async def identify_quick_wins(request: Dict[str, Any]):
    """Identify quick win decarbonization actions."""
    try:
        engine = QuickWinsEngine()
        result = await engine.calculate(request)

        # Track metrics
        QUICK_WINS_GENERATED.inc(len(result.get("quick_wins", [])))

        return result
    except Exception as e:
        logger.error(f"Quick wins identification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/engines/scope3", tags=["Engines"])
async def estimate_scope3(request: Dict[str, Any]):
    """Estimate Scope 3 emissions using spend-based method."""
    try:
        engine = Scope3EstimatorEngine()
        result = await engine.calculate(request)
        return result
    except Exception as e:
        logger.error(f"Scope 3 estimation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/engines/prioritize", tags=["Engines"])
async def prioritize_actions(request: Dict[str, Any]):
    """Prioritize actions using MACC-lite analysis."""
    try:
        engine = ActionPrioritizationEngine()
        result = await engine.calculate(request)
        return result
    except Exception as e:
        logger.error(f"Action prioritization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/engines/cost-benefit", tags=["Engines"])
async def analyze_cost_benefit(request: Dict[str, Any]):
    """Analyze cost-benefit of decarbonization portfolio."""
    try:
        engine = CostBenefitEngine()
        result = await engine.calculate(request)
        return result
    except Exception as e:
        logger.error(f"Cost-benefit analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/engines/grants", tags=["Engines"])
async def find_grants(request: Dict[str, Any]):
    """Find matching grants and funding opportunities."""
    try:
        engine = GrantFinderEngine()
        result = await engine.calculate(request)

        # Track metrics
        for grant in result.get("matched_grants", []):
            GRANTS_MATCHED.labels(region=grant.get("region", "unknown")).inc()

        return result
    except Exception as e:
        logger.error(f"Grant finding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/engines/certification", tags=["Engines"])
async def assess_certification_readiness(request: Dict[str, Any]):
    """Assess readiness for climate certifications."""
    try:
        engine = CertificationReadinessEngine()
        result = await engine.calculate(request)
        return result
    except Exception as e:
        logger.error(f"Certification assessment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Workflow endpoints
@app.post("/workflows/express", tags=["Workflows"])
async def express_onboarding(request: Dict[str, Any]):
    """Express onboarding workflow (15-20 minutes)."""
    try:
        workflow = ExpressOnboardingWorkflow()
        result = await workflow.execute(request)
        return result
    except Exception as e:
        logger.error(f"Express onboarding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/workflows/standard", tags=["Workflows"])
async def standard_setup(request: Dict[str, Any]):
    """Standard setup workflow (1-2 hours)."""
    try:
        workflow = StandardSetupWorkflow()
        result = await workflow.execute(request)
        return result
    except Exception as e:
        logger.error(f"Standard setup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Template endpoints
@app.get("/templates", tags=["Templates"])
async def list_templates():
    """List all available report templates."""
    return {
        "templates": app_state.template_registry.list_templates(),
        "count": app_state.template_registry.template_count,
    }


@app.post("/templates/{template_name}/render", tags=["Templates"])
async def render_template(template_name: str, request: Dict[str, Any]):
    """Render a report template."""
    try:
        format_type = request.pop("format", "markdown")
        result = app_state.template_registry.render(
            template_name=template_name,
            data=request,
            format=format_type,
        )
        return {"template": template_name, "format": format_type, "output": result}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Template '{template_name}' not found")
    except Exception as e:
        logger.error(f"Template rendering failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Root endpoint
@app.get("/", tags=["Info"])
async def root():
    """Pack information."""
    return {
        "pack_id": "PACK-026",
        "pack_name": "SME Net Zero Pack",
        "version": "1.0.0",
        "description": "Comprehensive net zero solution for Small and Medium Enterprises",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "engines": "/engines/*",
            "workflows": "/workflows/*",
            "templates": "/templates",
        },
        "uptime_seconds": (datetime.utcnow() - app_state.start_time).total_seconds(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        workers=int(os.getenv("MAX_WORKERS", "4")),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
