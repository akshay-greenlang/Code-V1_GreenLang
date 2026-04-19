#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PACK-027: Enterprise Net Zero Pack - FastAPI Application Entrypoint
====================================================================

Production FastAPI application for the Enterprise Net Zero Pack.
Provides REST API endpoints for all 12 engines, 10 workflows, and 12 templates.

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-027 Enterprise Net Zero Pack
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
    "pack027_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)
REQUEST_DURATION = Histogram(
    "pack027_http_request_duration_seconds",
    "HTTP request duration",
    ["method", "endpoint"],
)
BASELINE_CALCULATIONS = Counter(
    "pack027_baseline_calculations_total",
    "Total baseline calculations",
    ["consolidation_method"],
)
SBTI_TARGETS_SET = Counter(
    "pack027_sbti_targets_set_total",
    "Total SBTi targets set",
    ["target_type"],
)
SCENARIOS_MODELED = Counter(
    "pack027_scenarios_modeled_total",
    "Total scenarios modeled",
    ["pathway"],
)
ASSURANCE_CHECKS = Counter(
    "pack027_assurance_checks_total",
    "Total assurance readiness checks",
)

# Import pack components
try:
    from config import PackConfig, load_preset
    from engines import (
        EnterpriseBaselineEngine,
        SBTiTargetEngine,
        ScenarioModelingEngine,
        CarbonPricingEngine,
        Scope4AvoidedEmissionsEngine,
        SupplyChainMappingEngine,
        MultiEntityConsolidationEngine,
        FinancialIntegrationEngine,
        DataQualityGuardianEngine,
        RegulatoryComplianceEngine,
        AssuranceReadinessEngine,
        RiskAssessmentEngine,
    )
    from workflows import (
        ComprehensiveBaselineWorkflow,
        SBTiSubmissionWorkflow,
        AnnualInventoryWorkflow,
        ScenarioAnalysisWorkflow,
        SupplyChainEngagementWorkflow,
        InternalCarbonPricingWorkflow,
        MultiEntityRollupWorkflow,
        ExternalAssuranceWorkflow,
        BoardReportingWorkflow,
        RegulatoryFilingWorkflow,
    )
    from templates import TemplateRegistry
    from integrations import (
        PackOrchestrator,
        HealthCheck,
    )

    logger.info("✓ All PACK-027 components imported successfully")
except ImportError as e:
    logger.error(f"✗ Failed to import PACK-027 components: {e}")
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
    logger.info("🚀 Starting PACK-027 Enterprise Net Zero Pack...")
    app_state.start_time = datetime.utcnow()

    # Load configuration
    preset = os.getenv("PACK_PRESET", "manufacturing")
    app_state.config = load_preset(preset)
    logger.info(f"✓ Loaded preset: {preset}")

    # Initialize health check
    app_state.health_check = HealthCheck(config=app_state.config)
    logger.info("✓ Health check initialized")

    # Initialize template registry
    app_state.template_registry = TemplateRegistry()
    logger.info(f"✓ Template registry initialized ({app_state.template_registry.template_count} templates)")

    logger.info("✅ PACK-027 startup complete")

    yield

    # Shutdown
    logger.info("🛑 Shutting down PACK-027 Enterprise Net Zero Pack...")
    logger.info("✅ PACK-027 shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="PACK-027: Enterprise Net Zero Pack",
    description="Comprehensive net zero solution for large enterprises (>250 employees, >$50M revenue)",
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
            "pack_id": "PACK-027",
            "pack_name": "Enterprise Net Zero Pack",
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
async def calculate_enterprise_baseline(request: Dict[str, Any]):
    """Calculate enterprise emissions baseline with multi-entity consolidation."""
    try:
        engine = EnterpriseBaselineEngine()
        result = await engine.calculate(request)

        # Track metrics
        BASELINE_CALCULATIONS.labels(
            consolidation_method=request.get("consolidation_method", "financial_control")
        ).inc()

        return result
    except Exception as e:
        logger.error(f"Enterprise baseline calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/engines/sbti-target", tags=["Engines"])
async def set_sbti_target(request: Dict[str, Any]):
    """Set SBTi Corporate Standard aligned net zero target."""
    try:
        engine = SBTiTargetEngine()
        result = await engine.calculate(request)

        # Track metrics
        SBTI_TARGETS_SET.labels(target_type=request.get("target_type", "absolute")).inc()

        return result
    except Exception as e:
        logger.error(f"SBTi target setting failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/engines/scenario", tags=["Engines"])
async def model_scenarios(request: Dict[str, Any]):
    """Model climate scenarios (1.5°C, 2°C, BAU pathways)."""
    try:
        engine = ScenarioModelingEngine()
        result = await engine.calculate(request)

        # Track metrics
        for pathway in request.get("pathways", ["1.5C", "2C", "BAU"]):
            SCENARIOS_MODELED.labels(pathway=pathway).inc()

        return result
    except Exception as e:
        logger.error(f"Scenario modeling failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/engines/carbon-pricing", tags=["Engines"])
async def calculate_carbon_price(request: Dict[str, Any]):
    """Calculate internal carbon price for investment decisions."""
    try:
        engine = CarbonPricingEngine()
        result = await engine.calculate(request)
        return result
    except Exception as e:
        logger.error(f"Carbon pricing calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/engines/scope4", tags=["Engines"])
async def calculate_scope4(request: Dict[str, Any]):
    """Calculate Scope 4 avoided emissions."""
    try:
        engine = Scope4AvoidedEmissionsEngine()
        result = await engine.calculate(request)
        return result
    except Exception as e:
        logger.error(f"Scope 4 calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/engines/supply-chain", tags=["Engines"])
async def map_supply_chain(request: Dict[str, Any]):
    """Map multi-tier supply chain emissions."""
    try:
        engine = SupplyChainMappingEngine()
        result = await engine.calculate(request)
        return result
    except Exception as e:
        logger.error(f"Supply chain mapping failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/engines/consolidation", tags=["Engines"])
async def consolidate_entities(request: Dict[str, Any]):
    """Consolidate emissions from multiple entities."""
    try:
        engine = MultiEntityConsolidationEngine()
        result = await engine.calculate(request)
        return result
    except Exception as e:
        logger.error(f"Multi-entity consolidation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/engines/financial-integration", tags=["Engines"])
async def integrate_financial_data(request: Dict[str, Any]):
    """Integrate with enterprise financial systems (SAP/Oracle/Workday)."""
    try:
        engine = FinancialIntegrationEngine()
        result = await engine.calculate(request)
        return result
    except Exception as e:
        logger.error(f"Financial integration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/engines/data-quality", tags=["Engines"])
async def assess_data_quality(request: Dict[str, Any]):
    """Assess data quality (financial-grade ±3% target)."""
    try:
        engine = DataQualityGuardianEngine()
        result = await engine.calculate(request)
        return result
    except Exception as e:
        logger.error(f"Data quality assessment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/engines/compliance", tags=["Engines"])
async def check_compliance(request: Dict[str, Any]):
    """Check regulatory compliance (SEC/CSRD/SB253/ISSB)."""
    try:
        engine = RegulatoryComplianceEngine()
        result = await engine.calculate(request)
        return result
    except Exception as e:
        logger.error(f"Compliance check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/engines/assurance", tags=["Engines"])
async def assess_assurance_readiness(request: Dict[str, Any]):
    """Assess external assurance readiness (ISO 14064-3)."""
    try:
        engine = AssuranceReadinessEngine()
        result = await engine.calculate(request)

        # Track metrics
        ASSURANCE_CHECKS.inc()

        return result
    except Exception as e:
        logger.error(f"Assurance readiness assessment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/engines/risk", tags=["Engines"])
async def assess_climate_risk(request: Dict[str, Any]):
    """Assess climate-related financial risks (TCFD)."""
    try:
        engine = RiskAssessmentEngine()
        result = await engine.calculate(request)
        return result
    except Exception as e:
        logger.error(f"Risk assessment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Workflow endpoints
@app.post("/workflows/comprehensive-baseline", tags=["Workflows"])
async def comprehensive_baseline_workflow(request: Dict[str, Any]):
    """Comprehensive baseline workflow (multi-entity Scope 1+2+3)."""
    try:
        workflow = ComprehensiveBaselineWorkflow()
        result = await workflow.execute(request)
        return result
    except Exception as e:
        logger.error(f"Comprehensive baseline workflow failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/workflows/sbti-submission", tags=["Workflows"])
async def sbti_submission_workflow(request: Dict[str, Any]):
    """SBTi target submission workflow."""
    try:
        workflow = SBTiSubmissionWorkflow()
        result = await workflow.execute(request)
        return result
    except Exception as e:
        logger.error(f"SBTi submission workflow failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/workflows/annual-inventory", tags=["Workflows"])
async def annual_inventory_workflow(request: Dict[str, Any]):
    """Annual GHG inventory workflow."""
    try:
        workflow = AnnualInventoryWorkflow()
        result = await workflow.execute(request)
        return result
    except Exception as e:
        logger.error(f"Annual inventory workflow failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/workflows/scenario-analysis", tags=["Workflows"])
async def scenario_analysis_workflow(request: Dict[str, Any]):
    """Scenario analysis workflow (1.5°C, 2°C, BAU)."""
    try:
        workflow = ScenarioAnalysisWorkflow()
        result = await workflow.execute(request)
        return result
    except Exception as e:
        logger.error(f"Scenario analysis workflow failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/workflows/supply-chain-engagement", tags=["Workflows"])
async def supply_chain_engagement_workflow(request: Dict[str, Any]):
    """Supply chain engagement workflow."""
    try:
        workflow = SupplyChainEngagementWorkflow()
        result = await workflow.execute(request)
        return result
    except Exception as e:
        logger.error(f"Supply chain engagement workflow failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/workflows/carbon-pricing", tags=["Workflows"])
async def carbon_pricing_workflow(request: Dict[str, Any]):
    """Internal carbon pricing workflow."""
    try:
        workflow = InternalCarbonPricingWorkflow()
        result = await workflow.execute(request)
        return result
    except Exception as e:
        logger.error(f"Carbon pricing workflow failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/workflows/multi-entity-rollup", tags=["Workflows"])
async def multi_entity_rollup_workflow(request: Dict[str, Any]):
    """Multi-entity rollup workflow (100+ subsidiaries)."""
    try:
        workflow = MultiEntityRollupWorkflow()
        result = await workflow.execute(request)
        return result
    except Exception as e:
        logger.error(f"Multi-entity rollup workflow failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/workflows/external-assurance", tags=["Workflows"])
async def external_assurance_workflow(request: Dict[str, Any]):
    """External assurance workflow (ISO 14064-3)."""
    try:
        workflow = ExternalAssuranceWorkflow()
        result = await workflow.execute(request)
        return result
    except Exception as e:
        logger.error(f"External assurance workflow failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/workflows/board-reporting", tags=["Workflows"])
async def board_reporting_workflow(request: Dict[str, Any]):
    """Board reporting workflow."""
    try:
        workflow = BoardReportingWorkflow()
        result = await workflow.execute(request)
        return result
    except Exception as e:
        logger.error(f"Board reporting workflow failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/workflows/regulatory-filing", tags=["Workflows"])
async def regulatory_filing_workflow(request: Dict[str, Any]):
    """Regulatory filing workflow (SEC/CSRD/SB253)."""
    try:
        workflow = RegulatoryFilingWorkflow()
        result = await workflow.execute(request)
        return result
    except Exception as e:
        logger.error(f"Regulatory filing workflow failed: {e}")
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
        "pack_id": "PACK-027",
        "pack_name": "Enterprise Net Zero Pack",
        "version": "1.0.0",
        "tier": "Enterprise",
        "description": "Comprehensive net zero solution for large enterprises (>250 employees, >$50M revenue)",
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
