# -*- coding: utf-8 -*-
"""
GL-010 EMISSIONWATCH - FastAPI Application Entry Point.

This module provides the FastAPI application for the EmissionsComplianceAgent
with REST API endpoints for all 8 operation modes, health checks, and
Prometheus metrics.

Operation Modes:
- MONITOR: Real-time CEMS data monitoring
- REPORT: Generate regulatory reports (EPA, EU, China)
- ALERT: Violation detection and notification
- ANALYZE: Emissions trend analysis
- PREDICT: Exceedance prediction
- AUDIT: Compliance audit trail
- BENCHMARK: Compare against permit limits
- VALIDATE: Data quality validation

Standards Compliance:
- EPA 40 CFR Parts 60, 75 - Continuous Emissions Monitoring
- EU Industrial Emissions Directive 2010/75/EU
- China MEE Emission Standards (GB 13223-2011)

Author: GreenLang Foundation
Version: 1.0.0
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException, Request, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, PlainTextResponse
    from pydantic import BaseModel, Field
except ImportError:
    # Fallback for environments without FastAPI
    FastAPI = None
    HTTPException = None
    BaseModel = object
    Field = lambda **kwargs: None

from .config import EmissionsComplianceConfig, create_config
from .emissions_compliance_orchestrator import (
    EmissionsComplianceOrchestrator,
    OperationMode,
    create_orchestrator
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class CEMSDataModel(BaseModel):
    """CEMS data input model."""
    nox_ppm: float = Field(default=0.0, ge=0, description="NOx concentration in ppm")
    sox_ppm: float = Field(default=0.0, ge=0, description="SOx concentration in ppm")
    co2_percent: float = Field(default=0.0, ge=0, le=25, description="CO2 concentration in percent")
    co_ppm: float = Field(default=0.0, ge=0, description="CO concentration in ppm")
    o2_percent: float = Field(default=3.0, ge=0, le=21, description="O2 concentration in percent")
    pm_mg_m3: float = Field(default=0.0, ge=0, description="PM concentration in mg/m3")
    opacity_percent: float = Field(default=0.0, ge=0, le=100, description="Stack opacity in percent")
    flow_rate_dscfm: float = Field(default=10000, ge=0, description="Stack flow rate in dscfm")
    temperature_f: float = Field(default=300, description="Stack temperature in Fahrenheit")
    quality_code: str = Field(default="valid", description="Data quality code")


class FuelDataModel(BaseModel):
    """Fuel data input model."""
    fuel_type: str = Field(default="natural_gas", description="Type of fuel")
    heat_input_mmbtu_hr: float = Field(default=100.0, ge=0, description="Heat input in MMBtu/hr")
    heating_value_btu_lb: float = Field(default=23000, ge=0, description="Heating value in Btu/lb")
    sulfur_percent: float = Field(default=0.0006, ge=0, le=5, description="Fuel sulfur content")
    nitrogen_percent: float = Field(default=0.01, ge=0, le=5, description="Fuel nitrogen content")
    carbon_percent: float = Field(default=75.0, ge=0, le=100, description="Fuel carbon content")
    ash_percent: float = Field(default=0.0, ge=0, le=50, description="Fuel ash content")


class ProcessParametersModel(BaseModel):
    """Process parameters model."""
    combustion_temperature_f: float = Field(default=2800, description="Combustion temperature")
    excess_air_percent: float = Field(default=15.0, ge=0, description="Excess air percentage")
    load_percent: float = Field(default=100.0, ge=0, le=100, description="Load percentage")
    scr_efficiency_percent: float = Field(default=0.0, ge=0, le=100, description="SCR efficiency")
    fgd_efficiency_percent: float = Field(default=0.0, ge=0, le=100, description="FGD efficiency")
    baghouse_efficiency_percent: float = Field(default=99.0, ge=0, le=100, description="Baghouse efficiency")


class PermitLimitsModel(BaseModel):
    """Permit limits model."""
    nox_limit: float = Field(default=0.1, description="NOx limit (lb/MMBtu)")
    sox_limit: float = Field(default=0.15, description="SOx limit (lb/MMBtu)")
    co2_limit: float = Field(default=50.0, description="CO2 limit (tons/hr)")
    pm_limit: float = Field(default=0.03, description="PM limit (lb/MMBtu)")


class FacilityDataModel(BaseModel):
    """Facility data model."""
    facility_id: str = Field(default="FAC-001", description="Facility identifier")
    unit_id: str = Field(default="UNIT-001", description="Unit identifier")
    facility_name: Optional[str] = Field(default=None, description="Facility name")
    designated_representative: Optional[str] = Field(default=None, description="Designated representative")
    state: Optional[str] = Field(default=None, description="State code")


class ReportingPeriodModel(BaseModel):
    """Reporting period model."""
    start_date: str = Field(..., description="Period start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="Period end date (YYYY-MM-DD)")
    quarter: Optional[int] = Field(default=None, ge=1, le=4, description="Quarter number")
    year: Optional[int] = Field(default=None, ge=2000, description="Year")


class MonitoringRequest(BaseModel):
    """Request model for monitoring mode."""
    operation_mode: str = Field(default="monitor", description="Operation mode")
    cems_data: CEMSDataModel = Field(..., description="CEMS measurements")
    fuel_data: FuelDataModel = Field(default_factory=FuelDataModel, description="Fuel data")
    process_parameters: ProcessParametersModel = Field(
        default_factory=ProcessParametersModel,
        description="Process parameters"
    )
    jurisdiction: str = Field(default="EPA", description="Regulatory jurisdiction")


class ReportRequest(BaseModel):
    """Request model for report generation."""
    operation_mode: str = Field(default="report", description="Operation mode")
    report_format: str = Field(default="EPA_ECMPS", description="Report format")
    reporting_period: ReportingPeriodModel = Field(..., description="Reporting period")
    facility_data: FacilityDataModel = Field(
        default_factory=FacilityDataModel,
        description="Facility data"
    )
    emissions_data: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Historical emissions data"
    )


class AlertRequest(BaseModel):
    """Request model for alert mode."""
    operation_mode: str = Field(default="alert", description="Operation mode")
    cems_data: CEMSDataModel = Field(..., description="CEMS measurements")
    fuel_data: FuelDataModel = Field(default_factory=FuelDataModel, description="Fuel data")
    process_parameters: ProcessParametersModel = Field(
        default_factory=ProcessParametersModel,
        description="Process parameters"
    )
    permit_limits: PermitLimitsModel = Field(
        default_factory=PermitLimitsModel,
        description="Permit limits"
    )


class AnalysisRequest(BaseModel):
    """Request model for analysis mode."""
    operation_mode: str = Field(default="analyze", description="Operation mode")
    historical_data: List[Dict[str, Any]] = Field(..., description="Historical emissions data")
    analysis_period: Dict[str, str] = Field(default_factory=dict, description="Analysis period")
    process_parameters: ProcessParametersModel = Field(
        default_factory=ProcessParametersModel,
        description="Process parameters"
    )


class PredictionRequest(BaseModel):
    """Request model for prediction mode."""
    operation_mode: str = Field(default="predict", description="Operation mode")
    current_data: Dict[str, Any] = Field(..., description="Current emissions data")
    historical_data: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Historical data"
    )
    forecast_horizon_hours: int = Field(default=24, ge=1, le=168, description="Forecast horizon")
    permit_limits: PermitLimitsModel = Field(
        default_factory=PermitLimitsModel,
        description="Permit limits"
    )


class AuditRequest(BaseModel):
    """Request model for audit mode."""
    operation_mode: str = Field(default="audit", description="Operation mode")
    audit_period: Dict[str, str] = Field(..., description="Audit period")
    facility_data: FacilityDataModel = Field(
        default_factory=FacilityDataModel,
        description="Facility data"
    )
    emissions_records: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Emissions records"
    )
    compliance_events: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Compliance events"
    )


class BenchmarkRequest(BaseModel):
    """Request model for benchmark mode."""
    operation_mode: str = Field(default="benchmark", description="Operation mode")
    current_emissions: Dict[str, float] = Field(..., description="Current emissions values")
    permit_limits: PermitLimitsModel = Field(
        default_factory=PermitLimitsModel,
        description="Permit limits"
    )
    industry_benchmarks: Dict[str, float] = Field(
        default_factory=dict,
        description="Industry benchmarks"
    )
    process_type: str = Field(default="boiler", description="Process type")


class ValidateRequest(BaseModel):
    """Request model for validation mode."""
    operation_mode: str = Field(default="validate", description="Operation mode")
    cems_data: CEMSDataModel = Field(..., description="CEMS data to validate")
    validation_period: Dict[str, str] = Field(
        default_factory=dict,
        description="Validation period"
    )
    qapp_requirements: Dict[str, Any] = Field(
        default_factory=dict,
        description="QAPP requirements"
    )


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    agent_id: str
    version: str
    jurisdiction: str
    timestamp: str
    checks: Dict[str, bool]


class MetricsResponse(BaseModel):
    """Metrics response model."""
    calculations_performed: int
    avg_calculation_time_ms: float
    cache_hit_rate_percent: float
    nox_calculations: int
    sox_calculations: int
    co2_calculations: int
    pm_calculations: int
    violations_detected: int
    alerts_sent: int
    errors_encountered: int
    timestamp: str


# ============================================================================
# APPLICATION LIFECYCLE
# ============================================================================

# Global orchestrator instance
orchestrator: Optional[EmissionsComplianceOrchestrator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown of the orchestrator.
    """
    global orchestrator

    # Startup
    logger.info("Starting GL-010 EMISSIONWATCH EmissionsComplianceAgent...")

    try:
        config = create_config()
        orchestrator = create_orchestrator(config)
        logger.info(f"Orchestrator initialized: {orchestrator.config.agent_id}")
    except Exception as e:
        logger.error(f"Failed to initialize orchestrator: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down GL-010 EMISSIONWATCH...")
    if orchestrator:
        await orchestrator.shutdown()
    logger.info("Shutdown complete")


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

if FastAPI is not None:
    app = FastAPI(
        title="GL-010 EMISSIONWATCH - EmissionsComplianceAgent",
        description=(
            "Zero-hallucination emissions compliance monitoring for industrial processes. "
            "Supports EPA, EU IED, and China MEE regulations with real-time CEMS monitoring, "
            "violation detection, predictive analytics, and regulatory reporting."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )

    # CORS middleware - Restricted to trusted domains for security
    # SEC-001: Fixed wildcard CORS configuration
    ALLOWED_ORIGINS = [
        "https://dashboard.greenlang.io",
        "https://admin.greenlang.io",
        "https://api.greenlang.io",
        "https://epa.gov",
        "https://www.epa.gov",
        "https://ec.europa.eu",
        os.getenv("CORS_ALLOWED_ORIGIN", "https://localhost:3000"),
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ========================================================================
    # HEALTH ENDPOINTS
    # ========================================================================

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check():
        """
        Health check endpoint for Kubernetes probes.

        Returns:
            Health status with component checks
        """
        if orchestrator is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Orchestrator not initialized"
            )

        health = orchestrator.get_health()

        return HealthResponse(
            status=health['status'],
            agent_id=orchestrator.config.agent_id,
            version=orchestrator.config.version,
            jurisdiction=orchestrator.config.jurisdiction,
            timestamp=datetime.now(timezone.utc).isoformat(),
            checks=health['checks']
        )

    @app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
    async def health_check_v1():
        """API v1 health check endpoint."""
        return await health_check()

    @app.get("/ready", tags=["Health"])
    async def readiness_check():
        """
        Readiness check endpoint for Kubernetes.

        Returns:
            Ready status
        """
        if orchestrator is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not ready"
            )

        state = orchestrator.get_state()
        if state['state'] in ['ready', 'executing']:
            return {"status": "ready", "state": state['state']}

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service not ready: {state['state']}"
        )

    @app.get("/api/v1/ready", tags=["Health"])
    async def readiness_check_v1():
        """API v1 readiness check endpoint."""
        return await readiness_check()

    @app.get("/metrics", response_class=PlainTextResponse, tags=["Monitoring"])
    async def get_prometheus_metrics():
        """
        Prometheus-compatible metrics endpoint.

        Returns:
            Prometheus format metrics
        """
        if orchestrator is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Orchestrator not initialized"
            )

        metrics = orchestrator.metrics.get_metrics()

        # Format as Prometheus metrics
        lines = [
            "# HELP gl010_calculations_total Total number of calculations performed",
            "# TYPE gl010_calculations_total counter",
            f"gl010_calculations_total {metrics['calculations_performed']}",
            "",
            "# HELP gl010_calculation_time_ms Average calculation time in milliseconds",
            "# TYPE gl010_calculation_time_ms gauge",
            f"gl010_calculation_time_ms {metrics['avg_calculation_time_ms']:.2f}",
            "",
            "# HELP gl010_cache_hit_rate Cache hit rate percentage",
            "# TYPE gl010_cache_hit_rate gauge",
            f"gl010_cache_hit_rate {metrics.get('cache_hit_rate_percent', 0):.2f}",
            "",
            "# HELP gl010_violations_detected Total violations detected",
            "# TYPE gl010_violations_detected counter",
            f"gl010_violations_detected {metrics['violations_detected']}",
            "",
            "# HELP gl010_alerts_sent Total alerts sent",
            "# TYPE gl010_alerts_sent counter",
            f"gl010_alerts_sent {metrics['alerts_sent']}",
            "",
            "# HELP gl010_errors_total Total errors encountered",
            "# TYPE gl010_errors_total counter",
            f"gl010_errors_total {metrics['errors_encountered']}",
        ]

        return "\n".join(lines)

    @app.get("/api/v1/metrics", response_model=MetricsResponse, tags=["Monitoring"])
    async def get_metrics():
        """
        JSON metrics endpoint.

        Returns:
            Current performance metrics
        """
        if orchestrator is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Orchestrator not initialized"
            )

        metrics = orchestrator.metrics.get_metrics()

        return MetricsResponse(
            calculations_performed=metrics['calculations_performed'],
            avg_calculation_time_ms=round(metrics['avg_calculation_time_ms'], 2),
            cache_hit_rate_percent=round(metrics.get('cache_hit_rate_percent', 0), 2),
            nox_calculations=metrics['nox_calculations'],
            sox_calculations=metrics['sox_calculations'],
            co2_calculations=metrics['co2_calculations'],
            pm_calculations=metrics['pm_calculations'],
            violations_detected=metrics['violations_detected'],
            alerts_sent=metrics['alerts_sent'],
            errors_encountered=metrics['errors_encountered'],
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    # ========================================================================
    # OPERATION MODE ENDPOINTS
    # ========================================================================

    @app.post("/api/v1/monitor", tags=["Operations"])
    async def monitor_emissions(request: MonitoringRequest):
        """
        Real-time CEMS data monitoring.

        Processes continuous emissions monitoring data to calculate
        current emissions levels and check compliance.

        Args:
            request: Monitoring request with CEMS and fuel data

        Returns:
            Current emissions and compliance status
        """
        if orchestrator is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Orchestrator not initialized"
            )

        try:
            input_data = {
                'operation_mode': 'monitor',
                'cems_data': request.cems_data.model_dump(),
                'fuel_data': request.fuel_data.model_dump(),
                'process_parameters': request.process_parameters.model_dump(),
                'jurisdiction': request.jurisdiction
            }

            result = await orchestrator.execute(input_data)
            return JSONResponse(content=result)

        except ValueError as e:
            logger.error(f"Validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except Exception as e:
            logger.error(f"Monitoring failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Monitoring failed: {str(e)}"
            )

    @app.post("/api/v1/report", tags=["Operations"])
    async def generate_report(request: ReportRequest):
        """
        Generate regulatory compliance report.

        Creates compliance reports in EPA ECMPS, EU ELED, or China MEE formats.

        Args:
            request: Report request with period and emissions data

        Returns:
            Generated regulatory report
        """
        if orchestrator is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Orchestrator not initialized"
            )

        try:
            input_data = {
                'operation_mode': 'report',
                'report_format': request.report_format,
                'reporting_period': request.reporting_period.model_dump(),
                'facility_data': request.facility_data.model_dump(),
                'emissions_data': request.emissions_data
            }

            result = await orchestrator.execute(input_data)
            return JSONResponse(content=result)

        except ValueError as e:
            logger.error(f"Validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except Exception as e:
            logger.error(f"Report generation failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Report generation failed: {str(e)}"
            )

    @app.post("/api/v1/alert", tags=["Operations"])
    async def detect_alerts(request: AlertRequest):
        """
        Violation detection and alerting.

        Detects emissions violations and generates alerts
        based on permit limits.

        Args:
            request: Alert request with emissions and limits

        Returns:
            Detected violations and generated alerts
        """
        if orchestrator is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Orchestrator not initialized"
            )

        try:
            input_data = {
                'operation_mode': 'alert',
                'cems_data': request.cems_data.model_dump(),
                'fuel_data': request.fuel_data.model_dump(),
                'process_parameters': request.process_parameters.model_dump(),
                'permit_limits': request.permit_limits.model_dump()
            }

            result = await orchestrator.execute(input_data)
            return JSONResponse(content=result)

        except ValueError as e:
            logger.error(f"Validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except Exception as e:
            logger.error(f"Alert detection failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Alert detection failed: {str(e)}"
            )

    @app.post("/api/v1/analyze", tags=["Operations"])
    async def analyze_emissions(request: AnalysisRequest):
        """
        Emissions trend analysis.

        Analyzes historical emissions data to identify trends
        and patterns.

        Args:
            request: Analysis request with historical data

        Returns:
            Trend analysis results
        """
        if orchestrator is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Orchestrator not initialized"
            )

        try:
            input_data = {
                'operation_mode': 'analyze',
                'historical_data': request.historical_data,
                'analysis_period': request.analysis_period,
                'process_parameters': request.process_parameters.model_dump()
            }

            result = await orchestrator.execute(input_data)
            return JSONResponse(content=result)

        except ValueError as e:
            logger.error(f"Validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Analysis failed: {str(e)}"
            )

    @app.post("/api/v1/predict", tags=["Operations"])
    async def predict_exceedances(request: PredictionRequest):
        """
        Exceedance prediction.

        Predicts potential emissions exceedances based on
        current trends and operating conditions.

        Args:
            request: Prediction request with current and historical data

        Returns:
            Exceedance predictions
        """
        if orchestrator is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Orchestrator not initialized"
            )

        try:
            input_data = {
                'operation_mode': 'predict',
                'current_data': request.current_data,
                'historical_data': request.historical_data,
                'forecast_horizon_hours': request.forecast_horizon_hours,
                'permit_limits': request.permit_limits.model_dump()
            }

            result = await orchestrator.execute(input_data)
            return JSONResponse(content=result)

        except ValueError as e:
            logger.error(f"Validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {str(e)}"
            )

    @app.post("/api/v1/audit", tags=["Operations"])
    async def generate_audit_trail(request: AuditRequest):
        """
        Compliance audit trail generation.

        Generates complete audit trail with SHA-256 provenance
        hashes for regulatory verification.

        Args:
            request: Audit request with emissions records

        Returns:
            Audit trail with hash chain
        """
        if orchestrator is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Orchestrator not initialized"
            )

        try:
            input_data = {
                'operation_mode': 'audit',
                'audit_period': request.audit_period,
                'facility_data': request.facility_data.model_dump(),
                'emissions_records': request.emissions_records,
                'compliance_events': request.compliance_events
            }

            result = await orchestrator.execute(input_data)
            return JSONResponse(content=result)

        except ValueError as e:
            logger.error(f"Validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except Exception as e:
            logger.error(f"Audit generation failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Audit generation failed: {str(e)}"
            )

    @app.post("/api/v1/benchmark", tags=["Operations"])
    async def benchmark_emissions(request: BenchmarkRequest):
        """
        Permit limits benchmark comparison.

        Compares current emissions against permit limits
        and industry benchmarks.

        Args:
            request: Benchmark request with emissions and limits

        Returns:
            Benchmark comparison results
        """
        if orchestrator is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Orchestrator not initialized"
            )

        try:
            input_data = {
                'operation_mode': 'benchmark',
                'current_emissions': request.current_emissions,
                'permit_limits': request.permit_limits.model_dump(),
                'industry_benchmarks': request.industry_benchmarks,
                'process_type': request.process_type
            }

            result = await orchestrator.execute(input_data)
            return JSONResponse(content=result)

        except ValueError as e:
            logger.error(f"Validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except Exception as e:
            logger.error(f"Benchmarking failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Benchmarking failed: {str(e)}"
            )

    @app.post("/api/v1/validate", tags=["Operations"])
    async def validate_data(request: ValidateRequest):
        """
        CEMS data quality validation.

        Validates CEMS data quality according to EPA Part 75
        requirements.

        Args:
            request: Validation request with CEMS data

        Returns:
            Data quality validation results
        """
        if orchestrator is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Orchestrator not initialized"
            )

        try:
            input_data = {
                'operation_mode': 'validate',
                'cems_data': request.cems_data.model_dump(),
                'validation_period': request.validation_period,
                'qapp_requirements': request.qapp_requirements
            }

            result = await orchestrator.execute(input_data)
            return JSONResponse(content=result)

        except ValueError as e:
            logger.error(f"Validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except Exception as e:
            logger.error(f"Data validation failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Data validation failed: {str(e)}"
            )

    # ========================================================================
    # INFO ENDPOINTS
    # ========================================================================

    @app.get("/api/v1/info", tags=["Info"])
    async def get_info():
        """
        Get agent information.

        Returns:
            Agent identification and capabilities
        """
        if orchestrator is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Orchestrator not initialized"
            )

        return {
            "agent_id": orchestrator.config.agent_id,
            "codename": orchestrator.config.codename,
            "full_name": orchestrator.config.full_name,
            "version": orchestrator.config.version,
            "deterministic": orchestrator.config.deterministic,
            "jurisdiction": orchestrator.config.jurisdiction,
            "standards": [
                "EPA 40 CFR Part 60",
                "EPA 40 CFR Part 75",
                "EU IED 2010/75/EU",
                "China MEE GB 13223-2011"
            ],
            "operation_modes": [mode.value for mode in OperationMode],
            "pollutants": ["NOx", "SOx", "CO2", "PM", "CO", "Opacity"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    @app.get("/api/v1/state", tags=["Info"])
    async def get_state():
        """
        Get current orchestrator state.

        Returns:
            Current state and performance metrics
        """
        if orchestrator is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Orchestrator not initialized"
            )

        return orchestrator.get_state()

    @app.get("/api/v1/tools", tags=["Info"])
    async def get_tools():
        """
        Get available calculation tools.

        Returns:
            List of available tools with schemas
        """
        if orchestrator is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Orchestrator not initialized"
            )

        return orchestrator.tools.get_tool_schemas()

    @app.get("/api/v1/alerts", tags=["Info"])
    async def get_active_alerts():
        """
        Get active alerts.

        Returns:
            List of active violation alerts
        """
        if orchestrator is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Orchestrator not initialized"
            )

        return {
            "active_alerts": orchestrator.get_active_alerts(),
            "count": len(orchestrator.get_active_alerts()),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    @app.delete("/api/v1/alerts/{alert_id}", tags=["Info"])
    async def clear_alert(alert_id: str):
        """
        Clear an active alert.

        Args:
            alert_id: Alert identifier to clear

        Returns:
            Clear status
        """
        if orchestrator is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Orchestrator not initialized"
            )

        cleared = orchestrator.clear_alert(alert_id)
        if cleared:
            return {"status": "cleared", "alert_id": alert_id}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Alert {alert_id} not found"
            )

else:
    # Fallback for environments without FastAPI
    app = None


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    """
    Main entry point for CLI execution.

    Starts the FastAPI server with Uvicorn.
    """
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8010"))
    workers = int(os.environ.get("WORKERS", "1"))
    reload = os.environ.get("RELOAD", "false").lower() == "true"

    logger.info(f"Starting GL-010 EMISSIONWATCH on {host}:{port}")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
