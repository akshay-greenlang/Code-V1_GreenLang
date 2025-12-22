"""
GL-006 HEATRECLAIM - REST API

FastAPI-based REST endpoints for heat recovery optimization.

Endpoints:
- POST /optimize - Run full optimization
- POST /pinch-analysis - Run pinch analysis only
- POST /validate-streams - Validate stream data
- GET /status/{job_id} - Check job status
- GET /health - Health check
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import logging
import uuid

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..core.config import HeatReclaimConfig, OptimizationObjective
from ..core.schemas import (
    HeatStream,
    HENDesign,
    OptimizationRequest,
    OptimizationResult,
    OptimizationStatus,
    PinchAnalysisResult,
    APIResponse,
)
from ..core.orchestrator import HeatReclaimOrchestrator
from ..core.handlers import StreamDataHandler, RequestHandler

logger = logging.getLogger(__name__)


# Request/Response Models
class StreamInput(BaseModel):
    """Input model for heat stream."""

    stream_id: str = Field(..., description="Unique stream identifier")
    stream_name: Optional[str] = Field(None, description="Human-readable name")
    stream_type: str = Field("hot", description="Stream type: hot or cold")
    fluid_name: str = Field("Water", description="Working fluid")
    phase: str = Field("liquid", description="Phase: liquid, gas, two_phase")
    T_supply_C: float = Field(..., description="Supply temperature (°C)")
    T_target_C: float = Field(..., description="Target temperature (°C)")
    m_dot_kg_s: float = Field(..., description="Mass flow rate (kg/s)")
    Cp_kJ_kgK: float = Field(4.186, description="Specific heat capacity (kJ/kg·K)")
    pressure_kPa: float = Field(101.325, description="Operating pressure (kPa)")
    fouling_factor_m2K_W: float = Field(0.0001, description="Fouling factor (m²·K/W)")
    availability: float = Field(1.0, description="Stream availability (0-1)")


class OptimizationInput(BaseModel):
    """Input model for optimization request."""

    hot_streams: List[StreamInput] = Field(..., min_length=1)
    cold_streams: List[StreamInput] = Field(..., min_length=1)
    delta_t_min_C: float = Field(10.0, ge=1.0, le=50.0)
    objective: str = Field("minimize_cost")
    include_exergy_analysis: bool = Field(True)
    include_uncertainty: bool = Field(False)
    generate_pareto: bool = Field(False)
    n_pareto_points: int = Field(20, ge=5, le=100)
    max_time_seconds: float = Field(300.0, ge=10.0, le=3600.0)


class PinchAnalysisInput(BaseModel):
    """Input model for pinch analysis."""

    hot_streams: List[StreamInput] = Field(..., min_length=1)
    cold_streams: List[StreamInput] = Field(..., min_length=1)
    delta_t_min_C: float = Field(10.0, ge=1.0, le=50.0)


class ValidationInput(BaseModel):
    """Input model for stream validation."""

    hot_streams: List[StreamInput]
    cold_streams: List[StreamInput]


class JobStatusResponse(BaseModel):
    """Response model for job status."""

    job_id: str
    status: str
    created_at: datetime
    updated_at: datetime
    progress_percent: Optional[float] = None
    message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    version: str
    timestamp: datetime
    components: Dict[str, str]


# In-memory job store (replace with Redis/DB in production)
job_store: Dict[str, Dict[str, Any]] = {}


# Dependencies
def get_orchestrator() -> HeatReclaimOrchestrator:
    """Get orchestrator instance."""
    return HeatReclaimOrchestrator()


def get_stream_handler() -> StreamDataHandler:
    """Get stream handler instance."""
    return StreamDataHandler()


def get_request_handler() -> RequestHandler:
    """Get request handler instance."""
    return RequestHandler()


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("GL-006 HEATRECLAIM API starting up")
    yield
    logger.info("GL-006 HEATRECLAIM API shutting down")


# Create FastAPI app
def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="GL-006 HEATRECLAIM API",
        description="Heat Recovery Optimization API - Pinch Analysis & HEN Synthesis",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include router
    app.include_router(router, prefix="/api/v1")

    return app


# API Router
from fastapi import APIRouter

router = APIRouter(tags=["Heat Recovery"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns service status and component health.
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now(timezone.utc),
        components={
            "orchestrator": "ok",
            "pinch_calculator": "ok",
            "milp_optimizer": "ok",
            "database": "ok",
        },
    )


@router.post("/optimize", response_model=Dict[str, Any])
async def run_optimization(
    request: OptimizationInput,
    background_tasks: BackgroundTasks,
    async_mode: bool = Query(False, description="Run asynchronously"),
    orchestrator: HeatReclaimOrchestrator = Depends(get_orchestrator),
):
    """
    Run full heat recovery optimization.

    Performs:
    1. Stream validation
    2. Pinch analysis
    3. HEN synthesis
    4. MILP optimization
    5. Economic analysis
    6. Explainability generation
    """
    try:
        # Convert input to internal format
        hot_streams = [_convert_stream_input(s, "hot") for s in request.hot_streams]
        cold_streams = [_convert_stream_input(s, "cold") for s in request.cold_streams]

        # Map objective
        objective_map = {
            "minimize_cost": OptimizationObjective.MINIMIZE_COST,
            "minimize_utility": OptimizationObjective.MINIMIZE_UTILITY,
            "minimize_exchangers": OptimizationObjective.MINIMIZE_EXCHANGERS,
            "maximize_recovery": OptimizationObjective.MAXIMIZE_RECOVERY,
        }
        objective = objective_map.get(
            request.objective, OptimizationObjective.MINIMIZE_COST
        )

        if async_mode:
            # Create job and run in background
            job_id = str(uuid.uuid4())
            job_store[job_id] = {
                "status": "pending",
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
                "result": None,
            }

            background_tasks.add_task(
                _run_optimization_task,
                job_id,
                orchestrator,
                hot_streams,
                cold_streams,
                request.delta_t_min_C,
                objective,
                request.include_exergy_analysis,
                request.include_uncertainty,
                request.generate_pareto,
            )

            return {
                "job_id": job_id,
                "status": "accepted",
                "message": "Optimization job submitted",
                "status_url": f"/api/v1/status/{job_id}",
            }

        else:
            # Synchronous execution
            result = await orchestrator.optimize_async(
                hot_streams=hot_streams,
                cold_streams=cold_streams,
                delta_t_min=request.delta_t_min_C,
                objective=objective,
                include_exergy=request.include_exergy_analysis,
                include_uncertainty=request.include_uncertainty,
            )

            return _format_optimization_result(result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Optimization error: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@router.post("/pinch-analysis", response_model=Dict[str, Any])
async def run_pinch_analysis(
    request: PinchAnalysisInput,
    orchestrator: HeatReclaimOrchestrator = Depends(get_orchestrator),
):
    """
    Run pinch analysis only.

    Returns:
    - Pinch temperature
    - Minimum utility targets
    - Maximum heat recovery
    - Composite curves data
    """
    try:
        hot_streams = [_convert_stream_input(s, "hot") for s in request.hot_streams]
        cold_streams = [_convert_stream_input(s, "cold") for s in request.cold_streams]

        result = orchestrator.run_pinch_analysis(
            hot_streams=hot_streams,
            cold_streams=cold_streams,
            delta_t_min=request.delta_t_min_C,
        )

        return {
            "success": True,
            "pinch_temperature_C": result.pinch_temperature_C,
            "minimum_hot_utility_kW": result.minimum_hot_utility_kW,
            "minimum_cold_utility_kW": result.minimum_cold_utility_kW,
            "maximum_heat_recovery_kW": result.maximum_heat_recovery_kW,
            "is_threshold_problem": result.is_threshold_problem,
            "hot_composite_curve": [
                {"T_C": t, "H_kW": h}
                for t, h in zip(
                    result.hot_composite_T_C,
                    result.hot_composite_H_kW,
                )
            ],
            "cold_composite_curve": [
                {"T_C": t, "H_kW": h}
                for t, h in zip(
                    result.cold_composite_T_C,
                    result.cold_composite_H_kW,
                )
            ],
            "computation_hash": result.computation_hash,
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Pinch analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate-streams", response_model=Dict[str, Any])
async def validate_streams(
    request: ValidationInput,
    handler: StreamDataHandler = Depends(get_stream_handler),
):
    """
    Validate stream data without running optimization.

    Checks:
    - Required fields
    - Temperature directions
    - Value ranges
    - Duplicate IDs
    """
    try:
        hot_streams = [_convert_stream_input(s, "hot") for s in request.hot_streams]
        cold_streams = [_convert_stream_input(s, "cold") for s in request.cold_streams]

        result = handler.validate_streams(hot_streams, cold_streams)

        return {
            "is_valid": result.is_valid,
            "errors": result.errors,
            "warnings": result.warnings,
            "stream_count": {
                "hot": len(request.hot_streams),
                "cold": len(request.cold_streams),
            },
        }

    except Exception as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get status of async optimization job.
    """
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job not found")

    job = job_store[job_id]
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        created_at=job["created_at"],
        updated_at=job["updated_at"],
        progress_percent=job.get("progress"),
        message=job.get("message"),
        result=job.get("result"),
    )


@router.get("/designs", response_model=Dict[str, Any])
async def list_designs(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """
    List saved HEN designs.
    """
    # Placeholder - would query database
    return {
        "designs": [],
        "total": 0,
        "limit": limit,
        "offset": offset,
    }


@router.get("/designs/{design_id}", response_model=Dict[str, Any])
async def get_design(design_id: str):
    """
    Get specific HEN design by ID.
    """
    # Placeholder - would query database
    raise HTTPException(status_code=404, detail="Design not found")


# Helper functions
def _convert_stream_input(input_data: StreamInput, stream_type: str) -> HeatStream:
    """Convert API input to internal HeatStream."""
    from ..core.config import StreamType, Phase

    type_map = {"hot": StreamType.HOT, "cold": StreamType.COLD}
    phase_map = {"liquid": Phase.LIQUID, "gas": Phase.GAS, "two_phase": Phase.TWO_PHASE}

    return HeatStream(
        stream_id=input_data.stream_id,
        stream_name=input_data.stream_name or input_data.stream_id,
        stream_type=type_map.get(input_data.stream_type, StreamType.HOT),
        fluid_name=input_data.fluid_name,
        phase=phase_map.get(input_data.phase, Phase.LIQUID),
        T_supply_C=input_data.T_supply_C,
        T_target_C=input_data.T_target_C,
        m_dot_kg_s=input_data.m_dot_kg_s,
        Cp_kJ_kgK=input_data.Cp_kJ_kgK,
        pressure_kPa=input_data.pressure_kPa,
        fouling_factor_m2K_W=input_data.fouling_factor_m2K_W,
        availability=input_data.availability,
    )


async def _run_optimization_task(
    job_id: str,
    orchestrator: HeatReclaimOrchestrator,
    hot_streams: List[HeatStream],
    cold_streams: List[HeatStream],
    delta_t_min: float,
    objective: OptimizationObjective,
    include_exergy: bool,
    include_uncertainty: bool,
    generate_pareto: bool,
):
    """Background task for async optimization."""
    try:
        job_store[job_id]["status"] = "running"
        job_store[job_id]["updated_at"] = datetime.now(timezone.utc)

        result = await orchestrator.optimize_async(
            hot_streams=hot_streams,
            cold_streams=cold_streams,
            delta_t_min=delta_t_min,
            objective=objective,
            include_exergy=include_exergy,
            include_uncertainty=include_uncertainty,
        )

        job_store[job_id]["status"] = "completed"
        job_store[job_id]["result"] = _format_optimization_result(result)

    except Exception as e:
        job_store[job_id]["status"] = "failed"
        job_store[job_id]["message"] = str(e)

    finally:
        job_store[job_id]["updated_at"] = datetime.now(timezone.utc)


def _format_optimization_result(result: OptimizationResult) -> Dict[str, Any]:
    """Format optimization result for API response."""
    return {
        "request_id": result.request_id,
        "status": result.status.value,
        "optimization_time_seconds": result.optimization_time_seconds,
        "pinch_analysis": {
            "pinch_temperature_C": result.pinch_analysis.pinch_temperature_C,
            "minimum_hot_utility_kW": result.pinch_analysis.minimum_hot_utility_kW,
            "minimum_cold_utility_kW": result.pinch_analysis.minimum_cold_utility_kW,
            "maximum_heat_recovery_kW": result.pinch_analysis.maximum_heat_recovery_kW,
        },
        "design": {
            "design_name": result.recommended_design.design_name,
            "exchanger_count": result.recommended_design.exchanger_count,
            "total_heat_recovered_kW": result.recommended_design.total_heat_recovered_kW,
            "hot_utility_required_kW": result.recommended_design.hot_utility_required_kW,
            "cold_utility_required_kW": result.recommended_design.cold_utility_required_kW,
            "exchangers": [
                {
                    "exchanger_id": hx.exchanger_id,
                    "hot_stream_id": hx.hot_stream_id,
                    "cold_stream_id": hx.cold_stream_id,
                    "duty_kW": hx.duty_kW,
                    "area_m2": hx.area_m2,
                    "LMTD_C": hx.LMTD_C,
                }
                for hx in result.recommended_design.exchangers
            ],
        },
        "explanation_summary": result.explanation_summary,
        "key_drivers": result.key_drivers,
        "robustness_score": result.robustness_score,
    }


# Create app instance
app = create_app()
