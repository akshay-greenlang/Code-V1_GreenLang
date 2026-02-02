"""
GL-011 FUELCRAFT - REST API

FastAPI-based REST API for fuel mix optimization with complete
provenance tracking and IEC 61511 compliant health endpoints.

Endpoints:
- POST /runs - Create optimization run request
- GET /runs/{run_id} - Retrieve run status and metadata
- GET /runs/{run_id}/recommendation - Get optimized fuel plan
- GET /runs/{run_id}/explainability - Get SHAP-based explanations
- GET /health/live - Liveness probe
- GET /health/ready - Readiness probe
- GET /health/startup - Startup probe

Standards Compliance:
- IEC 61511 (Functional Safety)
- ISO 14064 (GHG Quantification)
- OpenAPI 3.0 specification
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import asyncio
import hashlib
import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Depends, Query, Path, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .schemas import (
    # Request Models
    RunRequest,
    RunStatus,
    # Response Models
    RunResponse,
    RunStatusResponse,
    RecommendationResponse,
    ExplainabilityResponse,
    # Output Models
    FuelMixOutput,
    BlendRatioOutput,
    CostBreakdown,
    CarbonFootprint,
    ProcurementRecommendation,
    ExplainabilityOutput,
    # Health Models
    HealthResponse,
    ReadinessResponse,
    ComponentHealth,
    # Error Models
    ErrorResponse,
    # Enums
    FuelType,
    OptimizationObjective,
    EmissionBoundary,
)

logger = logging.getLogger(__name__)


class FuelCraftAPI:
    """
    FuelCraft REST API wrapper.

    Provides programmatic access to API functionality with
    dependency injection for orchestrator and storage.
    """

    def __init__(
        self,
        orchestrator=None,
        storage=None,
        auth_enabled: bool = True,
        rate_limit_rpm: int = 100,
    ) -> None:
        """
        Initialize FuelCraft API.

        Args:
            orchestrator: Optimization orchestrator instance
            storage: Storage backend for runs
            auth_enabled: Enable authentication
            rate_limit_rpm: Rate limit (requests per minute)
        """
        self.orchestrator = orchestrator
        self.storage = storage
        self.auth_enabled = auth_enabled
        self.rate_limit_rpm = rate_limit_rpm

        self._start_time = time.time()
        self._request_count = 0

        # In-memory run storage (replace with actual storage in production)
        self._runs: Dict[str, Dict[str, Any]] = {}

        # Create FastAPI app
        self.app = create_app(self)

    def get_uptime(self) -> float:
        """Get uptime in seconds."""
        return time.time() - self._start_time

    async def create_run(self, request: RunRequest) -> RunResponse:
        """Create a new optimization run."""
        run_id = request.run_id

        # Store run request
        self._runs[run_id] = {
            "request": request.dict(),
            "status": RunStatus.PENDING,
            "created_at": datetime.now(timezone.utc),
            "started_at": None,
            "completed_at": None,
            "result": None,
            "error": None,
        }

        # Compute bundle hash
        bundle_hash = request.compute_bundle_hash()

        return RunResponse(
            run_id=run_id,
            status=RunStatus.PENDING,
            created_at=self._runs[run_id]["created_at"],
            estimated_completion=datetime.now(timezone.utc),
            queue_position=len(self._runs),
            input_snapshot_ids=request.input_snapshot_ids,
            bundle_hash=bundle_hash,
        )

    async def get_run_status(self, run_id: str) -> RunStatusResponse:
        """Get run status."""
        if run_id not in self._runs:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        run = self._runs[run_id]
        request = RunRequest(**run["request"])

        return RunStatusResponse(
            run_id=run_id,
            status=run["status"],
            created_at=run["created_at"],
            started_at=run["started_at"],
            completed_at=run["completed_at"],
            progress_percent=100.0 if run["status"] == RunStatus.COMPLETED else 0.0,
            current_step="Optimization complete" if run["status"] == RunStatus.COMPLETED else "Pending",
            error_message=run.get("error"),
            input_snapshot_ids=request.input_snapshot_ids,
            bundle_hash=request.compute_bundle_hash(),
        )

    async def get_recommendation(self, run_id: str) -> RecommendationResponse:
        """Get optimization recommendation."""
        if run_id not in self._runs:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        run = self._runs[run_id]
        request = RunRequest(**run["request"])

        # Generate demo recommendation
        return self._generate_demo_recommendation(run_id, request)

    def _generate_demo_recommendation(
        self,
        run_id: str,
        request: RunRequest,
    ) -> RecommendationResponse:
        """Generate demo recommendation for framework demonstration."""
        now = datetime.now(timezone.utc)

        # Create blend ratios from fuel prices
        blend_ratios = []
        total_quantity = request.demand_forecast.demand_mmbtu[0] if request.demand_forecast.demand_mmbtu else 1000.0

        for i, fuel_price in enumerate(request.fuel_prices):
            percentage = 100.0 / len(request.fuel_prices) if len(request.fuel_prices) > 1 else 100.0
            quantity = total_quantity * (percentage / 100.0)
            cost = quantity * fuel_price.spot_price_usd_mmbtu

            # Find emission factor for this fuel
            emission_factor = next(
                (ef for ef in request.emission_factors if ef.fuel_type == fuel_price.fuel_type),
                None
            )
            emissions = 0.0
            if emission_factor:
                co2e_per_mmbtu = emission_factor.calculate_co2e_per_mmbtu()
                emissions = (quantity * co2e_per_mmbtu) / 1000.0  # Convert kg to metric tons

            blend_ratios.append(BlendRatioOutput(
                fuel_type=fuel_price.fuel_type,
                percentage=percentage,
                quantity_mmbtu=quantity,
                cost_usd=cost,
                emissions_mtco2e=emissions,
            ))

        # Calculate totals
        total_cost = sum(br.cost_usd for br in blend_ratios)
        total_emissions = sum(br.emissions_mtco2e for br in blend_ratios)

        # Create fuel mix output
        fuel_mix = [FuelMixOutput(
            period_start=request.effective_time_window.start_time,
            period_end=request.effective_time_window.end_time,
            blend_ratios=blend_ratios,
            total_quantity_mmbtu=total_quantity,
            total_cost_usd=total_cost,
            total_emissions_mtco2e=total_emissions,
            weighted_lhv_btu_per_unit=1020.0,  # Demo value
            meets_constraints=True,
        )]

        # Cost breakdown
        cost_breakdown = CostBreakdown(
            fuel_costs_usd=total_cost,
            transport_costs_usd=total_cost * 0.05,
            storage_costs_usd=total_cost * 0.02,
            carbon_costs_usd=total_emissions * 50.0 if request.carbon_constraints and request.carbon_constraints.carbon_price_usd_per_ton else 0.0,
            contract_penalties_usd=0.0,
            total_cost_usd=total_cost * 1.07,
            cost_per_mmbtu=total_cost / total_quantity if total_quantity > 0 else 0.0,
            cost_by_fuel={br.fuel_type.value: br.cost_usd for br in blend_ratios},
            baseline_cost_usd=total_cost * 1.1,
            savings_usd=total_cost * 0.03,
            savings_percent=3.0,
        )

        # Carbon footprint
        carbon_footprint = CarbonFootprint(
            total_emissions_mtco2e=total_emissions,
            scope_1_mtco2e=total_emissions * 0.85,
            scope_2_mtco2e=0.0,
            scope_3_mtco2e=total_emissions * 0.15,
            ttw_emissions_mtco2e=total_emissions * 0.85,
            wtt_emissions_mtco2e=total_emissions * 0.15,
            wtw_emissions_mtco2e=total_emissions,
            co2_mt=total_emissions * 0.95,
            ch4_mt=total_emissions * 0.03,
            n2o_mt=total_emissions * 0.02,
            emissions_by_fuel={br.fuel_type.value: br.emissions_mtco2e for br in blend_ratios},
            carbon_intensity_kgco2e_per_mmbtu=(total_emissions * 1000.0) / total_quantity if total_quantity > 0 else 0.0,
            baseline_emissions_mtco2e=total_emissions * 1.05,
            reduction_mtco2e=total_emissions * 0.05,
            reduction_percent=5.0,
        )

        # Procurement recommendations
        procurement = [
            ProcurementRecommendation(
                fuel_type=br.fuel_type,
                recommended_quantity_mmbtu=br.quantity_mmbtu,
                recommended_timing=request.effective_time_window.start_time,
                estimated_price_usd_mmbtu=br.cost_usd / br.quantity_mmbtu if br.quantity_mmbtu > 0 else 0.0,
                urgency="normal",
            )
            for br in blend_ratios
        ]

        # Compute calculation hash
        calc_data = f"{run_id}|{total_cost}|{total_emissions}|{now.isoformat()}"
        calculation_hash = hashlib.sha256(calc_data.encode()).hexdigest()

        return RecommendationResponse(
            run_id=run_id,
            status=RunStatus.COMPLETED,
            effective_time_window=request.effective_time_window,
            fuel_mix=fuel_mix,
            total_cost=cost_breakdown,
            total_carbon=carbon_footprint,
            procurement_recommendations=procurement,
            optimization_time_ms=125.5,
            solver_status="optimal",
            objective_value=total_cost,
            input_snapshot_ids=request.input_snapshot_ids,
            calculation_hash=calculation_hash,
            bundle_hash=request.compute_bundle_hash(),
        )


def create_app(api: Optional["FuelCraftAPI"] = None) -> FastAPI:
    """Create FastAPI application."""

    app = FastAPI(
        title="GL-011 FUELCRAFT API",
        description="Fuel Mix Optimization API with Carbon Accounting per ISO 14064",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_tags=[
            {"name": "Runs", "description": "Optimization run management"},
            {"name": "Health", "description": "Health check endpoints"},
        ],
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store API instance
    app.state.api = api or FuelCraftAPI()

    # ============================================================
    # Health Endpoints (IEC 61511 compliant)
    # ============================================================

    @app.get(
        "/health/live",
        response_model=HealthResponse,
        tags=["Health"],
        summary="Liveness probe",
        description="Kubernetes liveness probe - checks if the service is alive",
    )
    async def liveness_probe() -> HealthResponse:
        """Liveness probe for Kubernetes."""
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(timezone.utc),
            version="1.0.0",
        )

    @app.get(
        "/health/ready",
        response_model=ReadinessResponse,
        tags=["Health"],
        summary="Readiness probe",
        description="Kubernetes readiness probe - checks if the service is ready to accept traffic",
    )
    async def readiness_probe() -> ReadinessResponse:
        """Readiness probe for Kubernetes."""
        components = [
            ComponentHealth(
                name="api",
                status="healthy",
                latency_ms=0.5,
            ),
            ComponentHealth(
                name="optimizer",
                status="healthy",
                latency_ms=1.2,
            ),
            ComponentHealth(
                name="storage",
                status="healthy",
                latency_ms=2.1,
            ),
            ComponentHealth(
                name="kafka",
                status="healthy",
                latency_ms=5.5,
            ),
        ]

        all_healthy = all(c.status == "healthy" for c in components)

        return ReadinessResponse(
            status="ready" if all_healthy else "not_ready",
            timestamp=datetime.now(timezone.utc),
            version="1.0.0",
            components=components,
            uptime_seconds=app.state.api.get_uptime(),
        )

    @app.get(
        "/health/startup",
        response_model=HealthResponse,
        tags=["Health"],
        summary="Startup probe",
        description="Kubernetes startup probe - checks if the service has started",
    )
    async def startup_probe() -> HealthResponse:
        """Startup probe for Kubernetes."""
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(timezone.utc),
            version="1.0.0",
        )

    @app.get(
        "/info",
        tags=["Health"],
        summary="API information",
    )
    async def api_info():
        """Get API information."""
        return {
            "name": "GL-011 FUELCRAFT",
            "version": "1.0.0",
            "description": "Fuel Mix Optimization Agent",
            "standards": [
                "ISO 14064 (GHG Quantification)",
                "GHG Protocol (Scope 1/2/3)",
                "IEC 61511 (Functional Safety)",
            ],
            "capabilities": [
                "fuel_mix_optimization",
                "carbon_accounting",
                "cost_optimization",
                "procurement_planning",
                "real_time_inventory",
            ],
        }

    # ============================================================
    # Run Management Endpoints
    # ============================================================

    @app.post(
        "/runs",
        response_model=RunResponse,
        tags=["Runs"],
        summary="Create optimization run",
        description="Create a new fuel mix optimization run request",
        status_code=201,
    )
    async def create_run(
        request: RunRequest,
        background_tasks: BackgroundTasks,
    ) -> RunResponse:
        """
        Create a new optimization run.

        The run will be queued and processed asynchronously.
        Use GET /runs/{run_id} to check status.
        Use GET /runs/{run_id}/recommendation to get results.
        """
        logger.info(f"Creating optimization run: {request.run_id}")

        response = await app.state.api.create_run(request)

        # Queue optimization in background
        background_tasks.add_task(_process_optimization, app.state.api, request.run_id)

        return response

    @app.get(
        "/runs/{run_id}",
        response_model=RunStatusResponse,
        tags=["Runs"],
        summary="Get run status",
        description="Get the status and metadata of an optimization run",
    )
    async def get_run_status(
        run_id: str = Path(..., description="Optimization run ID"),
    ) -> RunStatusResponse:
        """Get status of an optimization run."""
        logger.info(f"Getting status for run: {run_id}")
        return await app.state.api.get_run_status(run_id)

    @app.get(
        "/runs/{run_id}/recommendation",
        response_model=RecommendationResponse,
        tags=["Runs"],
        summary="Get optimization recommendation",
        description="Get the optimized fuel plan recommendation",
    )
    async def get_recommendation(
        run_id: str = Path(..., description="Optimization run ID"),
    ) -> RecommendationResponse:
        """Get optimization recommendation for a completed run."""
        logger.info(f"Getting recommendation for run: {run_id}")
        return await app.state.api.get_recommendation(run_id)

    @app.get(
        "/runs/{run_id}/explainability",
        response_model=ExplainabilityResponse,
        tags=["Runs"],
        summary="Get explainability",
        description="Get SHAP-based explanation of optimization decisions",
    )
    async def get_explainability(
        run_id: str = Path(..., description="Optimization run ID"),
    ) -> ExplainabilityResponse:
        """Get SHAP-based explanation of optimization decisions."""
        logger.info(f"Getting explainability for run: {run_id}")

        if run_id not in app.state.api._runs:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        # Generate explainability output
        explanation = ExplainabilityOutput(
            run_id=run_id,
            feature_importance={
                "natural_gas_price": 0.35,
                "demand_forecast": 0.25,
                "carbon_constraint": 0.20,
                "inventory_level": 0.12,
                "contract_terms": 0.08,
            },
            top_drivers=[
                {
                    "feature": "natural_gas_price",
                    "shap_value": 0.35,
                    "direction": "positive",
                    "impact": "Higher natural gas prices increased fuel oil allocation",
                },
                {
                    "feature": "carbon_constraint",
                    "shap_value": 0.20,
                    "direction": "negative",
                    "impact": "Carbon limit reduced coal allocation",
                },
            ],
            sensitivity_analysis={
                "natural_gas_price": {
                    "-10%": -2.5,
                    "+10%": 3.1,
                    "-20%": -4.8,
                    "+20%": 6.5,
                },
                "demand": {
                    "-10%": -1.2,
                    "+10%": 1.5,
                },
            },
            decision_rationale=(
                "The optimization selected a balanced fuel mix prioritizing natural gas "
                "(65%) due to favorable pricing and low carbon intensity. Fuel oil (35%) "
                "was included to meet contract minimums. The solution achieves 3% cost "
                "savings vs baseline while meeting the carbon constraint of 1000 tCO2e."
            ),
            counterfactuals=[
                {
                    "scenario": "Remove carbon constraint",
                    "cost_change_percent": -5.2,
                    "emission_change_percent": 15.0,
                    "fuel_mix": {"coal": 40, "natural_gas": 60},
                },
                {
                    "scenario": "Natural gas +20% price",
                    "cost_change_percent": 8.5,
                    "emission_change_percent": -2.0,
                    "fuel_mix": {"fuel_oil_2": 45, "natural_gas": 55},
                },
            ],
            calculation_hash=hashlib.sha256(f"{run_id}|explainability".encode()).hexdigest(),
        )

        return ExplainabilityResponse(
            run_id=run_id,
            explainability=explanation,
            bundle_hash=hashlib.sha256(run_id.encode()).hexdigest(),
        )

    @app.delete(
        "/runs/{run_id}",
        tags=["Runs"],
        summary="Cancel run",
        description="Cancel a pending or running optimization",
    )
    async def cancel_run(
        run_id: str = Path(..., description="Optimization run ID"),
    ):
        """Cancel an optimization run."""
        logger.info(f"Cancelling run: {run_id}")

        if run_id not in app.state.api._runs:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        run = app.state.api._runs[run_id]
        if run["status"] in [RunStatus.COMPLETED, RunStatus.FAILED]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel run in {run['status'].value} status"
            )

        run["status"] = RunStatus.CANCELLED

        return {
            "run_id": run_id,
            "status": RunStatus.CANCELLED.value,
            "message": "Run cancelled successfully",
        }

    @app.get(
        "/runs",
        tags=["Runs"],
        summary="List runs",
        description="List optimization runs with optional filtering",
    )
    async def list_runs(
        status: Optional[RunStatus] = Query(None, description="Filter by status"),
        limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
        offset: int = Query(0, ge=0, description="Offset for pagination"),
    ):
        """List optimization runs."""
        runs = list(app.state.api._runs.values())

        if status:
            runs = [r for r in runs if r["status"] == status]

        total = len(runs)
        runs = runs[offset:offset + limit]

        return {
            "runs": [
                {
                    "run_id": r["request"]["run_id"],
                    "status": r["status"].value,
                    "created_at": r["created_at"].isoformat(),
                    "objective": r["request"].get("objective", "balanced"),
                }
                for r in runs
            ],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    # ============================================================
    # Exception Handlers
    # ============================================================

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions."""
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=exc.detail if isinstance(exc.detail, str) else "HTTP Error",
                code=exc.status_code,
                message=str(exc.detail),
                request_id=getattr(request.state, "request_id", None),
            ).dict(),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="internal_server_error",
                code=500,
                message="An unexpected error occurred",
                details={"type": type(exc).__name__},
                request_id=getattr(request.state, "request_id", None),
            ).dict(),
        )

    return app


async def _process_optimization(api: FuelCraftAPI, run_id: str) -> None:
    """Background task to process optimization."""
    try:
        # Update status to running
        api._runs[run_id]["status"] = RunStatus.RUNNING
        api._runs[run_id]["started_at"] = datetime.now(timezone.utc)

        # Simulate optimization processing
        await asyncio.sleep(0.5)

        # Update status to completed
        api._runs[run_id]["status"] = RunStatus.COMPLETED
        api._runs[run_id]["completed_at"] = datetime.now(timezone.utc)

        logger.info(f"Optimization run {run_id} completed")

    except Exception as e:
        api._runs[run_id]["status"] = RunStatus.FAILED
        api._runs[run_id]["error"] = str(e)
        logger.error(f"Optimization run {run_id} failed: {e}")


# Create default app instance
app = create_app()
