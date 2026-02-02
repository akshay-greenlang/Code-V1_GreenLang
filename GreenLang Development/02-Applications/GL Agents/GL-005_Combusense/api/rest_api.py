"""
GL-005 COMBUSENSE REST API

FastAPI-based REST endpoints for combustion control and optimization.

Endpoints:
- /health - Health check
- /ready - Readiness probe
- /api/v1/optimize - Run combustion optimization
- /api/v1/control/pid - PID control setpoints
- /api/v1/emissions - Emissions calculations
- /api/v1/stability - Combustion stability analysis
- /api/v1/audit/{hash} - Audit record retrieval
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger(__name__)


# =============================================================================
# Request/Response Models
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str = "1.0.0"
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ReadinessResponse(BaseModel):
    """Readiness probe response."""
    ready: bool = True
    checks: Dict[str, bool] = Field(default_factory=dict)


class CombustionInputs(BaseModel):
    """Combustion optimization input data."""
    burner_id: str = Field(..., description="Burner identifier")
    fuel_type: str = Field("natural_gas", description="Fuel type")
    fuel_flow_kg_h: float = Field(..., gt=0, description="Fuel flow rate (kg/h)")
    air_flow_kg_h: float = Field(..., gt=0, description="Air flow rate (kg/h)")
    stack_o2_percent: float = Field(..., ge=0, le=21, description="Stack O2 (%)")
    stack_co_ppm: Optional[float] = Field(None, ge=0, description="Stack CO (ppm)")
    stack_temp_c: float = Field(..., ge=0, description="Stack temperature (C)")
    ambient_temp_c: float = Field(25.0, description="Ambient temperature (C)")


class OptimizationResult(BaseModel):
    """Combustion optimization result."""
    result_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    burner_id: str

    # Current state
    current_efficiency_percent: float
    current_excess_air_percent: float
    current_lambda: float

    # Recommended setpoints
    recommended_o2_setpoint: float
    recommended_air_fuel_ratio: float
    expected_efficiency_percent: float

    # Savings potential
    fuel_savings_percent: float
    annual_savings_usd: float
    co2_reduction_kg_year: float

    # Provenance
    computation_hash: str
    processing_time_ms: float

    model_config = ConfigDict(use_enum_values=True)


class PIDSetpointRequest(BaseModel):
    """PID control setpoint request."""
    controller_id: str = Field(..., description="Controller identifier")
    setpoint: float = Field(..., description="New setpoint value")
    setpoint_type: str = Field(..., description="Type: o2, temperature, pressure")
    ramp_rate: Optional[float] = Field(None, description="Ramp rate per minute")


class PIDSetpointResponse(BaseModel):
    """PID control setpoint response."""
    controller_id: str
    previous_setpoint: float
    new_setpoint: float
    accepted: bool
    reason: Optional[str] = None
    estimated_settling_time_s: float


class EmissionsRequest(BaseModel):
    """Emissions calculation request."""
    burner_id: str
    fuel_type: str
    fuel_flow_kg_h: float
    stack_o2_percent: float
    stack_temp_c: float
    operating_hours: float = 8760.0


class EmissionsResult(BaseModel):
    """Emissions calculation result."""
    result_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    burner_id: str

    # Emissions
    co2_kg_h: float
    nox_kg_h: float
    so2_kg_h: float
    co2_annual_tonnes: float
    nox_annual_kg: float

    # Compliance
    nox_compliant: bool
    co_compliant: bool
    compliance_standard: str = "EPA 40 CFR Part 60"

    # Provenance
    computation_hash: str


class StabilityRequest(BaseModel):
    """Combustion stability analysis request."""
    burner_id: str
    flame_temperature_readings: List[float] = Field(..., min_length=10)
    o2_readings: List[float] = Field(..., min_length=10)
    pressure_readings: List[float] = Field(..., min_length=10)
    sample_rate_hz: float = 1.0


class StabilityResult(BaseModel):
    """Combustion stability analysis result."""
    result_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    burner_id: str

    # Stability metrics
    combustion_quality_index: float = Field(..., ge=0, le=100)
    flame_stability_index: float = Field(..., ge=0, le=100)
    o2_variability_percent: float
    pressure_variability_percent: float

    # Assessment
    stability_status: str  # stable, marginal, unstable
    risk_factors: List[str]
    recommendations: List[str]

    # Provenance
    computation_hash: str


class AuditRecord(BaseModel):
    """Audit record response."""
    computation_hash: str
    timestamp: str
    calculation_type: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    execution_time_ms: float
    version: str


# =============================================================================
# FastAPI Application
# =============================================================================

def create_app() -> FastAPI:
    """Create and configure FastAPI application."""

    app = FastAPI(
        title="GL-005 COMBUSENSE API",
        description="Combustion Control & Optimization Agent",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # In-memory audit store (production: use persistent storage)
    audit_store: Dict[str, AuditRecord] = {}

    def compute_hash(data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash for provenance."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    # -------------------------------------------------------------------------
    # Health Endpoints
    # -------------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check():
        """Health check endpoint."""
        return HealthResponse()

    @app.get("/ready", response_model=ReadinessResponse, tags=["Health"])
    async def readiness_probe():
        """Readiness probe for Kubernetes."""
        checks = {
            "database": True,  # Would check actual DB connection
            "kafka": True,     # Would check Kafka connectivity
            "calculators": True,
        }
        return ReadinessResponse(ready=all(checks.values()), checks=checks)

    # -------------------------------------------------------------------------
    # Optimization Endpoints
    # -------------------------------------------------------------------------

    @app.post("/api/v1/optimize", response_model=OptimizationResult, tags=["Optimization"])
    async def optimize_combustion(inputs: CombustionInputs):
        """
        Run combustion optimization analysis.

        Calculates optimal O2 setpoint and air-fuel ratio for maximum efficiency
        while maintaining emissions compliance.
        """
        start_time = time.time()

        # Calculate stoichiometric values
        stoich_afr = 17.2 if inputs.fuel_type == "natural_gas" else 14.7
        actual_afr = inputs.air_flow_kg_h / inputs.fuel_flow_kg_h
        current_lambda = actual_afr / stoich_afr
        current_excess_air = (current_lambda - 1) * 100

        # Calculate current efficiency (simplified Siegert formula)
        stack_loss = 0.38 * (inputs.stack_temp_c - inputs.ambient_temp_c) / (21 - inputs.stack_o2_percent)
        current_efficiency = 100 - stack_loss - 2  # -2% for other losses

        # Optimize: find O2 setpoint that balances efficiency vs CO
        optimal_o2 = 2.5 if inputs.stack_co_ppm is None or inputs.stack_co_ppm < 50 else 3.0
        optimal_lambda = 1 + (optimal_o2 / (21 - optimal_o2))
        optimal_afr = stoich_afr * optimal_lambda

        # Calculate expected efficiency at optimal O2
        optimal_stack_loss = 0.38 * (inputs.stack_temp_c - inputs.ambient_temp_c) / (21 - optimal_o2)
        expected_efficiency = 100 - optimal_stack_loss - 2

        # Calculate savings
        efficiency_improvement = expected_efficiency - current_efficiency
        fuel_savings_percent = efficiency_improvement / current_efficiency * 100 if current_efficiency > 0 else 0
        fuel_cost_usd_kg = 0.5
        annual_fuel_kg = inputs.fuel_flow_kg_h * 8760
        annual_savings = annual_fuel_kg * fuel_cost_usd_kg * fuel_savings_percent / 100
        co2_factor = 2.75  # kg CO2 per kg natural gas
        co2_reduction = annual_fuel_kg * fuel_savings_percent / 100 * co2_factor

        processing_time = (time.time() - start_time) * 1000

        # Compute provenance hash
        inputs_dict = inputs.model_dump()
        computation_hash = compute_hash(inputs_dict)

        result = OptimizationResult(
            burner_id=inputs.burner_id,
            current_efficiency_percent=round(current_efficiency, 2),
            current_excess_air_percent=round(current_excess_air, 2),
            current_lambda=round(current_lambda, 4),
            recommended_o2_setpoint=optimal_o2,
            recommended_air_fuel_ratio=round(optimal_afr, 2),
            expected_efficiency_percent=round(expected_efficiency, 2),
            fuel_savings_percent=round(fuel_savings_percent, 2),
            annual_savings_usd=round(annual_savings, 2),
            co2_reduction_kg_year=round(co2_reduction, 2),
            computation_hash=computation_hash,
            processing_time_ms=round(processing_time, 2)
        )

        # Store audit record
        audit_store[computation_hash] = AuditRecord(
            computation_hash=computation_hash,
            timestamp=result.timestamp,
            calculation_type="combustion_optimization",
            inputs=inputs_dict,
            outputs=result.model_dump(),
            execution_time_ms=processing_time,
            version="1.0.0"
        )

        return result

    # -------------------------------------------------------------------------
    # PID Control Endpoints
    # -------------------------------------------------------------------------

    @app.post("/api/v1/control/pid", response_model=PIDSetpointResponse, tags=["Control"])
    async def update_pid_setpoint(request: PIDSetpointRequest):
        """
        Update PID controller setpoint.

        Validates setpoint against safety limits before acceptance.
        """
        # Safety limits by type
        limits = {
            "o2": (1.5, 8.0),
            "temperature": (100, 500),
            "pressure": (0.5, 10.0),
        }

        if request.setpoint_type not in limits:
            raise HTTPException(400, f"Unknown setpoint type: {request.setpoint_type}")

        min_val, max_val = limits[request.setpoint_type]

        if not (min_val <= request.setpoint <= max_val):
            return PIDSetpointResponse(
                controller_id=request.controller_id,
                previous_setpoint=3.0,  # Mock current value
                new_setpoint=request.setpoint,
                accepted=False,
                reason=f"Setpoint {request.setpoint} outside limits [{min_val}, {max_val}]",
                estimated_settling_time_s=0
            )

        # Estimate settling time based on ramp rate
        previous_setpoint = 3.0  # Mock
        delta = abs(request.setpoint - previous_setpoint)
        ramp_rate = request.ramp_rate or 1.0
        settling_time = delta / ramp_rate * 60 + 30  # +30s for settling

        return PIDSetpointResponse(
            controller_id=request.controller_id,
            previous_setpoint=previous_setpoint,
            new_setpoint=request.setpoint,
            accepted=True,
            estimated_settling_time_s=round(settling_time, 1)
        )

    # -------------------------------------------------------------------------
    # Emissions Endpoints
    # -------------------------------------------------------------------------

    @app.post("/api/v1/emissions", response_model=EmissionsResult, tags=["Emissions"])
    async def calculate_emissions(request: EmissionsRequest):
        """
        Calculate emissions for GHG reporting.

        Uses EPA 40 CFR Part 98 emission factors.
        """
        # EPA emission factors for natural gas (kg/kg fuel)
        co2_factor = 2.75
        nox_factor = 0.0016  # Varies with combustion conditions
        so2_factor = 0.00001  # Very low for natural gas

        co2_kg_h = request.fuel_flow_kg_h * co2_factor
        nox_kg_h = request.fuel_flow_kg_h * nox_factor * (1 + request.stack_o2_percent / 10)
        so2_kg_h = request.fuel_flow_kg_h * so2_factor

        co2_annual = co2_kg_h * request.operating_hours / 1000  # tonnes
        nox_annual = nox_kg_h * request.operating_hours

        # Compliance check (example limits)
        nox_limit_kg_h = 0.05 * request.fuel_flow_kg_h / 100
        nox_compliant = nox_kg_h <= nox_limit_kg_h

        computation_hash = compute_hash(request.model_dump())

        return EmissionsResult(
            burner_id=request.burner_id,
            co2_kg_h=round(co2_kg_h, 2),
            nox_kg_h=round(nox_kg_h, 4),
            so2_kg_h=round(so2_kg_h, 6),
            co2_annual_tonnes=round(co2_annual, 2),
            nox_annual_kg=round(nox_annual, 2),
            nox_compliant=nox_compliant,
            co_compliant=True,  # Would check actual CO measurements
            computation_hash=computation_hash
        )

    # -------------------------------------------------------------------------
    # Stability Analysis Endpoints
    # -------------------------------------------------------------------------

    @app.post("/api/v1/stability", response_model=StabilityResult, tags=["Stability"])
    async def analyze_stability(request: StabilityRequest):
        """
        Analyze combustion stability from sensor data.

        Calculates Combustion Quality Index (CQI) and identifies risk factors.
        """
        import statistics

        # Calculate variability metrics
        temp_std = statistics.stdev(request.flame_temperature_readings)
        temp_mean = statistics.mean(request.flame_temperature_readings)
        temp_cv = (temp_std / temp_mean * 100) if temp_mean > 0 else 0

        o2_std = statistics.stdev(request.o2_readings)
        o2_mean = statistics.mean(request.o2_readings)
        o2_cv = (o2_std / o2_mean * 100) if o2_mean > 0 else 0

        pressure_std = statistics.stdev(request.pressure_readings)
        pressure_mean = statistics.mean(request.pressure_readings)
        pressure_cv = (pressure_std / pressure_mean * 100) if pressure_mean > 0 else 0

        # Calculate Combustion Quality Index (CQI)
        # Lower variability = higher quality
        cqi = max(0, 100 - (temp_cv * 2 + o2_cv * 3 + pressure_cv * 2))

        # Flame Stability Index
        flame_stability = max(0, 100 - temp_cv * 5)

        # Determine status
        if cqi >= 80:
            status = "stable"
        elif cqi >= 60:
            status = "marginal"
        else:
            status = "unstable"

        # Identify risk factors
        risk_factors = []
        recommendations = []

        if temp_cv > 5:
            risk_factors.append("High flame temperature variability")
            recommendations.append("Check burner alignment and fuel quality")

        if o2_cv > 10:
            risk_factors.append("Unstable O2 readings")
            recommendations.append("Inspect damper actuators and air leaks")

        if pressure_cv > 5:
            risk_factors.append("Pressure fluctuations detected")
            recommendations.append("Check fuel supply pressure regulation")

        computation_hash = compute_hash(request.model_dump())

        return StabilityResult(
            burner_id=request.burner_id,
            combustion_quality_index=round(cqi, 1),
            flame_stability_index=round(flame_stability, 1),
            o2_variability_percent=round(o2_cv, 2),
            pressure_variability_percent=round(pressure_cv, 2),
            stability_status=status,
            risk_factors=risk_factors,
            recommendations=recommendations,
            computation_hash=computation_hash
        )

    # -------------------------------------------------------------------------
    # Audit Endpoints
    # -------------------------------------------------------------------------

    @app.get("/api/v1/audit/{hash}", response_model=AuditRecord, tags=["Audit"])
    async def get_audit_record(hash: str):
        """
        Retrieve audit record by computation hash.

        Provides complete provenance trail for regulatory compliance.
        """
        if hash not in audit_store:
            raise HTTPException(404, f"Audit record not found: {hash}")
        return audit_store[hash]

    return app


# Create default app instance
app = create_app()
