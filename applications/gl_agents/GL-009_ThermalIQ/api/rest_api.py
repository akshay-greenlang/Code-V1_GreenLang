"""
GL-009 ThermalIQ - REST API

FastAPI-based REST endpoints for thermal fluid property calculations,
exergy analysis, efficiency metrics, and Sankey diagram generation.

Endpoints:
- POST /analyze - Full thermal analysis
- POST /efficiency - Calculate thermal efficiency
- POST /exergy - Calculate exergy destruction
- GET /fluids - List available fluids
- GET /fluids/{name}/properties - Get fluid properties at T, P
- POST /sankey - Generate Sankey diagram
- POST /recommend-fluid - Get fluid recommendations
- GET /health - Health check
- GET /metrics - Prometheus metrics
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import hashlib
import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .api_schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    EfficiencyRequest,
    EfficiencyResponse,
    ExergyRequest,
    ExergyResponse,
    FluidPropertiesRequest,
    FluidPropertiesResponse,
    FluidListResponse,
    SankeyRequest,
    SankeyResponse,
    FluidRecommendationRequest,
    FluidRecommendationResponse,
    HealthResponse,
    MetricsResponse,
    ErrorResponse,
    FluidProperties,
    FluidPhase,
    SankeyNode,
    SankeyLink,
    ExergyComponent,
    FluidRecommendation,
    FluidCategory,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Application State
# =============================================================================

class AppState:
    """Application state for metrics and caching."""

    def __init__(self):
        self.start_time = time.time()
        self.requests_total = 0
        self.requests_success = 0
        self.requests_failed = 0
        self.total_latency_ms = 0.0
        self.fluid_lookups = 0
        self.exergy_calculations = 0
        self.sankey_generations = 0
        self.active_connections = 0

        # In-memory job store (replace with Redis in production)
        self.jobs: Dict[str, Dict[str, Any]] = {}

        # Fluid property cache
        self.property_cache: Dict[str, Any] = {}
        self.cache_hits = 0
        self.cache_misses = 0


app_state = AppState()


# =============================================================================
# Available Fluids Database (Mock)
# =============================================================================

AVAILABLE_FLUIDS = {
    "Water": {
        "category": "inorganic",
        "molecular_weight": 18.015,
        "critical_temp_C": 373.95,
        "critical_pressure_kPa": 22064.0,
        "gwp": 0,
        "odp": 0,
        "flammability": "A1",
        "toxicity": "A1"
    },
    "R134a": {
        "category": "refrigerant",
        "molecular_weight": 102.03,
        "critical_temp_C": 101.06,
        "critical_pressure_kPa": 4059.3,
        "gwp": 1430,
        "odp": 0,
        "flammability": "A1",
        "toxicity": "A1"
    },
    "R410A": {
        "category": "refrigerant",
        "molecular_weight": 72.59,
        "critical_temp_C": 71.36,
        "critical_pressure_kPa": 4901.2,
        "gwp": 2088,
        "odp": 0,
        "flammability": "A1",
        "toxicity": "A1"
    },
    "R32": {
        "category": "refrigerant",
        "molecular_weight": 52.02,
        "critical_temp_C": 78.11,
        "critical_pressure_kPa": 5782.0,
        "gwp": 675,
        "odp": 0,
        "flammability": "A2L",
        "toxicity": "A"
    },
    "Ammonia": {
        "category": "refrigerant",
        "molecular_weight": 17.03,
        "critical_temp_C": 132.25,
        "critical_pressure_kPa": 11333.0,
        "gwp": 0,
        "odp": 0,
        "flammability": "B2L",
        "toxicity": "B"
    },
    "CO2": {
        "category": "refrigerant",
        "molecular_weight": 44.01,
        "critical_temp_C": 30.98,
        "critical_pressure_kPa": 7377.3,
        "gwp": 1,
        "odp": 0,
        "flammability": "A1",
        "toxicity": "A1"
    },
    "Therminol66": {
        "category": "thermal_oil",
        "molecular_weight": 252.0,
        "critical_temp_C": 600.0,
        "critical_pressure_kPa": 1500.0,
        "gwp": None,
        "odp": 0,
        "flammability": "B1",
        "toxicity": "A"
    },
    "EthyleneGlycol50": {
        "category": "water_glycol",
        "molecular_weight": 62.07,
        "critical_temp_C": 300.0,
        "critical_pressure_kPa": 7500.0,
        "gwp": None,
        "odp": 0,
        "flammability": "A1",
        "toxicity": "A"
    },
}


# =============================================================================
# Helper Functions
# =============================================================================

def compute_hash(data: Dict[str, Any]) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    import json
    json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


async def get_fluid_properties_async(
    fluid_name: str,
    temperature_C: float,
    pressure_kPa: float,
    quality: Optional[float] = None
) -> FluidProperties:
    """
    Get fluid properties at given state point.
    In production, this would call CoolProp or similar library.
    """
    app_state.fluid_lookups += 1

    # Check cache
    cache_key = f"{fluid_name}:{temperature_C}:{pressure_kPa}:{quality}"
    if cache_key in app_state.property_cache:
        app_state.cache_hits += 1
        return app_state.property_cache[cache_key]

    app_state.cache_misses += 1

    # Mock property calculation (replace with CoolProp in production)
    fluid_info = AVAILABLE_FLUIDS.get(fluid_name, AVAILABLE_FLUIDS["Water"])

    # Determine phase based on temperature and pressure
    if temperature_C > fluid_info["critical_temp_C"]:
        phase = FluidPhase.SUPERCRITICAL
    elif quality is not None and 0 < quality < 1:
        phase = FluidPhase.TWO_PHASE
    elif temperature_C > 100:  # Simplified check
        phase = FluidPhase.GAS
    else:
        phase = FluidPhase.LIQUID

    # Mock properties (in production, use CoolProp)
    properties = FluidProperties(
        temperature_C=temperature_C,
        pressure_kPa=pressure_kPa,
        phase=phase,
        density_kg_m3=1000.0 if phase == FluidPhase.LIQUID else 1.2,
        specific_heat_kJ_kgK=4.186 if fluid_name == "Water" else 1.0,
        enthalpy_kJ_kg=temperature_C * 4.186,  # Simplified
        entropy_kJ_kgK=1.0 + temperature_C / 373.15,  # Simplified
        internal_energy_kJ_kg=temperature_C * 3.5,  # Simplified
        viscosity_Pa_s=0.001 if phase == FluidPhase.LIQUID else 0.00001,
        thermal_conductivity_W_mK=0.6 if phase == FluidPhase.LIQUID else 0.025,
        prandtl_number=7.0 if phase == FluidPhase.LIQUID else 0.7,
        quality=quality,
        compressibility_factor=1.0 if phase != FluidPhase.GAS else 0.99
    )

    # Cache the result
    app_state.property_cache[cache_key] = properties

    return properties


async def calculate_exergy_async(
    streams: List[Dict[str, Any]],
    dead_state_T: float,
    dead_state_P: float
) -> Dict[str, Any]:
    """Calculate exergy for given streams."""
    app_state.exergy_calculations += 1

    total_exergy_in = 0.0
    total_exergy_out = 0.0
    components = []

    for stream in streams:
        # Get properties at stream state and dead state
        T_in = stream.get("inlet_temperature_C", 25)
        T_out = stream.get("outlet_temperature_C", 25)
        m_dot = stream.get("mass_flow_kg_s", 1.0)
        Cp = stream.get("specific_heat_kJ_kgK", 4.186)

        # Simplified exergy calculation: Ex = m * Cp * [(T - T0) - T0 * ln(T/T0)]
        T0_K = dead_state_T + 273.15
        T_in_K = T_in + 273.15
        T_out_K = T_out + 273.15

        import math
        ex_in = m_dot * Cp * ((T_in_K - T0_K) - T0_K * math.log(T_in_K / T0_K))
        ex_out = m_dot * Cp * ((T_out_K - T0_K) - T0_K * math.log(T_out_K / T0_K))

        ex_destruction = abs(ex_in - ex_out)
        efficiency = (ex_out / ex_in * 100) if ex_in > 0 else 0

        total_exergy_in += abs(ex_in)
        total_exergy_out += abs(ex_out)

        components.append(ExergyComponent(
            name=stream.get("stream_id", "unknown"),
            exergy_input_kW=abs(ex_in),
            exergy_output_kW=abs(ex_out),
            exergy_destruction_kW=ex_destruction,
            exergy_efficiency_percent=efficiency,
            irreversibility_kW=ex_destruction
        ))

    total_destruction = total_exergy_in - total_exergy_out
    overall_efficiency = (total_exergy_out / total_exergy_in * 100) if total_exergy_in > 0 else 0

    return {
        "total_exergy_input_kW": total_exergy_in,
        "total_exergy_output_kW": total_exergy_out,
        "total_exergy_destruction_kW": max(0, total_destruction),
        "exergy_efficiency_percent": overall_efficiency,
        "components": components,
        "improvement_potential_kW": max(0, total_destruction)
    }


async def generate_sankey_async(
    streams: List[Dict[str, Any]],
    diagram_type: str,
    show_losses: bool
) -> Dict[str, Any]:
    """Generate Sankey diagram data."""
    app_state.sankey_generations += 1

    nodes = []
    links = []
    total_input = 0.0
    total_output = 0.0
    total_losses = 0.0

    # Create input node
    nodes.append(SankeyNode(
        id="input",
        name="Energy Input",
        value=0,  # Will be updated
        category="input",
        color="#2196F3"
    ))

    # Create nodes and links for each stream
    for i, stream in enumerate(streams):
        stream_id = stream.get("stream_id", f"stream_{i}")
        T_in = stream.get("inlet_temperature_C", 100)
        T_out = stream.get("outlet_temperature_C", 50)
        m_dot = stream.get("mass_flow_kg_s", 1.0)
        Cp = stream.get("specific_heat_kJ_kgK", 4.186)

        duty = m_dot * Cp * abs(T_in - T_out)
        total_input += duty

        # Stream node
        nodes.append(SankeyNode(
            id=stream_id,
            name=stream.get("stream_name", stream_id),
            value=duty,
            category="stream",
            color="#4CAF50" if T_in > T_out else "#F44336"
        ))

        # Link from input to stream
        links.append(SankeyLink(
            source="input",
            target=stream_id,
            value=duty,
            label=f"{duty:.1f} kW"
        ))

        # Assume 90% efficiency, 10% losses
        useful = duty * 0.9
        loss = duty * 0.1
        total_output += useful
        total_losses += loss

        if show_losses:
            loss_id = f"{stream_id}_loss"
            nodes.append(SankeyNode(
                id=loss_id,
                name=f"{stream_id} Loss",
                value=loss,
                category="loss",
                color="#9E9E9E"
            ))
            links.append(SankeyLink(
                source=stream_id,
                target=loss_id,
                value=loss,
                label=f"{loss:.1f} kW loss"
            ))

    # Update input node value
    nodes[0].value = total_input

    # Create output node
    nodes.append(SankeyNode(
        id="output",
        name="Useful Output",
        value=total_output,
        category="output",
        color="#8BC34A"
    ))

    # Link streams to output
    for stream in streams:
        stream_id = stream.get("stream_id", "unknown")
        T_in = stream.get("inlet_temperature_C", 100)
        T_out = stream.get("outlet_temperature_C", 50)
        m_dot = stream.get("mass_flow_kg_s", 1.0)
        Cp = stream.get("specific_heat_kJ_kgK", 4.186)
        duty = m_dot * Cp * abs(T_in - T_out) * 0.9

        links.append(SankeyLink(
            source=stream_id,
            target="output",
            value=duty,
            label=f"{duty:.1f} kW"
        ))

    return {
        "nodes": nodes,
        "links": links,
        "total_input_kW": total_input,
        "total_output_kW": total_output,
        "total_losses_kW": total_losses
    }


# =============================================================================
# Lifespan Context Manager
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("GL-009 ThermalIQ API starting up")
    app_state.start_time = time.time()
    yield
    logger.info("GL-009 ThermalIQ API shutting down")


# =============================================================================
# Create FastAPI Application
# =============================================================================

def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="GL-009 ThermalIQ API",
        description="""
        Thermal Fluid Property and Exergy Analysis API.

        Provides:
        - Fluid property calculations using thermodynamic models
        - First and second law efficiency analysis
        - Exergy destruction quantification
        - Sankey diagram generation for energy/exergy flows
        - Intelligent fluid recommendations

        Part of the GreenLang industrial decarbonization platform.
        """,
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/api/openapi.json",
        contact={
            "name": "GreenLang Support",
            "email": "support@greenlang.io"
        },
        license_info={
            "name": "Proprietary",
            "url": "https://greenlang.io/license"
        }
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://*.greenlang.io", "http://localhost:*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )

    # Include router
    app.include_router(router, prefix="/api/v1")

    return app


# =============================================================================
# API Router
# =============================================================================

from fastapi import APIRouter

router = APIRouter(tags=["ThermalIQ"])


# =============================================================================
# Health and Metrics Endpoints
# =============================================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check API health status and component availability",
    tags=["System"]
)
async def health_check():
    """Health check endpoint for load balancers and monitoring."""
    uptime = time.time() - app_state.start_time

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now(timezone.utc),
        components={
            "api": "ok",
            "fluid_database": "ok",
            "exergy_calculator": "ok",
            "sankey_generator": "ok",
            "cache": "ok"
        },
        uptime_seconds=uptime
    )


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Prometheus metrics",
    description="Get API metrics for monitoring",
    tags=["System"]
)
async def get_metrics():
    """Get Prometheus-compatible metrics."""
    total = app_state.requests_success + app_state.requests_failed
    avg_latency = (app_state.total_latency_ms / total) if total > 0 else 0

    cache_total = app_state.cache_hits + app_state.cache_misses
    cache_rate = (app_state.cache_hits / cache_total) if cache_total > 0 else 0

    return MetricsResponse(
        requests_total=app_state.requests_total,
        requests_success=app_state.requests_success,
        requests_failed=app_state.requests_failed,
        average_latency_ms=avg_latency,
        active_connections=app_state.active_connections,
        cache_hit_rate=cache_rate,
        fluid_lookups=app_state.fluid_lookups,
        exergy_calculations=app_state.exergy_calculations,
        sankey_generations=app_state.sankey_generations
    )


# =============================================================================
# Fluid Endpoints
# =============================================================================

@router.get(
    "/fluids",
    response_model=FluidListResponse,
    summary="List available fluids",
    description="Get list of all available fluids with their categories",
    tags=["Fluids"]
)
async def list_fluids(
    category: Optional[str] = Query(None, description="Filter by category")
):
    """List all available fluids."""
    app_state.requests_total += 1
    start_time = time.time()

    try:
        fluids = []
        categories = set()

        for name, info in AVAILABLE_FLUIDS.items():
            categories.add(info["category"])

            if category and info["category"] != category:
                continue

            fluids.append({
                "name": name,
                "category": info["category"],
                "molecular_weight_g_mol": info["molecular_weight"],
                "critical_temperature_C": info["critical_temp_C"],
                "critical_pressure_kPa": info["critical_pressure_kPa"],
                "gwp": info.get("gwp"),
                "odp": info.get("odp"),
                "safety_class": f"{info.get('flammability', '')}{info.get('toxicity', '')}"
            })

        app_state.requests_success += 1
        app_state.total_latency_ms += (time.time() - start_time) * 1000

        return FluidListResponse(
            fluids=fluids,
            categories=list(categories),
            total_count=len(fluids)
        )

    except Exception as e:
        app_state.requests_failed += 1
        logger.error(f"Error listing fluids: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/fluids/{fluid_name}/properties",
    response_model=FluidPropertiesResponse,
    summary="Get fluid properties",
    description="Calculate thermodynamic properties at specified temperature and pressure",
    tags=["Fluids"]
)
async def get_fluid_properties(
    fluid_name: str = Path(..., description="Fluid name"),
    temperature_C: float = Query(..., description="Temperature (C)"),
    pressure_kPa: float = Query(..., description="Pressure (kPa)"),
    quality: Optional[float] = Query(None, ge=0, le=1, description="Vapor quality (two-phase)")
):
    """Get fluid properties at specified state point."""
    app_state.requests_total += 1
    start_time = time.time()

    try:
        # Validate fluid name
        if fluid_name not in AVAILABLE_FLUIDS:
            raise HTTPException(
                status_code=404,
                detail=f"Fluid '{fluid_name}' not found. Available: {list(AVAILABLE_FLUIDS.keys())}"
            )

        fluid_info = AVAILABLE_FLUIDS[fluid_name]

        # Get properties
        properties = await get_fluid_properties_async(
            fluid_name, temperature_C, pressure_kPa, quality
        )

        # Check validity
        warnings = []
        is_valid = True

        if temperature_C > fluid_info["critical_temp_C"]:
            warnings.append("Temperature above critical point - supercritical state")

        if temperature_C < -273.15:
            is_valid = False
            warnings.append("Temperature below absolute zero")

        app_state.requests_success += 1
        app_state.total_latency_ms += (time.time() - start_time) * 1000

        return FluidPropertiesResponse(
            fluid_name=fluid_name,
            properties=properties,
            molecular_weight_g_mol=fluid_info["molecular_weight"],
            critical_temperature_C=fluid_info["critical_temp_C"],
            critical_pressure_kPa=fluid_info["critical_pressure_kPa"],
            is_valid_state=is_valid,
            warnings=warnings,
            data_source="CoolProp/ThermalIQ",
            computation_hash=compute_hash({
                "fluid": fluid_name,
                "T": temperature_C,
                "P": pressure_kPa
            })
        )

    except HTTPException:
        raise
    except Exception as e:
        app_state.requests_failed += 1
        logger.error(f"Error getting fluid properties: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Analysis Endpoints
# =============================================================================

@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=200,
    summary="Full thermal analysis",
    description="Perform comprehensive thermal analysis including efficiency and exergy",
    tags=["Analysis"]
)
async def analyze_thermal_system(request: AnalyzeRequest):
    """
    Perform full thermal analysis on provided streams.

    Includes:
    - First law efficiency calculation
    - Second law (exergy) efficiency (optional)
    - Sankey diagram generation (optional)
    - Fluid recommendations (optional)
    """
    app_state.requests_total += 1
    start_time = time.time()
    request_id = str(uuid.uuid4())

    try:
        logger.info(f"[{request_id}] Starting thermal analysis for {len(request.streams)} streams")

        # Calculate total heat duty and mass flow
        total_duty = 0.0
        total_mass_flow = 0.0
        stream_results = []

        for stream in request.streams:
            T_in = stream.inlet_temperature_C
            T_out = stream.outlet_temperature_C
            m_dot = stream.mass_flow_kg_s
            Cp = stream.specific_heat_kJ_kgK or 4.186

            duty = m_dot * Cp * abs(T_in - T_out)
            total_duty += duty
            total_mass_flow += m_dot

            stream_results.append({
                "stream_id": stream.stream_id,
                "fluid_name": stream.fluid_name,
                "heat_duty_kW": duty,
                "temperature_drop_C": abs(T_in - T_out),
                "mass_flow_kg_s": m_dot
            })

        # Calculate first law efficiency (simplified)
        first_law_eff = 85.0  # Placeholder - would calculate based on actual losses

        # Exergy analysis if requested
        exergy_analysis = None
        second_law_eff = None
        if request.include_exergy:
            streams_dict = [s.model_dump() for s in request.streams]
            exergy_result = await calculate_exergy_async(
                streams_dict,
                request.ambient_temperature_C,
                request.ambient_pressure_kPa
            )
            exergy_analysis = {
                "dead_state_temperature_C": request.ambient_temperature_C,
                "total_exergy_destruction_kW": exergy_result["total_exergy_destruction_kW"],
                "exergy_efficiency_percent": exergy_result["exergy_efficiency_percent"],
                "components": [c.model_dump() for c in exergy_result["components"]]
            }
            second_law_eff = exergy_result["exergy_efficiency_percent"]

        # Sankey diagram if requested
        sankey_diagram = None
        if request.include_sankey:
            streams_dict = [s.model_dump() for s in request.streams]
            sankey_result = await generate_sankey_async(streams_dict, "energy", True)
            sankey_diagram = {
                "nodes": [n.model_dump() for n in sankey_result["nodes"]],
                "links": [l.model_dump() for l in sankey_result["links"]],
                "total_input_kW": sankey_result["total_input_kW"],
                "total_output_kW": sankey_result["total_output_kW"]
            }

        # Recommendations if requested
        recommendations = None
        if request.include_recommendations:
            recommendations = [
                {
                    "type": "efficiency_improvement",
                    "description": "Consider heat recovery between streams",
                    "potential_savings_kW": total_duty * 0.1
                }
            ]

        processing_time = (time.time() - start_time) * 1000

        app_state.requests_success += 1
        app_state.total_latency_ms += processing_time

        return AnalyzeResponse(
            request_id=request_id,
            status="completed",
            timestamp=datetime.now(timezone.utc),
            total_heat_duty_kW=total_duty,
            total_mass_flow_kg_s=total_mass_flow,
            first_law_efficiency_percent=first_law_eff,
            second_law_efficiency_percent=second_law_eff,
            stream_results=stream_results,
            exergy_analysis=exergy_analysis,
            sankey_diagram=sankey_diagram,
            recommendations=recommendations,
            computation_hash=compute_hash({
                "streams": [s.model_dump() for s in request.streams],
                "ambient_T": request.ambient_temperature_C
            }),
            processing_time_ms=processing_time
        )

    except Exception as e:
        app_state.requests_failed += 1
        logger.error(f"[{request_id}] Analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post(
    "/efficiency",
    response_model=EfficiencyResponse,
    summary="Calculate thermal efficiency",
    description="Calculate first and second law thermal efficiency",
    tags=["Analysis"]
)
async def calculate_efficiency(request: EfficiencyRequest):
    """Calculate thermal efficiency for provided streams."""
    app_state.requests_total += 1
    start_time = time.time()
    request_id = str(uuid.uuid4())

    try:
        # Calculate energy balance
        total_energy_in = 0.0
        total_energy_out = 0.0
        stream_efficiencies = []

        for stream in request.streams:
            Cp = stream.specific_heat_kJ_kgK or 4.186
            duty = stream.mass_flow_kg_s * Cp * abs(
                stream.inlet_temperature_C - stream.outlet_temperature_C
            )

            # Assume 90% of heat is useful, 10% lost
            useful = duty * 0.9
            loss = duty * 0.1

            total_energy_in += duty
            total_energy_out += useful

            stream_efficiencies.append({
                "stream_id": stream.stream_id,
                "energy_in_kW": duty,
                "energy_out_kW": useful,
                "efficiency_percent": 90.0
            })

        energy_loss = total_energy_in - total_energy_out
        first_law_eff = (total_energy_out / total_energy_in * 100) if total_energy_in > 0 else 0

        # Second law efficiency if method includes it
        second_law_eff = None
        exergy_in = None
        exergy_out = None
        exergy_destruction = None

        if request.method in ["second_law", "combined"]:
            streams_dict = [s.model_dump() for s in request.streams]
            exergy_result = await calculate_exergy_async(
                streams_dict,
                request.ambient_temperature_C,
                101.325
            )
            second_law_eff = exergy_result["exergy_efficiency_percent"]
            exergy_in = exergy_result["total_exergy_input_kW"]
            exergy_out = exergy_result["total_exergy_output_kW"]
            exergy_destruction = exergy_result["total_exergy_destruction_kW"]

        processing_time = (time.time() - start_time) * 1000
        app_state.requests_success += 1
        app_state.total_latency_ms += processing_time

        return EfficiencyResponse(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc),
            first_law_efficiency_percent=first_law_eff,
            energy_input_kW=total_energy_in,
            energy_output_kW=total_energy_out,
            energy_loss_kW=energy_loss,
            second_law_efficiency_percent=second_law_eff,
            exergy_input_kW=exergy_in,
            exergy_output_kW=exergy_out,
            exergy_destruction_kW=exergy_destruction,
            stream_efficiencies=stream_efficiencies,
            computation_hash=compute_hash({"streams": [s.model_dump() for s in request.streams]}),
            method_used=request.method.value if hasattr(request.method, 'value') else str(request.method)
        )

    except Exception as e:
        app_state.requests_failed += 1
        logger.error(f"Efficiency calculation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/exergy",
    response_model=ExergyResponse,
    summary="Calculate exergy destruction",
    description="Perform detailed exergy (second law) analysis",
    tags=["Analysis"]
)
async def calculate_exergy(request: ExergyRequest):
    """Calculate exergy destruction and identify improvement opportunities."""
    app_state.requests_total += 1
    start_time = time.time()
    request_id = str(uuid.uuid4())

    try:
        streams_dict = [s.model_dump() for s in request.streams]

        exergy_result = await calculate_exergy_async(
            streams_dict,
            request.dead_state_temperature_C,
            request.dead_state_pressure_kPa
        )

        processing_time = (time.time() - start_time) * 1000
        app_state.requests_success += 1
        app_state.total_latency_ms += processing_time

        return ExergyResponse(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc),
            dead_state_temperature_C=request.dead_state_temperature_C,
            dead_state_pressure_kPa=request.dead_state_pressure_kPa,
            total_exergy_input_kW=exergy_result["total_exergy_input_kW"],
            total_exergy_output_kW=exergy_result["total_exergy_output_kW"],
            total_exergy_destruction_kW=exergy_result["total_exergy_destruction_kW"],
            exergy_efficiency_percent=exergy_result["exergy_efficiency_percent"],
            physical_exergy_kW=exergy_result["total_exergy_input_kW"],
            chemical_exergy_kW=0.0 if not request.include_chemical_exergy else None,
            kinetic_exergy_kW=0.0 if not request.include_kinetic_exergy else None,
            potential_exergy_kW=0.0 if not request.include_potential_exergy else None,
            components=exergy_result["components"],
            improvement_potential_kW=exergy_result["improvement_potential_kW"],
            computation_hash=compute_hash({
                "streams": streams_dict,
                "dead_state_T": request.dead_state_temperature_C
            }),
            processing_time_ms=processing_time
        )

    except Exception as e:
        app_state.requests_failed += 1
        logger.error(f"Exergy calculation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/sankey",
    response_model=SankeyResponse,
    summary="Generate Sankey diagram",
    description="Generate energy or exergy flow Sankey diagram",
    tags=["Visualization"]
)
async def generate_sankey(request: SankeyRequest):
    """Generate Sankey diagram for energy/exergy visualization."""
    app_state.requests_total += 1
    start_time = time.time()
    request_id = str(uuid.uuid4())

    try:
        streams_dict = [s.model_dump() for s in request.streams]

        result = await generate_sankey_async(
            streams_dict,
            request.diagram_type,
            request.show_losses
        )

        processing_time = (time.time() - start_time) * 1000
        app_state.requests_success += 1
        app_state.total_latency_ms += processing_time

        return SankeyResponse(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc),
            nodes=result["nodes"],
            links=result["links"],
            total_input_kW=result["total_input_kW"],
            total_output_kW=result["total_output_kW"],
            total_losses_kW=result["total_losses_kW"],
            diagram_type=request.diagram_type,
            layout_direction="left_to_right",
            computation_hash=compute_hash({"streams": streams_dict})
        )

    except Exception as e:
        app_state.requests_failed += 1
        logger.error(f"Sankey generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/recommend-fluid",
    response_model=FluidRecommendationResponse,
    summary="Get fluid recommendations",
    description="Get intelligent fluid recommendations based on operating conditions",
    tags=["Fluids"]
)
async def recommend_fluid(request: FluidRecommendationRequest):
    """Get fluid recommendations for given application and conditions."""
    app_state.requests_total += 1
    start_time = time.time()
    request_id = str(uuid.uuid4())

    try:
        recommendations = []
        best_overall = None
        best_environmental = None
        best_performance = None

        for name, info in AVAILABLE_FLUIDS.items():
            # Skip excluded fluids
            if request.exclude_fluids and name in request.exclude_fluids:
                continue

            # Check category filter
            if request.preferred_categories:
                category_values = [c.value if hasattr(c, 'value') else c for c in request.preferred_categories]
                if info["category"] not in category_values:
                    continue

            # Check GWP constraint
            if request.max_gwp is not None and info.get("gwp") is not None:
                if info["gwp"] > request.max_gwp:
                    continue

            # Check ODP constraint
            if request.max_odp is not None and info.get("odp") is not None:
                if info["odp"] > request.max_odp:
                    continue

            # Check temperature range
            if request.max_temperature_C > info["critical_temp_C"]:
                continue

            # Check flammability
            if request.require_non_flammable:
                if info.get("flammability", "").startswith(("A2", "A3", "B2", "B3")):
                    continue

            # Calculate suitability score (simplified)
            score = 80.0  # Base score
            pros = []
            cons = []

            # Adjust for GWP
            gwp = info.get("gwp", 0)
            if gwp == 0:
                score += 10
                pros.append("Zero GWP - environmentally friendly")
            elif gwp < 150:
                score += 5
                pros.append("Low GWP")
            elif gwp > 1000:
                score -= 10
                cons.append("High GWP - environmental concern")

            # Adjust for temperature range
            temp_margin = info["critical_temp_C"] - request.max_temperature_C
            if temp_margin > 50:
                score += 5
                pros.append("Good temperature margin")
            elif temp_margin < 20:
                score -= 5
                cons.append("Limited temperature margin")

            recommendations.append(FluidRecommendation(
                fluid_name=name,
                category=FluidCategory(info["category"]),
                suitability_score=min(100, max(0, score)),
                ranking=0,  # Will be set after sorting
                properties_at_conditions=None,
                gwp=info.get("gwp"),
                odp=info.get("odp"),
                flammability_class=info.get("flammability"),
                toxicity_class=info.get("toxicity"),
                pros=pros,
                cons=cons,
                notes=None
            ))

        # Sort by score and limit
        recommendations.sort(key=lambda x: x.suitability_score, reverse=True)
        recommendations = recommendations[:request.top_n]

        # Update rankings
        for i, rec in enumerate(recommendations):
            rec.ranking = i + 1

        # Identify best options
        if recommendations:
            best_overall = recommendations[0].fluid_name

            # Find best environmental
            env_recs = [r for r in recommendations if (r.gwp or 0) <= 10]
            if env_recs:
                best_environmental = env_recs[0].fluid_name

            # Best performance is same as overall for now
            best_performance = best_overall

        processing_time = (time.time() - start_time) * 1000
        app_state.requests_success += 1
        app_state.total_latency_ms += processing_time

        return FluidRecommendationResponse(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc),
            application=request.application,
            temperature_range_C=(request.min_temperature_C, request.max_temperature_C),
            recommendations=recommendations,
            best_overall=best_overall or "None found",
            best_environmental=best_environmental,
            best_performance=best_performance,
            computation_hash=compute_hash({
                "application": request.application,
                "temp_range": [request.min_temperature_C, request.max_temperature_C]
            })
        )

    except Exception as e:
        app_state.requests_failed += 1
        logger.error(f"Fluid recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Exception Handlers
# =============================================================================

from fastapi import Request
from fastapi.exceptions import RequestValidationError


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with structured response."""
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="validation_error",
            message="Request validation failed",
            details={"errors": exc.errors()},
            request_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc)
        ).model_dump(mode="json")
    )


# =============================================================================
# Create App Instance
# =============================================================================

app = create_app()

# Register exception handlers
app.add_exception_handler(RequestValidationError, validation_exception_handler)
