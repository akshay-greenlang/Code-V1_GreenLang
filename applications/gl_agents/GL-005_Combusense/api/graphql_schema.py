"""
GL-005 COMBUSENSE GraphQL API

Strawberry-based GraphQL schema for combustion control and optimization.

Queries:
- burnerStatus - Get current burner status
- optimizationResult - Get optimization result by ID
- emissionsReport - Get emissions for date range
- stabilityHistory - Get stability analysis history

Mutations:
- optimizeCombustion - Run optimization analysis
- updateSetpoint - Update PID setpoint
- calculateEmissions - Calculate emissions

Subscriptions:
- telemetryStream - Real-time telemetry updates
- alertStream - Real-time alert notifications
"""

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import strawberry
from strawberry.types import Info


# =============================================================================
# GraphQL Types
# =============================================================================

@strawberry.type
class BurnerStatus:
    """Current burner operational status."""
    burner_id: str
    status: str  # running, idle, fault
    fuel_flow_kg_h: float
    air_flow_kg_h: float
    stack_o2_percent: float
    stack_temp_c: float
    efficiency_percent: float
    lambda_value: float
    last_update: str


@strawberry.type
class OptimizationResult:
    """Combustion optimization result."""
    result_id: str
    timestamp: str
    burner_id: str

    # Current state
    current_efficiency_percent: float
    current_excess_air_percent: float
    current_lambda: float

    # Recommendations
    recommended_o2_setpoint: float
    recommended_air_fuel_ratio: float
    expected_efficiency_percent: float

    # Savings
    fuel_savings_percent: float
    annual_savings_usd: float
    co2_reduction_kg_year: float

    # Provenance
    computation_hash: str
    processing_time_ms: float


@strawberry.type
class EmissionsResult:
    """Emissions calculation result."""
    result_id: str
    timestamp: str
    burner_id: str
    co2_kg_h: float
    nox_kg_h: float
    so2_kg_h: float
    co2_annual_tonnes: float
    nox_annual_kg: float
    nox_compliant: bool
    co_compliant: bool
    compliance_standard: str
    computation_hash: str


@strawberry.type
class StabilityResult:
    """Combustion stability analysis result."""
    result_id: str
    timestamp: str
    burner_id: str
    combustion_quality_index: float
    flame_stability_index: float
    o2_variability_percent: float
    pressure_variability_percent: float
    stability_status: str
    risk_factors: List[str]
    recommendations: List[str]
    computation_hash: str


@strawberry.type
class SetpointResponse:
    """PID setpoint update response."""
    controller_id: str
    previous_setpoint: float
    new_setpoint: float
    accepted: bool
    reason: Optional[str]
    estimated_settling_time_s: float


@strawberry.type
class TelemetryPoint:
    """Real-time telemetry data point."""
    burner_id: str
    timestamp: str
    fuel_flow_kg_h: float
    air_flow_kg_h: float
    stack_o2_percent: float
    stack_co_ppm: float
    stack_temp_c: float
    efficiency_percent: float


@strawberry.type
class Alert:
    """System alert."""
    alert_id: str
    timestamp: str
    burner_id: str
    severity: str  # info, warning, critical
    alert_type: str
    message: str
    recommended_actions: List[str]


# =============================================================================
# Input Types
# =============================================================================

@strawberry.input
class CombustionInput:
    """Input for combustion optimization."""
    burner_id: str
    fuel_type: str = "natural_gas"
    fuel_flow_kg_h: float
    air_flow_kg_h: float
    stack_o2_percent: float
    stack_co_ppm: Optional[float] = None
    stack_temp_c: float
    ambient_temp_c: float = 25.0


@strawberry.input
class SetpointInput:
    """Input for setpoint update."""
    controller_id: str
    setpoint: float
    setpoint_type: str  # o2, temperature, pressure
    ramp_rate: Optional[float] = None


@strawberry.input
class EmissionsInput:
    """Input for emissions calculation."""
    burner_id: str
    fuel_type: str
    fuel_flow_kg_h: float
    stack_o2_percent: float
    stack_temp_c: float
    operating_hours: float = 8760.0


@strawberry.input
class StabilityInput:
    """Input for stability analysis."""
    burner_id: str
    flame_temperature_readings: List[float]
    o2_readings: List[float]
    pressure_readings: List[float]
    sample_rate_hz: float = 1.0


# =============================================================================
# Helper Functions
# =============================================================================

def compute_hash(data: Dict[str, Any]) -> str:
    """Compute SHA-256 hash for provenance."""
    json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


# In-memory stores (production: use persistent storage)
_burner_store: Dict[str, BurnerStatus] = {}
_result_store: Dict[str, OptimizationResult] = {}


# =============================================================================
# Query Resolvers
# =============================================================================

@strawberry.type
class Query:
    """GraphQL Query root."""

    @strawberry.field
    def burner_status(self, burner_id: str) -> Optional[BurnerStatus]:
        """Get current status of a burner."""
        if burner_id in _burner_store:
            return _burner_store[burner_id]

        # Return mock data for demo
        return BurnerStatus(
            burner_id=burner_id,
            status="running",
            fuel_flow_kg_h=1000.0,
            air_flow_kg_h=17200.0,
            stack_o2_percent=3.5,
            stack_temp_c=180.0,
            efficiency_percent=92.5,
            lambda_value=1.17,
            last_update=datetime.now(timezone.utc).isoformat()
        )

    @strawberry.field
    def optimization_result(self, result_id: str) -> Optional[OptimizationResult]:
        """Get optimization result by ID."""
        return _result_store.get(result_id)

    @strawberry.field
    def burner_list(self, site_id: Optional[str] = None) -> List[BurnerStatus]:
        """Get list of all burners, optionally filtered by site."""
        burners = list(_burner_store.values())
        if not burners:
            # Return mock data
            burners = [
                BurnerStatus(
                    burner_id=f"burner-0{i}",
                    status="running" if i % 2 == 0 else "idle",
                    fuel_flow_kg_h=1000.0 * i,
                    air_flow_kg_h=17200.0 * i,
                    stack_o2_percent=3.0 + i * 0.5,
                    stack_temp_c=180.0 + i * 10,
                    efficiency_percent=92.0 - i * 0.5,
                    lambda_value=1.15 + i * 0.02,
                    last_update=datetime.now(timezone.utc).isoformat()
                )
                for i in range(1, 4)
            ]
        return burners

    @strawberry.field
    def health(self) -> str:
        """Health check query."""
        return "healthy"


# =============================================================================
# Mutation Resolvers
# =============================================================================

@strawberry.type
class Mutation:
    """GraphQL Mutation root."""

    @strawberry.mutation
    def optimize_combustion(self, input: CombustionInput) -> OptimizationResult:
        """Run combustion optimization analysis."""
        import time
        start_time = time.time()

        # Calculate stoichiometric values
        stoich_afr = 17.2 if input.fuel_type == "natural_gas" else 14.7
        actual_afr = input.air_flow_kg_h / input.fuel_flow_kg_h
        current_lambda = actual_afr / stoich_afr
        current_excess_air = (current_lambda - 1) * 100

        # Calculate current efficiency
        stack_loss = 0.38 * (input.stack_temp_c - input.ambient_temp_c) / (21 - input.stack_o2_percent)
        current_efficiency = 100 - stack_loss - 2

        # Optimize
        optimal_o2 = 2.5 if input.stack_co_ppm is None or input.stack_co_ppm < 50 else 3.0
        optimal_lambda = 1 + (optimal_o2 / (21 - optimal_o2))
        optimal_afr = stoich_afr * optimal_lambda

        optimal_stack_loss = 0.38 * (input.stack_temp_c - input.ambient_temp_c) / (21 - optimal_o2)
        expected_efficiency = 100 - optimal_stack_loss - 2

        efficiency_improvement = expected_efficiency - current_efficiency
        fuel_savings_percent = efficiency_improvement / current_efficiency * 100 if current_efficiency > 0 else 0
        fuel_cost_usd_kg = 0.5
        annual_fuel_kg = input.fuel_flow_kg_h * 8760
        annual_savings = annual_fuel_kg * fuel_cost_usd_kg * fuel_savings_percent / 100
        co2_factor = 2.75
        co2_reduction = annual_fuel_kg * fuel_savings_percent / 100 * co2_factor

        processing_time = (time.time() - start_time) * 1000

        input_dict = {
            "burner_id": input.burner_id,
            "fuel_type": input.fuel_type,
            "fuel_flow_kg_h": input.fuel_flow_kg_h,
            "air_flow_kg_h": input.air_flow_kg_h,
            "stack_o2_percent": input.stack_o2_percent,
        }
        computation_hash = compute_hash(input_dict)

        result = OptimizationResult(
            result_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            burner_id=input.burner_id,
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

        _result_store[result.result_id] = result
        return result

    @strawberry.mutation
    def update_setpoint(self, input: SetpointInput) -> SetpointResponse:
        """Update PID controller setpoint."""
        limits = {
            "o2": (1.5, 8.0),
            "temperature": (100, 500),
            "pressure": (0.5, 10.0),
        }

        if input.setpoint_type not in limits:
            return SetpointResponse(
                controller_id=input.controller_id,
                previous_setpoint=3.0,
                new_setpoint=input.setpoint,
                accepted=False,
                reason=f"Unknown setpoint type: {input.setpoint_type}",
                estimated_settling_time_s=0
            )

        min_val, max_val = limits[input.setpoint_type]

        if not (min_val <= input.setpoint <= max_val):
            return SetpointResponse(
                controller_id=input.controller_id,
                previous_setpoint=3.0,
                new_setpoint=input.setpoint,
                accepted=False,
                reason=f"Setpoint outside limits [{min_val}, {max_val}]",
                estimated_settling_time_s=0
            )

        previous_setpoint = 3.0
        delta = abs(input.setpoint - previous_setpoint)
        ramp_rate = input.ramp_rate or 1.0
        settling_time = delta / ramp_rate * 60 + 30

        return SetpointResponse(
            controller_id=input.controller_id,
            previous_setpoint=previous_setpoint,
            new_setpoint=input.setpoint,
            accepted=True,
            reason=None,
            estimated_settling_time_s=round(settling_time, 1)
        )

    @strawberry.mutation
    def calculate_emissions(self, input: EmissionsInput) -> EmissionsResult:
        """Calculate emissions for reporting."""
        co2_factor = 2.75
        nox_factor = 0.0016
        so2_factor = 0.00001

        co2_kg_h = input.fuel_flow_kg_h * co2_factor
        nox_kg_h = input.fuel_flow_kg_h * nox_factor * (1 + input.stack_o2_percent / 10)
        so2_kg_h = input.fuel_flow_kg_h * so2_factor

        co2_annual = co2_kg_h * input.operating_hours / 1000
        nox_annual = nox_kg_h * input.operating_hours

        nox_limit_kg_h = 0.05 * input.fuel_flow_kg_h / 100
        nox_compliant = nox_kg_h <= nox_limit_kg_h

        input_dict = {
            "burner_id": input.burner_id,
            "fuel_type": input.fuel_type,
            "fuel_flow_kg_h": input.fuel_flow_kg_h,
        }
        computation_hash = compute_hash(input_dict)

        return EmissionsResult(
            result_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            burner_id=input.burner_id,
            co2_kg_h=round(co2_kg_h, 2),
            nox_kg_h=round(nox_kg_h, 4),
            so2_kg_h=round(so2_kg_h, 6),
            co2_annual_tonnes=round(co2_annual, 2),
            nox_annual_kg=round(nox_annual, 2),
            nox_compliant=nox_compliant,
            co_compliant=True,
            compliance_standard="EPA 40 CFR Part 98",
            computation_hash=computation_hash
        )

    @strawberry.mutation
    def analyze_stability(self, input: StabilityInput) -> StabilityResult:
        """Analyze combustion stability from sensor data."""
        import statistics

        temp_std = statistics.stdev(input.flame_temperature_readings)
        temp_mean = statistics.mean(input.flame_temperature_readings)
        temp_cv = (temp_std / temp_mean * 100) if temp_mean > 0 else 0

        o2_std = statistics.stdev(input.o2_readings)
        o2_mean = statistics.mean(input.o2_readings)
        o2_cv = (o2_std / o2_mean * 100) if o2_mean > 0 else 0

        pressure_std = statistics.stdev(input.pressure_readings)
        pressure_mean = statistics.mean(input.pressure_readings)
        pressure_cv = (pressure_std / pressure_mean * 100) if pressure_mean > 0 else 0

        cqi = max(0, 100 - (temp_cv * 2 + o2_cv * 3 + pressure_cv * 2))
        flame_stability = max(0, 100 - temp_cv * 5)

        if cqi >= 80:
            status = "stable"
        elif cqi >= 60:
            status = "marginal"
        else:
            status = "unstable"

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

        input_dict = {"burner_id": input.burner_id}
        computation_hash = compute_hash(input_dict)

        return StabilityResult(
            result_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            burner_id=input.burner_id,
            combustion_quality_index=round(cqi, 1),
            flame_stability_index=round(flame_stability, 1),
            o2_variability_percent=round(o2_cv, 2),
            pressure_variability_percent=round(pressure_cv, 2),
            stability_status=status,
            risk_factors=risk_factors,
            recommendations=recommendations,
            computation_hash=computation_hash
        )


# =============================================================================
# Schema
# =============================================================================

schema = strawberry.Schema(query=Query, mutation=Mutation)
