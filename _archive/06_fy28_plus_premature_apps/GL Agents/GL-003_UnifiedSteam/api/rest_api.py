"""
GL-003 UnifiedSteam REST API

FastAPI REST endpoints for steam system optimization.
Provides HTTP endpoints for steam properties, optimization, diagnostics, and analytics.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse

from .api_auth import (
    Permission,
    SteamSystemUser,
    get_current_user,
    require_permissions,
    log_security_event,
)
from .api_schemas import (
    # Steam Properties
    SteamPropertiesRequest,
    SteamPropertiesResponse,
    SteamState,
    SteamPhase,
    SteamRegion,
    # Balance
    EnthalpyBalanceRequest,
    EnthalpyBalanceResponse,
    MassBalanceRequest,
    MassBalanceResponse,
    # Optimization
    DesuperheaterOptimizationRequest,
    DesuperheaterOptimizationResponse,
    CondensateOptimizationRequest,
    CondensateOptimizationResponse,
    NetworkOptimizationRequest,
    NetworkOptimizationResponse,
    OptimizationRecommendation,
    OptimizationType,
    RecommendationPriority,
    RecommendationStatus,
    # Trap Diagnostics
    TrapDiagnosticsRequest,
    TrapDiagnosticsResponse,
    BatchTrapDiagnosticsRequest,
    BatchTrapDiagnosticsResponse,
    TrapStatus,
    TrapCondition,
    TrapType,
    TrapFailurePrediction,
    # RCA
    RCARequest,
    RCAResponse,
    CausalFactor,
    # Explainability
    ExplainabilityRequest,
    ExplainabilityResponse,
    FeatureContribution,
    # KPIs
    KPIDashboardResponse,
    ClimateImpactResponse,
    KPIValue,
    KPICategory,
    EnergyMetrics,
    EmissionsMetrics,
    # Common
    ErrorResponse,
    PaginationParams,
    TimeRangeFilter,
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["Steam System Optimizer"])


# =============================================================================
# Steam Properties Endpoints
# =============================================================================

@router.post(
    "/steam/properties",
    response_model=SteamPropertiesResponse,
    status_code=status.HTTP_200_OK,
    summary="Compute steam properties",
    description="Compute thermodynamic properties of steam/water using IAPWS IF97 formulation",
    tags=["Steam Properties"],
)
async def compute_steam_properties(
    request: Request,
    steam_request: SteamPropertiesRequest,
    current_user: SteamSystemUser = Depends(
        require_permissions(Permission.STEAM_PROPERTIES_COMPUTE)
    ),
) -> SteamPropertiesResponse:
    """
    Compute steam properties from given inputs.

    Requires at least two independent properties (P, T, h, s, or x).
    Returns complete thermodynamic state including phase information.
    """
    logger.info(f"Steam properties request from {current_user.username}")

    try:
        start_time = datetime.utcnow()

        # Mock computation - in production, use CoolProp or IAPWS library
        # Determine phase and compute properties
        steam_state = _compute_steam_state(steam_request)

        computation_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        await log_security_event(
            event_type="operation",
            action="compute",
            resource_type="steam_properties",
            request=request,
            user=current_user,
            success=True,
        )

        return SteamPropertiesResponse(
            request_id=steam_request.request_id,
            success=True,
            steam_state=steam_state,
            computation_time_ms=computation_time,
        )

    except ValueError as e:
        logger.warning(f"Invalid steam properties input: {e}")
        return SteamPropertiesResponse(
            request_id=steam_request.request_id,
            success=False,
            computation_time_ms=0,
            error_message=str(e),
        )
    except Exception as e:
        logger.error(f"Steam properties computation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Steam properties computation failed",
        )


@router.post(
    "/steam/balance",
    response_model=EnthalpyBalanceResponse,
    status_code=status.HTTP_200_OK,
    summary="Compute enthalpy balance",
    description="Compute enthalpy balance for steam system equipment",
    tags=["Steam Properties"],
)
async def compute_enthalpy_balance(
    request: Request,
    balance_request: EnthalpyBalanceRequest,
    current_user: SteamSystemUser = Depends(
        require_permissions(Permission.BALANCE_COMPUTE)
    ),
) -> EnthalpyBalanceResponse:
    """
    Compute enthalpy balance across equipment.

    Calculates inlet/outlet enthalpy flows and identifies imbalances.
    """
    logger.info(f"Enthalpy balance request for {balance_request.equipment_id}")

    try:
        # Mock computation
        total_inlet = 0.0
        total_outlet = 0.0
        stream_enthalpies = {}

        for stream in balance_request.streams:
            # Compute enthalpy for each stream
            h = stream.specific_enthalpy_kj_kg or 2800.0  # Mock value
            enthalpy_flow = stream.mass_flow_kg_s * h

            stream_enthalpies[stream.stream_id] = enthalpy_flow

            if stream.is_inlet:
                total_inlet += enthalpy_flow
            else:
                total_outlet += enthalpy_flow

        # Add heat input/loss
        total_inlet += balance_request.heat_input_kw
        total_outlet += balance_request.heat_loss_kw

        imbalance = total_inlet - total_outlet
        imbalance_percent = abs(imbalance / total_inlet * 100) if total_inlet > 0 else 0

        return EnthalpyBalanceResponse(
            request_id=balance_request.request_id,
            equipment_id=balance_request.equipment_id,
            success=True,
            total_inlet_enthalpy_kw=total_inlet,
            total_outlet_enthalpy_kw=total_outlet,
            enthalpy_imbalance_kw=imbalance,
            enthalpy_imbalance_percent=imbalance_percent,
            stream_enthalpies=stream_enthalpies,
            balance_closed=imbalance_percent < 2.0,
            data_quality_score=95.0,
        )

    except Exception as e:
        logger.error(f"Enthalpy balance computation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Enthalpy balance computation failed",
        )


# =============================================================================
# Optimization Endpoints
# =============================================================================

@router.post(
    "/optimization/desuperheater",
    response_model=DesuperheaterOptimizationResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Optimize desuperheater operation",
    description="Optimize spray water flow for desuperheater temperature control",
    tags=["Optimization"],
)
async def optimize_desuperheater(
    request: Request,
    opt_request: DesuperheaterOptimizationRequest,
    current_user: SteamSystemUser = Depends(
        require_permissions(Permission.OPTIMIZATION_REQUEST)
    ),
) -> DesuperheaterOptimizationResponse:
    """
    Optimize desuperheater spray water flow.

    Determines optimal spray water flow rate to achieve target outlet temperature
    while maintaining minimum superheat above saturation.
    """
    logger.info(f"Desuperheater optimization for {opt_request.desuperheater_id}")

    try:
        start_time = datetime.utcnow()

        # Mock optimization - compute spray water requirement
        # Q_steam + Q_spray = Q_outlet (energy balance)
        inlet_h = 3050.0  # Superheated steam enthalpy (mock)
        outlet_h = 2850.0  # Target outlet enthalpy (mock)
        spray_h = 200.0  # Spray water enthalpy (mock)

        # Mass balance: m_steam + m_spray = m_outlet
        # Energy balance: m_steam*h_steam + m_spray*h_spray = m_outlet*h_outlet
        # Solving: m_spray = m_steam * (h_steam - h_outlet) / (h_outlet - h_spray)

        m_steam = opt_request.inlet_steam_flow_kg_s
        optimal_spray = m_steam * (inlet_h - outlet_h) / (outlet_h - spray_h)
        optimal_spray = max(0, min(optimal_spray, opt_request.max_spray_water_flow_kg_s or float('inf')))

        computation_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        # Generate recommendations if applicable
        recommendations = []
        if optimal_spray > m_steam * 0.3:
            recommendations.append(OptimizationRecommendation(
                recommendation_type=OptimizationType.DESUPERHEATER,
                priority=RecommendationPriority.MEDIUM,
                title="High spray water usage detected",
                description="Spray water flow exceeds 30% of steam flow. Consider checking inlet steam conditions.",
                rationale="High spray water usage reduces overall system efficiency.",
                estimated_energy_savings_kw=optimal_spray * 50,
                confidence_score=0.85,
                affected_equipment=[opt_request.desuperheater_id],
            ))

        return DesuperheaterOptimizationResponse(
            request_id=opt_request.request_id,
            desuperheater_id=opt_request.desuperheater_id,
            success=True,
            optimal_spray_water_flow_kg_s=optimal_spray,
            optimal_outlet_temperature_c=opt_request.target_outlet_temperature_c,
            spray_water_energy_kw=optimal_spray * spray_h,
            desuperheating_efficiency=0.95,
            recommendations=recommendations,
            computation_time_ms=computation_time,
        )

    except Exception as e:
        logger.error(f"Desuperheater optimization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Desuperheater optimization failed",
        )


@router.post(
    "/optimization/condensate",
    response_model=CondensateOptimizationResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Optimize condensate recovery",
    description="Optimize condensate recovery system for maximum energy and water savings",
    tags=["Optimization"],
)
async def optimize_condensate(
    request: Request,
    opt_request: CondensateOptimizationRequest,
    current_user: SteamSystemUser = Depends(
        require_permissions(Permission.OPTIMIZATION_REQUEST)
    ),
) -> CondensateOptimizationResponse:
    """
    Optimize condensate recovery system.

    Analyzes current recovery rates and recommends improvements for
    water, energy, and cost savings.
    """
    logger.info(f"Condensate optimization for system {opt_request.system_id}")

    try:
        start_time = datetime.utcnow()

        # Calculate optimal recovery (mock)
        current_rate = opt_request.current_recovery_rate_percent
        optimal_rate = min(95.0, current_rate + 15.0)  # Target improvement

        # Calculate savings
        delta_temp = opt_request.condensate_return_temperature_c - opt_request.makeup_water_temperature_c
        energy_value_per_kg = delta_temp * 4.186 / 3600  # kWh per kg saved

        # Assume 1000 kg/h condensate production
        condensate_flow = 1000.0  # kg/h
        additional_recovery = (optimal_rate - current_rate) / 100 * condensate_flow

        annual_hours = 8760 * 0.9  # 90% utilization
        annual_water_savings = additional_recovery * annual_hours / 1000  # m3
        annual_energy_savings = additional_recovery * energy_value_per_kg * annual_hours / 1000  # MWh
        annual_cost_savings = (
            annual_water_savings * opt_request.makeup_water_cost_usd_m3 +
            annual_energy_savings * 50  # $50/MWh assumed
        )

        # CO2 reduction (natural gas boiler assumed)
        annual_co2_reduction = annual_energy_savings * 0.2  # tonnes CO2 per MWh

        computation_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        recommendations = [
            OptimizationRecommendation(
                recommendation_type=OptimizationType.CONDENSATE_RECOVERY,
                priority=RecommendationPriority.HIGH,
                title=f"Increase condensate recovery to {optimal_rate:.1f}%",
                description=f"Current recovery rate of {current_rate:.1f}% can be improved.",
                rationale="Higher condensate recovery reduces makeup water and energy costs.",
                estimated_energy_savings_kw=annual_energy_savings * 1000 / annual_hours,
                estimated_cost_savings_usd_year=annual_cost_savings,
                estimated_emissions_reduction_kg_co2_year=annual_co2_reduction * 1000,
                confidence_score=0.88,
                affected_equipment=[opt_request.system_id],
            ),
        ]

        return CondensateOptimizationResponse(
            request_id=opt_request.request_id,
            system_id=opt_request.system_id,
            success=True,
            optimal_recovery_rate_percent=optimal_rate,
            current_vs_optimal_delta_percent=optimal_rate - current_rate,
            annual_water_savings_m3=annual_water_savings,
            annual_energy_savings_mwh=annual_energy_savings,
            annual_cost_savings_usd=annual_cost_savings,
            annual_co2_reduction_tonnes=annual_co2_reduction,
            recommendations=recommendations,
            computation_time_ms=computation_time,
        )

    except Exception as e:
        logger.error(f"Condensate optimization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Condensate optimization failed",
        )


@router.post(
    "/optimization/network",
    response_model=NetworkOptimizationResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Optimize steam network",
    description="Optimize steam distribution network for cost and emissions balance",
    tags=["Optimization"],
)
async def optimize_network(
    request: Request,
    opt_request: NetworkOptimizationRequest,
    current_user: SteamSystemUser = Depends(
        require_permissions(Permission.OPTIMIZATION_EXECUTE)
    ),
) -> NetworkOptimizationResponse:
    """
    Optimize steam distribution network.

    Determines optimal header pressures, generator loading, and letdown flows
    to minimize cost and emissions while meeting demand.
    """
    logger.info(f"Network optimization for {opt_request.network_id}")

    try:
        start_time = datetime.utcnow()

        # Mock network optimization results
        optimal_pressures = {h["header_id"]: h.get("design_pressure_kpa", 1000) for h in opt_request.headers}
        optimal_generation = {g["generator_id"]: opt_request.total_demand_kg_s / len(opt_request.generators)
                            for g in opt_request.generators}
        optimal_letdowns = {}

        total_cost = opt_request.total_demand_kg_s * 25.0  # Mock cost
        total_emissions = opt_request.total_demand_kg_s * 0.5  # Mock emissions

        computation_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return NetworkOptimizationResponse(
            request_id=opt_request.request_id,
            network_id=opt_request.network_id,
            success=True,
            optimal_header_pressures_kpa=optimal_pressures,
            optimal_generator_outputs_kg_s=optimal_generation,
            optimal_letdown_flows_kg_s=optimal_letdowns,
            total_generation_kg_s=opt_request.total_demand_kg_s,
            total_consumption_kg_s=opt_request.total_demand_kg_s,
            network_efficiency_percent=92.5,
            total_operating_cost_usd_h=total_cost,
            marginal_cost_by_header_usd_kg={},
            total_emissions_kg_co2_h=total_emissions,
            emissions_by_source={},
            all_constraints_satisfied=True,
            recommendations=[],
            computation_time_ms=computation_time,
            solver_status="optimal",
        )

    except Exception as e:
        logger.error(f"Network optimization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Network optimization failed",
        )


# =============================================================================
# Trap Diagnostics Endpoints
# =============================================================================

@router.get(
    "/traps/{trap_id}/diagnostics",
    response_model=TrapDiagnosticsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get trap diagnostics",
    description="Retrieve diagnostics and status for a specific steam trap",
    tags=["Trap Diagnostics"],
)
async def get_trap_diagnostics(
    request: Request,
    trap_id: str,
    include_prediction: bool = Query(default=True, description="Include failure prediction"),
    current_user: SteamSystemUser = Depends(
        require_permissions(Permission.TRAP_READ)
    ),
) -> TrapDiagnosticsResponse:
    """
    Get diagnostics for a single steam trap.

    Returns current status, condition assessment, and optionally failure prediction.
    """
    logger.info(f"Trap diagnostics request for {trap_id}")

    try:
        # Mock trap status
        trap_status = TrapStatus(
            trap_id=trap_id,
            trap_name=f"Steam Trap {trap_id}",
            trap_type=TrapType.THERMODYNAMIC,
            condition=TrapCondition.GOOD,
            condition_confidence=0.92,
            location="Building A, Level 2",
            inlet_pressure_kpa=800.0,
            outlet_pressure_kpa=101.325,
            inlet_temperature_c=175.0,
            outlet_temperature_c=100.0,
            differential_temperature_c=75.0,
            estimated_steam_loss_kg_h=0.0,
            estimated_energy_loss_kw=0.0,
            estimated_annual_cost_loss_usd=0.0,
        )

        failure_prediction = None
        if include_prediction:
            failure_prediction = TrapFailurePrediction(
                trap_id=trap_id,
                failure_probability_30d=0.05,
                failure_probability_90d=0.12,
                predicted_failure_mode=TrapCondition.LEAKING,
                predicted_remaining_life_days=365,
                risk_factors=["Age > 3 years", "High cycling frequency"],
                risk_score=15.0,
                recommended_action="Continue monitoring",
                priority=RecommendationPriority.LOW,
                model_confidence=0.88,
            )

        return TrapDiagnosticsResponse(
            request_id=uuid4(),
            trap_id=trap_id,
            success=True,
            status=trap_status,
            failure_prediction=failure_prediction,
            diagnostic_method="multi-sensor",
            diagnostic_confidence=0.92,
            anomalies_detected=[],
            computation_time_ms=15.5,
        )

    except Exception as e:
        logger.error(f"Trap diagnostics failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Trap diagnostics failed",
        )


@router.post(
    "/traps/batch-diagnostics",
    response_model=BatchTrapDiagnosticsResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch trap diagnostics",
    description="Perform diagnostics on multiple steam traps",
    tags=["Trap Diagnostics"],
)
async def batch_trap_diagnostics(
    request: Request,
    batch_request: BatchTrapDiagnosticsRequest,
    current_user: SteamSystemUser = Depends(
        require_permissions(Permission.TRAP_DIAGNOSE)
    ),
) -> BatchTrapDiagnosticsResponse:
    """
    Perform batch diagnostics on multiple traps.

    Returns individual results plus summary statistics and prioritized actions.
    """
    logger.info(f"Batch diagnostics for {len(batch_request.traps)} traps")

    try:
        start_time = datetime.utcnow()

        results = []
        traps_good = 0
        traps_leaking = 0
        traps_blocked = 0
        traps_failed = 0
        total_steam_loss = 0.0
        total_energy_loss = 0.0

        for trap_req in batch_request.traps:
            # Mock individual diagnosis
            import random
            condition = random.choice([TrapCondition.GOOD] * 8 + [TrapCondition.LEAKING] + [TrapCondition.BLOCKED])

            steam_loss = 0.0 if condition == TrapCondition.GOOD else random.uniform(5, 50)
            energy_loss = steam_loss * 2.5  # Approximate

            trap_status = TrapStatus(
                trap_id=trap_req.trap_id,
                trap_name=f"Trap {trap_req.trap_id}",
                trap_type=TrapType.THERMODYNAMIC,
                condition=condition,
                condition_confidence=0.85,
                location="Various",
                inlet_pressure_kpa=trap_req.inlet_pressure_kpa,
                outlet_pressure_kpa=trap_req.outlet_pressure_kpa,
                inlet_temperature_c=trap_req.inlet_temperature_c,
                outlet_temperature_c=trap_req.outlet_temperature_c,
                estimated_steam_loss_kg_h=steam_loss,
                estimated_energy_loss_kw=energy_loss,
                estimated_annual_cost_loss_usd=steam_loss * 100 * 8760,
            )

            results.append(TrapDiagnosticsResponse(
                request_id=trap_req.request_id,
                trap_id=trap_req.trap_id,
                success=True,
                status=trap_status,
                diagnostic_confidence=0.85,
                computation_time_ms=5.0,
            ))

            if condition == TrapCondition.GOOD:
                traps_good += 1
            elif condition == TrapCondition.LEAKING:
                traps_leaking += 1
            elif condition == TrapCondition.BLOCKED:
                traps_blocked += 1
            else:
                traps_failed += 1

            total_steam_loss += steam_loss
            total_energy_loss += energy_loss

        computation_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        # Generate prioritized actions
        prioritized_actions = []
        for result in results:
            if result.status and result.status.condition != TrapCondition.GOOD:
                prioritized_actions.append({
                    "trap_id": result.trap_id,
                    "condition": result.status.condition.value,
                    "priority": "high" if result.status.estimated_steam_loss_kg_h > 20 else "medium",
                    "action": "Replace" if result.status.condition == TrapCondition.BLOCKED else "Repair",
                    "estimated_savings_usd_year": result.status.estimated_annual_cost_loss_usd,
                })

        prioritized_actions.sort(key=lambda x: x.get("estimated_savings_usd_year", 0), reverse=True)

        return BatchTrapDiagnosticsResponse(
            request_id=batch_request.request_id,
            success=True,
            results=results,
            total_traps=len(batch_request.traps),
            traps_good=traps_good,
            traps_leaking=traps_leaking,
            traps_blocked=traps_blocked,
            traps_failed=traps_failed,
            total_estimated_steam_loss_kg_h=total_steam_loss,
            total_estimated_energy_loss_kw=total_energy_loss,
            total_estimated_annual_cost_loss_usd=total_steam_loss * 100 * 8760,
            prioritized_actions=prioritized_actions[:10],
            computation_time_ms=computation_time,
        )

    except Exception as e:
        logger.error(f"Batch trap diagnostics failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch trap diagnostics failed",
        )


# =============================================================================
# Recommendations Endpoints
# =============================================================================

@router.get(
    "/recommendations",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Get recommendations list",
    description="Retrieve list of optimization recommendations",
    tags=["Recommendations"],
)
async def get_recommendations(
    request: Request,
    status_filter: Optional[RecommendationStatus] = Query(default=None),
    priority_filter: Optional[RecommendationPriority] = Query(default=None),
    type_filter: Optional[OptimizationType] = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    current_user: SteamSystemUser = Depends(
        require_permissions(Permission.RECOMMENDATION_READ)
    ),
) -> Dict[str, Any]:
    """
    Get list of recommendations with optional filtering and pagination.
    """
    logger.info(f"Recommendations request from {current_user.username}")

    # Mock recommendations
    recommendations = [
        OptimizationRecommendation(
            recommendation_id=uuid4(),
            recommendation_type=OptimizationType.CONDENSATE_RECOVERY,
            priority=RecommendationPriority.HIGH,
            status=RecommendationStatus.PENDING,
            title="Increase condensate recovery rate",
            description="Current recovery rate of 65% can be increased to 85%.",
            rationale="Higher recovery reduces makeup water and energy costs.",
            estimated_energy_savings_kw=150.0,
            estimated_cost_savings_usd_year=75000.0,
            estimated_emissions_reduction_kg_co2_year=50000.0,
            confidence_score=0.92,
        ),
        OptimizationRecommendation(
            recommendation_id=uuid4(),
            recommendation_type=OptimizationType.DESUPERHEATER,
            priority=RecommendationPriority.MEDIUM,
            status=RecommendationStatus.PENDING,
            title="Optimize desuperheater control",
            description="Temperature oscillations indicate suboptimal control tuning.",
            rationale="Better control reduces thermal stress and improves efficiency.",
            estimated_energy_savings_kw=25.0,
            estimated_cost_savings_usd_year=12000.0,
            confidence_score=0.85,
        ),
    ]

    return {
        "recommendations": [r.model_dump() for r in recommendations],
        "total_count": len(recommendations),
        "page": page,
        "page_size": page_size,
        "total_pages": 1,
    }


@router.get(
    "/recommendations/{rec_id}/explanation",
    response_model=ExplainabilityResponse,
    status_code=status.HTTP_200_OK,
    summary="Get recommendation explanation",
    description="Get SHAP/LIME explainability for a recommendation",
    tags=["Recommendations"],
)
async def get_recommendation_explanation(
    request: Request,
    rec_id: UUID,
    explanation_type: str = Query(default="shap", pattern="^(shap|lime|both)$"),
    current_user: SteamSystemUser = Depends(
        require_permissions(Permission.EXPLAINABILITY_READ)
    ),
) -> ExplainabilityResponse:
    """
    Get explainability information for a recommendation.

    Returns SHAP values, LIME explanations, and natural language summary.
    """
    logger.info(f"Explainability request for recommendation {rec_id}")

    try:
        shap_contributions = [
            FeatureContribution(
                feature_name="condensate_recovery_rate",
                feature_value=0.65,
                contribution_score=0.35,
                contribution_direction="positive",
                explanation="Current low recovery rate is main driver for recommendation",
            ),
            FeatureContribution(
                feature_name="makeup_water_cost",
                feature_value=2.5,
                contribution_score=0.25,
                contribution_direction="positive",
                explanation="Higher makeup water cost increases savings potential",
            ),
            FeatureContribution(
                feature_name="condensate_temperature",
                feature_value=85.0,
                contribution_score=0.20,
                contribution_direction="positive",
                explanation="High condensate temperature means significant energy recovery potential",
            ),
        ]

        return ExplainabilityResponse(
            request_id=uuid4(),
            recommendation_id=rec_id,
            success=True,
            shap_feature_contributions=shap_contributions if explanation_type in ["shap", "both"] else [],
            shap_base_value=0.5,
            shap_output_value=0.92,
            lime_feature_contributions=shap_contributions if explanation_type in ["lime", "both"] else [],
            plain_english_explanation=(
                "This recommendation is primarily driven by the current low condensate recovery rate of 65%. "
                "Given the high cost of makeup water and the high temperature of available condensate, "
                "increasing recovery would significantly reduce both water and energy costs."
            ),
            technical_explanation=(
                "SHAP analysis indicates condensate_recovery_rate contributes 35% to the recommendation score. "
                "The model predicts 92% confidence in potential savings based on similar facilities."
            ),
            key_drivers=[
                "Low condensate recovery rate (65%)",
                "High makeup water cost ($2.50/m3)",
                "High condensate temperature (85C)",
            ],
            computation_time_ms=45.0,
        )

    except Exception as e:
        logger.error(f"Explainability request failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Explainability request failed",
        )


# =============================================================================
# RCA Endpoints
# =============================================================================

@router.post(
    "/rca/analyze",
    response_model=RCAResponse,
    status_code=status.HTTP_200_OK,
    summary="Root cause analysis",
    description="Perform causal root cause analysis for an event or anomaly",
    tags=["Root Cause Analysis"],
)
async def analyze_root_cause(
    request: Request,
    rca_request: RCARequest,
    current_user: SteamSystemUser = Depends(
        require_permissions(Permission.RCA_ANALYZE)
    ),
) -> RCAResponse:
    """
    Perform root cause analysis using causal inference.

    Identifies root causes and contributing factors for events/anomalies.
    """
    logger.info(f"RCA request for event: {rca_request.target_event}")

    try:
        start_time = datetime.utcnow()

        # Mock RCA results
        root_causes = [
            CausalFactor(
                factor_name="Steam trap failure",
                factor_description="Upstream steam trap failed in open position",
                causal_strength=0.85,
                confidence=0.82,
                is_root_cause=True,
                is_contributing_factor=False,
                supporting_evidence=[
                    "Temperature spike detected 15 min before event",
                    "Trap acoustic signature changed 2 hours prior",
                ],
                related_variables=["trap_temperature", "trap_acoustic"],
            ),
        ]

        contributing_factors = [
            CausalFactor(
                factor_name="High system load",
                factor_description="System operating at 95% capacity",
                causal_strength=0.45,
                confidence=0.75,
                is_root_cause=False,
                is_contributing_factor=True,
                supporting_evidence=["Steam demand 15% above normal"],
            ),
        ]

        computation_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return RCAResponse(
            request_id=rca_request.request_id,
            success=True,
            target_event=rca_request.target_event,
            event_timestamp=rca_request.event_timestamp,
            root_causes=root_causes,
            contributing_factors=contributing_factors,
            causal_chain=["trap_failure", "steam_loss", "pressure_drop", "temperature_spike"],
            executive_summary=(
                "The event was primarily caused by a steam trap failure that resulted in "
                "steam loss and subsequent pressure drop. High system load contributed to "
                "the severity of the impact."
            ),
            recommended_actions=[
                "Replace failed steam trap immediately",
                "Inspect adjacent traps for similar degradation",
                "Review trap maintenance schedule",
            ],
            analysis_method="causal_discovery",
            model_confidence=0.82,
            computation_time_ms=computation_time,
        )

    except Exception as e:
        logger.error(f"RCA failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Root cause analysis failed",
        )


# =============================================================================
# KPI and Analytics Endpoints
# =============================================================================

@router.get(
    "/kpis",
    response_model=KPIDashboardResponse,
    status_code=status.HTTP_200_OK,
    summary="Get KPI dashboard",
    description="Retrieve KPI metrics for steam system performance",
    tags=["Analytics"],
)
async def get_kpi_dashboard(
    request: Request,
    period_hours: int = Query(default=24, ge=1, le=720),
    current_user: SteamSystemUser = Depends(
        require_permissions(Permission.KPI_READ)
    ),
) -> KPIDashboardResponse:
    """
    Get KPI dashboard with key performance metrics.
    """
    now = datetime.utcnow()

    energy_kpis = [
        KPIValue(
            kpi_name="Total Steam Consumption",
            category=KPICategory.ENERGY,
            current_value=15000,
            target_value=14000,
            unit="kg/h",
            trend="stable",
            is_on_target=False,
        ),
        KPIValue(
            kpi_name="Boiler Efficiency",
            category=KPICategory.EFFICIENCY,
            current_value=88.5,
            target_value=90.0,
            unit="%",
            trend="up",
            trend_percent=0.5,
            is_on_target=False,
        ),
    ]

    efficiency_kpis = [
        KPIValue(
            kpi_name="Condensate Recovery Rate",
            category=KPICategory.EFFICIENCY,
            current_value=72.0,
            target_value=85.0,
            unit="%",
            trend="up",
            is_on_target=False,
        ),
    ]

    emissions_kpis = [
        KPIValue(
            kpi_name="CO2 Emissions",
            category=KPICategory.EMISSIONS,
            current_value=2500,
            target_value=2200,
            unit="kg/h",
            trend="down",
            is_on_target=False,
        ),
    ]

    return KPIDashboardResponse(
        success=True,
        period_start=datetime(now.year, now.month, now.day),
        period_end=now,
        energy_kpis=energy_kpis,
        efficiency_kpis=efficiency_kpis,
        cost_kpis=[],
        emissions_kpis=emissions_kpis,
        reliability_kpis=[],
        overall_performance_score=78.5,
        kpis_on_target=1,
        kpis_off_target=4,
        kpis_improving=3,
        kpis_declining=1,
        kpi_alerts=[],
    )


@router.get(
    "/climate-impact",
    response_model=ClimateImpactResponse,
    status_code=status.HTTP_200_OK,
    summary="Get climate impact",
    description="Retrieve climate and energy impact metrics",
    tags=["Analytics"],
)
async def get_climate_impact(
    request: Request,
    period_days: int = Query(default=30, ge=1, le=365),
    current_user: SteamSystemUser = Depends(
        require_permissions(Permission.CLIMATE_READ)
    ),
) -> ClimateImpactResponse:
    """
    Get climate and energy impact metrics.

    Returns emissions, energy consumption, and environmental KPIs.
    """
    now = datetime.utcnow()

    energy_metrics = EnergyMetrics(
        total_steam_consumption_kg_h=15000,
        total_steam_generation_kg_h=15500,
        total_energy_consumption_mw=12.5,
        boiler_efficiency_percent=88.5,
        system_efficiency_percent=82.0,
        condensate_recovery_percent=72.0,
        flash_steam_recovery_percent=45.0,
        energy_intensity_mj_per_unit=3.2,
    )

    emissions_metrics = EmissionsMetrics(
        total_co2_emissions_kg_h=2500,
        co2_emissions_by_source={
            "boiler_1": 1500,
            "boiler_2": 1000,
        },
        total_nox_emissions_kg_h=2.5,
        total_sox_emissions_kg_h=0.5,
        carbon_intensity_kg_co2_per_mwh=200,
        avoided_emissions_kg_co2_h=150,
    )

    return ClimateImpactResponse(
        success=True,
        period_start=datetime(now.year, now.month, now.day - period_days),
        period_end=now,
        energy_metrics=energy_metrics,
        emissions_metrics=emissions_metrics,
        annual_emissions_target_tonnes_co2=20000,
        ytd_emissions_tonnes_co2=18000,
        on_track_for_target=True,
        improvement_opportunities=[],
        reporting_standard="GHG Protocol",
        verification_status="unverified",
    )


# =============================================================================
# Health Check Endpoints
# =============================================================================

@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="API health check endpoint",
    tags=["System"],
)
async def health_check():
    """Health check for load balancers."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
    }


@router.get(
    "/ready",
    status_code=status.HTTP_200_OK,
    summary="Readiness check",
    description="API readiness check endpoint",
    tags=["System"],
)
async def readiness_check():
    """Readiness check for Kubernetes deployments."""
    return {
        "status": "ready",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "steam_properties": "ready",
            "optimization": "ready",
            "diagnostics": "ready",
        },
    }


# =============================================================================
# Helper Functions
# =============================================================================

def _compute_steam_state(request: SteamPropertiesRequest) -> SteamState:
    """
    Compute steam state from request inputs.

    This is a mock implementation. In production, use CoolProp or IAPWS library.
    """
    # Use provided values or defaults
    P = request.pressure_kpa or 1000.0
    T = request.temperature_c

    # Determine saturation temperature at this pressure (simplified)
    T_sat = 100 + (P - 101.325) * 0.03  # Very rough approximation

    # Determine phase
    if T is not None:
        if T < T_sat - 1:
            phase = SteamPhase.SUBCOOLED_LIQUID
            region = SteamRegion.REGION_1
        elif abs(T - T_sat) <= 1:
            if request.quality is not None and 0 < request.quality < 1:
                phase = SteamPhase.TWO_PHASE
                region = SteamRegion.REGION_4
            elif request.quality == 0:
                phase = SteamPhase.SATURATED_LIQUID
                region = SteamRegion.REGION_4
            else:
                phase = SteamPhase.SATURATED_VAPOR
                region = SteamRegion.REGION_4
        else:
            phase = SteamPhase.SUPERHEATED_VAPOR
            region = SteamRegion.REGION_2
    else:
        T = T_sat
        phase = SteamPhase.SATURATED_VAPOR
        region = SteamRegion.REGION_4

    # Mock property values (would use real thermodynamic calculations)
    if phase == SteamPhase.SUPERHEATED_VAPOR:
        h = 2676 + 2.0 * (T - T_sat)  # Rough approximation
        s = 7.36 + 0.002 * (T - T_sat)
        v = 0.001 + 0.002 * (T - T_sat) / P
        rho = 1 / v
    elif phase == SteamPhase.SUBCOOLED_LIQUID:
        h = 4.186 * T  # cp * T
        s = 0.3 + 0.001 * T
        v = 0.001  # Liquid is nearly incompressible
        rho = 1000.0
    else:  # Saturated
        h = 2676.0
        s = 7.36
        v = 1.67 / (P / 101.325)
        rho = 1 / v

    return SteamState(
        pressure_kpa=P,
        temperature_c=T,
        specific_enthalpy_kj_kg=h,
        specific_entropy_kj_kg_k=s,
        specific_volume_m3_kg=v,
        density_kg_m3=rho,
        quality=request.quality,
        phase=phase,
        region=region,
    )
