"""
GL-006 HEATRECLAIM - GraphQL Schema

Strawberry-based GraphQL schema for flexible queries
on heat recovery optimization data.

Types:
- HeatStreamType
- HeatExchangerType
- HENDesignType
- PinchAnalysisType
- OptimizationResultType
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import logging

try:
    import strawberry
    from strawberry.types import Info
    HAS_STRAWBERRY = True
except ImportError:
    HAS_STRAWBERRY = False
    # Create dummy decorators for import compatibility
    class strawberry:
        @staticmethod
        def type(cls):
            return cls
        @staticmethod
        def input(cls):
            return cls
        @staticmethod
        def field(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
        @staticmethod
        def mutation(*args, **kwargs):
            def decorator(func):
                return func
            return decorator

from ..core.config import HeatReclaimConfig, OptimizationObjective
from ..core.schemas import (
    HeatStream,
    HeatExchanger,
    HENDesign,
    PinchAnalysisResult,
    OptimizationResult,
)
from ..core.orchestrator import HeatReclaimOrchestrator

logger = logging.getLogger(__name__)


# GraphQL Types
@strawberry.type
class HeatStreamType:
    """GraphQL type for heat stream."""

    stream_id: str
    stream_name: str
    stream_type: str
    fluid_name: str
    phase: str
    T_supply_C: float
    T_target_C: float
    m_dot_kg_s: float
    Cp_kJ_kgK: float
    duty_kW: float
    FCp_kW_K: float
    pressure_kPa: float


@strawberry.type
class HeatExchangerType:
    """GraphQL type for heat exchanger."""

    exchanger_id: str
    exchanger_name: str
    exchanger_type: str
    hot_stream_id: str
    cold_stream_id: str
    duty_kW: float
    hot_inlet_T_C: float
    hot_outlet_T_C: float
    cold_inlet_T_C: float
    cold_outlet_T_C: float
    LMTD_C: Optional[float]
    area_m2: Optional[float]
    U_W_m2K: Optional[float]


@strawberry.type
class EconomicAnalysisType:
    """GraphQL type for economic analysis."""

    total_capital_cost_usd: float
    annual_utility_savings_usd: float
    payback_period_years: float
    npv_usd: float
    irr_percent: float
    levelized_cost_usd_gj: float


@strawberry.type
class HENDesignType:
    """GraphQL type for HEN design."""

    design_name: str
    mode: str
    exchangers: List[HeatExchangerType]
    total_heat_recovered_kW: float
    hot_utility_required_kW: float
    cold_utility_required_kW: float
    exchanger_count: int
    new_exchanger_count: int
    total_area_m2: Optional[float]
    economic_analysis: Optional[EconomicAnalysisType]


@strawberry.type
class CompositePointType:
    """Point on composite curve."""

    temperature_C: float
    enthalpy_kW: float


@strawberry.type
class PinchAnalysisType:
    """GraphQL type for pinch analysis result."""

    pinch_temperature_C: float
    minimum_hot_utility_kW: float
    minimum_cold_utility_kW: float
    maximum_heat_recovery_kW: float
    is_threshold_problem: bool
    hot_composite_curve: List[CompositePointType]
    cold_composite_curve: List[CompositePointType]
    computation_hash: str


@strawberry.type
class FeatureImportanceType:
    """Feature importance from explainability."""

    feature_name: str
    importance_value: float
    rank: int
    direction: str
    category: str


@strawberry.type
class ExplainabilityType:
    """GraphQL type for explainability report."""

    method: str
    feature_importance: List[FeatureImportanceType]
    key_drivers: List[str]
    summary: str
    confidence_score: float


@strawberry.type
class OptimizationResultType:
    """GraphQL type for optimization result."""

    request_id: str
    status: str
    optimization_time_seconds: float
    pinch_analysis: PinchAnalysisType
    recommended_design: HENDesignType
    alternative_designs: List[HENDesignType]
    explanation_summary: str
    key_drivers: List[str]
    robustness_score: float


@strawberry.type
class ValidationResultType:
    """GraphQL type for validation result."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]


# Input Types
@strawberry.input
class HeatStreamInput:
    """GraphQL input for heat stream."""

    stream_id: str
    stream_name: Optional[str] = None
    stream_type: str = "hot"
    fluid_name: str = "Water"
    phase: str = "liquid"
    T_supply_C: float
    T_target_C: float
    m_dot_kg_s: float
    Cp_kJ_kgK: float = 4.186
    pressure_kPa: float = 101.325
    fouling_factor_m2K_W: float = 0.0001
    availability: float = 1.0


@strawberry.input
class OptimizationOptionsInput:
    """GraphQL input for optimization options."""

    delta_t_min_C: float = 10.0
    objective: str = "minimize_cost"
    include_exergy_analysis: bool = True
    include_uncertainty: bool = False
    generate_pareto: bool = False
    n_pareto_points: int = 20
    max_time_seconds: float = 300.0


# Query Resolvers
@strawberry.type
class HENQuery:
    """GraphQL queries for heat recovery optimization."""

    @strawberry.field
    def health(self) -> str:
        """Health check."""
        return "GL-006 HEATRECLAIM GraphQL API is healthy"

    @strawberry.field
    def pinch_analysis(
        self,
        hot_streams: List[HeatStreamInput],
        cold_streams: List[HeatStreamInput],
        delta_t_min_C: float = 10.0,
    ) -> PinchAnalysisType:
        """
        Run pinch analysis on provided streams.
        """
        # Convert inputs
        hot = [_convert_stream_input(s) for s in hot_streams]
        cold = [_convert_stream_input(s) for s in cold_streams]

        # Run analysis
        orchestrator = HeatReclaimOrchestrator()
        result = orchestrator.run_pinch_analysis(hot, cold, delta_t_min_C)

        return _convert_pinch_result(result)

    @strawberry.field
    def validate_streams(
        self,
        hot_streams: List[HeatStreamInput],
        cold_streams: List[HeatStreamInput],
    ) -> ValidationResultType:
        """
        Validate stream data.
        """
        from ..core.handlers import StreamDataHandler

        hot = [_convert_stream_input(s) for s in hot_streams]
        cold = [_convert_stream_input(s) for s in cold_streams]

        handler = StreamDataHandler()
        result = handler.validate_streams(hot, cold)

        return ValidationResultType(
            is_valid=result.is_valid,
            errors=result.errors,
            warnings=result.warnings,
        )

    @strawberry.field
    def design(self, design_id: str) -> Optional[HENDesignType]:
        """
        Get saved design by ID.
        """
        # Placeholder - would query database
        return None

    @strawberry.field
    def designs(
        self,
        limit: int = 10,
        offset: int = 0,
    ) -> List[HENDesignType]:
        """
        List saved designs.
        """
        # Placeholder - would query database
        return []


# Mutation Resolvers
@strawberry.type
class HENMutation:
    """GraphQL mutations for heat recovery optimization."""

    @strawberry.mutation
    def optimize(
        self,
        hot_streams: List[HeatStreamInput],
        cold_streams: List[HeatStreamInput],
        options: Optional[OptimizationOptionsInput] = None,
    ) -> OptimizationResultType:
        """
        Run full heat recovery optimization.
        """
        options = options or OptimizationOptionsInput()

        # Convert inputs
        hot = [_convert_stream_input(s) for s in hot_streams]
        cold = [_convert_stream_input(s) for s in cold_streams]

        # Map objective
        objective_map = {
            "minimize_cost": OptimizationObjective.MINIMIZE_COST,
            "minimize_utility": OptimizationObjective.MINIMIZE_UTILITY,
            "minimize_exchangers": OptimizationObjective.MINIMIZE_EXCHANGERS,
            "maximize_recovery": OptimizationObjective.MAXIMIZE_RECOVERY,
        }
        objective = objective_map.get(
            options.objective, OptimizationObjective.MINIMIZE_COST
        )

        # Run optimization
        orchestrator = HeatReclaimOrchestrator()
        result = orchestrator.optimize(
            hot_streams=hot,
            cold_streams=cold,
            delta_t_min=options.delta_t_min_C,
            objective=objective,
            include_exergy=options.include_exergy_analysis,
            include_uncertainty=options.include_uncertainty,
        )

        return _convert_optimization_result(result)

    @strawberry.mutation
    def save_design(
        self,
        design_name: str,
        hot_streams: List[HeatStreamInput],
        cold_streams: List[HeatStreamInput],
        options: Optional[OptimizationOptionsInput] = None,
    ) -> str:
        """
        Optimize and save design to database.
        Returns design ID.
        """
        # Placeholder - would run optimization and save
        import uuid
        return str(uuid.uuid4())

    @strawberry.mutation
    def delete_design(self, design_id: str) -> bool:
        """
        Delete saved design.
        """
        # Placeholder - would delete from database
        return True


# Helper Functions
def _convert_stream_input(input_data: HeatStreamInput) -> HeatStream:
    """Convert GraphQL input to internal HeatStream."""
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


def _convert_pinch_result(result: PinchAnalysisResult) -> PinchAnalysisType:
    """Convert internal result to GraphQL type."""
    hot_curve = [
        CompositePointType(temperature_C=t, enthalpy_kW=h)
        for t, h in zip(result.hot_composite_T_C, result.hot_composite_H_kW)
    ]
    cold_curve = [
        CompositePointType(temperature_C=t, enthalpy_kW=h)
        for t, h in zip(result.cold_composite_T_C, result.cold_composite_H_kW)
    ]

    return PinchAnalysisType(
        pinch_temperature_C=result.pinch_temperature_C,
        minimum_hot_utility_kW=result.minimum_hot_utility_kW,
        minimum_cold_utility_kW=result.minimum_cold_utility_kW,
        maximum_heat_recovery_kW=result.maximum_heat_recovery_kW,
        is_threshold_problem=result.is_threshold_problem,
        hot_composite_curve=hot_curve,
        cold_composite_curve=cold_curve,
        computation_hash=result.computation_hash,
    )


def _convert_optimization_result(result: OptimizationResult) -> OptimizationResultType:
    """Convert internal result to GraphQL type."""
    pinch = _convert_pinch_result(result.pinch_analysis)

    design = _convert_design(result.recommended_design)

    alternatives = [_convert_design(d) for d in result.alternative_designs]

    return OptimizationResultType(
        request_id=result.request_id,
        status=result.status.value,
        optimization_time_seconds=result.optimization_time_seconds,
        pinch_analysis=pinch,
        recommended_design=design,
        alternative_designs=alternatives,
        explanation_summary=result.explanation_summary,
        key_drivers=result.key_drivers,
        robustness_score=result.robustness_score,
    )


def _convert_design(design: HENDesign) -> HENDesignType:
    """Convert internal design to GraphQL type."""
    exchangers = [
        HeatExchangerType(
            exchanger_id=hx.exchanger_id,
            exchanger_name=hx.exchanger_name,
            exchanger_type=hx.exchanger_type.value if hasattr(hx.exchanger_type, 'value') else str(hx.exchanger_type),
            hot_stream_id=hx.hot_stream_id,
            cold_stream_id=hx.cold_stream_id,
            duty_kW=hx.duty_kW,
            hot_inlet_T_C=hx.hot_inlet_T_C,
            hot_outlet_T_C=hx.hot_outlet_T_C,
            cold_inlet_T_C=hx.cold_inlet_T_C,
            cold_outlet_T_C=hx.cold_outlet_T_C,
            LMTD_C=hx.LMTD_C,
            area_m2=hx.area_m2,
            U_W_m2K=hx.U_W_m2K,
        )
        for hx in design.exchangers
    ]

    economic = None
    if design.economic_analysis:
        ea = design.economic_analysis
        economic = EconomicAnalysisType(
            total_capital_cost_usd=ea.total_capital_cost_usd,
            annual_utility_savings_usd=ea.annual_utility_savings_usd,
            payback_period_years=ea.payback_period_years,
            npv_usd=ea.npv_usd,
            irr_percent=ea.irr_percent,
            levelized_cost_usd_gj=ea.levelized_cost_usd_gj,
        )

    return HENDesignType(
        design_name=design.design_name,
        mode=design.mode.value if hasattr(design.mode, 'value') else str(design.mode),
        exchangers=exchangers,
        total_heat_recovered_kW=design.total_heat_recovered_kW,
        hot_utility_required_kW=design.hot_utility_required_kW,
        cold_utility_required_kW=design.cold_utility_required_kW,
        exchanger_count=design.exchanger_count,
        new_exchanger_count=design.new_exchanger_count,
        total_area_m2=design.total_area_m2,
        economic_analysis=economic,
    )


# Create Schema
if HAS_STRAWBERRY:
    schema = strawberry.Schema(query=HENQuery, mutation=HENMutation)
else:
    schema = None
    logger.warning("Strawberry not installed - GraphQL unavailable")
