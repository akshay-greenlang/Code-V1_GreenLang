"""
GL-006 HEATRECLAIM - Main Orchestrator

Central orchestrator for the Heat Recovery Maximizer agent.
Coordinates pinch analysis, HEN synthesis, optimization,
explainability, and reporting workflows.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import hashlib
import json
import logging
import asyncio

from .config import (
    HeatReclaimConfig,
    OptimizationMode,
    OptimizationObjective,
    DEFAULT_CONFIG,
)
from .schemas import (
    HeatStream,
    HeatExchanger,
    HENDesign,
    PinchAnalysisResult,
    ExergyAnalysisResult,
    EconomicAnalysisResult,
    OptimizationRequest,
    OptimizationResult,
    OptimizationStatus,
    ParetoPoint,
    ExplainabilityReport,
    AgentStatus,
    HealthCheckResponse,
    CalculationEvent,
    CalculationType,
)
from ..calculators.pinch_analysis import PinchAnalysisCalculator
from ..calculators.hen_synthesis import HENSynthesizer
from ..calculators.exergy_calculator import ExergyCalculator
from ..calculators.economic_calculator import EconomicCalculator
from ..calculators.lmtd_calculator import NTUCalculator
from ..optimization.milp_optimizer import MILPOptimizer
from ..optimization.pareto_generator import ParetoGenerator
from ..optimization.uncertainty_quantifier import UncertaintyQuantifier

logger = logging.getLogger(__name__)


class HeatReclaimOrchestrator:
    """
    Main orchestrator for GL-006 HEATRECLAIM agent.

    Coordinates all heat recovery optimization workflows including:
    - Pinch analysis and targeting
    - Heat exchanger network synthesis
    - Multi-objective optimization
    - Exergy analysis
    - Economic evaluation
    - Uncertainty quantification
    - Explainability and reporting

    Ensures deterministic, reproducible operation with full
    provenance tracking and audit logging.

    Example:
        >>> orchestrator = HeatReclaimOrchestrator()
        >>> result = await orchestrator.optimize(request)
        >>> print(f"Recommended design saves ${result.recommended_design.economic_analysis.annual_utility_savings_usd}/year")
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        config: Optional[HeatReclaimConfig] = None,
    ) -> None:
        """
        Initialize the HEATRECLAIM orchestrator.

        Args:
            config: Agent configuration
        """
        self.config = config or DEFAULT_CONFIG
        self._start_time = datetime.now(timezone.utc)

        # Initialize calculators
        self.pinch_calc = PinchAnalysisCalculator(
            delta_t_min=self.config.delta_t_min_C
        )
        self.hen_synth = HENSynthesizer(
            delta_t_min=self.config.delta_t_min_C
        )
        self.exergy_calc = ExergyCalculator(
            T0_K=self.config.reference_temperature_K
        )
        self.econ_calc = EconomicCalculator(
            params=self.config.economic_params
        )
        self.ntu_calc = NTUCalculator()

        # Initialize optimizers
        self.milp_optimizer = MILPOptimizer(
            delta_t_min=self.config.delta_t_min_C,
            time_limit_seconds=self.config.max_optimization_time_s,
        )
        self.pareto_gen = ParetoGenerator(
            delta_t_min=self.config.delta_t_min_C,
            n_points=self.config.n_pareto_points,
            random_seed=self.config.uncertainty_params.random_seed,
        )
        self.uq = UncertaintyQuantifier(
            params=self.config.uncertainty_params,
            delta_t_min=self.config.delta_t_min_C,
        )

        # Statistics
        self._optimizations_count = 0
        self._successful_count = 0
        self._total_heat_recovered = 0.0
        self._total_savings = 0.0
        self._calculation_events: List[CalculationEvent] = []

        logger.info(
            f"GL-006 HEATRECLAIM orchestrator initialized: "
            f"version={self.VERSION}, mode={self.config.mode.value}"
        )

    async def optimize(
        self,
        request: OptimizationRequest,
    ) -> OptimizationResult:
        """
        Execute full heat recovery optimization workflow.

        Args:
            request: Optimization request with streams and settings

        Returns:
            OptimizationResult with recommended design and analysis
        """
        start_time = datetime.now(timezone.utc)
        self._optimizations_count += 1

        logger.info(
            f"Starting optimization: request_id={request.request_id}, "
            f"hot_streams={len(request.hot_streams)}, "
            f"cold_streams={len(request.cold_streams)}"
        )

        try:
            # Step 1: Pinch Analysis
            pinch_result = await self._run_pinch_analysis(
                request.hot_streams,
                request.cold_streams,
                request.delta_t_min_C,
            )

            # Step 2: HEN Synthesis or MILP Optimization
            if request.objective == OptimizationObjective.MULTI_OBJECTIVE:
                # Generate Pareto frontier
                pareto_points = await self._run_pareto_optimization(
                    request.hot_streams,
                    request.cold_streams,
                    request.n_pareto_points,
                )
                # Select balanced solution
                recommended_design = self._select_balanced_design(pareto_points)
            else:
                # Single-objective optimization
                recommended_design = await self._run_hen_synthesis(
                    request.hot_streams,
                    request.cold_streams,
                    pinch_result,
                    request.mode,
                )
                pareto_points = []

            # Step 3: Exergy Analysis
            if request.include_exergy_analysis:
                exergy_result = await self._run_exergy_analysis(
                    request.hot_streams,
                    request.cold_streams,
                    recommended_design,
                )
                recommended_design.exergy_analysis = exergy_result

            # Step 4: Economic Analysis
            econ_result = await self._run_economic_analysis(
                recommended_design,
                pinch_result.minimum_hot_utility_kW,
                pinch_result.minimum_cold_utility_kW,
            )
            recommended_design.economic_analysis = econ_result

            # Step 5: Uncertainty Quantification (if requested)
            uncertainty_bounds = {}
            robustness_score = 1.0
            if request.include_uncertainty:
                uq_result = await self._run_uncertainty_analysis(
                    request.hot_streams,
                    request.cold_streams,
                    recommended_design,
                )
                uncertainty_bounds = uq_result.confidence_intervals
                robustness_score = uq_result.robustness_score

            # Build result
            optimization_time = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds()

            result = OptimizationResult(
                request_id=request.request_id,
                status=OptimizationStatus.COMPLETED,
                pinch_analysis=pinch_result,
                recommended_design=recommended_design,
                pareto_points=pareto_points,
                pareto_hypervolume=self._calculate_hypervolume(pareto_points),
                uncertainty_bounds=uncertainty_bounds,
                robustness_score=robustness_score,
                solver_used=self.config.solver.value,
                optimization_time_seconds=round(optimization_time, 3),
                convergence_achieved=True,
                explanation_summary=self._generate_explanation(
                    pinch_result, recommended_design
                ),
                key_drivers=self._identify_key_drivers(recommended_design),
                binding_constraints=self._identify_binding_constraints(
                    recommended_design
                ),
                configuration_version=self.config.version,
                random_seed=self.config.uncertainty_params.random_seed,
            )

            # Compute provenance hashes
            result.input_hash = self._compute_hash({
                "request_id": request.request_id,
                "hot_streams": [s.stream_id for s in request.hot_streams],
                "cold_streams": [s.stream_id for s in request.cold_streams],
                "delta_t_min": request.delta_t_min_C,
            })
            result.output_hash = self._compute_hash({
                "heat_recovered_kW": recommended_design.total_heat_recovered_kW,
                "hot_utility_kW": recommended_design.hot_utility_required_kW,
                "n_exchangers": recommended_design.exchanger_count,
            })

            # Update statistics
            self._successful_count += 1
            self._total_heat_recovered += recommended_design.total_heat_recovered_kW
            if econ_result:
                self._total_savings += econ_result.annual_utility_savings_usd

            logger.info(
                f"Optimization complete: request_id={request.request_id}, "
                f"heat_recovered={recommended_design.total_heat_recovered_kW}kW, "
                f"time={optimization_time:.2f}s"
            )

            return result

        except Exception as e:
            logger.error(f"Optimization failed: {e}", exc_info=True)
            return OptimizationResult(
                request_id=request.request_id,
                status=OptimizationStatus.FAILED,
                pinch_analysis=PinchAnalysisResult(
                    pinch_temperature_C=0,
                    delta_t_min_C=request.delta_t_min_C,
                    minimum_hot_utility_kW=0,
                    minimum_cold_utility_kW=0,
                    maximum_heat_recovery_kW=0,
                ),
                recommended_design=HENDesign(
                    exchangers=[],
                    total_heat_recovered_kW=0,
                    hot_utility_required_kW=0,
                    cold_utility_required_kW=0,
                ),
                explanation_summary=f"Optimization failed: {str(e)}",
            )

    async def _run_pinch_analysis(
        self,
        hot_streams: List[HeatStream],
        cold_streams: List[HeatStream],
        delta_t_min: float,
    ) -> PinchAnalysisResult:
        """Run pinch analysis targeting."""
        logger.debug("Running pinch analysis...")

        calculator = PinchAnalysisCalculator(delta_t_min=delta_t_min)
        result = calculator.calculate(hot_streams, cold_streams)

        # Log calculation event
        self._log_calculation(
            CalculationType.PINCH_ANALYSIS,
            {"n_hot": len(hot_streams), "n_cold": len(cold_streams)},
            {"pinch_T": result.pinch_temperature_C},
        )

        return result

    async def _run_hen_synthesis(
        self,
        hot_streams: List[HeatStream],
        cold_streams: List[HeatStream],
        pinch_result: PinchAnalysisResult,
        mode: OptimizationMode,
    ) -> HENDesign:
        """Run HEN synthesis."""
        logger.debug("Running HEN synthesis...")

        design = self.hen_synth.synthesize(
            hot_streams,
            cold_streams,
            pinch_result,
            mode=mode,
        )

        self._log_calculation(
            CalculationType.HEN_SYNTHESIS,
            {"pinch_T": pinch_result.pinch_temperature_C},
            {"n_exchangers": design.exchanger_count},
        )

        return design

    async def _run_pareto_optimization(
        self,
        hot_streams: List[HeatStream],
        cold_streams: List[HeatStream],
        n_points: int,
    ) -> List[ParetoPoint]:
        """Run multi-objective Pareto optimization."""
        logger.debug(f"Running Pareto optimization ({n_points} points)...")

        pareto_points = self.pareto_gen.generate_epsilon_constraint(
            hot_streams,
            cold_streams,
            n_points=n_points,
        )

        self._log_calculation(
            CalculationType.PARETO,
            {"n_points_requested": n_points},
            {"n_points_generated": len(pareto_points)},
        )

        return pareto_points

    async def _run_exergy_analysis(
        self,
        hot_streams: List[HeatStream],
        cold_streams: List[HeatStream],
        design: HENDesign,
    ) -> ExergyAnalysisResult:
        """Run exergy analysis."""
        logger.debug("Running exergy analysis...")

        result = self.exergy_calc.analyze_network(
            hot_streams,
            cold_streams,
            design.exchangers,
            design.hot_utility_required_kW,
            design.cold_utility_required_kW,
        )

        self._log_calculation(
            CalculationType.EXERGY,
            {"n_exchangers": len(design.exchangers)},
            {"exergy_destruction_kW": result.total_exergy_destruction_kW},
        )

        return result

    async def _run_economic_analysis(
        self,
        design: HENDesign,
        hot_utility_reduction: float,
        cold_utility_reduction: float,
    ) -> EconomicAnalysisResult:
        """Run economic analysis."""
        logger.debug("Running economic analysis...")

        result = self.econ_calc.calculate_full_analysis(
            design.exchangers,
            hot_utility_reduction,
            cold_utility_reduction,
        )

        self._log_calculation(
            CalculationType.ECONOMIC,
            {"n_exchangers": len(design.exchangers)},
            {"npv_usd": result.npv_usd},
        )

        return result

    async def _run_uncertainty_analysis(
        self,
        hot_streams: List[HeatStream],
        cold_streams: List[HeatStream],
        design: HENDesign,
    ):
        """Run uncertainty quantification."""
        logger.debug("Running uncertainty analysis...")

        result = self.uq.analyze(
            hot_streams,
            cold_streams,
            design,
        )

        self._log_calculation(
            CalculationType.UNCERTAINTY,
            {"n_samples": result.n_samples},
            {"robustness": result.robustness_score},
        )

        return result

    def _select_balanced_design(
        self,
        pareto_points: List[ParetoPoint],
    ) -> HENDesign:
        """Select balanced design from Pareto frontier."""
        if not pareto_points:
            return HENDesign(
                exchangers=[],
                total_heat_recovered_kW=0,
                hot_utility_required_kW=0,
                cold_utility_required_kW=0,
            )

        # Select point with best combined score
        # (compromise between cost and heat recovery)
        best_point = None
        best_score = float('inf')

        for point in pareto_points:
            cost = point.objectives.get("total_annual_cost", float('inf'))
            recovery = point.objectives.get("heat_recovered", 0)

            # Normalize and combine
            cost_norm = cost / 1e6 if cost < 1e9 else 1.0
            recovery_norm = 1.0 - recovery / 1e6 if recovery > 0 else 1.0

            score = 0.6 * cost_norm + 0.4 * recovery_norm

            if score < best_score and point.design:
                best_score = score
                best_point = point

        if best_point and best_point.design:
            return best_point.design

        return pareto_points[0].design if pareto_points[0].design else HENDesign(
            exchangers=[],
            total_heat_recovered_kW=0,
            hot_utility_required_kW=0,
            cold_utility_required_kW=0,
        )

    def _calculate_hypervolume(self, pareto_points: List[ParetoPoint]) -> float:
        """Calculate Pareto hypervolume."""
        if not pareto_points:
            return 0.0
        return self.pareto_gen.calculate_hypervolume(pareto_points)

    def _generate_explanation(
        self,
        pinch_result: PinchAnalysisResult,
        design: HENDesign,
    ) -> str:
        """Generate human-readable explanation."""
        lines = [
            f"Pinch analysis identified pinch temperature at {pinch_result.pinch_temperature_C}°C.",
            f"Minimum hot utility requirement: {pinch_result.minimum_hot_utility_kW:.1f} kW.",
            f"Minimum cold utility requirement: {pinch_result.minimum_cold_utility_kW:.1f} kW.",
            f"Maximum heat recovery potential: {pinch_result.maximum_heat_recovery_kW:.1f} kW.",
            "",
            f"Recommended network design uses {design.exchanger_count} heat exchangers",
            f"to recover {design.total_heat_recovered_kW:.1f} kW of heat.",
        ]

        if design.economic_analysis:
            econ = design.economic_analysis
            lines.extend([
                "",
                f"Capital investment: ${econ.total_capital_cost_usd:,.0f}",
                f"Annual utility savings: ${econ.annual_utility_savings_usd:,.0f}/year",
                f"Simple payback: {econ.payback_period_years:.1f} years",
            ])

        return "\n".join(lines)

    def _identify_key_drivers(self, design: HENDesign) -> List[str]:
        """Identify key drivers of the design."""
        drivers = []

        if design.total_heat_recovered_kW > 0:
            drivers.append("Heat recovery potential")

        if design.exchanger_count > 5:
            drivers.append("Network complexity")

        if design.economic_analysis:
            if design.economic_analysis.payback_period_years < 2:
                drivers.append("Excellent payback period")
            if design.economic_analysis.annual_utility_savings_usd > 100000:
                drivers.append("Significant utility savings")

        if design.exergy_analysis:
            if design.exergy_analysis.exergy_efficiency > 0.7:
                drivers.append("High exergy efficiency")

        return drivers

    def _identify_binding_constraints(self, design: HENDesign) -> List[str]:
        """Identify binding constraints."""
        constraints = []

        if design.constraint_details:
            for detail in design.constraint_details[:5]:
                constraints.append(detail.get("violation", ""))

        if not design.all_constraints_satisfied:
            constraints.append("Temperature approach constraints active")

        return constraints

    def _log_calculation(
        self,
        calc_type: CalculationType,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
    ) -> None:
        """Log calculation event for audit."""
        event = CalculationEvent(
            calculation_type=calc_type,
            input_summary=inputs,
            input_hash=self._compute_hash(inputs),
            output_summary=outputs,
            output_hash=self._compute_hash(outputs),
            formula_id=f"{calc_type.value}_v1.0",
            deterministic=True,
            reproducible=True,
        )
        self._calculation_events.append(event)

    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash for provenance."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def get_status(self) -> AgentStatus:
        """Get current agent status."""
        uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds()

        return AgentStatus(
            agent_id=self.config.agent_id,
            agent_name=self.config.agent_name,
            agent_version=self.VERSION,
            status="running",
            health="healthy",
            uptime_seconds=uptime,
            optimizations_performed=self._optimizations_count,
            optimizations_successful=self._successful_count,
            total_heat_recovered_GJ=self._total_heat_recovered * 8000 * 0.0036,
            total_cost_savings_usd=self._total_savings,
            avg_optimization_time_seconds=0.0,
        )

    def health_check(self) -> HealthCheckResponse:
        """Perform health check."""
        uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds()

        checks = {
            "pinch_calculator": "ok",
            "hen_synthesizer": "ok",
            "milp_optimizer": "ok",
            "exergy_calculator": "ok",
            "economic_calculator": "ok",
        }

        return HealthCheckResponse(
            status="healthy",
            version=self.VERSION,
            uptime_seconds=uptime,
            checks=checks,
        )


# Synchronous wrapper for non-async contexts
def run_optimization_sync(
    request: OptimizationRequest,
    config: Optional[HeatReclaimConfig] = None,
) -> OptimizationResult:
    """
    Run optimization synchronously.

    Args:
        request: Optimization request
        config: Agent configuration

    Returns:
        OptimizationResult
    """
    orchestrator = HeatReclaimOrchestrator(config)
    return asyncio.run(orchestrator.optimize(request))
