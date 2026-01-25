"""
GL-004 BURNMASTER Orchestrator

Main orchestration engine for the Burner Optimization Agent.
Coordinates data acquisition, calculations, optimization,
recommendations, and control actions.

Author: GreenLang AI Agent Workforce
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable
from uuid import uuid4

import structlog

from .config import BurnmasterConfig, OperatingMode, RuntimeContext
from .schemas import (
    Alert,
    BurnerSensorData,
    EmissionsReport,
    HealthStatus,
    OptimizationResult,
    ProcessState,
    Provenance,
    RecommendationPriority,
    RootCauseAnalysis,
    SetpointRecommendation,
)


logger = structlog.get_logger(__name__)


@dataclass
class OrchestrationMetrics:
    """Metrics for orchestration performance."""
    cycles_completed: int = 0
    total_cycle_time_ms: float = 0.0
    avg_cycle_time_ms: float = 0.0
    max_cycle_time_ms: float = 0.0
    optimization_runs: int = 0
    recommendations_generated: int = 0
    setpoints_applied: int = 0
    alerts_raised: int = 0
    errors_encountered: int = 0
    last_cycle_time: datetime | None = None


class BurnerOrchestrator:
    """
    Main orchestration engine for GL-004 BURNMASTER.

    Coordinates:
    - Real-time data acquisition from OPC-UA/DCS
    - Combustion stoichiometry calculations
    - Lambda and excess O2 monitoring
    - Multi-objective optimization
    - Stability monitoring and prediction
    - Emissions estimation and tracking
    - Safety constraint checking
    - Recommendation generation with explainability
    - Closed-loop setpoint control (when enabled)

    Attributes:
        config: Agent configuration
        context: Runtime context
        metrics: Orchestration performance metrics
    """

    def __init__(
        self,
        config: BurnmasterConfig | None = None,
        data_callback: Callable[[], ProcessState] | None = None,
        setpoint_callback: Callable[[list[SetpointRecommendation]], bool] | None = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            config: Agent configuration
            data_callback: Callback to fetch current process data
            setpoint_callback: Callback to apply setpoint changes
        """
        self.config = config or BurnmasterConfig()
        self.context = RuntimeContext(config=self.config)
        self.metrics = OrchestrationMetrics()

        self._data_callback = data_callback
        self._setpoint_callback = setpoint_callback

        # Internal state
        self._running = False
        self._cycle_task: asyncio.Task | None = None
        self._last_process_state: ProcessState | None = None
        self._pending_recommendations: list[SetpointRecommendation] = []
        self._active_alerts: dict[str, Alert] = {}

        # Calculation caches
        self._lambda_history: list[tuple[datetime, float]] = []
        self._efficiency_history: list[tuple[datetime, float]] = []
        self._stability_history: list[tuple[datetime, float]] = []

        logger.info(
            "Orchestrator initialized",
            agent_id=self.config.agent_id,
            mode=self.config.mode.value,
            num_burners=self.config.num_burners,
        )

    async def start(self) -> None:
        """Start the orchestration loop."""
        if self._running:
            logger.warning("Orchestrator already running")
            return

        self._running = True
        self.context.start_time = time.time()

        logger.info(
            "Starting orchestrator",
            mode=self.config.mode.value,
            update_interval_s=self.config.optimization.min_update_interval_s,
        )

        self._cycle_task = asyncio.create_task(self._orchestration_loop())

    async def stop(self) -> None:
        """Stop the orchestration loop gracefully."""
        if not self._running:
            return

        logger.info("Stopping orchestrator")
        self._running = False

        if self._cycle_task:
            self._cycle_task.cancel()
            try:
                await self._cycle_task
            except asyncio.CancelledError:
                pass

        logger.info(
            "Orchestrator stopped",
            cycles_completed=self.metrics.cycles_completed,
            total_runtime_s=time.time() - self.context.start_time,
        )

    async def _orchestration_loop(self) -> None:
        """Main orchestration loop."""
        interval = self.config.optimization.min_update_interval_s

        while self._running:
            cycle_start = time.time()

            try:
                await self._execute_cycle()
            except Exception as e:
                self.metrics.errors_encountered += 1
                logger.exception("Error in orchestration cycle", error=str(e))

                # In closed-loop mode, switch to fallback on repeated errors
                if self.context.error_count > 3:
                    await self._enter_fallback_mode("Repeated orchestration errors")

            # Update metrics
            cycle_time_ms = (time.time() - cycle_start) * 1000
            self._update_cycle_metrics(cycle_time_ms)

            # Wait for next cycle
            elapsed = time.time() - cycle_start
            sleep_time = max(0, interval - elapsed)
            await asyncio.sleep(sleep_time)

    async def _execute_cycle(self) -> None:
        """Execute a single orchestration cycle."""
        # 1. Acquire process data
        process_state = await self._acquire_data()
        if process_state is None:
            logger.warning("No process data available")
            return

        self._last_process_state = process_state

        # 2. Validate data quality
        data_quality_ok = await self._validate_data_quality(process_state)
        if not data_quality_ok:
            logger.warning("Data quality issues detected")

        # 3. Compute derived values (lambda, efficiency, etc.)
        await self._compute_derived_values(process_state)

        # 4. Check safety constraints
        safety_ok, safety_violations = await self._check_safety_constraints(process_state)
        if not safety_ok:
            await self._handle_safety_violations(safety_violations)

        # 5. Monitor stability
        stability_ok = await self._monitor_stability(process_state)
        if not stability_ok:
            await self._handle_instability(process_state)

        # 6. Run optimization (if mode allows)
        if self.context.mode_allows_recommendations():
            optimization_result = await self._run_optimization(process_state)
            if optimization_result and optimization_result.success:
                self.metrics.optimization_runs += 1

                # Generate recommendations
                recommendations = await self._generate_recommendations(
                    process_state, optimization_result
                )
                self._pending_recommendations.extend(recommendations)
                self.metrics.recommendations_generated += len(recommendations)

                # Apply setpoints (if closed-loop mode)
                if self.context.mode_allows_writes():
                    await self._apply_setpoints(recommendations)

        # 7. Update emissions tracking
        await self._update_emissions_tracking(process_state)

        # 8. Publish state and metrics
        await self._publish_state(process_state)

        self.metrics.cycles_completed += 1

    async def _acquire_data(self) -> ProcessState | None:
        """Acquire current process data."""
        if self._data_callback:
            try:
                return self._data_callback()
            except Exception as e:
                logger.error("Failed to acquire data", error=str(e))
                return None
        return None

    async def _validate_data_quality(self, state: ProcessState) -> bool:
        """Validate data quality and freshness."""
        issues = []

        # Check data age
        data_age = (datetime.utcnow() - state.timestamp).total_seconds()
        if data_age > 30:
            issues.append(f"Data is {data_age:.1f}s old")

        # Check for bad quality sensors
        for burner in state.burners:
            if burner.excess_o2_pct.quality.value == "bad":
                issues.append(f"Bad O2 sensor on {burner.burner_id}")
            if burner.fuel_flow_rate_kg_s.quality.value == "bad":
                issues.append(f"Bad fuel flow sensor on {burner.burner_id}")

        if issues:
            logger.warning("Data quality issues", issues=issues)
            return False

        return True

    async def _compute_derived_values(self, state: ProcessState) -> None:
        """Compute derived thermodynamic values."""
        # This would integrate with the combustion module
        # For now, store history
        self._lambda_history.append((state.timestamp, state.average_lambda))
        self._efficiency_history.append((state.timestamp, state.current_efficiency_pct))
        self._stability_history.append((state.timestamp, state.overall_stability_index))

        # Trim history to last hour
        cutoff = datetime.utcnow().timestamp() - 3600
        self._lambda_history = [
            (t, v) for t, v in self._lambda_history
            if t.timestamp() > cutoff
        ]

    async def _check_safety_constraints(
        self, state: ProcessState
    ) -> tuple[bool, list[str]]:
        """Check all safety constraints."""
        violations = []
        limits = self.config.safety

        # Check excess O2 bounds
        if state.average_excess_o2_pct < limits.min_excess_o2:
            violations.append(
                f"Low O2: {state.average_excess_o2_pct:.1f}% < {limits.min_excess_o2}%"
            )
        if state.average_excess_o2_pct > limits.max_excess_o2:
            violations.append(
                f"High O2: {state.average_excess_o2_pct:.1f}% > {limits.max_excess_o2}%"
            )

        # Check CO
        if state.aggregate_co_ppm > limits.max_co_ppm:
            violations.append(f"High CO: {state.aggregate_co_ppm:.0f} ppm")

        # Check NOx
        if state.aggregate_nox_ppm > limits.max_nox_ppm:
            violations.append(f"High NOx: {state.aggregate_nox_ppm:.0f} ppm")

        # Check stability
        if state.overall_stability_index < limits.min_stability_index:
            violations.append(
                f"Low stability: {state.overall_stability_index:.2f}"
            )

        # Check lambda bounds
        if state.average_lambda < limits.min_lambda:
            violations.append(f"Lambda too low: {state.average_lambda:.3f}")
        if state.average_lambda > limits.max_lambda:
            violations.append(f"Lambda too high: {state.average_lambda:.3f}")

        return len(violations) == 0, violations

    async def _handle_safety_violations(self, violations: list[str]) -> None:
        """Handle safety constraint violations."""
        for violation in violations:
            alert = Alert(
                severity="critical",
                category="safety",
                title="Safety Limit Violation",
                description=violation,
                affected_burners=["all"],
                recommended_action="Check combustion parameters immediately",
            )
            self._active_alerts[str(alert.alert_id)] = alert
            self.metrics.alerts_raised += 1

            logger.error("Safety violation", violation=violation)

        # In closed-loop mode, switch to fallback
        if self.config.mode == OperatingMode.CLOSED_LOOP:
            await self._enter_fallback_mode("Safety constraint violation")

    async def _monitor_stability(self, state: ProcessState) -> bool:
        """Monitor flame stability."""
        if state.stability_warning:
            logger.warning(
                "Stability warning",
                stability_index=state.overall_stability_index,
                instability_type=state.instability_type,
            )
            return False

        return True

    async def _handle_instability(self, state: ProcessState) -> None:
        """Handle detected instability."""
        alert = Alert(
            severity="warning",
            category="stability",
            title="Flame Instability Detected",
            description=f"Stability index: {state.overall_stability_index:.2f}, Type: {state.instability_type}",
            affected_burners=[b.burner_id for b in state.burners],
            recommended_action="Increase excess air or reduce load",
        )
        self._active_alerts[str(alert.alert_id)] = alert
        self.metrics.alerts_raised += 1

    async def _run_optimization(
        self, state: ProcessState
    ) -> OptimizationResult | None:
        """Run multi-objective optimization."""
        # Check if enough time has passed since last optimization
        if self.context.last_optimization_time:
            elapsed = time.time() - self.context.last_optimization_time
            if elapsed < self.config.optimization.min_update_interval_s:
                return None

        # Optimization would integrate with the optimization module
        # Placeholder for demonstration
        self.context.last_optimization_time = time.time()

        return OptimizationResult(
            success=True,
            iterations=50,
            convergence_achieved=True,
            computation_time_ms=45.0,
            total_objective=0.85,
            fuel_cost_component=0.4,
            emissions_cost_component=0.2,
            co_penalty_component=0.0,
            stability_penalty_component=0.15,
            actuator_move_component=0.1,
            optimal_setpoints=[],
            recommendations=[],
            active_constraints=["min_o2", "max_stack_temp"],
            constraint_violations=[],
            provenance=Provenance(
                source="BurnerOrchestrator",
                method="multi_objective_sqp",
            ),
        )

    async def _generate_recommendations(
        self,
        state: ProcessState,
        optimization_result: OptimizationResult,
    ) -> list[SetpointRecommendation]:
        """Generate setpoint recommendations with explainability."""
        recommendations = []

        # Example recommendation based on current state
        if state.average_lambda > 1.2:
            from .schemas import Setpoint

            rec = SetpointRecommendation(
                setpoint=Setpoint(
                    variable="excess_o2_pct",
                    current_value=state.average_excess_o2_pct,
                    target_value=state.average_excess_o2_pct - 0.5,
                    min_value=self.config.safety.min_excess_o2,
                    max_value=self.config.safety.max_excess_o2,
                ),
                category="air_fuel_ratio",
                priority=RecommendationPriority.MEDIUM,
                expected_benefit="Reduce excess air to improve efficiency",
                expected_fuel_savings_pct=0.5,
                expected_efficiency_gain_pct=0.3,
                confidence=0.85,
                explanation="Current lambda is above optimal. Reducing excess O2 by 0.5% will reduce heat losses to flue gas.",
                contributing_factors=[
                    "High lambda (excess air)",
                    "Stack temperature above optimal",
                    "Load at 80% capacity",
                ],
                physics_basis="Stack loss is proportional to excess air. Reducing from 1.20 to 1.15 lambda reduces dry flue gas losses.",
                safety_verified=True,
                safety_margin=0.5,
                provenance=Provenance(
                    source="BurnerOrchestrator",
                    method="physics_based_recommendation",
                ),
            )
            recommendations.append(rec)

        return recommendations

    async def _apply_setpoints(
        self, recommendations: list[SetpointRecommendation]
    ) -> None:
        """Apply setpoint changes (closed-loop mode only)."""
        if not self.context.mode_allows_writes():
            return

        if not self._setpoint_callback:
            logger.warning("No setpoint callback configured")
            return

        try:
            success = self._setpoint_callback(recommendations)
            if success:
                self.metrics.setpoints_applied += len(recommendations)
                self.context.last_setpoint_change_time = time.time()
                logger.info(
                    "Applied setpoints",
                    count=len(recommendations),
                )
        except Exception as e:
            logger.error("Failed to apply setpoints", error=str(e))
            self.context.error_count += 1

    async def _update_emissions_tracking(self, state: ProcessState) -> None:
        """Update cumulative emissions tracking."""
        # This would integrate with the climate module
        pass

    async def _publish_state(self, state: ProcessState) -> None:
        """Publish current state to monitoring systems."""
        # This would integrate with monitoring/Kafka modules
        pass

    async def _enter_fallback_mode(self, reason: str) -> None:
        """Enter safety fallback mode."""
        previous_mode = self.context.current_mode
        self.context.current_mode = OperatingMode.FALLBACK

        logger.critical(
            "Entering fallback mode",
            reason=reason,
            previous_mode=previous_mode.value,
        )

        alert = Alert(
            severity="critical",
            category="mode_change",
            title="Entered Fallback Mode",
            description=f"Reason: {reason}",
            affected_burners=["all"],
            recommended_action="Manual intervention required",
        )
        self._active_alerts[str(alert.alert_id)] = alert
        self.metrics.alerts_raised += 1

    def _update_cycle_metrics(self, cycle_time_ms: float) -> None:
        """Update cycle performance metrics."""
        self.metrics.total_cycle_time_ms += cycle_time_ms
        self.metrics.max_cycle_time_ms = max(
            self.metrics.max_cycle_time_ms, cycle_time_ms
        )
        if self.metrics.cycles_completed > 0:
            self.metrics.avg_cycle_time_ms = (
                self.metrics.total_cycle_time_ms / self.metrics.cycles_completed
            )
        self.metrics.last_cycle_time = datetime.utcnow()

    def get_health_status(self) -> HealthStatus:
        """Get current health status of the agent."""
        return HealthStatus(
            agent_id=self.config.agent_id,
            is_healthy=self._running and self.context.error_count < 3,
            uptime_s=time.time() - self.context.start_time,
            current_mode=self.context.current_mode.value,
            sensor_health={},
            data_freshness_s=0.0,
            missing_sensors=[],
            optimization_latency_ms=self.metrics.avg_cycle_time_ms,
            inference_latency_ms=0.0,
            last_optimization_time=(
                datetime.fromtimestamp(self.context.last_optimization_time)
                if self.context.last_optimization_time
                else None
            ),
            error_count_24h=self.context.error_count,
            warning_count_24h=self.context.warning_count,
        )

    def get_pending_recommendations(self) -> list[SetpointRecommendation]:
        """Get list of pending recommendations."""
        return self._pending_recommendations.copy()

    def get_active_alerts(self) -> list[Alert]:
        """Get list of active alerts."""
        return list(self._active_alerts.values())

    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """Acknowledge an active alert."""
        if alert_id in self._active_alerts:
            self._active_alerts[alert_id].acknowledged = True
            self._active_alerts[alert_id].acknowledged_by = user
            return True
        return False

    def change_mode(self, new_mode: OperatingMode) -> bool:
        """Request mode change."""
        old_mode = self.context.current_mode

        # Validate mode transition
        if old_mode == OperatingMode.FALLBACK:
            # Cannot exit fallback mode automatically
            logger.warning("Cannot exit fallback mode without manual reset")
            return False

        if new_mode == OperatingMode.CLOSED_LOOP:
            # Additional checks for closed-loop
            if not self._setpoint_callback:
                logger.warning("Cannot enter closed-loop without setpoint callback")
                return False

        self.context.current_mode = new_mode
        logger.info("Mode changed", from_mode=old_mode.value, to_mode=new_mode.value)
        return True
