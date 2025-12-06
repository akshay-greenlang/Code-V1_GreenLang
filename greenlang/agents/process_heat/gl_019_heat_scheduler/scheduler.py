"""
GL-019 HEATSCHEDULER - Main Heat Scheduler Agent

Main Heat Scheduler Agent implementation that orchestrates load forecasting,
thermal storage optimization, demand charge management, and production
planning for industrial process heat systems.

Key Features:
    - ML-based 24-48 hour load forecasting
    - Thermal storage charge/discharge optimization
    - Peak demand reduction and load shifting
    - Production schedule integration
    - Weather-based load adjustment
    - Zero-hallucination: All optimizations are deterministic
    - Complete provenance tracking with SHA-256 hashing

Example:
    >>> from greenlang.agents.process_heat.gl_019_heat_scheduler import (
    ...     HeatSchedulerAgent,
    ...     HeatSchedulerConfig,
    ... )
    >>> config = HeatSchedulerConfig(...)
    >>> agent = HeatSchedulerAgent(config)
    >>> result = agent.process(input_data)
    >>> print(f"Savings: ${result.total_savings_usd}")

Author: GreenLang Team
Version: 1.0.0
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
import hashlib
import json
import logging

from greenlang.agents.process_heat.shared.base_agent import (
    AgentCapability,
    AgentConfig,
    BaseProcessHeatAgent,
    ProcessingError,
    SafetyLevel,
    ValidationError,
)
from greenlang.agents.process_heat.shared.provenance import ProvenanceTracker
from greenlang.agents.process_heat.shared.audit import AuditLogger

from greenlang.agents.process_heat.gl_019_heat_scheduler.config import (
    HeatSchedulerConfig,
    TariffConfiguration,
    ThermalStorageConfiguration,
    LoadForecastingConfiguration,
    DemandChargeConfiguration,
    WeatherConfiguration,
)
from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import (
    HeatSchedulerInput,
    HeatSchedulerOutput,
    ScheduleStatus,
    ScheduleActionItem,
    ScheduleAction,
    LoadForecastResult,
    ThermalStorageResult,
    DemandChargeResult,
    ProductionScheduleResult,
    WeatherForecastResult,
)
from greenlang.agents.process_heat.gl_019_heat_scheduler.load_forecasting import (
    EnsembleForecaster,
    HistoricalDataPoint,
)
from greenlang.agents.process_heat.gl_019_heat_scheduler.thermal_storage import (
    ThermalStorageOptimizer,
)
from greenlang.agents.process_heat.gl_019_heat_scheduler.demand_charge import (
    DemandChargeOptimizer,
)
from greenlang.agents.process_heat.gl_019_heat_scheduler.production_planning import (
    ProductionPlanner,
    ProductionShift,
)
from greenlang.agents.process_heat.gl_019_heat_scheduler.weather_integration import (
    WeatherService,
)

logger = logging.getLogger(__name__)


# =============================================================================
# HEAT SCHEDULER AGENT
# =============================================================================

class HeatSchedulerAgent(
    BaseProcessHeatAgent[HeatSchedulerInput, HeatSchedulerOutput]
):
    """
    GL-019 HEATSCHEDULER Heat Scheduling Agent.

    Provides comprehensive heat load scheduling optimization for industrial
    process heat applications including:
    - ML-based load forecasting (24-48 hour horizon)
    - Thermal storage optimization (hot water tanks, PCM)
    - Demand charge optimization (peak reduction, load shifting)
    - Production schedule integration
    - Weather forecast integration

    All calculations are DETERMINISTIC with ZERO HALLUCINATION guarantees.
    ML models use ensemble methods (Gradient Boosting, Random Forest, ARIMA)
    for predictions, but all scheduling decisions are rule-based.

    Attributes:
        config: Heat scheduler configuration
        forecaster: ML-based load forecaster
        storage_optimizer: Thermal storage optimizer
        demand_optimizer: Demand charge optimizer
        production_planner: Production schedule integrator
        weather_service: Weather forecast service

    Example:
        >>> config = HeatSchedulerConfig(
        ...     tariffs=[TariffConfiguration(...)],
        ...     equipment=[EquipmentConfiguration(...)],
        ...     thermal_storage=[ThermalStorageConfiguration(...)],
        ... )
        >>> agent = HeatSchedulerAgent(config)
        >>> result = agent.process(input_data)
        >>> print(f"Peak: {result.peak_demand_kw}kW, Savings: ${result.total_savings_usd}")
    """

    def __init__(
        self,
        scheduler_config: HeatSchedulerConfig,
        safety_level: SafetyLevel = SafetyLevel.SIL_2,
    ) -> None:
        """
        Initialize the Heat Scheduler Agent.

        Args:
            scheduler_config: Complete heat scheduler configuration
            safety_level: Safety Integrity Level (default SIL-2)
        """
        # Create agent config
        agent_config = AgentConfig(
            agent_id=f"GL-019-{scheduler_config.agent_name.replace(' ', '-')}",
            agent_type="GL-019",
            name=scheduler_config.agent_name,
            version=scheduler_config.version,
            capabilities={
                AgentCapability.OPTIMIZATION,
                AgentCapability.PREDICTIVE_ANALYTICS,
                AgentCapability.REAL_TIME_MONITORING,
                AgentCapability.ML_INFERENCE,
            },
        )

        super().__init__(
            config=agent_config,
            safety_level=safety_level,
        )

        self.scheduler_config = scheduler_config

        # Initialize components
        self._init_components()

        # Initialize provenance tracker
        self.provenance_tracker = ProvenanceTracker(
            agent_id=agent_config.agent_id,
            agent_version=scheduler_config.version,
        )

        # Initialize audit logger
        self.audit_logger = AuditLogger(
            agent_id=agent_config.agent_id,
            agent_version=scheduler_config.version,
        )

        logger.info(
            f"HeatSchedulerAgent initialized: {scheduler_config.agent_name}"
        )

    def _init_components(self) -> None:
        """Initialize all sub-components."""
        config = self.scheduler_config

        # Load forecaster
        self.forecaster = EnsembleForecaster(config.load_forecasting)

        # Thermal storage optimizer
        self.storage_optimizer = ThermalStorageOptimizer(
            storage_configs=config.thermal_storage,
            tariff_config=config.tariffs[0] if config.tariffs else None,
        )

        # Demand charge optimizer
        self.demand_optimizer = DemandChargeOptimizer(
            config=config.demand_charge,
            tariff_config=config.tariffs[0] if config.tariffs else None,
        )

        # Production planner
        shifts = self._create_shifts_from_config()
        self.production_planner = ProductionPlanner(
            shifts=shifts,
            baseline_load_kw=config.demand_charge.peak_demand_limit_kw * 0.5,
            demand_limit_kw=config.demand_charge.peak_demand_limit_kw,
        )

        # Weather service
        self.weather_service = WeatherService(config.weather)

        logger.info("Heat scheduler components initialized")

    def _create_shifts_from_config(self) -> List[ProductionShift]:
        """Create production shifts from configuration."""
        # Default shifts if none configured
        return [
            ProductionShift(
                shift_id="day_shift",
                name="Day Shift",
                start_time_hour=6,
                duration_hours=8,
                days_active=[0, 1, 2, 3, 4],
                heat_load_factor=1.0,
                ramp_up_minutes=30,
            ),
            ProductionShift(
                shift_id="evening_shift",
                name="Evening Shift",
                start_time_hour=14,
                duration_hours=8,
                days_active=[0, 1, 2, 3, 4],
                heat_load_factor=0.9,
                ramp_up_minutes=15,
            ),
        ]

    def process(
        self,
        input_data: HeatSchedulerInput,
    ) -> HeatSchedulerOutput:
        """
        Process heat scheduling optimization request.

        This is the main entry point for heat scheduling. It:
        1. Validates input data
        2. Generates or uses provided load forecast
        3. Fetches weather forecast
        4. Optimizes thermal storage dispatch
        5. Optimizes demand charges
        6. Integrates production schedule
        7. Generates comprehensive schedule
        8. Returns output with provenance tracking

        Args:
            input_data: Heat scheduler input data

        Returns:
            HeatSchedulerOutput with schedule and savings

        Raises:
            ValidationError: If input validation fails
            ProcessingError: If processing fails
        """
        start_time = datetime.now(timezone.utc)
        logger.info(
            f"Processing heat schedule for {input_data.facility_id}, "
            f"horizon={input_data.optimization_horizon_hours}h"
        )

        try:
            # Validate input
            if not self.validate_input(input_data):
                raise ValidationError("Input validation failed")

            with self.safety_guard():
                # Step 1: Get or generate load forecast
                load_forecast = self._get_load_forecast(input_data)

                # Step 2: Get weather forecast
                weather_forecast = None
                if self.scheduler_config.weather.enabled:
                    weather_forecast = self._get_weather_forecast(input_data)

                # Step 3: Optimize thermal storage
                storage_result = None
                storage_capacity_kw = 0.0
                if self.scheduler_config.thermal_storage:
                    storage_result = self._optimize_storage(
                        load_forecast,
                        input_data.optimization_horizon_hours,
                    )
                    storage_capacity_kw = sum(
                        s.config.max_discharge_rate_kw
                        for s in self.storage_optimizer._units.values()
                    )

                # Step 4: Optimize demand charges
                demand_result = self._optimize_demand(
                    load_forecast,
                    storage_capacity_kw,
                )

                # Step 5: Schedule production
                production_result = None
                if input_data.production_orders:
                    production_result = self._schedule_production(
                        input_data.production_orders,
                        input_data.timestamp,
                        input_data.timestamp + timedelta(
                            hours=input_data.optimization_horizon_hours
                        ),
                    )

                # Step 6: Generate schedule actions
                schedule_actions = self._generate_schedule_actions(
                    load_forecast,
                    storage_result,
                    demand_result,
                    production_result,
                )

                # Step 7: Calculate costs and savings
                cost_results = self._calculate_costs(
                    load_forecast,
                    storage_result,
                    demand_result,
                )

                # Step 8: Calculate KPIs
                kpis = self._calculate_kpis(
                    load_forecast,
                    demand_result,
                    storage_result,
                )

                # Step 9: Generate alerts
                alerts = self._generate_alerts(
                    demand_result,
                    load_forecast,
                )

                # Calculate processing time
                processing_time = (
                    datetime.now(timezone.utc) - start_time
                ).total_seconds() * 1000

                # Create output
                output = HeatSchedulerOutput(
                    facility_id=input_data.facility_id,
                    request_id=input_data.request_id,
                    timestamp=datetime.now(timezone.utc),
                    status=ScheduleStatus.OPTIMAL,
                    processing_time_ms=round(processing_time, 2),
                    schedule_horizon_hours=input_data.optimization_horizon_hours,
                    schedule_actions=schedule_actions,
                    load_forecast=load_forecast,
                    storage_result=storage_result,
                    demand_result=demand_result,
                    production_result=production_result,
                    weather_forecast=weather_forecast,
                    baseline_cost_usd=cost_results["baseline"],
                    optimized_cost_usd=cost_results["optimized"],
                    total_savings_usd=cost_results["savings"],
                    savings_breakdown=cost_results["breakdown"],
                    total_energy_kwh=load_forecast.total_energy_kwh or 0.0,
                    peak_demand_kw=demand_result.optimized_peak_kw if demand_result else 0.0,
                    average_load_kw=load_forecast.avg_load_kw or 0.0,
                    load_factor_pct=self._calculate_load_factor(load_forecast),
                    kpis=kpis,
                    alerts=alerts,
                    metadata={
                        "agent_version": self.scheduler_config.version,
                        "optimization_mode": self.scheduler_config.optimization_parameters.primary_objective.value,
                    },
                )

                # Calculate provenance hash
                output.provenance_hash = self._calculate_provenance_hash(
                    input_data,
                    output,
                )
                output.input_hash = self._hash_object(input_data)

                # Validate output
                if not self.validate_output(output):
                    raise ValidationError("Output validation failed")

                # Audit log
                self.audit_logger.log_calculation(
                    calculation_type="heat_scheduling",
                    inputs={
                        "facility_id": input_data.facility_id,
                        "horizon_hours": input_data.optimization_horizon_hours,
                    },
                    outputs={
                        "total_savings": output.total_savings_usd,
                        "peak_demand_kw": output.peak_demand_kw,
                    },
                    formula_id="GL019_SCHEDULER",
                    duration_ms=processing_time,
                    provenance_hash=output.provenance_hash,
                )

                logger.info(
                    f"Heat scheduling complete: "
                    f"savings=${output.total_savings_usd:.2f}, "
                    f"peak={output.peak_demand_kw:.0f}kW "
                    f"({processing_time:.1f}ms)"
                )

                return output

        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Heat scheduling failed: {e}", exc_info=True)
            raise ProcessingError(f"Heat scheduling failed: {str(e)}") from e

    def validate_input(self, input_data: HeatSchedulerInput) -> bool:
        """
        Validate heat scheduler input data.

        Args:
            input_data: Input to validate

        Returns:
            True if valid
        """
        errors: List[str] = []

        # Check required fields
        if not input_data.facility_id:
            errors.append("Missing facility_id")

        if input_data.optimization_horizon_hours < 1:
            errors.append("Invalid optimization horizon")

        if input_data.current_load_kw < 0:
            errors.append("Invalid current load")

        if errors:
            logger.warning(f"Input validation errors: {errors}")
            return False

        return True

    def validate_output(self, output_data: HeatSchedulerOutput) -> bool:
        """
        Validate heat scheduler output data.

        Args:
            output_data: Output to validate

        Returns:
            True if valid
        """
        # Check required components
        if output_data.load_forecast is None:
            logger.error("Missing load forecast")
            return False

        # Check cost sanity
        if output_data.optimized_cost_usd < 0:
            logger.error("Negative optimized cost")
            return False

        if output_data.total_savings_usd < 0:
            logger.warning("Negative savings - suboptimal schedule")

        return True

    def _get_load_forecast(
        self,
        input_data: HeatSchedulerInput,
    ) -> LoadForecastResult:
        """Get or generate load forecast."""
        if input_data.load_forecast:
            return input_data.load_forecast

        # Generate forecast
        return self.forecaster.forecast(
            forecast_start=input_data.timestamp,
            horizon_hours=input_data.optimization_horizon_hours,
            weather_forecast=input_data.weather_forecast,
        )

    async def _get_weather_forecast_async(
        self,
        input_data: HeatSchedulerInput,
    ) -> Optional[WeatherForecastResult]:
        """Get weather forecast asynchronously."""
        if input_data.weather_forecast:
            return input_data.weather_forecast

        try:
            return await self.weather_service.get_forecast()
        except Exception as e:
            logger.warning(f"Weather forecast failed: {e}")
            return None

    def _get_weather_forecast(
        self,
        input_data: HeatSchedulerInput,
    ) -> Optional[WeatherForecastResult]:
        """Get weather forecast (sync wrapper)."""
        if input_data.weather_forecast:
            return input_data.weather_forecast

        # Note: In production, this would be async
        return None

    def _optimize_storage(
        self,
        load_forecast: LoadForecastResult,
        horizon_hours: int,
    ) -> Optional[ThermalStorageResult]:
        """Optimize thermal storage dispatch."""
        if not self.scheduler_config.thermal_storage:
            return None

        return self.storage_optimizer.optimize_dispatch(
            load_forecast=load_forecast,
            horizon_hours=horizon_hours,
            demand_limit_kw=self.scheduler_config.demand_charge.peak_demand_limit_kw,
        )

    def _optimize_demand(
        self,
        load_forecast: LoadForecastResult,
        storage_capacity_kw: float,
    ) -> DemandChargeResult:
        """Optimize demand charges."""
        return self.demand_optimizer.optimize(
            load_forecast=load_forecast,
            storage_capacity_kw=storage_capacity_kw,
        )

    def _schedule_production(
        self,
        orders,
        horizon_start: datetime,
        horizon_end: datetime,
    ) -> ProductionScheduleResult:
        """Schedule production orders."""
        return self.production_planner.schedule_orders(
            orders=orders,
            horizon_start=horizon_start,
            horizon_end=horizon_end,
        )

    def _generate_schedule_actions(
        self,
        load_forecast: LoadForecastResult,
        storage_result: Optional[ThermalStorageResult],
        demand_result: DemandChargeResult,
        production_result: Optional[ProductionScheduleResult],
    ) -> List[ScheduleActionItem]:
        """Generate schedule actions from optimization results."""
        actions: List[ScheduleActionItem] = []

        # Add storage actions
        if storage_result:
            for schedule in storage_result.unit_schedules:
                for point in schedule.dispatch_points:
                    if point.power_kw != 0:
                        action_type = (
                            ScheduleAction.STORAGE_CHARGE
                            if point.power_kw > 0
                            else ScheduleAction.STORAGE_DISCHARGE
                        )
                        actions.append(ScheduleActionItem(
                            timestamp=point.timestamp,
                            action_type=action_type,
                            storage_id=schedule.storage_id,
                            power_setpoint_kw=abs(point.power_kw),
                            duration_minutes=15,
                            reason=f"Storage {point.mode.value}",
                            priority=6,
                        ))

        # Add production actions
        if production_result:
            prod_actions = self.production_planner.generate_schedule_actions(
                production_result.scheduled_orders
            )
            actions.extend(prod_actions)

        # Sort by timestamp
        return sorted(actions, key=lambda a: a.timestamp)

    def _calculate_costs(
        self,
        load_forecast: LoadForecastResult,
        storage_result: Optional[ThermalStorageResult],
        demand_result: DemandChargeResult,
    ) -> Dict[str, Any]:
        """Calculate baseline and optimized costs."""
        # Get tariff
        tariff = (
            self.scheduler_config.tariffs[0]
            if self.scheduler_config.tariffs else None
        )

        # Calculate baseline energy cost
        baseline_energy_cost = 0.0
        if load_forecast.total_energy_kwh and tariff:
            # Simplified: use average rate
            avg_rate = (tariff.peak_rate_per_kwh + tariff.off_peak_rate_per_kwh) / 2
            baseline_energy_cost = load_forecast.total_energy_kwh * avg_rate

        # Baseline demand cost
        baseline_demand_cost = demand_result.baseline_demand_charge_usd

        baseline_total = baseline_energy_cost + baseline_demand_cost

        # Optimized costs
        optimized_demand_cost = demand_result.optimized_demand_charge_usd

        # Storage savings
        storage_savings = 0.0
        if storage_result:
            storage_savings = storage_result.total_savings_usd

        optimized_total = (
            baseline_energy_cost - storage_savings +
            optimized_demand_cost
        )

        savings = baseline_total - optimized_total

        return {
            "baseline": round(baseline_total, 2),
            "optimized": round(optimized_total, 2),
            "savings": round(savings, 2),
            "breakdown": {
                "energy_cost": round(baseline_energy_cost, 2),
                "demand_savings": round(demand_result.demand_charge_savings_usd, 2),
                "storage_arbitrage": round(
                    storage_result.total_energy_arbitrage_usd if storage_result else 0, 2
                ),
                "load_shift_savings": round(demand_result.load_shift_savings_usd, 2),
            },
        }

    def _calculate_kpis(
        self,
        load_forecast: LoadForecastResult,
        demand_result: DemandChargeResult,
        storage_result: Optional[ThermalStorageResult],
    ) -> Dict[str, float]:
        """Calculate key performance indicators."""
        kpis = {
            "peak_demand_kw": demand_result.optimized_peak_kw,
            "peak_reduction_pct": demand_result.peak_reduction_pct,
            "average_load_kw": load_forecast.avg_load_kw or 0.0,
            "load_factor_pct": self._calculate_load_factor(load_forecast),
            "total_energy_kwh": load_forecast.total_energy_kwh or 0.0,
        }

        if storage_result:
            kpis["storage_utilization_pct"] = (
                storage_result.current_soc_kwh /
                storage_result.total_storage_capacity_kwh * 100
                if storage_result.total_storage_capacity_kwh > 0 else 0
            )

        return {k: round(v, 2) for k, v in kpis.items()}

    def _calculate_load_factor(
        self,
        load_forecast: LoadForecastResult,
    ) -> float:
        """Calculate load factor percentage."""
        if not load_forecast.peak_load_kw or load_forecast.peak_load_kw == 0:
            return 0.0

        avg = load_forecast.avg_load_kw or 0.0
        return (avg / load_forecast.peak_load_kw) * 100

    def _generate_alerts(
        self,
        demand_result: DemandChargeResult,
        load_forecast: LoadForecastResult,
    ) -> List[Dict[str, Any]]:
        """Generate alerts from results."""
        alerts: List[Dict[str, Any]] = []

        # Demand alerts
        if demand_result.peak_limit_exceeded:
            alerts.append({
                "type": "DEMAND_LIMIT",
                "severity": demand_result.alert_level.value if demand_result.alert_level else "warning",
                "message": demand_result.alert_message or "Peak demand exceeds limit",
                "value": demand_result.optimized_peak_kw,
            })

        # Forecast quality alerts
        if load_forecast.data_quality_score < 0.7:
            alerts.append({
                "type": "DATA_QUALITY",
                "severity": "warning",
                "message": f"Low forecast data quality: {load_forecast.data_quality_score:.0%}",
                "value": load_forecast.data_quality_score,
            })

        return alerts

    def _calculate_provenance_hash(
        self,
        input_data: HeatSchedulerInput,
        output: HeatSchedulerOutput,
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        provenance_data = {
            "agent_id": self.config.agent_id,
            "agent_version": self.scheduler_config.version,
            "input_hash": self._hash_object(input_data),
            "timestamp": output.timestamp.isoformat(),
            "baseline_cost": output.baseline_cost_usd,
            "optimized_cost": output.optimized_cost_usd,
            "peak_demand": output.peak_demand_kw,
        }

        data_str = json.dumps(provenance_data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def _hash_object(self, obj: Any) -> str:
        """Hash an object for provenance tracking."""
        if hasattr(obj, "json"):
            data_str = obj.json()
        elif hasattr(obj, "dict"):
            data_str = json.dumps(obj.dict(), sort_keys=True, default=str)
        else:
            data_str = json.dumps(obj, sort_keys=True, default=str)

        return hashlib.sha256(data_str.encode()).hexdigest()[:16]

    def add_historical_data(
        self,
        data_point: HistoricalDataPoint,
    ) -> None:
        """Add historical data point for forecasting."""
        self.forecaster.add_historical_point(data_point)

    def train_forecaster(
        self,
        historical_data: List[HistoricalDataPoint],
    ) -> None:
        """Train load forecaster on historical data."""
        self.forecaster.train(historical_data)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "HeatSchedulerAgent",
]
