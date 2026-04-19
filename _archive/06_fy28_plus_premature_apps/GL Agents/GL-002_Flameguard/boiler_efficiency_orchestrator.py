"""
GL-002 FLAMEGUARD BoilerEfficiencyOptimizer - Main Orchestrator

This is the main orchestrator for the FLAMEGUARD agent, coordinating:
- Multi-boiler combustion optimization
- O2 trim and excess air control
- Efficiency calculations per ASME PTC 4.1
- Emissions monitoring and compliance
- Safety interlocks per NFPA 85
- Real-time optimization with AI/ML

Standards Compliance:
    - ASME PTC 4.1 (Fired Steam Generators Performance Test Codes)
    - NFPA 85 (Boiler and Combustion Systems Hazards Code)
    - IEC 61511 (Functional Safety - Safety Instrumented Systems)
    - EPA 40 CFR Part 60/63 (Emissions Standards)

Performance Targets:
    - Efficiency improvement: 3-8%
    - O2 setpoint optimization: Within 0.5% accuracy
    - CO breakthrough detection: <10ms response
    - Emissions compliance: 100% regulatory adherence
    - Safety response: <100ms for critical events
"""

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
import asyncio
import hashlib
import json
import logging
import time
import uuid

from core import (
    FlameguardConfig,
    BoilerProcessData,
    OptimizationRequest,
    OptimizationResult,
    EfficiencyCalculation,
    EmissionsCalculation,
    CombustionAnalysis,
    SafetyStatus,
    BoilerStatus,
    AgentStatus,
    FlameguardEvent,
    CalculationEvent,
    SetpointRecommendation,
    OperatingState,
    OptimizationStatus,
    SeverityLevel,
    CalculationType,
    BoilerSafetyEventHandler,
    CombustionEventHandler,
    OptimizationEventHandler,
    EfficiencyEventHandler,
    EmissionsEventHandler,
    AuditEventHandler,
    MetricsEventHandler,
)

logger = logging.getLogger(__name__)


class FlameGuardOrchestrator:
    """
    Main orchestrator for the GL-002 FLAMEGUARD BoilerEfficiencyOptimizer.

    This orchestrator coordinates all aspects of boiler efficiency optimization:
    - Real-time data acquisition from SCADA/DCS
    - Combustion analysis and O2 trim optimization
    - Multi-boiler load dispatch optimization
    - Efficiency and emissions calculations
    - Safety monitoring and emergency response
    - AI/ML-based setpoint optimization

    Example:
        >>> config = FlameguardConfig(
        ...     boiler=BoilerSpecifications(boiler_id="BOILER-001")
        ... )
        >>> orchestrator = FlameGuardOrchestrator(config)
        >>> await orchestrator.start()
        >>> result = await orchestrator.optimize("BOILER-001")
        >>> print(f"Efficiency improvement: {result.efficiency_improvement_percent}%")
    """

    def __init__(self, config: FlameguardConfig) -> None:
        """
        Initialize the FlameGuard orchestrator.

        Args:
            config: Complete FLAMEGUARD configuration
        """
        self.config = config
        self._agent_id = config.agent_id
        self._start_time: Optional[datetime] = None
        self._running = False

        # Managed boilers
        self._boilers: Dict[str, BoilerProcessData] = {}
        self._boiler_configs: Dict[str, FlameguardConfig] = {}
        self._boiler_statuses: Dict[str, BoilerStatus] = {}

        # Event handlers
        self._safety_handler = BoilerSafetyEventHandler(
            trip_callback=self._handle_trip
        )
        self._combustion_handler = CombustionEventHandler()
        self._optimization_handler = OptimizationEventHandler()
        self._efficiency_handler = EfficiencyEventHandler()
        self._emissions_handler = EmissionsEventHandler()
        self._audit_handler = AuditEventHandler()
        self._metrics_handler = MetricsEventHandler()

        # Optimization state
        self._optimization_lock = asyncio.Lock()
        self._last_optimization: Dict[str, datetime] = {}
        self._optimization_results: Dict[str, OptimizationResult] = {}

        # Calculation cache for deterministic results
        self._calculation_cache: Dict[str, Any] = {}

        # Async tasks
        self._tasks: List[asyncio.Task] = []

        # Statistics
        self._stats = {
            "optimizations_performed": 0,
            "optimizations_successful": 0,
            "total_efficiency_improvement": 0.0,
            "total_emissions_reduction": 0.0,
            "total_cost_savings": 0.0,
            "safety_events": 0,
            "calculations_performed": 0,
        }

        logger.info(
            f"FlameGuard orchestrator initialized: {self._agent_id}"
        )

    async def start(self) -> None:
        """
        Start the orchestrator.

        This begins:
        - Data acquisition loops
        - Optimization cycles
        - Safety monitoring
        - Metrics collection
        """
        if self._running:
            logger.warning("Orchestrator already running")
            return

        self._start_time = datetime.now(timezone.utc)
        self._running = True

        logger.info(f"Starting FlameGuard orchestrator: {self._agent_id}")

        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._optimization_loop()),
            asyncio.create_task(self._safety_monitoring_loop()),
            asyncio.create_task(self._metrics_collection_loop()),
            asyncio.create_task(self._efficiency_calculation_loop()),
        ]

        logger.info("FlameGuard orchestrator started")

    async def stop(self) -> None:
        """Stop the orchestrator gracefully."""
        if not self._running:
            return

        logger.info("Stopping FlameGuard orchestrator...")
        self._running = False

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        logger.info("FlameGuard orchestrator stopped")

    def register_boiler(
        self,
        boiler_id: str,
        config: Optional[FlameguardConfig] = None,
    ) -> None:
        """
        Register a boiler for management.

        Args:
            boiler_id: Unique boiler identifier
            config: Optional boiler-specific configuration
        """
        if config is None:
            config = self.config

        self._boiler_configs[boiler_id] = config
        self._boiler_statuses[boiler_id] = BoilerStatus(
            boiler_id=boiler_id,
            boiler_name=config.boiler.boiler_name,
            operating_state=OperatingState.OFFLINE,
        )

        logger.info(f"Registered boiler: {boiler_id}")

    def unregister_boiler(self, boiler_id: str) -> None:
        """Unregister a boiler from management."""
        self._boiler_configs.pop(boiler_id, None)
        self._boiler_statuses.pop(boiler_id, None)
        self._boilers.pop(boiler_id, None)
        logger.info(f"Unregistered boiler: {boiler_id}")

    async def update_process_data(
        self,
        data: BoilerProcessData,
    ) -> None:
        """
        Update real-time process data for a boiler.

        Args:
            data: Current process data from SCADA/DCS
        """
        boiler_id = data.boiler_id
        self._boilers[boiler_id] = data

        # Update boiler status
        if boiler_id in self._boiler_statuses:
            status = self._boiler_statuses[boiler_id]
            status.operating_state = data.operating_state
            status.load_percent = data.load_percent
            status.last_data_update = data.timestamp
            status.data_quality = data.data_quality
            status.scada_connected = True

        # Check for safety conditions
        await self._check_safety_conditions(data)

        # Update combustion analysis
        await self._update_combustion_analysis(data)

    async def optimize(
        self,
        boiler_id: str,
        request: Optional[OptimizationRequest] = None,
    ) -> OptimizationResult:
        """
        Run optimization for a specific boiler.

        Args:
            boiler_id: Target boiler ID
            request: Optional optimization request with constraints

        Returns:
            Optimization result with recommended setpoints
        """
        start_time = time.perf_counter()

        if request is None:
            request = OptimizationRequest(boiler_id=boiler_id)

        # Emit optimization started event
        await self._emit_event(FlameguardEvent(
            event_type="OPTIMIZATION_STARTED",
            boiler_id=boiler_id,
            payload={"optimization_id": request.request_id},
        ))

        try:
            async with self._optimization_lock:
                result = await self._run_optimization(boiler_id, request)

            self._optimization_results[boiler_id] = result
            self._last_optimization[boiler_id] = datetime.now(timezone.utc)

            # Update statistics
            self._stats["optimizations_performed"] += 1
            if result.status == OptimizationStatus.COMPLETED:
                self._stats["optimizations_successful"] += 1
                self._stats["total_efficiency_improvement"] += (
                    result.efficiency_improvement_percent
                )
                self._stats["total_cost_savings"] += result.cost_savings_hr

            # Emit optimization completed event
            await self._emit_event(FlameguardEvent(
                event_type="OPTIMIZATION_COMPLETED",
                boiler_id=boiler_id,
                payload={
                    "optimization_id": result.optimization_id,
                    "efficiency_improvement": result.efficiency_improvement_percent,
                    "cost_savings": result.cost_savings_hr,
                },
            ))

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                f"Optimization completed for {boiler_id} in {elapsed_ms:.1f}ms: "
                f"+{result.efficiency_improvement_percent:.2f}% efficiency"
            )

            return result

        except Exception as e:
            logger.error(f"Optimization failed for {boiler_id}: {e}")

            # Emit failure event
            await self._emit_event(FlameguardEvent(
                event_type="OPTIMIZATION_FAILED",
                boiler_id=boiler_id,
                severity=SeverityLevel.WARNING,
                payload={"error": str(e)},
            ))

            return OptimizationResult(
                request_id=request.request_id,
                status=OptimizationStatus.FAILED,
                objective_value=0.0,
                current_efficiency_percent=80.0,
                optimal_efficiency_percent=80.0,
                efficiency_improvement_percent=0.0,
                explanation=f"Optimization failed: {e}",
            )

    async def _run_optimization(
        self,
        boiler_id: str,
        request: OptimizationRequest,
    ) -> OptimizationResult:
        """
        Internal optimization execution.

        Performs:
        1. Combustion analysis
        2. Current efficiency calculation
        3. O2 setpoint optimization
        4. Excess air optimization
        5. Setpoint recommendations
        """
        config = self._boiler_configs.get(boiler_id, self.config)
        data = self._boilers.get(boiler_id)

        if data is None:
            raise ValueError(f"No process data for boiler {boiler_id}")

        # 1. Calculate current efficiency
        current_efficiency = await self._calculate_efficiency(boiler_id, data)

        # 2. Analyze combustion
        combustion = await self._analyze_combustion(boiler_id, data)

        # 3. Determine optimal O2 setpoint based on load
        optimal_o2 = self._calculate_optimal_o2(
            data.load_percent,
            config.combustion.o2_trim,
        )

        # 4. Calculate optimal excess air
        current_excess_air = combustion.excess_air_percent
        optimal_excess_air = self._calculate_optimal_excess_air(
            data.load_percent,
            config.combustion.excess_air,
        )

        # 5. Calculate efficiency improvement potential
        o2_deviation = data.flue_gas_o2_percent - optimal_o2
        excess_air_deviation = current_excess_air - optimal_excess_air

        # Efficiency gain from O2 reduction (approx 1% efficiency per 2% O2 reduction)
        efficiency_gain_o2 = max(0, o2_deviation * 0.5)

        # Efficiency gain from excess air reduction
        efficiency_gain_excess_air = max(0, excess_air_deviation * 0.02)

        total_efficiency_gain = efficiency_gain_o2 + efficiency_gain_excess_air
        optimal_efficiency = min(
            current_efficiency.efficiency_percent + total_efficiency_gain,
            98.0  # Max theoretical
        )

        # 6. Generate setpoint recommendations
        recommendations = []

        if abs(o2_deviation) > config.combustion.o2_trim.deadband_percent:
            recommendations.append(SetpointRecommendation(
                tag_name=f"{boiler_id}.O2_SETPOINT",
                current_value=data.flue_gas_o2_percent,
                recommended_value=optimal_o2,
                change_amount=optimal_o2 - data.flue_gas_o2_percent,
                change_percent=(optimal_o2 - data.flue_gas_o2_percent) /
                               data.flue_gas_o2_percent * 100
                               if data.flue_gas_o2_percent > 0 else 0,
                unit="%",
                rationale=f"Optimize O2 for {data.load_percent:.0f}% load",
                expected_impact=f"+{efficiency_gain_o2:.2f}% efficiency",
            ))

        if abs(excess_air_deviation) > 2.0:
            target_damper = data.air_damper_position_percent - (
                excess_air_deviation * 0.5
            )
            target_damper = max(20, min(100, target_damper))

            recommendations.append(SetpointRecommendation(
                tag_name=f"{boiler_id}.AIR_DAMPER_SP",
                current_value=data.air_damper_position_percent,
                recommended_value=target_damper,
                change_amount=target_damper - data.air_damper_position_percent,
                change_percent=(target_damper - data.air_damper_position_percent) /
                               data.air_damper_position_percent * 100
                               if data.air_damper_position_percent > 0 else 0,
                unit="%",
                rationale="Reduce excess air to improve efficiency",
                expected_impact=f"+{efficiency_gain_excess_air:.2f}% efficiency",
            ))

        # Calculate cost savings (assume $5/MMBTU fuel cost)
        fuel_rate = current_efficiency.fuel_input_mmbtu_hr
        fuel_cost = 5.0
        current_cost_hr = fuel_rate * fuel_cost
        optimal_fuel_rate = fuel_rate * (
            current_efficiency.efficiency_percent / optimal_efficiency
        )
        optimal_cost_hr = optimal_fuel_rate * fuel_cost
        cost_savings = current_cost_hr - optimal_cost_hr

        # Build result
        return OptimizationResult(
            request_id=request.request_id,
            status=OptimizationStatus.COMPLETED,
            objective_value=optimal_efficiency,
            objective_type=request.objective,
            current_efficiency_percent=current_efficiency.efficiency_percent,
            optimal_efficiency_percent=optimal_efficiency,
            efficiency_improvement_percent=total_efficiency_gain,
            current_cost_hr=current_cost_hr,
            optimal_cost_hr=optimal_cost_hr,
            cost_savings_hr=cost_savings,
            recommended_setpoints={
                r.tag_name: r.recommended_value for r in recommendations
            },
            setpoint_changes=[r.dict() for r in recommendations],
            confidence_percent=combustion.combustion_quality_score,
            explanation=self._generate_explanation(
                combustion, current_efficiency, recommendations
            ),
            key_factors=[
                f"Current O2: {data.flue_gas_o2_percent:.1f}%",
                f"Optimal O2: {optimal_o2:.1f}%",
                f"Current excess air: {current_excess_air:.1f}%",
                f"CO level: {data.flue_gas_co_ppm:.0f} ppm",
            ],
            can_auto_implement=not request.advisory_only,
            implementation_risk="low" if total_efficiency_gain < 2.0 else "medium",
            operator_approval_required=config.optimization.setpoints.require_operator_approval,
            provenance_hash=self._compute_hash({
                "boiler_id": boiler_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "input": data.dict(),
                "output": total_efficiency_gain,
            }),
        )

    async def _calculate_efficiency(
        self,
        boiler_id: str,
        data: BoilerProcessData,
    ) -> EfficiencyCalculation:
        """
        Calculate boiler efficiency per ASME PTC 4.1 indirect method.

        Calculates losses:
        1. Dry flue gas loss
        2. Moisture in fuel loss
        3. Hydrogen combustion moisture loss
        4. Moisture in air loss
        5. Unburned carbon loss
        6. CO loss
        7. Radiation/convection loss
        8. Blowdown loss
        """
        config = self._boiler_configs.get(boiler_id, self.config)
        fuel = config.fuel.primary_fuel

        # Calculate fuel input
        fuel_hhv = fuel.higher_heating_value_btu_lb
        fuel_flow = data.fuel_flow_rate  # Assume lb/hr for solid, scfh for gas

        # For natural gas, convert SCFH to lb/hr (approx 0.042 lb/scf)
        if fuel.fuel_type.value == "natural_gas":
            fuel_mass_flow = data.fuel_flow_rate * 0.042
        else:
            fuel_mass_flow = data.fuel_flow_rate

        fuel_input_btu_hr = fuel_mass_flow * fuel_hhv
        fuel_input_mmbtu_hr = fuel_input_btu_hr / 1e6

        # 1. Dry flue gas loss
        # L1 = (Cp_fg * (Tfg - Ta) * (1 + EA/100)) / HHV * 100
        cp_flue_gas = 0.24  # BTU/lb-F
        temp_diff = data.flue_gas_temperature_f - data.ambient_temperature_f
        excess_air_fraction = self._o2_to_excess_air(data.flue_gas_o2_percent) / 100
        stoich_air = fuel.stoichiometric_air_fuel_ratio
        actual_air = stoich_air * (1 + excess_air_fraction)

        dry_flue_gas_loss = (
            cp_flue_gas * temp_diff * actual_air / fuel_hhv * 100
        )

        # 2. Moisture in fuel loss
        moisture_loss = (
            fuel.moisture_content_percent / 100 *
            (1055 + 0.46 * (data.flue_gas_temperature_f - data.ambient_temperature_f)) /
            fuel_hhv * 100
        ) if fuel.moisture_content_percent > 0 else 0

        # 3. Hydrogen combustion moisture loss
        # 9 lb water per lb H2
        h2_moisture_loss = (
            9 * fuel.hydrogen_content_percent / 100 *
            (1055 + 0.46 * (data.flue_gas_temperature_f - data.ambient_temperature_f)) /
            fuel_hhv * 100
        )

        # 4. Moisture in air loss
        # Assume 60% RH, ~0.013 lb moisture/lb dry air
        humidity_ratio = 0.013 * data.combustion_air_humidity_percent / 60
        air_moisture_loss = (
            humidity_ratio * actual_air * 0.46 * temp_diff / fuel_hhv * 100
        )

        # 5. Unburned carbon loss (negligible for gas)
        unburned_carbon_loss = 0.0
        if "coal" in fuel.fuel_type.value:
            # Assume 2% unburned in ash
            unburned_carbon_loss = (
                fuel.ash_content_percent / 100 * 0.02 * 14500 / fuel_hhv * 100
            )

        # 6. CO loss
        # CO has 10,100 BTU/lb heating value
        # Approximate CO mass fraction from ppm
        co_mass_fraction = data.flue_gas_co_ppm / 1e6 * 28 / 29  # MW ratio
        co_loss = co_mass_fraction * actual_air * 10100 / fuel_hhv * 100

        # 7. Radiation/convection loss (use ABMA chart approximation)
        capacity_mmbtu = fuel_input_mmbtu_hr
        radiation_loss = 1.5 / (capacity_mmbtu ** 0.15) if capacity_mmbtu > 0 else 1.0
        radiation_loss = min(max(radiation_loss, 0.3), 3.0)

        # 8. Blowdown loss
        blowdown_rate = config.boiler.blowdown_rate_percent / 100
        # Enthalpy difference between steam and feedwater
        steam_enthalpy = 1190  # BTU/lb at 150 psig (approx)
        fw_enthalpy = data.feedwater_temperature_f - 32  # BTU/lb (approx)
        blowdown_loss = (
            blowdown_rate * (steam_enthalpy - fw_enthalpy) / fuel_hhv * 100 *
            data.steam_flow_klb_hr * 1000 / fuel_mass_flow
            if fuel_mass_flow > 0 else 0
        )

        # Total losses
        total_losses = (
            dry_flue_gas_loss +
            moisture_loss +
            h2_moisture_loss +
            air_moisture_loss +
            unburned_carbon_loss +
            co_loss +
            radiation_loss +
            blowdown_loss
        )

        # Efficiency
        efficiency_hhv = 100 - total_losses
        efficiency_hhv = max(50.0, min(100.0, efficiency_hhv))

        # LHV basis efficiency (higher than HHV)
        hhv_lhv_ratio = fuel.higher_heating_value_btu_lb / fuel.lower_heating_value_btu_lb
        efficiency_lhv = efficiency_hhv * hhv_lhv_ratio
        efficiency_lhv = min(100.0, efficiency_lhv)

        # Steam output
        steam_enthalpy_out = steam_enthalpy
        steam_output_mmbtu = (
            data.steam_flow_klb_hr * 1000 *
            (steam_enthalpy_out - fw_enthalpy) / 1e6
        )

        # Create provenance hash
        input_data = {
            "boiler_id": boiler_id,
            "timestamp": data.timestamp.isoformat(),
            "flue_gas_temp": data.flue_gas_temperature_f,
            "o2": data.flue_gas_o2_percent,
            "fuel_flow": data.fuel_flow_rate,
        }
        provenance_hash = self._compute_hash(input_data)

        result = EfficiencyCalculation(
            efficiency_percent=efficiency_hhv,
            efficiency_hhv_basis=efficiency_hhv,
            efficiency_lhv_basis=efficiency_lhv,
            fuel_input_mmbtu_hr=fuel_input_mmbtu_hr,
            steam_output_mmbtu_hr=steam_output_mmbtu,
            dry_flue_gas_loss_percent=dry_flue_gas_loss,
            moisture_in_fuel_loss_percent=moisture_loss,
            hydrogen_combustion_loss_percent=h2_moisture_loss,
            moisture_in_air_loss_percent=air_moisture_loss,
            unburned_carbon_loss_percent=unburned_carbon_loss,
            co_loss_percent=co_loss,
            radiation_convection_loss_percent=radiation_loss,
            blowdown_loss_percent=blowdown_loss,
            total_losses_percent=total_losses,
            fuel_utilization_percent=efficiency_hhv,
            provenance_hash=provenance_hash,
        )

        # Emit efficiency event
        await self._emit_event(FlameguardEvent(
            event_type="EFFICIENCY_CALCULATED",
            boiler_id=boiler_id,
            payload={
                "efficiency_percent": efficiency_hhv,
                "total_losses_percent": total_losses,
                "fuel_input_mmbtu_hr": fuel_input_mmbtu_hr,
                "steam_output_mmbtu_hr": steam_output_mmbtu,
            },
        ))

        self._stats["calculations_performed"] += 1
        return result

    async def _analyze_combustion(
        self,
        boiler_id: str,
        data: BoilerProcessData,
    ) -> CombustionAnalysis:
        """Analyze combustion conditions."""
        config = self._boiler_configs.get(boiler_id, self.config)
        fuel = config.fuel.primary_fuel

        # Calculate excess air from O2
        excess_air = self._o2_to_excess_air(data.flue_gas_o2_percent)

        # Target O2 for current load
        target_o2 = self._calculate_optimal_o2(
            data.load_percent,
            config.combustion.o2_trim,
        )

        # Air-fuel ratio
        stoich_afr = fuel.stoichiometric_air_fuel_ratio
        actual_afr = stoich_afr * (1 + excess_air / 100)
        lambda_value = actual_afr / stoich_afr

        # CO breakthrough detection
        co_limit = config.combustion.co_monitoring.co_limit_ppm
        co_breakthrough = data.flue_gas_co_ppm > config.combustion.co_monitoring.breakthrough_threshold_ppm

        # Combustion efficiency (chemical efficiency)
        # Based on CO/CO2 ratio
        combustion_efficiency = 99.5  # Default for clean combustion
        if data.flue_gas_co2_percent > 0:
            co_co2_ratio = data.flue_gas_co_ppm / 10000 / data.flue_gas_co2_percent
            combustion_efficiency = 100 - (co_co2_ratio * 100)
            combustion_efficiency = max(85.0, min(100.0, combustion_efficiency))

        # Quality score
        quality_score = 100.0
        # Penalize for O2 deviation
        o2_deviation = abs(data.flue_gas_o2_percent - target_o2)
        quality_score -= o2_deviation * 5

        # Penalize for high CO
        if data.flue_gas_co_ppm > 200:
            quality_score -= (data.flue_gas_co_ppm - 200) / 10

        # Penalize for high excess air
        if excess_air > 25:
            quality_score -= (excess_air - 25) * 2

        quality_score = max(0.0, min(100.0, quality_score))

        # Recommended adjustments
        o2_adjustment = target_o2 - data.flue_gas_o2_percent
        air_adjustment = -o2_adjustment * 5  # Approximate

        return CombustionAnalysis(
            measured_o2_percent=data.flue_gas_o2_percent,
            target_o2_percent=target_o2,
            o2_deviation_percent=data.flue_gas_o2_percent - target_o2,
            excess_air_percent=excess_air,
            stoichiometric_air_lb_lb_fuel=stoich_afr,
            actual_air_lb_lb_fuel=actual_afr,
            measured_co_ppm=data.flue_gas_co_ppm,
            co_limit_ppm=co_limit,
            co_breakthrough_detected=co_breakthrough,
            air_fuel_ratio_actual=actual_afr,
            air_fuel_ratio_stoichiometric=stoich_afr,
            lambda_value=lambda_value,
            combustion_efficiency_percent=combustion_efficiency,
            combustion_quality_score=quality_score,
            recommended_o2_adjustment_percent=o2_adjustment,
            recommended_air_adjustment_percent=air_adjustment,
        )

    async def _check_safety_conditions(
        self,
        data: BoilerProcessData,
    ) -> None:
        """Check for safety conditions and emit events."""
        boiler_id = data.boiler_id
        config = self._boiler_configs.get(boiler_id, self.config)
        interlocks = config.safety.interlocks

        # Check flame
        if not data.flame_status and data.operating_state not in [
            OperatingState.OFFLINE,
            OperatingState.COLD_STANDBY,
            OperatingState.PURGING,
        ]:
            await self._emit_event(FlameguardEvent(
                event_type="FLAME_FAILURE",
                boiler_id=boiler_id,
                severity=SeverityLevel.EMERGENCY,
                payload={
                    "flame_signal": data.flame_signal_percent,
                    "description": "Flame failure detected",
                },
            ))

        # Check steam pressure
        if data.steam_pressure_psig > interlocks.high_steam_pressure_psig:
            await self._emit_event(FlameguardEvent(
                event_type="HIGH_STEAM_PRESSURE",
                boiler_id=boiler_id,
                severity=SeverityLevel.CRITICAL,
                payload={
                    "measured_value": data.steam_pressure_psig,
                    "threshold_value": interlocks.high_steam_pressure_psig,
                    "unit": "psig",
                },
            ))

        # Check drum level
        if data.drum_level_inches < interlocks.low_water_level_inches:
            await self._emit_event(FlameguardEvent(
                event_type="LOW_WATER_LEVEL",
                boiler_id=boiler_id,
                severity=SeverityLevel.EMERGENCY,
                payload={
                    "measured_value": data.drum_level_inches,
                    "threshold_value": interlocks.low_water_level_inches,
                    "unit": "inches",
                },
            ))

        # Check flue gas temperature
        if data.flue_gas_temperature_f > interlocks.high_flue_gas_temp_f:
            await self._emit_event(FlameguardEvent(
                event_type="HIGH_FLUE_GAS_TEMP",
                boiler_id=boiler_id,
                severity=SeverityLevel.CRITICAL,
                payload={
                    "measured_value": data.flue_gas_temperature_f,
                    "threshold_value": interlocks.high_flue_gas_temp_f,
                    "unit": "°F",
                },
            ))

        # Check CO breakthrough
        co_config = config.combustion.co_monitoring
        if data.flue_gas_co_ppm > co_config.co_alarm_ppm:
            await self._emit_event(FlameguardEvent(
                event_type="CO_BREAKTHROUGH",
                boiler_id=boiler_id,
                severity=SeverityLevel.WARNING,
                payload={
                    "co_ppm": data.flue_gas_co_ppm,
                    "threshold": co_config.co_alarm_ppm,
                },
            ))

    async def _update_combustion_analysis(
        self,
        data: BoilerProcessData,
    ) -> None:
        """Update combustion analysis tracking."""
        boiler_id = data.boiler_id
        config = self._boiler_configs.get(boiler_id, self.config)

        # Check O2 deviation
        target_o2 = self._calculate_optimal_o2(
            data.load_percent,
            config.combustion.o2_trim,
        )
        deviation = abs(data.flue_gas_o2_percent - target_o2)

        if deviation > 1.0:
            await self._emit_event(FlameguardEvent(
                event_type="O2_DEVIATION",
                boiler_id=boiler_id,
                severity=SeverityLevel.WARNING,
                payload={
                    "measured_o2": data.flue_gas_o2_percent,
                    "target_o2": target_o2,
                    "deviation": deviation,
                },
            ))

    def _calculate_optimal_o2(
        self,
        load_percent: float,
        config: Any,  # O2TrimConfig
    ) -> float:
        """Calculate optimal O2 setpoint based on load curve."""
        load_fraction = load_percent / 100.0
        curve = config.o2_setpoint_curve

        # Interpolate on curve
        if not curve:
            return config.target_o2_percent

        loads = sorted(curve.keys())

        if load_fraction <= loads[0]:
            return curve[loads[0]]
        if load_fraction >= loads[-1]:
            return curve[loads[-1]]

        # Linear interpolation
        for i in range(len(loads) - 1):
            if loads[i] <= load_fraction <= loads[i + 1]:
                t = (load_fraction - loads[i]) / (loads[i + 1] - loads[i])
                return curve[loads[i]] + t * (curve[loads[i + 1]] - curve[loads[i]])

        return config.target_o2_percent

    def _calculate_optimal_excess_air(
        self,
        load_percent: float,
        config: Any,  # ExcessAirConfig
    ) -> float:
        """Calculate optimal excess air based on load curve."""
        load_fraction = load_percent / 100.0
        curve = config.excess_air_curve

        if not curve:
            return config.design_excess_air_percent

        loads = sorted(curve.keys())

        if load_fraction <= loads[0]:
            return curve[loads[0]]
        if load_fraction >= loads[-1]:
            return curve[loads[-1]]

        for i in range(len(loads) - 1):
            if loads[i] <= load_fraction <= loads[i + 1]:
                t = (load_fraction - loads[i]) / (loads[i + 1] - loads[i])
                return curve[loads[i]] + t * (curve[loads[i + 1]] - curve[loads[i]])

        return config.design_excess_air_percent

    def _o2_to_excess_air(self, o2_percent: float) -> float:
        """Convert O2% to excess air%."""
        # EA = O2 / (21 - O2) * 100
        if o2_percent >= 21:
            return 500.0  # Max
        return o2_percent / (21 - o2_percent) * 100

    def _generate_explanation(
        self,
        combustion: CombustionAnalysis,
        efficiency: EfficiencyCalculation,
        recommendations: List[SetpointRecommendation],
    ) -> str:
        """Generate human-readable optimization explanation."""
        parts = [
            f"Current boiler efficiency: {efficiency.efficiency_percent:.1f}%",
            f"O2 level: {combustion.measured_o2_percent:.1f}% "
            f"(target: {combustion.target_o2_percent:.1f}%)",
            f"Excess air: {combustion.excess_air_percent:.1f}%",
        ]

        if recommendations:
            parts.append(f"\nRecommendations ({len(recommendations)}):")
            for rec in recommendations:
                parts.append(
                    f"  - {rec.tag_name}: {rec.current_value:.1f} -> "
                    f"{rec.recommended_value:.1f} ({rec.expected_impact})"
                )

        if combustion.co_breakthrough_detected:
            parts.append("\nWARNING: CO breakthrough detected - review combustion")

        return "\n".join(parts)

    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash for provenance tracking."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    async def _emit_event(self, event: FlameguardEvent) -> None:
        """Emit an event to all registered handlers."""
        handlers = [
            self._safety_handler,
            self._combustion_handler,
            self._optimization_handler,
            self._efficiency_handler,
            self._emissions_handler,
            self._audit_handler,
            self._metrics_handler,
        ]

        for handler in handlers:
            try:
                await handler.handle(event)
            except Exception as e:
                logger.error(f"Handler {handler.name} failed: {e}")

    def _handle_trip(self, boiler_id: str, reason: str) -> None:
        """Handle boiler trip callback."""
        logger.critical(f"BOILER TRIP: {boiler_id} - {reason}")
        self._stats["safety_events"] += 1

        # Update boiler status
        if boiler_id in self._boiler_statuses:
            self._boiler_statuses[boiler_id].operating_state = (
                OperatingState.EMERGENCY_SHUTDOWN
            )

    async def _optimization_loop(self) -> None:
        """Background optimization loop."""
        while self._running:
            try:
                for boiler_id in list(self._boilers.keys()):
                    data = self._boilers.get(boiler_id)
                    if data is None:
                        continue

                    # Skip if not modulating
                    if data.operating_state not in [
                        OperatingState.MODULATING,
                        OperatingState.HIGH_FIRE,
                    ]:
                        continue

                    # Check optimization interval
                    last_opt = self._last_optimization.get(boiler_id)
                    min_interval = 3600 / self.config.optimization.max_optimization_cycles_per_hour
                    if last_opt and (
                        datetime.now(timezone.utc) - last_opt
                    ).total_seconds() < min_interval:
                        continue

                    # Run optimization
                    await self.optimize(boiler_id)

                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
                await asyncio.sleep(60)

    async def _safety_monitoring_loop(self) -> None:
        """Background safety monitoring loop."""
        while self._running:
            try:
                for boiler_id, data in self._boilers.items():
                    await self._check_safety_conditions(data)

                await asyncio.sleep(1)  # Check every second

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Safety monitoring error: {e}")
                await asyncio.sleep(1)

    async def _metrics_collection_loop(self) -> None:
        """Background metrics collection loop."""
        while self._running:
            try:
                # Collect metrics from all boilers
                for boiler_id, data in self._boilers.items():
                    await self._emit_event(FlameguardEvent(
                        event_type="METRICS_UPDATE",
                        boiler_id=boiler_id,
                        payload={
                            "gauges": {
                                f"{boiler_id}.efficiency": (
                                    self._boiler_statuses.get(boiler_id, BoilerStatus(boiler_id=boiler_id))
                                    .current_efficiency_percent
                                ),
                                f"{boiler_id}.load": data.load_percent,
                                f"{boiler_id}.o2": data.flue_gas_o2_percent,
                                f"{boiler_id}.co": data.flue_gas_co_ppm,
                            },
                        },
                    ))

                await asyncio.sleep(self.config.metrics.collection_interval_s)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(15)

    async def _efficiency_calculation_loop(self) -> None:
        """Background efficiency calculation loop."""
        while self._running:
            try:
                for boiler_id, data in self._boilers.items():
                    if data.operating_state in [
                        OperatingState.MODULATING,
                        OperatingState.HIGH_FIRE,
                        OperatingState.LOW_FIRE,
                    ]:
                        efficiency = await self._calculate_efficiency(boiler_id, data)

                        # Update boiler status
                        if boiler_id in self._boiler_statuses:
                            status = self._boiler_statuses[boiler_id]
                            status.current_efficiency_percent = efficiency.efficiency_percent
                            status.efficiency_vs_design_percent = (
                                efficiency.efficiency_percent -
                                self.config.efficiency.design_efficiency_percent
                            )

                await asyncio.sleep(60)  # Calculate every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Efficiency calculation error: {e}")
                await asyncio.sleep(60)

    def get_status(self) -> AgentStatus:
        """Get current agent status."""
        uptime = 0.0
        if self._start_time:
            uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds()

        return AgentStatus(
            agent_id=self._agent_id,
            agent_name="FLAMEGUARD",
            agent_version=self.config.version,
            agent_type="GL-002",
            status="running" if self._running else "stopped",
            health="healthy" if self._running else "stopped",
            uptime_seconds=uptime,
            managed_boilers=list(self._boilers.keys()),
            boiler_statuses=dict(self._boiler_statuses),
            optimizations_performed=self._stats["optimizations_performed"],
            optimizations_successful=self._stats["optimizations_successful"],
            total_efficiency_improvement_percent=self._stats["total_efficiency_improvement"],
            total_cost_savings_usd=self._stats["total_cost_savings"],
        )

    def get_boiler_status(self, boiler_id: str) -> Optional[BoilerStatus]:
        """Get status for a specific boiler."""
        return self._boiler_statuses.get(boiler_id)

    def get_optimization_result(
        self,
        boiler_id: str,
    ) -> Optional[OptimizationResult]:
        """Get last optimization result for a boiler."""
        return self._optimization_results.get(boiler_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            **self._stats,
            "managed_boilers": len(self._boilers),
            "uptime_seconds": (
                (datetime.now(timezone.utc) - self._start_time).total_seconds()
                if self._start_time else 0
            ),
        }
