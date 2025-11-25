# -*- coding: utf-8 -*-
"""
GL-004 BurnerOptimizationAgent - Main Orchestrator
Optimizes burner settings for complete combustion and reduced emissions

This agent implements zero-hallucination combustion optimization using:
- Physics-based stoichiometric calculations
- Multi-objective optimization (efficiency vs emissions)
- Real-time adaptive control
- Safety interlocks and fail-safes

Agent Specification:
- Agent ID: GL-004
- Agent Name: BurnerOptimizationAgent
- Category: Combustion
- Type: Optimizer
- Primary Function: Optimize burner settings for complete combustion and minimal emissions
- Inputs: Air-fuel ratio, flame temperature, O2 levels
- Outputs: Optimal burner settings, emissions reduction percentage
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field, validator

from calculators.stoichiometric_calculator import StoichiometricCalculator
from calculators.combustion_efficiency_calculator import CombustionEfficiencyCalculator
from calculators.emissions_calculator import EmissionsCalculator
from calculators.air_fuel_optimizer import AirFuelOptimizer
from calculators.flame_analysis_calculator import FlameAnalysisCalculator
from calculators.burner_performance_calculator import BurnerPerformanceCalculator

from integrations.burner_controller_connector import BurnerControllerConnector
from integrations.o2_analyzer_connector import O2AnalyzerConnector
from integrations.emissions_monitor_connector import EmissionsMonitorConnector
from integrations.flame_scanner_connector import FlameScannerConnector
from integrations.temperature_sensor_connector import TemperatureSensorConnector
from integrations.scada_integration import SCADAIntegration

from config import settings
from monitoring.metrics import metrics_collector
from greenlang.determinism import deterministic_uuid, DeterministicClock

logger = logging.getLogger(__name__)


class BurnerState(BaseModel):
    """Current state of the burner system"""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    fuel_flow_rate: float = Field(..., description="Fuel flow rate (kg/hr or m3/hr)")
    air_flow_rate: float = Field(..., description="Combustion air flow rate (m3/hr)")
    air_fuel_ratio: float = Field(..., description="Actual air-fuel ratio")
    o2_level: float = Field(..., description="O2 concentration in flue gas (%)")
    co_level: Optional[float] = Field(None, description="CO concentration (ppm)")
    nox_level: Optional[float] = Field(None, description="NOx concentration (ppm)")
    flame_temperature: Optional[float] = Field(None, description="Flame temperature (°C)")
    furnace_temperature: float = Field(..., description="Furnace temperature (°C)")
    flue_gas_temperature: float = Field(..., description="Flue gas temperature (°C)")
    burner_load: float = Field(..., description="Burner load (%)")
    combustion_efficiency: Optional[float] = Field(None, description="Calculated efficiency (%)")

    @validator('o2_level')
    def validate_o2(cls, v: float) -> float:
        """Validate O2 level is within physical bounds"""
        if not 0 <= v <= 21:
            raise ValueError(f"O2 level {v}% is outside valid range [0, 21]%")
        return v

    @validator('burner_load')
    def validate_load(cls, v: float) -> float:
        """Validate burner load percentage"""
        if not 0 <= v <= 100:
            raise ValueError(f"Burner load {v}% is outside valid range [0, 100]%")
        return v


class OptimizationResult(BaseModel):
    """Result of burner optimization"""
    optimization_id: str = Field(default_factory=lambda: str(deterministic_uuid(__name__, str(DeterministicClock.now()))))
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Current state
    current_air_fuel_ratio: float
    current_efficiency: float
    current_nox: float
    current_co: float

    # Optimized settings
    optimal_air_fuel_ratio: float
    optimal_fuel_flow: float
    optimal_air_flow: float
    optimal_excess_air: float

    # Predicted improvements
    predicted_efficiency: float
    predicted_nox: float
    predicted_co: float

    # Benefits
    efficiency_improvement: float = Field(..., description="Efficiency gain (%)")
    nox_reduction: float = Field(..., description="NOx reduction (%)")
    co_reduction: float = Field(..., description="CO reduction (%)")
    fuel_savings: float = Field(..., description="Fuel savings (kg/hr or m3/hr)")

    # Optimization metadata
    iterations: int
    convergence_status: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)

    # Provenance
    hash: str = Field(..., description="SHA-256 hash for determinism verification")

    def calculate_hash(self) -> str:
        """Calculate deterministic hash of optimization result"""
        hashable_data = {
            'current_air_fuel_ratio': round(self.current_air_fuel_ratio, 6),
            'optimal_air_fuel_ratio': round(self.optimal_air_fuel_ratio, 6),
            'predicted_efficiency': round(self.predicted_efficiency, 6),
            'predicted_nox': round(self.predicted_nox, 6),
            'predicted_co': round(self.predicted_co, 6)
        }
        hash_input = json.dumps(hashable_data, sort_keys=True)
        return hashlib.sha256(hash_input.encode()).hexdigest()


class SafetyInterlocks(BaseModel):
    """Safety interlock status"""
    flame_present: bool = Field(..., description="Flame detection status")
    fuel_pressure_ok: bool = Field(..., description="Fuel pressure within limits")
    air_pressure_ok: bool = Field(..., description="Air pressure within limits")
    purge_complete: bool = Field(..., description="Pre-purge completed")
    temperature_ok: bool = Field(..., description="Temperature within safe limits")
    emergency_stop_clear: bool = Field(..., description="No emergency stop active")

    def all_safe(self) -> bool:
        """Check if all interlocks are satisfied"""
        return all([
            self.flame_present,
            self.fuel_pressure_ok,
            self.air_pressure_ok,
            self.purge_complete,
            self.temperature_ok,
            self.emergency_stop_clear
        ])


class BurnerOptimizationOrchestrator:
    """
    Main orchestrator for GL-004 BurnerOptimizationAgent

    Orchestrates burner optimization workflow:
    1. Collect real-time data from sensors and controllers
    2. Analyze combustion efficiency and emissions
    3. Optimize air-fuel ratio for multi-objective goals
    4. Implement optimized settings with safety checks
    5. Monitor and validate improvements
    """

    def __init__(self):
        """Initialize the burner optimization orchestrator"""
        self.agent_id = "GL-004"
        self.agent_name = "BurnerOptimizationAgent"
        self.version = "1.0.0"

        # Initialize calculators (zero-hallucination, physics-based)
        self.stoichiometric_calc = StoichiometricCalculator()
        self.efficiency_calc = CombustionEfficiencyCalculator()
        self.emissions_calc = EmissionsCalculator()
        self.optimizer = AirFuelOptimizer()
        self.flame_calc = FlameAnalysisCalculator()
        self.performance_calc = BurnerPerformanceCalculator()

        # Initialize integrations
        self.burner_controller: Optional[BurnerControllerConnector] = None
        self.o2_analyzer: Optional[O2AnalyzerConnector] = None
        self.emissions_monitor: Optional[EmissionsMonitorConnector] = None
        self.flame_scanner: Optional[FlameScannerConnector] = None
        self.temperature_sensors: Optional[TemperatureSensorConnector] = None
        self.scada: Optional[SCADAIntegration] = None

        # State
        self.current_state: Optional[BurnerState] = None
        self.optimization_history: List[OptimizationResult] = []
        self.is_running = False

        logger.info(f"Initialized {self.agent_name} v{self.version}")

    async def initialize_integrations(self) -> None:
        """Initialize all integration connectors"""
        try:
            logger.info("Initializing integrations...")

            # Initialize burner controller
            self.burner_controller = BurnerControllerConnector(
                host=settings.BURNER_CONTROLLER_HOST,
                port=settings.BURNER_CONTROLLER_PORT,
                protocol=settings.BURNER_CONTROLLER_PROTOCOL
            )
            await self.burner_controller.connect()

            # Initialize O2 analyzer
            self.o2_analyzer = O2AnalyzerConnector(
                host=settings.O2_ANALYZER_HOST,
                port=settings.O2_ANALYZER_PORT
            )
            await self.o2_analyzer.connect()

            # Initialize emissions monitor
            self.emissions_monitor = EmissionsMonitorConnector(
                host=settings.EMISSIONS_MONITOR_HOST,
                port=settings.EMISSIONS_MONITOR_PORT
            )
            await self.emissions_monitor.connect()

            # Initialize flame scanner
            self.flame_scanner = FlameScannerConnector(
                host=settings.FLAME_SCANNER_HOST,
                port=settings.FLAME_SCANNER_PORT
            )
            await self.flame_scanner.connect()

            # Initialize temperature sensors
            self.temperature_sensors = TemperatureSensorConnector(
                sensors=settings.TEMPERATURE_SENSORS
            )
            await self.temperature_sensors.connect()

            # Initialize SCADA integration
            self.scada = SCADAIntegration(
                opc_ua_endpoint=settings.SCADA_OPC_UA_ENDPOINT,
                mqtt_broker=settings.MQTT_BROKER_URL
            )
            await self.scada.connect()

            logger.info("All integrations initialized successfully")
            metrics_collector.integration_status.labels(
                agent=self.agent_id,
                integration="all"
            ).set(1)

        except Exception as e:
            logger.error(f"Failed to initialize integrations: {e}")
            metrics_collector.integration_status.labels(
                agent=self.agent_id,
                integration="all"
            ).set(0)
            raise

    async def collect_burner_state(self) -> BurnerState:
        """
        Collect current burner state from all sensors and controllers

        Returns:
            BurnerState: Current state of the burner system
        """
        try:
            logger.debug("Collecting burner state data...")

            # Collect data from all sources in parallel
            fuel_flow, air_flow, o2_level, emissions, flame_temp, furnace_temp, flue_temp, load = await asyncio.gather(
                self.burner_controller.get_fuel_flow_rate(),
                self.burner_controller.get_air_flow_rate(),
                self.o2_analyzer.get_o2_concentration(),
                self.emissions_monitor.get_emissions_data(),
                self.temperature_sensors.get_flame_temperature(),
                self.temperature_sensors.get_furnace_temperature(),
                self.temperature_sensors.get_flue_gas_temperature(),
                self.burner_controller.get_burner_load()
            )

            # Calculate air-fuel ratio
            air_fuel_ratio = air_flow / fuel_flow if fuel_flow > 0 else 0

            # Create burner state
            state = BurnerState(
                fuel_flow_rate=fuel_flow,
                air_flow_rate=air_flow,
                air_fuel_ratio=air_fuel_ratio,
                o2_level=o2_level,
                co_level=emissions.get('CO'),
                nox_level=emissions.get('NOx'),
                flame_temperature=flame_temp,
                furnace_temperature=furnace_temp,
                flue_gas_temperature=flue_temp,
                burner_load=load
            )

            self.current_state = state

            # Update metrics
            metrics_collector.air_fuel_ratio.labels(agent=self.agent_id).set(air_fuel_ratio)
            metrics_collector.o2_level.labels(agent=self.agent_id).set(o2_level)
            metrics_collector.burner_load.labels(agent=self.agent_id).set(load)

            logger.debug(f"Burner state collected: AFR={air_fuel_ratio:.2f}, O2={o2_level:.1f}%")
            return state

        except Exception as e:
            logger.error(f"Failed to collect burner state: {e}")
            metrics_collector.error_counter.labels(
                agent=self.agent_id,
                error_type="data_collection"
            ).inc()
            raise

    async def check_safety_interlocks(self) -> SafetyInterlocks:
        """
        Check all safety interlocks before making changes

        Returns:
            SafetyInterlocks: Status of all safety interlocks
        """
        try:
            logger.debug("Checking safety interlocks...")

            # Check all interlocks in parallel
            flame, fuel_press, air_press, purge, temp, estop = await asyncio.gather(
                self.flame_scanner.is_flame_present(),
                self.burner_controller.check_fuel_pressure(),
                self.burner_controller.check_air_pressure(),
                self.burner_controller.is_purge_complete(),
                self.burner_controller.check_temperature_limits(),
                self.burner_controller.is_emergency_stop_clear()
            )

            interlocks = SafetyInterlocks(
                flame_present=flame,
                fuel_pressure_ok=fuel_press,
                air_pressure_ok=air_press,
                purge_complete=purge,
                temperature_ok=temp,
                emergency_stop_clear=estop
            )

            if not interlocks.all_safe():
                logger.warning(f"Safety interlocks not satisfied: {interlocks.dict()}")
                metrics_collector.safety_interlock_counter.labels(
                    agent=self.agent_id
                ).inc()

            return interlocks

        except Exception as e:
            logger.error(f"Failed to check safety interlocks: {e}")
            raise

    async def analyze_combustion(self, state: BurnerState) -> Dict[str, Any]:
        """
        Analyze current combustion performance

        Args:
            state: Current burner state

        Returns:
            Dict with analysis results including efficiency and emissions
        """
        try:
            logger.debug("Analyzing combustion performance...")

            # Calculate stoichiometric parameters
            stoich_result = self.stoichiometric_calc.calculate(
                fuel_type=settings.FUEL_TYPE,
                fuel_composition=settings.FUEL_COMPOSITION,
                air_fuel_ratio=state.air_fuel_ratio
            )

            # Calculate combustion efficiency
            efficiency_result = self.efficiency_calc.calculate(
                fuel_type=settings.FUEL_TYPE,
                fuel_flow=state.fuel_flow_rate,
                air_flow=state.air_flow_rate,
                flue_gas_temp=state.flue_gas_temperature,
                ambient_temp=settings.AMBIENT_TEMPERATURE,
                o2_level=state.o2_level,
                co_ppm=state.co_level or 0
            )

            # Calculate emissions
            emissions_result = self.emissions_calc.calculate(
                fuel_type=settings.FUEL_TYPE,
                fuel_composition=settings.FUEL_COMPOSITION,
                air_fuel_ratio=state.air_fuel_ratio,
                flame_temperature=state.flame_temperature or 0,
                excess_air=stoich_result['excess_air_percent']
            )

            # Analyze flame characteristics
            flame_result = self.flame_calc.analyze(
                flame_temperature=state.flame_temperature or 0,
                fuel_type=settings.FUEL_TYPE,
                air_fuel_ratio=state.air_fuel_ratio
            )

            # Calculate burner performance metrics
            performance_result = self.performance_calc.calculate(
                fuel_flow=state.fuel_flow_rate,
                burner_load=state.burner_load,
                max_capacity=settings.BURNER_MAX_CAPACITY
            )

            analysis = {
                'stoichiometric': stoich_result,
                'efficiency': efficiency_result,
                'emissions': emissions_result,
                'flame': flame_result,
                'performance': performance_result,
                'timestamp': DeterministicClock.utcnow().isoformat()
            }

            # Update metrics
            metrics_collector.combustion_efficiency.labels(
                agent=self.agent_id
            ).set(efficiency_result['gross_efficiency'])

            metrics_collector.nox_level.labels(
                agent=self.agent_id
            ).set(emissions_result['nox_ppm'])

            metrics_collector.co_level.labels(
                agent=self.agent_id
            ).set(state.co_level or 0)

            logger.info(f"Combustion analysis: Efficiency={efficiency_result['gross_efficiency']:.2f}%, "
                       f"NOx={emissions_result['nox_ppm']:.1f}ppm, CO={state.co_level or 0:.1f}ppm")

            return analysis

        except Exception as e:
            logger.error(f"Failed to analyze combustion: {e}")
            raise

    async def optimize_burner_settings(
        self,
        state: BurnerState,
        analysis: Dict[str, Any]
    ) -> OptimizationResult:
        """
        Optimize burner settings for efficiency and emissions

        Args:
            state: Current burner state
            analysis: Combustion analysis results

        Returns:
            OptimizationResult: Optimized settings and predicted improvements
        """
        try:
            logger.info("Optimizing burner settings...")
            metrics_collector.optimization_counter.labels(agent=self.agent_id).inc()

            # Define optimization objectives
            objectives = {
                'maximize_efficiency': True,
                'minimize_nox': True,
                'minimize_co': True,
                'target_efficiency': settings.TARGET_EFFICIENCY_PERCENT,
                'max_nox_ppm': settings.MAX_NOX_PPM,
                'max_co_ppm': settings.MAX_CO_PPM
            }

            # Define constraints
            constraints = {
                'min_excess_air': settings.MIN_EXCESS_AIR_PERCENT,
                'max_excess_air': settings.MAX_EXCESS_AIR_PERCENT,
                'min_o2': settings.MIN_O2_PERCENT,
                'max_o2': settings.MAX_O2_PERCENT,
                'min_fuel_flow': settings.MIN_FUEL_FLOW,
                'max_fuel_flow': settings.MAX_FUEL_FLOW,
                'min_air_flow': settings.MIN_AIR_FLOW,
                'max_air_flow': settings.MAX_AIR_FLOW
            }

            # Run optimization
            optimization_result = self.optimizer.optimize(
                current_state=state,
                current_analysis=analysis,
                objectives=objectives,
                constraints=constraints,
                fuel_type=settings.FUEL_TYPE
            )

            # Create result object
            result = OptimizationResult(
                current_air_fuel_ratio=state.air_fuel_ratio,
                current_efficiency=analysis['efficiency']['gross_efficiency'],
                current_nox=analysis['emissions']['nox_ppm'],
                current_co=state.co_level or 0,

                optimal_air_fuel_ratio=optimization_result['optimal_afr'],
                optimal_fuel_flow=optimization_result['optimal_fuel_flow'],
                optimal_air_flow=optimization_result['optimal_air_flow'],
                optimal_excess_air=optimization_result['optimal_excess_air'],

                predicted_efficiency=optimization_result['predicted_efficiency'],
                predicted_nox=optimization_result['predicted_nox'],
                predicted_co=optimization_result['predicted_co'],

                efficiency_improvement=optimization_result['predicted_efficiency'] - analysis['efficiency']['gross_efficiency'],
                nox_reduction=((analysis['emissions']['nox_ppm'] - optimization_result['predicted_nox']) /
                              analysis['emissions']['nox_ppm'] * 100 if analysis['emissions']['nox_ppm'] > 0 else 0),
                co_reduction=((state.co_level - optimization_result['predicted_co']) /
                             state.co_level * 100 if state.co_level and state.co_level > 0 else 0),
                fuel_savings=optimization_result['fuel_savings'],

                iterations=optimization_result['iterations'],
                convergence_status=optimization_result['convergence_status'],
                confidence_score=optimization_result['confidence_score'],

                hash=""  # Will be calculated
            )

            # Calculate deterministic hash
            result.hash = result.calculate_hash()

            # Store in history
            self.optimization_history.append(result)

            # Update metrics
            metrics_collector.efficiency_improvement.labels(
                agent=self.agent_id
            ).set(result.efficiency_improvement)

            metrics_collector.emissions_reduction.labels(
                agent=self.agent_id,
                pollutant="NOx"
            ).set(result.nox_reduction)

            logger.info(f"Optimization complete: Efficiency +{result.efficiency_improvement:.2f}%, "
                       f"NOx -{result.nox_reduction:.1f}%, Fuel savings {result.fuel_savings:.2f} kg/hr")

            return result

        except Exception as e:
            logger.error(f"Failed to optimize burner settings: {e}")
            metrics_collector.error_counter.labels(
                agent=self.agent_id,
                error_type="optimization"
            ).inc()
            raise

    async def implement_optimized_settings(
        self,
        optimization: OptimizationResult,
        interlocks: SafetyInterlocks
    ) -> bool:
        """
        Implement optimized burner settings with safety checks

        Args:
            optimization: Optimization result with new settings
            interlocks: Safety interlock status

        Returns:
            bool: True if settings were successfully implemented
        """
        try:
            # Check safety interlocks
            if not interlocks.all_safe():
                logger.error("Cannot implement settings: Safety interlocks not satisfied")
                return False

            logger.info("Implementing optimized burner settings...")

            # Calculate new fuel and air flow setpoints
            new_fuel_flow = optimization.optimal_fuel_flow
            new_air_flow = optimization.optimal_air_flow

            # Validate setpoints are within safe limits
            if not (settings.MIN_FUEL_FLOW <= new_fuel_flow <= settings.MAX_FUEL_FLOW):
                logger.error(f"Fuel flow {new_fuel_flow} outside safe limits")
                return False

            if not (settings.MIN_AIR_FLOW <= new_air_flow <= settings.MAX_AIR_FLOW):
                logger.error(f"Air flow {new_air_flow} outside safe limits")
                return False

            # Implement changes gradually (ramp rate control)
            steps = 10
            current_fuel = self.current_state.fuel_flow_rate
            current_air = self.current_state.air_flow_rate

            for step in range(1, steps + 1):
                # Calculate intermediate setpoints
                intermediate_fuel = current_fuel + (new_fuel_flow - current_fuel) * step / steps
                intermediate_air = current_air + (new_air_flow - current_air) * step / steps

                # Write setpoints to burner controller
                await self.burner_controller.set_fuel_flow(intermediate_fuel)
                await self.burner_controller.set_air_flow(intermediate_air)

                # Wait for stabilization
                await asyncio.sleep(settings.SETPOINT_CHANGE_DELAY_SECONDS)

                # Check interlocks after each step
                current_interlocks = await self.check_safety_interlocks()
                if not current_interlocks.all_safe():
                    logger.error("Safety interlock tripped during implementation - reverting")
                    # Revert to original settings
                    await self.burner_controller.set_fuel_flow(current_fuel)
                    await self.burner_controller.set_air_flow(current_air)
                    return False

            logger.info(f"Successfully implemented optimized settings: "
                       f"Fuel={new_fuel_flow:.2f}, Air={new_air_flow:.2f}")

            metrics_collector.setpoint_change_counter.labels(
                agent=self.agent_id
            ).inc()

            # Publish to SCADA
            await self.scada.publish_optimization_result(optimization)

            return True

        except Exception as e:
            logger.error(f"Failed to implement optimized settings: {e}")
            metrics_collector.error_counter.labels(
                agent=self.agent_id,
                error_type="implementation"
            ).inc()
            return False

    async def validate_optimization_results(
        self,
        optimization: OptimizationResult,
        validation_duration_seconds: int = 300
    ) -> Dict[str, Any]:
        """
        Validate that optimization achieved predicted improvements

        Args:
            optimization: The optimization result to validate
            validation_duration_seconds: How long to monitor (default 5 minutes)

        Returns:
            Dict with validation results
        """
        try:
            logger.info(f"Validating optimization results over {validation_duration_seconds}s...")

            # Collect measurements over validation period
            measurements = []
            interval = 30  # Sample every 30 seconds
            samples = validation_duration_seconds // interval

            for i in range(samples):
                state = await self.collect_burner_state()
                analysis = await self.analyze_combustion(state)

                measurements.append({
                    'timestamp': DeterministicClock.utcnow(),
                    'efficiency': analysis['efficiency']['gross_efficiency'],
                    'nox': analysis['emissions']['nox_ppm'],
                    'co': state.co_level or 0,
                    'o2': state.o2_level
                })

                if i < samples - 1:  # Don't sleep on last iteration
                    await asyncio.sleep(interval)

            # Calculate average actual results
            avg_efficiency = sum(m['efficiency'] for m in measurements) / len(measurements)
            avg_nox = sum(m['nox'] for m in measurements) / len(measurements)
            avg_co = sum(m['co'] for m in measurements) / len(measurements)

            # Compare to predictions
            efficiency_error = abs(avg_efficiency - optimization.predicted_efficiency)
            nox_error = abs(avg_nox - optimization.predicted_nox)
            co_error = abs(avg_co - optimization.predicted_co)

            validation_passed = (
                efficiency_error < settings.VALIDATION_EFFICIENCY_TOLERANCE and
                nox_error < settings.VALIDATION_NOX_TOLERANCE and
                co_error < settings.VALIDATION_CO_TOLERANCE
            )

            validation_result = {
                'optimization_id': optimization.optimization_id,
                'validation_passed': validation_passed,
                'predicted_efficiency': optimization.predicted_efficiency,
                'actual_efficiency': avg_efficiency,
                'efficiency_error': efficiency_error,
                'predicted_nox': optimization.predicted_nox,
                'actual_nox': avg_nox,
                'nox_error': nox_error,
                'predicted_co': optimization.predicted_co,
                'actual_co': avg_co,
                'co_error': co_error,
                'measurements': measurements,
                'timestamp': DeterministicClock.utcnow().isoformat()
            }

            logger.info(f"Validation {'PASSED' if validation_passed else 'FAILED'}: "
                       f"Efficiency actual={avg_efficiency:.2f}% vs predicted={optimization.predicted_efficiency:.2f}%")

            return validation_result

        except Exception as e:
            logger.error(f"Failed to validate optimization results: {e}")
            raise

    async def run_optimization_cycle(self) -> OptimizationResult:
        """
        Run complete optimization cycle

        Returns:
            OptimizationResult: Result of the optimization
        """
        try:
            logger.info("=== Starting Optimization Cycle ===")

            # 1. Collect current burner state
            state = await self.collect_burner_state()

            # 2. Check safety interlocks
            interlocks = await self.check_safety_interlocks()
            if not interlocks.all_safe():
                logger.warning("Skipping optimization: Safety interlocks not satisfied")
                metrics_collector.optimization_skipped_counter.labels(
                    agent=self.agent_id,
                    reason="safety_interlocks"
                ).inc()
                raise ValueError("Safety interlocks not satisfied")

            # 3. Analyze current combustion performance
            analysis = await self.analyze_combustion(state)

            # 4. Optimize burner settings
            optimization = await self.optimize_burner_settings(state, analysis)

            # 5. Implement optimized settings
            success = await self.implement_optimized_settings(optimization, interlocks)

            if not success:
                logger.error("Failed to implement optimized settings")
                return optimization

            # 6. Validate optimization results
            validation = await self.validate_optimization_results(
                optimization,
                validation_duration_seconds=settings.VALIDATION_DURATION_SECONDS
            )

            if validation['validation_passed']:
                logger.info("✓ Optimization cycle completed successfully and validated")
            else:
                logger.warning("⚠ Optimization implemented but validation failed")

            return optimization

        except Exception as e:
            logger.error(f"Optimization cycle failed: {e}")
            metrics_collector.error_counter.labels(
                agent=self.agent_id,
                error_type="optimization_cycle"
            ).inc()
            raise

    async def start(self) -> None:
        """Start the burner optimization agent"""
        try:
            logger.info(f"Starting {self.agent_name}...")

            # Initialize all integrations
            await self.initialize_integrations()

            self.is_running = True
            logger.info(f"{self.agent_name} started successfully")

            # Run continuous optimization loop
            while self.is_running:
                try:
                    await self.run_optimization_cycle()

                    # Wait before next cycle
                    await asyncio.sleep(settings.OPTIMIZATION_INTERVAL_SECONDS)

                except Exception as e:
                    logger.error(f"Error in optimization cycle: {e}")
                    await asyncio.sleep(settings.ERROR_RETRY_DELAY_SECONDS)

        except Exception as e:
            logger.error(f"Failed to start {self.agent_name}: {e}")
            raise

    async def stop(self) -> None:
        """Stop the burner optimization agent"""
        logger.info(f"Stopping {self.agent_name}...")
        self.is_running = False

        # Close all integrations
        if self.burner_controller:
            await self.burner_controller.disconnect()
        if self.o2_analyzer:
            await self.o2_analyzer.disconnect()
        if self.emissions_monitor:
            await self.emissions_monitor.disconnect()
        if self.flame_scanner:
            await self.flame_scanner.disconnect()
        if self.temperature_sensors:
            await self.temperature_sensors.disconnect()
        if self.scada:
            await self.scada.disconnect()

        logger.info(f"{self.agent_name} stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            'agent_id': self.agent_id,
            'agent_name': self.agent_name,
            'version': self.version,
            'is_running': self.is_running,
            'current_state': self.current_state.dict() if self.current_state else None,
            'optimization_count': len(self.optimization_history),
            'last_optimization': self.optimization_history[-1].dict() if self.optimization_history else None
        }
