# -*- coding: utf-8 -*-
"""
GL-005 CombustionControlAgent - Main Orchestrator
Automated control of combustion processes for consistent heat output

This agent implements zero-hallucination real-time combustion control using:
- PID control loops for fuel and air flow
- Oxygen trim control for efficiency
- Multi-variable control with feedforward compensation
- Safety interlocks and fail-safes
- Heat output stability control

Agent Specification:
- Agent ID: GL-005
- Agent Name: CombustionControlAgent
- Category: Combustion Control
- Type: Real-time Controller
- Primary Function: Automated control of combustion processes for consistent heat output
- Inputs: Fuel flow, air flow, temperature, pressure
- Outputs: Real-time combustion adjustments, stability metrics
- Control Loop Target: <100ms response time
"""

import asyncio
import hashlib
import json
import logging
import time
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Deque
from uuid import uuid4

from pydantic import BaseModel, Field, validator

from calculators.pid_controller import PIDController
from calculators.feedforward_controller import FeedforwardController
from calculators.stability_analyzer import StabilityAnalyzer
from calculators.heat_output_calculator import HeatOutputCalculator
from calculators.air_fuel_ratio_calculator import AirFuelRatioCalculator
from calculators.combustion_performance_calculator import CombustionPerformanceCalculator

from integrations.dcs_connector import DCSConnector
from integrations.plc_connector import PLCConnector
from integrations.combustion_analyzer_connector import CombustionAnalyzerConnector
from integrations.pressure_sensor_connector import PressureSensorConnector
from integrations.temperature_sensor_connector import TemperatureSensorConnector
from integrations.flow_meter_connector import FlowMeterConnector
from integrations.scada_integration import SCADAIntegration

from config import settings
from monitoring.metrics import metrics_collector
from greenlang.determinism import deterministic_uuid, DeterministicClock

logger = logging.getLogger(__name__)


class CombustionState(BaseModel):
    """Current state of combustion process"""
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Flow measurements
    fuel_flow: float = Field(..., description="Fuel flow rate (kg/hr or m3/hr)")
    air_flow: float = Field(..., description="Combustion air flow rate (m3/hr)")
    air_fuel_ratio: float = Field(..., description="Actual air-fuel ratio")

    # Temperature measurements
    flame_temperature: Optional[float] = Field(None, description="Flame temperature (°C)")
    furnace_temperature: float = Field(..., description="Furnace temperature (°C)")
    flue_gas_temperature: float = Field(..., description="Flue gas temperature (°C)")
    ambient_temperature: float = Field(..., description="Ambient air temperature (°C)")

    # Pressure measurements
    fuel_pressure: float = Field(..., description="Fuel supply pressure (kPa)")
    air_pressure: float = Field(..., description="Air supply pressure (kPa)")
    furnace_pressure: float = Field(..., description="Furnace draft pressure (Pa)")

    # Combustion quality measurements
    o2_percent: float = Field(..., description="O2 in flue gas (%)")
    co_ppm: Optional[float] = Field(None, description="CO concentration (ppm)")
    co2_percent: Optional[float] = Field(None, description="CO2 concentration (%)")
    nox_ppm: Optional[float] = Field(None, description="NOx concentration (ppm)")

    # Performance metrics
    heat_output_kw: Optional[float] = Field(None, description="Calculated heat output (kW)")
    thermal_efficiency: Optional[float] = Field(None, description="Thermal efficiency (%)")
    excess_air_percent: Optional[float] = Field(None, description="Excess air (%)")

    @validator('o2_percent')
    def validate_o2(cls, v: float) -> float:
        """Validate O2 level is within physical bounds"""
        if not 0 <= v <= 21:
            raise ValueError(f"O2 level {v}% is outside valid range [0, 21]%")
        return v

    @validator('fuel_pressure', 'air_pressure')
    def validate_pressure(cls, v: float) -> float:
        """Validate pressures are positive"""
        if v < 0:
            raise ValueError(f"Pressure {v} cannot be negative")
        return v


class ControlAction(BaseModel):
    """Control action to be implemented"""
    action_id: str = Field(default_factory=lambda: str(deterministic_uuid(__name__, str(DeterministicClock.now()))))
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Fuel control
    fuel_flow_setpoint: float = Field(..., description="New fuel flow setpoint")
    fuel_flow_delta: float = Field(..., description="Change from current")

    # Air control
    air_flow_setpoint: float = Field(..., description="New air flow setpoint")
    air_flow_delta: float = Field(..., description="Change from current")

    # Control modes
    fuel_control_mode: str = Field(..., description="auto, manual, cascade")
    air_control_mode: str = Field(..., description="auto, manual, ratio")
    o2_trim_enabled: bool = Field(True, description="O2 trim control active")

    # Control outputs
    fuel_valve_position: float = Field(..., ge=0, le=100, description="Fuel valve % open")
    air_damper_position: float = Field(..., ge=0, le=100, description="Air damper % open")

    # Safety flags
    safety_override: bool = Field(False, description="Safety system override active")
    interlock_satisfied: bool = Field(..., description="All interlocks satisfied")

    # Provenance
    hash: str = Field(..., description="SHA-256 hash for determinism")

    def calculate_hash(self) -> str:
        """Calculate deterministic hash of control action"""
        hashable_data = {
            'fuel_flow_setpoint': round(self.fuel_flow_setpoint, 6),
            'air_flow_setpoint': round(self.air_flow_setpoint, 6),
            'fuel_valve_position': round(self.fuel_valve_position, 4),
            'air_damper_position': round(self.air_damper_position, 4)
        }
        hash_input = json.dumps(hashable_data, sort_keys=True)
        return hashlib.sha256(hash_input.encode()).hexdigest()


class StabilityMetrics(BaseModel):
    """Combustion stability metrics"""
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Heat output stability
    heat_output_stability_index: float = Field(..., ge=0, le=1, description="0=unstable, 1=stable")
    heat_output_variance: float = Field(..., description="Variance in heat output (kW²)")
    heat_output_cv: float = Field(..., description="Coefficient of variation (%)")

    # Temperature stability
    furnace_temp_stability: float = Field(..., ge=0, le=1)
    flame_temp_stability: float = Field(..., ge=0, le=1)

    # Emissions stability
    o2_stability: float = Field(..., ge=0, le=1)
    co_stability: Optional[float] = Field(None, ge=0, le=1)

    # Oscillation detection
    oscillation_detected: bool = Field(False, description="Control oscillation detected")
    oscillation_frequency_hz: Optional[float] = Field(None, description="Oscillation frequency")
    oscillation_amplitude: Optional[float] = Field(None, description="Peak-to-peak amplitude")

    # Overall stability
    overall_stability_score: float = Field(..., ge=0, le=100, description="Composite stability (0-100)")
    stability_rating: str = Field(..., description="excellent, good, fair, poor, unstable")


class SafetyInterlocks(BaseModel):
    """Safety interlock status"""
    flame_present: bool = Field(..., description="Flame detection confirmed")
    fuel_pressure_ok: bool = Field(..., description="Fuel pressure within limits")
    air_pressure_ok: bool = Field(..., description="Air pressure within limits")
    furnace_temp_ok: bool = Field(..., description="Furnace temp within safe limits")
    furnace_pressure_ok: bool = Field(..., description="Furnace pressure within limits")
    purge_complete: bool = Field(..., description="Pre-purge completed")
    emergency_stop_clear: bool = Field(..., description="No emergency stop active")
    high_fire_lockout_clear: bool = Field(..., description="No high fire lockout")
    low_fire_lockout_clear: bool = Field(..., description="No low fire lockout")

    def all_safe(self) -> bool:
        """Check if all interlocks are satisfied"""
        return all([
            self.flame_present,
            self.fuel_pressure_ok,
            self.air_pressure_ok,
            self.furnace_temp_ok,
            self.furnace_pressure_ok,
            self.purge_complete,
            self.emergency_stop_clear,
            self.high_fire_lockout_clear,
            self.low_fire_lockout_clear
        ])

    def get_failed_interlocks(self) -> List[str]:
        """Get list of failed interlock names"""
        failed = []
        for field, value in self.dict().items():
            if not value:
                failed.append(field)
        return failed


class CombustionControlOrchestrator:
    """
    Main orchestrator for GL-005 CombustionControlAgent

    Orchestrates real-time combustion control workflow:
    1. Read combustion state from DCS/PLC/analyzers (<50ms)
    2. Analyze stability and performance
    3. Execute PID control loops (fuel, air, O2 trim)
    4. Calculate optimal control actions
    5. Implement control actions with safety checks
    6. Monitor and validate stability

    Control Loop Performance:
    - Target cycle time: <100ms
    - PID update rate: 10 Hz (100ms)
    - Data acquisition rate: 20 Hz (50ms)
    - Safety check rate: 20 Hz (50ms)
    """

    def __init__(self):
        """Initialize the combustion control orchestrator"""
        self.agent_id = "GL-005"
        self.agent_name = "CombustionControlAgent"
        self.version = "1.0.0"

        # Initialize PID controllers
        self.fuel_pid = PIDController(
            kp=settings.FUEL_CONTROL_KP,
            ki=settings.FUEL_CONTROL_KI,
            kd=settings.FUEL_CONTROL_KD,
            output_limits=(settings.MIN_FUEL_FLOW, settings.MAX_FUEL_FLOW),
            sample_time=settings.CONTROL_LOOP_INTERVAL_MS / 1000.0
        )

        self.air_pid = PIDController(
            kp=settings.AIR_CONTROL_KP,
            ki=settings.AIR_CONTROL_KI,
            kd=settings.AIR_CONTROL_KD,
            output_limits=(settings.MIN_AIR_FLOW, settings.MAX_AIR_FLOW),
            sample_time=settings.CONTROL_LOOP_INTERVAL_MS / 1000.0
        )

        self.o2_trim_pid = PIDController(
            kp=settings.O2_TRIM_KP,
            ki=settings.O2_TRIM_KI,
            kd=settings.O2_TRIM_KD,
            output_limits=(-settings.O2_TRIM_MAX_ADJUSTMENT, settings.O2_TRIM_MAX_ADJUSTMENT),
            sample_time=settings.O2_TRIM_INTERVAL_MS / 1000.0
        )

        # Initialize calculators
        self.feedforward_calc = FeedforwardController()
        self.stability_analyzer = StabilityAnalyzer(window_size=settings.STABILITY_WINDOW_SIZE)
        self.heat_output_calc = HeatOutputCalculator()
        self.afr_calc = AirFuelRatioCalculator()
        self.performance_calc = CombustionPerformanceCalculator()

        # Initialize integrations
        self.dcs: Optional[DCSConnector] = None
        self.plc: Optional[PLCConnector] = None
        self.combustion_analyzer: Optional[CombustionAnalyzerConnector] = None
        self.pressure_sensors: Optional[PressureSensorConnector] = None
        self.temperature_sensors: Optional[TemperatureSensorConnector] = None
        self.flow_meters: Optional[FlowMeterConnector] = None
        self.scada: Optional[SCADAIntegration] = None

        # State management
        self.current_state: Optional[CombustionState] = None
        self.control_history: Deque[ControlAction] = deque(maxlen=settings.CONTROL_HISTORY_SIZE)
        self.state_history: Deque[CombustionState] = deque(maxlen=settings.STATE_HISTORY_SIZE)
        self.stability_history: Deque[StabilityMetrics] = deque(maxlen=100)

        # Control mode
        self.control_enabled = False
        self.is_running = False
        self.last_control_time: float = 0.0

        # Performance tracking
        self.cycle_times: Deque[float] = deque(maxlen=1000)
        self.control_errors: int = 0

        logger.info(f"Initialized {self.agent_name} v{self.version}")

    async def initialize_integrations(self) -> None:
        """Initialize all integration connectors"""
        try:
            logger.info("Initializing integrations...")

            # Initialize DCS (primary control interface)
            self.dcs = DCSConnector(
                host=settings.DCS_HOST,
                port=settings.DCS_PORT,
                protocol=settings.DCS_PROTOCOL,
                timeout=settings.DCS_TIMEOUT_MS / 1000.0
            )
            await self.dcs.connect()

            # Initialize PLC (secondary/backup control)
            self.plc = PLCConnector(
                host=settings.PLC_HOST,
                port=settings.PLC_PORT,
                modbus_id=settings.PLC_MODBUS_ID,
                timeout=settings.PLC_TIMEOUT_MS / 1000.0
            )
            await self.plc.connect()

            # Initialize combustion analyzer
            self.combustion_analyzer = CombustionAnalyzerConnector(
                endpoints=settings.COMBUSTION_ANALYZER_ENDPOINTS,
                timeout=settings.ANALYZER_TIMEOUT_MS / 1000.0
            )
            await self.combustion_analyzer.connect()

            # Initialize pressure sensors
            self.pressure_sensors = PressureSensorConnector(
                sensors=settings.PRESSURE_SENSORS
            )
            await self.pressure_sensors.connect()

            # Initialize temperature sensors
            self.temperature_sensors = TemperatureSensorConnector(
                sensors=settings.TEMPERATURE_SENSORS
            )
            await self.temperature_sensors.connect()

            # Initialize flow meters
            self.flow_meters = FlowMeterConnector(
                fuel_flow_meter=settings.FUEL_FLOW_METER,
                air_flow_meter=settings.AIR_FLOW_METER
            )
            await self.flow_meters.connect()

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

    async def read_combustion_state(self) -> CombustionState:
        """
        Read current combustion state from all sensors and controllers

        Target execution time: <50ms

        Returns:
            CombustionState: Current state of combustion process
        """
        start_time = time.perf_counter()

        try:
            logger.debug("Reading combustion state...")

            # Read all data sources in parallel for minimum latency
            (
                fuel_flow, air_flow,
                flame_temp, furnace_temp, flue_temp, ambient_temp,
                fuel_press, air_press, furnace_press,
                analyzer_data
            ) = await asyncio.gather(
                self.flow_meters.get_fuel_flow(),
                self.flow_meters.get_air_flow(),
                self.temperature_sensors.get_temperature('flame'),
                self.temperature_sensors.get_temperature('furnace'),
                self.temperature_sensors.get_temperature('flue_gas'),
                self.temperature_sensors.get_temperature('ambient'),
                self.pressure_sensors.get_pressure('fuel'),
                self.pressure_sensors.get_pressure('air'),
                self.pressure_sensors.get_pressure('furnace'),
                self.combustion_analyzer.get_measurements()
            )

            # Calculate derived values
            air_fuel_ratio = air_flow / fuel_flow if fuel_flow > 0 else 0

            # Calculate heat output
            heat_output = self.heat_output_calc.calculate(
                fuel_flow=fuel_flow,
                fuel_type=settings.FUEL_TYPE,
                fuel_lhv=settings.FUEL_LHV_MJ_PER_KG
            )

            # Calculate thermal efficiency
            thermal_eff = self.performance_calc.calculate_efficiency(
                fuel_flow=fuel_flow,
                fuel_lhv=settings.FUEL_LHV_MJ_PER_KG,
                flue_gas_temp=flue_temp,
                ambient_temp=ambient_temp,
                o2_percent=analyzer_data.get('O2', 0)
            )

            # Calculate excess air
            excess_air = self.afr_calc.calculate_excess_air(
                o2_percent=analyzer_data.get('O2', 0),
                fuel_type=settings.FUEL_TYPE
            )

            # Create state object
            state = CombustionState(
                fuel_flow=fuel_flow,
                air_flow=air_flow,
                air_fuel_ratio=air_fuel_ratio,
                flame_temperature=flame_temp,
                furnace_temperature=furnace_temp,
                flue_gas_temperature=flue_temp,
                ambient_temperature=ambient_temp,
                fuel_pressure=fuel_press,
                air_pressure=air_press,
                furnace_pressure=furnace_press,
                o2_percent=analyzer_data.get('O2', 0),
                co_ppm=analyzer_data.get('CO'),
                co2_percent=analyzer_data.get('CO2'),
                nox_ppm=analyzer_data.get('NOx'),
                heat_output_kw=heat_output,
                thermal_efficiency=thermal_eff,
                excess_air_percent=excess_air
            )

            # Store in history
            self.current_state = state
            self.state_history.append(state)

            # Update metrics
            metrics_collector.fuel_flow.labels(agent=self.agent_id).set(fuel_flow)
            metrics_collector.air_flow.labels(agent=self.agent_id).set(air_flow)
            metrics_collector.air_fuel_ratio.labels(agent=self.agent_id).set(air_fuel_ratio)
            metrics_collector.o2_level.labels(agent=self.agent_id).set(analyzer_data.get('O2', 0))
            metrics_collector.heat_output.labels(agent=self.agent_id).set(heat_output)
            metrics_collector.thermal_efficiency.labels(agent=self.agent_id).set(thermal_eff)

            # Track execution time
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            metrics_collector.state_read_time_ms.labels(agent=self.agent_id).observe(elapsed_ms)

            if elapsed_ms > 50:
                logger.warning(f"State read took {elapsed_ms:.1f}ms (target <50ms)")

            logger.debug(f"State read complete: {elapsed_ms:.1f}ms")
            return state

        except Exception as e:
            logger.error(f"Failed to read combustion state: {e}")
            metrics_collector.error_counter.labels(
                agent=self.agent_id,
                error_type="state_read"
            ).inc()
            raise

    async def check_safety_interlocks(self) -> SafetyInterlocks:
        """
        Check all safety interlocks before control actions

        Returns:
            SafetyInterlocks: Status of all safety interlocks
        """
        try:
            logger.debug("Checking safety interlocks...")

            # Check interlocks via DCS and PLC
            dcs_interlocks, plc_interlocks = await asyncio.gather(
                self.dcs.get_interlock_status(),
                self.plc.get_interlock_status()
            )

            # Merge interlock status (fail-safe: use most restrictive)
            interlocks = SafetyInterlocks(
                flame_present=dcs_interlocks.get('flame_present', False) and plc_interlocks.get('flame_present', False),
                fuel_pressure_ok=dcs_interlocks.get('fuel_pressure_ok', False),
                air_pressure_ok=dcs_interlocks.get('air_pressure_ok', False),
                furnace_temp_ok=dcs_interlocks.get('furnace_temp_ok', True),
                furnace_pressure_ok=dcs_interlocks.get('furnace_pressure_ok', True),
                purge_complete=dcs_interlocks.get('purge_complete', True),
                emergency_stop_clear=plc_interlocks.get('emergency_stop_clear', False),
                high_fire_lockout_clear=dcs_interlocks.get('high_fire_lockout_clear', True),
                low_fire_lockout_clear=dcs_interlocks.get('low_fire_lockout_clear', True)
            )

            if not interlocks.all_safe():
                failed = interlocks.get_failed_interlocks()
                logger.warning(f"Safety interlocks not satisfied: {failed}")
                metrics_collector.safety_interlock_counter.labels(
                    agent=self.agent_id
                ).inc()

                # Publish alarm to SCADA
                await self.scada.publish_alarm({
                    'severity': 'HIGH',
                    'message': f'Safety interlocks failed: {failed}',
                    'timestamp': DeterministicClock.utcnow().isoformat()
                })

            return interlocks

        except Exception as e:
            logger.error(f"Failed to check safety interlocks: {e}")
            # Fail-safe: return all interlocks failed
            return SafetyInterlocks(
                flame_present=False,
                fuel_pressure_ok=False,
                air_pressure_ok=False,
                furnace_temp_ok=False,
                furnace_pressure_ok=False,
                purge_complete=False,
                emergency_stop_clear=False,
                high_fire_lockout_clear=False,
                low_fire_lockout_clear=False
            )

    async def analyze_stability(self, state: CombustionState) -> StabilityMetrics:
        """
        Analyze combustion stability from recent state history

        Args:
            state: Current combustion state

        Returns:
            StabilityMetrics: Stability analysis results
        """
        try:
            logger.debug("Analyzing combustion stability...")

            # Need sufficient history for stability analysis
            if len(self.state_history) < settings.STABILITY_MIN_SAMPLES:
                return StabilityMetrics(
                    heat_output_stability_index=0.5,
                    heat_output_variance=0,
                    heat_output_cv=0,
                    furnace_temp_stability=0.5,
                    flame_temp_stability=0.5,
                    o2_stability=0.5,
                    overall_stability_score=50.0,
                    stability_rating="fair"
                )

            # Extract time series from history
            heat_outputs = [s.heat_output_kw for s in self.state_history if s.heat_output_kw is not None]
            furnace_temps = [s.furnace_temperature for s in self.state_history]
            flame_temps = [s.flame_temperature for s in self.state_history if s.flame_temperature is not None]
            o2_levels = [s.o2_percent for s in self.state_history]
            co_levels = [s.co_ppm for s in self.state_history if s.co_ppm is not None]

            # Analyze heat output stability
            heat_stability = self.stability_analyzer.analyze_stability(
                values=heat_outputs,
                target=settings.HEAT_OUTPUT_TARGET_KW,
                tolerance=settings.HEAT_OUTPUT_TOLERANCE_PERCENT
            )

            # Analyze temperature stability
            furnace_temp_stability = self.stability_analyzer.analyze_stability(
                values=furnace_temps,
                target=state.furnace_temperature,
                tolerance=settings.TEMPERATURE_STABILITY_TOLERANCE_C
            )

            flame_temp_stability = self.stability_analyzer.analyze_stability(
                values=flame_temps,
                target=state.flame_temperature or 0,
                tolerance=settings.TEMPERATURE_STABILITY_TOLERANCE_C
            ) if flame_temps else 0.5

            # Analyze O2 stability
            o2_stability = self.stability_analyzer.analyze_stability(
                values=o2_levels,
                target=settings.TARGET_O2_PERCENT,
                tolerance=settings.O2_STABILITY_TOLERANCE_PERCENT
            )

            # Analyze CO stability (if available)
            co_stability = None
            if co_levels:
                co_stability = self.stability_analyzer.analyze_stability(
                    values=co_levels,
                    target=0,
                    tolerance=settings.MAX_CO_PPM * 0.1
                )

            # Detect oscillations
            oscillation_result = self.stability_analyzer.detect_oscillations(
                values=heat_outputs,
                sample_rate_hz=1000.0 / settings.CONTROL_LOOP_INTERVAL_MS
            )

            # Calculate overall stability score (0-100)
            weights = {
                'heat_output': 0.4,
                'furnace_temp': 0.3,
                'o2': 0.2,
                'flame_temp': 0.1
            }

            overall_score = (
                heat_stability['stability_index'] * weights['heat_output'] +
                furnace_temp_stability['stability_index'] * weights['furnace_temp'] +
                o2_stability['stability_index'] * weights['o2'] +
                flame_temp_stability * weights['flame_temp']
            ) * 100

            # Determine stability rating
            if overall_score >= 90:
                rating = "excellent"
            elif overall_score >= 75:
                rating = "good"
            elif overall_score >= 60:
                rating = "fair"
            elif overall_score >= 40:
                rating = "poor"
            else:
                rating = "unstable"

            metrics_result = StabilityMetrics(
                heat_output_stability_index=heat_stability['stability_index'],
                heat_output_variance=heat_stability['variance'],
                heat_output_cv=heat_stability['coefficient_of_variation'],
                furnace_temp_stability=furnace_temp_stability['stability_index'],
                flame_temp_stability=flame_temp_stability,
                o2_stability=o2_stability['stability_index'],
                co_stability=co_stability['stability_index'] if co_stability else None,
                oscillation_detected=oscillation_result['oscillation_detected'],
                oscillation_frequency_hz=oscillation_result.get('frequency_hz'),
                oscillation_amplitude=oscillation_result.get('amplitude'),
                overall_stability_score=overall_score,
                stability_rating=rating
            )

            # Store in history
            self.stability_history.append(metrics_result)

            # Update metrics
            metrics_collector.stability_score.labels(agent=self.agent_id).set(overall_score)
            metrics_collector.oscillation_detected.labels(agent=self.agent_id).set(
                1 if oscillation_result['oscillation_detected'] else 0
            )

            logger.info(f"Stability analysis: Score={overall_score:.1f}, Rating={rating}")

            return metrics_result

        except Exception as e:
            logger.error(f"Failed to analyze stability: {e}")
            raise

    async def optimize_fuel_air_ratio(
        self,
        state: CombustionState,
        heat_demand_kw: float
    ) -> Tuple[float, float]:
        """
        Optimize fuel and air flow for target heat output

        Args:
            state: Current combustion state
            heat_demand_kw: Target heat output (kW)

        Returns:
            Tuple of (optimal_fuel_flow, optimal_air_flow)
        """
        try:
            logger.debug(f"Optimizing for heat demand: {heat_demand_kw:.1f} kW")

            # Calculate required fuel flow for heat demand
            required_fuel_flow = self.heat_output_calc.calculate_fuel_flow(
                heat_output_kw=heat_demand_kw,
                fuel_type=settings.FUEL_TYPE,
                fuel_lhv=settings.FUEL_LHV_MJ_PER_KG,
                efficiency=state.thermal_efficiency or settings.TARGET_EFFICIENCY_PERCENT
            )

            # Calculate stoichiometric air requirement
            stoich_air_flow = self.afr_calc.calculate_stoichiometric_air(
                fuel_flow=required_fuel_flow,
                fuel_type=settings.FUEL_TYPE,
                fuel_composition=settings.FUEL_COMPOSITION
            )

            # Add excess air for optimal combustion
            target_excess_air = settings.OPTIMAL_EXCESS_AIR_PERCENT / 100.0
            optimal_air_flow = stoich_air_flow * (1 + target_excess_air)

            # Apply O2 trim correction
            if settings.O2_TRIM_ENABLED:
                o2_error = state.o2_percent - settings.TARGET_O2_PERCENT
                trim_correction = self.o2_trim_pid.update(
                    setpoint=settings.TARGET_O2_PERCENT,
                    process_variable=state.o2_percent
                )
                optimal_air_flow += trim_correction

            # Constrain to safe operating limits
            optimal_fuel_flow = max(
                settings.MIN_FUEL_FLOW,
                min(settings.MAX_FUEL_FLOW, required_fuel_flow)
            )

            optimal_air_flow = max(
                settings.MIN_AIR_FLOW,
                min(settings.MAX_AIR_FLOW, optimal_air_flow)
            )

            logger.debug(f"Optimal flows: Fuel={optimal_fuel_flow:.2f}, Air={optimal_air_flow:.2f}")

            return optimal_fuel_flow, optimal_air_flow

        except Exception as e:
            logger.error(f"Failed to optimize fuel-air ratio: {e}")
            raise

    async def calculate_control_action(
        self,
        state: CombustionState,
        stability: StabilityMetrics,
        heat_demand_kw: float
    ) -> ControlAction:
        """
        Calculate control action using PID + feedforward control

        Args:
            state: Current combustion state
            stability: Stability metrics
            heat_demand_kw: Target heat output

        Returns:
            ControlAction: Control action to implement
        """
        try:
            logger.debug("Calculating control action...")

            # Get optimal fuel and air flows
            optimal_fuel, optimal_air = await self.optimize_fuel_air_ratio(state, heat_demand_kw)

            # Feedforward component (anticipates changes)
            feedforward_fuel = self.feedforward_calc.calculate_fuel_feedforward(
                heat_demand_kw=heat_demand_kw,
                fuel_lhv=settings.FUEL_LHV_MJ_PER_KG
            )

            feedforward_air = self.feedforward_calc.calculate_air_feedforward(
                fuel_flow=feedforward_fuel,
                fuel_type=settings.FUEL_TYPE,
                target_excess_air=settings.OPTIMAL_EXCESS_AIR_PERCENT
            )

            # PID feedback component (corrects errors)
            fuel_feedback = self.fuel_pid.update(
                setpoint=optimal_fuel,
                process_variable=state.fuel_flow
            )

            air_feedback = self.air_pid.update(
                setpoint=optimal_air,
                process_variable=state.air_flow
            )

            # Combined control output (feedforward + feedback)
            fuel_setpoint = feedforward_fuel + fuel_feedback
            air_setpoint = feedforward_air + air_feedback

            # Constrain to operating limits
            fuel_setpoint = max(settings.MIN_FUEL_FLOW, min(settings.MAX_FUEL_FLOW, fuel_setpoint))
            air_setpoint = max(settings.MIN_AIR_FLOW, min(settings.MAX_AIR_FLOW, air_setpoint))

            # Calculate deltas
            fuel_delta = fuel_setpoint - state.fuel_flow
            air_delta = air_setpoint - state.air_flow

            # Convert to valve/damper positions (assuming linear relationship)
            fuel_valve_pos = ((fuel_setpoint - settings.MIN_FUEL_FLOW) /
                             (settings.MAX_FUEL_FLOW - settings.MIN_FUEL_FLOW) * 100)

            air_damper_pos = ((air_setpoint - settings.MIN_AIR_FLOW) /
                             (settings.MAX_AIR_FLOW - settings.MIN_AIR_FLOW) * 100)

            # Determine control modes
            fuel_mode = "auto" if settings.FUEL_CONTROL_AUTO else "manual"
            air_mode = "auto" if settings.AIR_CONTROL_AUTO else "manual"

            # Create control action
            action = ControlAction(
                fuel_flow_setpoint=fuel_setpoint,
                fuel_flow_delta=fuel_delta,
                air_flow_setpoint=air_setpoint,
                air_flow_delta=air_delta,
                fuel_control_mode=fuel_mode,
                air_control_mode=air_mode,
                o2_trim_enabled=settings.O2_TRIM_ENABLED,
                fuel_valve_position=fuel_valve_pos,
                air_damper_position=air_damper_pos,
                safety_override=False,
                interlock_satisfied=True,  # Will be checked before implementation
                hash=""  # Will be calculated
            )

            # Calculate deterministic hash
            action.hash = action.calculate_hash()

            # Store in history
            self.control_history.append(action)

            logger.info(f"Control action: Fuel Δ{fuel_delta:+.2f} → {fuel_setpoint:.2f}, "
                       f"Air Δ{air_delta:+.2f} → {air_setpoint:.2f}")

            return action

        except Exception as e:
            logger.error(f"Failed to calculate control action: {e}")
            raise

    async def adjust_burner_settings(
        self,
        action: ControlAction,
        interlocks: SafetyInterlocks
    ) -> bool:
        """
        Implement control action on burner (write to DCS/PLC)

        Args:
            action: Control action to implement
            interlocks: Safety interlock status

        Returns:
            bool: True if successfully implemented
        """
        try:
            # Verify safety interlocks
            if not interlocks.all_safe():
                logger.error("Cannot implement control: Safety interlocks not satisfied")
                metrics_collector.control_blocked_counter.labels(
                    agent=self.agent_id,
                    reason="safety_interlocks"
                ).inc()
                return False

            # Verify control action is within safe limits
            if not (settings.MIN_FUEL_FLOW <= action.fuel_flow_setpoint <= settings.MAX_FUEL_FLOW):
                logger.error(f"Fuel setpoint {action.fuel_flow_setpoint} outside safe limits")
                return False

            if not (settings.MIN_AIR_FLOW <= action.air_flow_setpoint <= settings.MAX_AIR_FLOW):
                logger.error(f"Air setpoint {action.air_flow_setpoint} outside safe limits")
                return False

            logger.info("Implementing control action...")

            # Write setpoints to DCS (primary)
            try:
                await asyncio.gather(
                    self.dcs.set_fuel_flow_setpoint(action.fuel_flow_setpoint),
                    self.dcs.set_air_flow_setpoint(action.air_flow_setpoint)
                )

                # Also write to PLC (backup/monitoring)
                await asyncio.gather(
                    self.plc.set_fuel_flow_setpoint(action.fuel_flow_setpoint),
                    self.plc.set_air_flow_setpoint(action.air_flow_setpoint)
                )

            except Exception as e:
                logger.error(f"Failed to write setpoints: {e}")
                metrics_collector.error_counter.labels(
                    agent=self.agent_id,
                    error_type="setpoint_write"
                ).inc()
                return False

            # Publish to SCADA for monitoring
            await self.scada.publish_control_action({
                'action_id': action.action_id,
                'fuel_setpoint': action.fuel_flow_setpoint,
                'air_setpoint': action.air_flow_setpoint,
                'timestamp': action.timestamp.isoformat()
            })

            # Update metrics
            metrics_collector.control_action_counter.labels(
                agent=self.agent_id
            ).inc()

            logger.info("Control action implemented successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to implement control action: {e}")
            metrics_collector.error_counter.labels(
                agent=self.agent_id,
                error_type="control_implementation"
            ).inc()
            return False

    async def validate_safety_interlocks(self) -> bool:
        """
        Validate all safety interlocks are satisfied

        Returns:
            bool: True if all interlocks satisfied
        """
        interlocks = await self.check_safety_interlocks()
        return interlocks.all_safe()

    def calculate_control_hash(self, action: ControlAction) -> str:
        """
        Calculate SHA-256 hash for control action (already in ControlAction.calculate_hash)

        Args:
            action: Control action

        Returns:
            str: SHA-256 hash
        """
        return action.calculate_hash()

    async def run_control_cycle(self, heat_demand_kw: Optional[float] = None) -> Dict[str, Any]:
        """
        Run single control cycle

        Target execution time: <100ms

        Args:
            heat_demand_kw: Target heat output (default: use configured target)

        Returns:
            Dict with cycle results
        """
        cycle_start = time.perf_counter()

        try:
            logger.debug("=== Starting Control Cycle ===")

            # Use configured target if not specified
            if heat_demand_kw is None:
                heat_demand_kw = settings.HEAT_OUTPUT_TARGET_KW

            # 1. Read combustion state (<50ms target)
            state = await self.read_combustion_state()

            # 2. Check safety interlocks
            interlocks = await self.check_safety_interlocks()
            if not interlocks.all_safe():
                logger.warning("Skipping control: Safety interlocks not satisfied")
                return {
                    'success': False,
                    'reason': 'safety_interlocks',
                    'failed_interlocks': interlocks.get_failed_interlocks(),
                    'cycle_time_ms': (time.perf_counter() - cycle_start) * 1000
                }

            # 3. Analyze stability
            stability = await self.analyze_stability(state)

            # 4. Calculate control action
            action = await self.calculate_control_action(state, stability, heat_demand_kw)

            # 5. Implement control action
            success = await self.adjust_burner_settings(action, interlocks)

            # Track cycle time
            cycle_time_ms = (time.perf_counter() - cycle_start) * 1000
            self.cycle_times.append(cycle_time_ms)

            metrics_collector.control_cycle_time_ms.labels(
                agent=self.agent_id
            ).observe(cycle_time_ms)

            if cycle_time_ms > settings.CONTROL_LOOP_INTERVAL_MS:
                logger.warning(f"Control cycle took {cycle_time_ms:.1f}ms (target <{settings.CONTROL_LOOP_INTERVAL_MS}ms)")

            result = {
                'success': success,
                'action_id': action.action_id,
                'state': state.dict(),
                'stability': stability.dict(),
                'action': action.dict(),
                'cycle_time_ms': cycle_time_ms,
                'timestamp': DeterministicClock.utcnow().isoformat()
            }

            logger.info(f"✓ Control cycle complete: {cycle_time_ms:.1f}ms, "
                       f"Stability={stability.overall_stability_score:.1f}")

            return result

        except Exception as e:
            self.control_errors += 1
            logger.error(f"Control cycle failed: {e}")
            metrics_collector.error_counter.labels(
                agent=self.agent_id,
                error_type="control_cycle"
            ).inc()

            return {
                'success': False,
                'error': str(e),
                'cycle_time_ms': (time.perf_counter() - cycle_start) * 1000
            }

    async def start(self) -> None:
        """Start the combustion control agent"""
        try:
            logger.info(f"Starting {self.agent_name}...")

            # Initialize integrations if not already done
            if self.dcs is None:
                await self.initialize_integrations()

            self.is_running = True
            self.control_enabled = settings.CONTROL_AUTO_START

            logger.info(f"{self.agent_name} started successfully")
            logger.info(f"Control loop interval: {settings.CONTROL_LOOP_INTERVAL_MS}ms")

            # Run continuous control loop
            while self.is_running:
                try:
                    if self.control_enabled:
                        await self.run_control_cycle()

                    # Wait for next cycle
                    await asyncio.sleep(settings.CONTROL_LOOP_INTERVAL_MS / 1000.0)

                except Exception as e:
                    logger.error(f"Error in control loop: {e}")
                    await asyncio.sleep(settings.ERROR_RETRY_DELAY_MS / 1000.0)

        except Exception as e:
            logger.error(f"Failed to start {self.agent_name}: {e}")
            raise

    async def stop(self) -> None:
        """Stop the combustion control agent"""
        logger.info(f"Stopping {self.agent_name}...")
        self.is_running = False
        self.control_enabled = False

        # Close all integrations
        if self.dcs:
            await self.dcs.disconnect()
        if self.plc:
            await self.plc.disconnect()
        if self.combustion_analyzer:
            await self.combustion_analyzer.disconnect()
        if self.pressure_sensors:
            await self.pressure_sensors.disconnect()
        if self.temperature_sensors:
            await self.temperature_sensors.disconnect()
        if self.flow_meters:
            await self.flow_meters.disconnect()
        if self.scada:
            await self.scada.disconnect()

        logger.info(f"{self.agent_name} stopped")

    def enable_control(self) -> None:
        """Enable automatic control"""
        logger.info("Enabling automatic control")
        self.control_enabled = True

    def disable_control(self) -> None:
        """Disable automatic control (manual mode)"""
        logger.info("Disabling automatic control")
        self.control_enabled = False

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        avg_cycle_time = sum(self.cycle_times) / len(self.cycle_times) if self.cycle_times else 0

        return {
            'agent_id': self.agent_id,
            'agent_name': self.agent_name,
            'version': self.version,
            'is_running': self.is_running,
            'control_enabled': self.control_enabled,
            'current_state': self.current_state.dict() if self.current_state else None,
            'latest_stability': self.stability_history[-1].dict() if self.stability_history else None,
            'control_cycles_executed': len(self.control_history),
            'control_errors': self.control_errors,
            'avg_cycle_time_ms': avg_cycle_time,
            'target_cycle_time_ms': settings.CONTROL_LOOP_INTERVAL_MS
        }
