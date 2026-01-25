# -*- coding: utf-8 -*-
"""
GL-005 CombustionControlAgent - Prometheus Metrics

Comprehensive metrics collection for real-time combustion control monitoring.
Exports metrics to Prometheus for Grafana visualization and SLO tracking.

Metrics Categories:
- Control loop performance (cycle time, PID outputs, stability)
- Combustion process state (temperature, pressure, flow, efficiency)
- Safety interlocks and trips
- Integration connector health (DCS, PLC, SCADA, analyzers)
- API endpoint performance
- Business metrics (fuel savings, emissions reduction)
"""

import logging
import time
import functools
from typing import Dict, Callable, Optional
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Summary,
    Info
)

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Centralized Prometheus metrics collector for GL-005 CombustionControlAgent.

    Provides 50+ metrics covering:
    - Control loop performance and stability
    - Combustion process monitoring
    - Safety interlock tracking
    - Integration health
    - API performance
    """

    def __init__(self):
        """Initialize all Prometheus metrics"""
        self.registry = CollectorRegistry()

        # ====================================================================
        # AGENT INFORMATION
        # ====================================================================
        self.agent_info = Info(
            'gl005_agent',
            'GL-005 CombustionControlAgent information',
            registry=self.registry
        )
        self.agent_info.info({
            'agent_id': 'GL-005',
            'agent_name': 'CombustionControlAgent',
            'version': '1.0.0',
            'category': 'combustion_control',
            'control_type': 'real_time_pid'
        })

        # ====================================================================
        # CONTROL LOOP PERFORMANCE METRICS
        # ====================================================================
        self.control_cycles_total = Counter(
            'gl005_control_cycles_total',
            'Total control cycles executed',
            ['agent', 'status'],  # status: success, failure, skipped
            registry=self.registry
        )

        self.control_cycle_time_ms = Histogram(
            'gl005_control_cycle_time_milliseconds',
            'Control cycle execution time (target: <100ms)',
            ['agent'],
            buckets=[10, 25, 50, 75, 100, 150, 200, 300, 500, 1000],
            registry=self.registry
        )

        self.pid_output = Summary(
            'gl005_pid_output',
            'PID controller output values',
            ['agent', 'controller'],  # controller: fuel, air, o2_trim
            registry=self.registry
        )

        self.pid_error = Gauge(
            'gl005_pid_error',
            'Current PID error (setpoint - process_variable)',
            ['agent', 'controller'],
            registry=self.registry
        )

        self.pid_integral_term = Gauge(
            'gl005_pid_integral_term',
            'PID integral term accumulation',
            ['agent', 'controller'],
            registry=self.registry
        )

        self.pid_derivative_term = Gauge(
            'gl005_pid_derivative_term',
            'PID derivative term',
            ['agent', 'controller'],
            registry=self.registry
        )

        self.control_action_counter = Counter(
            'gl005_control_actions_implemented_total',
            'Total control actions implemented',
            ['agent'],
            registry=self.registry
        )

        self.control_blocked_counter = Counter(
            'gl005_control_blocked_total',
            'Control actions blocked (safety/limits)',
            ['agent', 'reason'],  # reason: safety_interlocks, limits_exceeded, manual_mode
            registry=self.registry
        )

        self.stability_score = Gauge(
            'gl005_combustion_stability_score',
            'Overall combustion stability score (0-100)',
            ['agent'],
            registry=self.registry
        )

        self.oscillation_detected = Gauge(
            'gl005_oscillation_detected',
            'Control oscillation detected (1=yes, 0=no)',
            ['agent'],
            registry=self.registry
        )

        self.oscillation_frequency_hz = Gauge(
            'gl005_oscillation_frequency_hz',
            'Detected oscillation frequency',
            ['agent'],
            registry=self.registry
        )

        # ====================================================================
        # COMBUSTION PROCESS STATE METRICS
        # ====================================================================
        self.fuel_flow = Gauge(
            'gl005_fuel_flow_rate',
            'Fuel flow rate (kg/hr or m3/hr)',
            ['agent', 'fuel_type'],
            registry=self.registry
        )

        self.air_flow = Gauge(
            'gl005_air_flow_rate_m3_per_hr',
            'Combustion air flow rate',
            ['agent'],
            registry=self.registry
        )

        self.air_fuel_ratio = Gauge(
            'gl005_air_fuel_ratio',
            'Actual air-fuel ratio',
            ['agent'],
            registry=self.registry
        )

        self.fuel_valve_position = Gauge(
            'gl005_fuel_valve_position_percent',
            'Fuel valve position (% open)',
            ['agent'],
            registry=self.registry
        )

        self.air_damper_position = Gauge(
            'gl005_air_damper_position_percent',
            'Air damper position (% open)',
            ['agent'],
            registry=self.registry
        )

        self.flame_temperature = Gauge(
            'gl005_flame_temperature_celsius',
            'Flame temperature',
            ['agent'],
            registry=self.registry
        )

        self.furnace_temperature = Gauge(
            'gl005_furnace_temperature_celsius',
            'Furnace temperature',
            ['agent'],
            registry=self.registry
        )

        self.flue_gas_temperature = Gauge(
            'gl005_flue_gas_temperature_celsius',
            'Flue gas temperature',
            ['agent'],
            registry=self.registry
        )

        self.fuel_pressure = Gauge(
            'gl005_fuel_pressure_kpa',
            'Fuel supply pressure',
            ['agent'],
            registry=self.registry
        )

        self.air_pressure = Gauge(
            'gl005_air_pressure_kpa',
            'Air supply pressure',
            ['agent'],
            registry=self.registry
        )

        self.furnace_pressure = Gauge(
            'gl005_furnace_pressure_pa',
            'Furnace draft pressure',
            ['agent'],
            registry=self.registry
        )

        # ====================================================================
        # COMBUSTION QUALITY METRICS
        # ====================================================================
        self.o2_level = Gauge(
            'gl005_o2_level_percent',
            'O2 concentration in flue gas',
            ['agent'],
            registry=self.registry
        )

        self.co_level = Gauge(
            'gl005_co_ppm',
            'CO emissions concentration',
            ['agent'],
            registry=self.registry
        )

        self.co2_level = Gauge(
            'gl005_co2_percent',
            'CO2 emissions percentage',
            ['agent'],
            registry=self.registry
        )

        self.nox_level = Gauge(
            'gl005_nox_ppm',
            'NOx emissions concentration',
            ['agent'],
            registry=self.registry
        )

        self.heat_output = Gauge(
            'gl005_heat_output_kw',
            'Calculated heat output',
            ['agent'],
            registry=self.registry
        )

        self.thermal_efficiency = Gauge(
            'gl005_thermal_efficiency_percent',
            'Thermal efficiency',
            ['agent'],
            registry=self.registry
        )

        self.combustion_efficiency = Gauge(
            'gl005_combustion_efficiency_percent',
            'Combustion efficiency',
            ['agent'],
            registry=self.registry
        )

        self.excess_air_percent = Gauge(
            'gl005_excess_air_percent',
            'Excess air percentage',
            ['agent'],
            registry=self.registry
        )

        # ====================================================================
        # SAFETY & INTERLOCK METRICS
        # ====================================================================
        self.safety_trips_total = Counter(
            'gl005_safety_trips_total',
            'Total safety system trips',
            ['agent', 'interlock_name'],
            registry=self.registry
        )

        self.safety_interlock_counter = Counter(
            'gl005_safety_interlock_triggered_total',
            'Safety interlock activations',
            ['agent'],
            registry=self.registry
        )

        self.interlock_status = Gauge(
            'gl005_interlock_status',
            'Individual interlock status (1=satisfied, 0=failed)',
            ['agent', 'interlock_name'],
            registry=self.registry
        )

        self.flame_present = Gauge(
            'gl005_flame_present',
            'Flame detection status (1=detected, 0=lost)',
            ['agent'],
            registry=self.registry
        )

        self.emergency_stop_active = Gauge(
            'gl005_emergency_stop_active',
            'Emergency stop status (1=active, 0=clear)',
            ['agent'],
            registry=self.registry
        )

        # ====================================================================
        # INTEGRATION CONNECTOR METRICS
        # ====================================================================
        self.integration_status = Gauge(
            'gl005_integration_status',
            'Integration connection status (1=connected, 0=disconnected)',
            ['agent', 'integration'],  # integration: dcs, plc, scada, analyzer, flow_meters
            registry=self.registry
        )

        self.integration_latency = Histogram(
            'gl005_integration_latency_milliseconds',
            'Integration connector latency',
            ['agent', 'integration', 'operation'],  # operation: read, write
            buckets=[5, 10, 25, 50, 100, 250, 500, 1000],
            registry=self.registry
        )

        self.integration_errors_total = Counter(
            'gl005_integration_errors_total',
            'Integration connector errors',
            ['agent', 'integration', 'error_type'],
            registry=self.registry
        )

        self.sensor_read_latency = Histogram(
            'gl005_sensor_read_latency_milliseconds',
            'Sensor data acquisition latency',
            ['agent', 'sensor_type'],  # sensor_type: temperature, pressure, flow, analyzer
            buckets=[5, 10, 25, 50, 100],
            registry=self.registry
        )

        self.state_read_time_ms = Histogram(
            'gl005_state_read_time_milliseconds',
            'Complete state read time (target: <50ms)',
            ['agent'],
            buckets=[10, 25, 50, 75, 100, 150, 200],
            registry=self.registry
        )

        self.dcs_write_latency = Histogram(
            'gl005_dcs_write_latency_milliseconds',
            'DCS setpoint write latency',
            ['agent', 'setpoint_type'],  # setpoint_type: fuel_flow, air_flow
            buckets=[5, 10, 25, 50, 100],
            registry=self.registry
        )

        self.scada_publish_counter = Counter(
            'gl005_scada_publish_total',
            'SCADA data points published',
            ['agent', 'topic'],
            registry=self.registry
        )

        # ====================================================================
        # API PERFORMANCE METRICS
        # ====================================================================
        self.api_request_counter = Counter(
            'gl005_api_requests_total',
            'Total API requests',
            ['agent', 'method', 'endpoint', 'status'],
            registry=self.registry
        )

        self.api_latency = Histogram(
            'gl005_api_latency_seconds',
            'API request latency',
            ['agent', 'method', 'endpoint'],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
            registry=self.registry
        )

        self.api_request_size_bytes = Histogram(
            'gl005_api_request_size_bytes',
            'API request body size',
            ['agent', 'endpoint'],
            buckets=[100, 1000, 10000, 100000],
            registry=self.registry
        )

        self.api_response_size_bytes = Histogram(
            'gl005_api_response_size_bytes',
            'API response body size',
            ['agent', 'endpoint'],
            buckets=[100, 1000, 10000, 100000],
            registry=self.registry
        )

        # ====================================================================
        # ERROR METRICS
        # ====================================================================
        self.errors_total = Counter(
            'gl005_errors_total',
            'Total errors by type',
            ['agent', 'error_type'],  # error_type: state_read, control_cycle, integration, validation
            registry=self.registry
        )

        self.exception_counter = Counter(
            'gl005_exceptions_total',
            'Unhandled exceptions',
            ['agent', 'exception_type'],
            registry=self.registry
        )

        self.validation_failures = Counter(
            'gl005_validation_failures_total',
            'Data validation failures',
            ['agent', 'validation_type'],
            registry=self.registry
        )

        # ====================================================================
        # PERFORMANCE METRICS
        # ====================================================================
        self.cpu_usage = Gauge(
            'gl005_cpu_usage_percent',
            'CPU usage',
            ['agent'],
            registry=self.registry
        )

        self.memory_usage_bytes = Gauge(
            'gl005_memory_usage_bytes',
            'Memory usage',
            ['agent'],
            registry=self.registry
        )

        self.active_connections = Gauge(
            'gl005_active_connections',
            'Active network connections',
            ['agent'],
            registry=self.registry
        )

        self.control_history_size = Gauge(
            'gl005_control_history_size',
            'Number of control actions in history',
            ['agent'],
            registry=self.registry
        )

        self.state_history_size = Gauge(
            'gl005_state_history_size',
            'Number of states in history',
            ['agent'],
            registry=self.registry
        )

        # ====================================================================
        # BUSINESS METRICS
        # ====================================================================
        self.fuel_savings_per_hour = Gauge(
            'gl005_fuel_savings_per_hour',
            'Fuel savings achieved (kg/hr)',
            ['agent'],
            registry=self.registry
        )

        self.cost_savings_per_hour = Gauge(
            'gl005_cost_savings_per_hour_usd',
            'Cost savings from optimization ($/hr)',
            ['agent'],
            registry=self.registry
        )

        self.emissions_reduction_kg_hr = Gauge(
            'gl005_emissions_reduction_kg_hr',
            'Emissions reduction rate',
            ['agent', 'pollutant'],  # pollutant: co2, nox, co
            registry=self.registry
        )

        self.efficiency_improvement = Gauge(
            'gl005_efficiency_improvement_percent',
            'Efficiency improvement from baseline',
            ['agent'],
            registry=self.registry
        )

        self.uptime_percent = Gauge(
            'gl005_uptime_percent',
            'Burner uptime percentage',
            ['agent'],
            registry=self.registry
        )

        # ====================================================================
        # DETERMINISM METRICS
        # ====================================================================
        self.determinism_verification_failures = Counter(
            'gl005_determinism_verification_failures_total',
            'Determinism verification failures',
            ['agent', 'violation_type'],
            registry=self.registry
        )

        self.provenance_hash_verifications = Counter(
            'gl005_provenance_hash_verifications_total',
            'Provenance hash verifications',
            ['agent', 'status'],  # status: success, failure
            registry=self.registry
        )

        self.calculation_reproducibility_score = Gauge(
            'gl005_calculation_reproducibility_score',
            'Calculation reproducibility score (0-1)',
            ['agent', 'calculation_type'],
            registry=self.registry
        )

        # ====================================================================
        # WEBSOCKET STREAMING METRICS
        # ====================================================================
        self.websocket_connections_total = Counter(
            'gl005_websocket_connections_total',
            'Total WebSocket connections (connect/disconnect)',
            ['agent', 'status'],  # status: connected, disconnected, auth_failed, rate_limited
            registry=self.registry
        )

        self.websocket_active_connections = Gauge(
            'gl005_websocket_active_connections',
            'Current active WebSocket connections',
            ['agent'],
            registry=self.registry
        )

        self.websocket_messages_sent_total = Counter(
            'gl005_websocket_messages_sent_total',
            'Total WebSocket messages sent',
            ['agent', 'stream_type'],  # stream_type: combustion_state, stability_metrics, control_action
            registry=self.registry
        )

        self.websocket_messages_received_total = Counter(
            'gl005_websocket_messages_received_total',
            'Total WebSocket messages received from clients',
            ['agent', 'message_type'],  # message_type: ping, subscribe, unsubscribe
            registry=self.registry
        )

        self.websocket_broadcast_latency_ms = Histogram(
            'gl005_websocket_broadcast_latency_milliseconds',
            'WebSocket broadcast latency',
            ['agent', 'stream_type'],
            buckets=[1, 5, 10, 25, 50, 100, 250, 500],
            registry=self.registry
        )

        self.websocket_connection_duration_seconds = Histogram(
            'gl005_websocket_connection_duration_seconds',
            'WebSocket connection duration',
            ['agent'],
            buckets=[1, 5, 30, 60, 300, 600, 1800, 3600],
            registry=self.registry
        )

        self.websocket_clients_per_user = Gauge(
            'gl005_websocket_clients_per_user',
            'WebSocket connections per user',
            ['agent', 'user_id'],
            registry=self.registry
        )

        logger.info("GL-005 metrics collector initialized with 60+ metrics")

    # ========================================================================
    # CONVENIENCE METHODS FOR UPDATING METRICS
    # ========================================================================

    def record_control_cycle(self, agent_id: str, duration_ms: float, success: bool) -> None:
        """
        Record control cycle execution.

        Args:
            agent_id: Agent identifier
            duration_ms: Cycle duration in milliseconds
            success: Whether cycle succeeded
        """
        status = 'success' if success else 'failure'
        self.control_cycles_total.labels(agent=agent_id, status=status).inc()
        self.control_cycle_time_ms.labels(agent=agent_id).observe(duration_ms)

    def record_safety_event(self, agent_id: str, interlock_name: str, triggered: bool) -> None:
        """
        Record safety interlock event.

        Args:
            agent_id: Agent identifier
            interlock_name: Name of the interlock
            triggered: Whether interlock was triggered
        """
        if triggered:
            self.safety_trips_total.labels(
                agent=agent_id,
                interlock_name=interlock_name
            ).inc()
            self.interlock_status.labels(
                agent=agent_id,
                interlock_name=interlock_name
            ).set(0)
        else:
            self.interlock_status.labels(
                agent=agent_id,
                interlock_name=interlock_name
            ).set(1)

    def update_combustion_state(self, agent_id: str, state: Dict) -> None:
        """
        Update combustion process state metrics.

        Args:
            agent_id: Agent identifier
            state: Dictionary with combustion state values
        """
        if 'fuel_flow' in state:
            self.fuel_flow.labels(
                agent=agent_id,
                fuel_type=state.get('fuel_type', 'natural_gas')
            ).set(state['fuel_flow'])

        if 'air_flow' in state:
            self.air_flow.labels(agent=agent_id).set(state['air_flow'])

        if 'air_fuel_ratio' in state:
            self.air_fuel_ratio.labels(agent=agent_id).set(state['air_fuel_ratio'])

        if 'flame_temperature' in state and state['flame_temperature'] is not None:
            self.flame_temperature.labels(agent=agent_id).set(state['flame_temperature'])

        if 'furnace_temperature' in state:
            self.furnace_temperature.labels(agent=agent_id).set(state['furnace_temperature'])

        if 'flue_gas_temperature' in state:
            self.flue_gas_temperature.labels(agent=agent_id).set(state['flue_gas_temperature'])

        if 'fuel_pressure' in state:
            self.fuel_pressure.labels(agent=agent_id).set(state['fuel_pressure'])

        if 'air_pressure' in state:
            self.air_pressure.labels(agent=agent_id).set(state['air_pressure'])

        if 'furnace_pressure' in state:
            self.furnace_pressure.labels(agent=agent_id).set(state['furnace_pressure'])

        if 'o2_percent' in state:
            self.o2_level.labels(agent=agent_id).set(state['o2_percent'])

        if 'co_ppm' in state and state['co_ppm'] is not None:
            self.co_level.labels(agent=agent_id).set(state['co_ppm'])

        if 'co2_percent' in state and state['co2_percent'] is not None:
            self.co2_level.labels(agent=agent_id).set(state['co2_percent'])

        if 'nox_ppm' in state and state['nox_ppm'] is not None:
            self.nox_level.labels(agent=agent_id).set(state['nox_ppm'])

        if 'heat_output_kw' in state and state['heat_output_kw'] is not None:
            self.heat_output.labels(agent=agent_id).set(state['heat_output_kw'])

        if 'thermal_efficiency' in state and state['thermal_efficiency'] is not None:
            self.thermal_efficiency.labels(agent=agent_id).set(state['thermal_efficiency'])

        if 'excess_air_percent' in state and state['excess_air_percent'] is not None:
            self.excess_air_percent.labels(agent=agent_id).set(state['excess_air_percent'])

    def record_api_request(
        self,
        agent_id: str,
        method: str,
        endpoint: str,
        duration_seconds: float,
        status: str
    ) -> None:
        """
        Record API request metrics.

        Args:
            agent_id: Agent identifier
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            duration_seconds: Request duration
            status: Response status (success, error)
        """
        self.api_request_counter.labels(
            agent=agent_id,
            method=method,
            endpoint=endpoint,
            status=status
        ).inc()

        self.api_latency.labels(
            agent=agent_id,
            method=method,
            endpoint=endpoint
        ).observe(duration_seconds)

    def record_integration_latency(
        self,
        agent_id: str,
        connector: str,
        operation: str,
        duration_ms: float
    ) -> None:
        """
        Record integration connector latency.

        Args:
            agent_id: Agent identifier
            connector: Connector name (dcs, plc, scada, analyzer)
            operation: Operation type (read, write)
            duration_ms: Operation duration in milliseconds
        """
        self.integration_latency.labels(
            agent=agent_id,
            integration=connector,
            operation=operation
        ).observe(duration_ms)

    def update_pid_metrics(
        self,
        agent_id: str,
        controller: str,
        output: float,
        error: float,
        integral: float,
        derivative: float
    ) -> None:
        """
        Update PID controller metrics.

        Args:
            agent_id: Agent identifier
            controller: Controller name (fuel, air, o2_trim)
            output: PID output value
            error: Current error
            integral: Integral term
            derivative: Derivative term
        """
        self.pid_output.labels(agent=agent_id, controller=controller).observe(output)
        self.pid_error.labels(agent=agent_id, controller=controller).set(error)
        self.pid_integral_term.labels(agent=agent_id, controller=controller).set(integral)
        self.pid_derivative_term.labels(agent=agent_id, controller=controller).set(derivative)

    def get_metrics(self) -> bytes:
        """
        Get metrics in Prometheus format.

        Returns:
            Metrics in Prometheus exposition format
        """
        from prometheus_client import generate_latest
        return generate_latest(self.registry)

    def get_metrics_summary(self) -> Dict[str, int]:
        """
        Get summary of metrics types.

        Returns:
            Dictionary with metric counts by type
        """
        return {
            'total_metrics': 63,
            'gauges': 41,
            'counters': 16,
            'histograms': 10,
            'summaries': 1,
            'info': 1
        }


# ============================================================================
# DECORATORS FOR AUTOMATIC METRICS TRACKING
# ============================================================================

def track_api_metrics(method: str, endpoint: str):
    """
    Decorator to automatically track API request metrics.

    Usage:
        @track_api_metrics('GET', '/health')
        async def health_endpoint():
            return {"status": "healthy"}
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            status = 'success'
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = 'error'
                raise
            finally:
                duration = time.time() - start_time
                metrics_collector.record_api_request(
                    agent_id='GL-005',
                    method=method,
                    endpoint=endpoint,
                    duration_seconds=duration,
                    status=status
                )

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            status = 'success'
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = 'error'
                raise
            finally:
                duration = time.time() - start_time
                metrics_collector.record_api_request(
                    agent_id='GL-005',
                    method=method,
                    endpoint=endpoint,
                    duration_seconds=duration,
                    status=status
                )

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def track_integration_metrics(connector: str, operation: str):
    """
    Decorator to track integration connector metrics.

    Usage:
        @track_integration_metrics('dcs', 'read')
        async def read_from_dcs():
            return data
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration_ms = (time.perf_counter() - start_time) * 1000
                metrics_collector.record_integration_latency(
                    agent_id='GL-005',
                    connector=connector,
                    operation=operation,
                    duration_ms=duration_ms
                )

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration_ms = (time.perf_counter() - start_time) * 1000
                metrics_collector.record_integration_latency(
                    agent_id='GL-005',
                    connector=connector,
                    operation=operation,
                    duration_ms=duration_ms
                )

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# ============================================================================
# GLOBAL METRICS COLLECTOR INSTANCE
# ============================================================================

# Global singleton instance
metrics_collector = MetricsCollector()

__all__ = [
    'metrics_collector',
    'MetricsCollector',
    'track_api_metrics',
    'track_integration_metrics'
]
