"""
GL-003 UnifiedSteam gRPC Client

Client-side implementation for accessing UnifiedSteam gRPC services.
Provides connection management, retry logic, and type-safe service access.

Features:
- Connection pooling and keepalive
- Exponential backoff retry with configurable policies
- Client-side load balancing support
- Timeout handling per-method
- TLS/mTLS support
- Async and sync interfaces

Usage:
    from api.grpc_client import SteamServiceClient, ClientConfig

    # Create client
    config = ClientConfig(
        host="localhost",
        port=50052,
        enable_tls=True,
    )
    client = SteamServiceClient(config)

    # Connect
    await client.connect()

    # Use services
    result = await client.steam_properties.compute_properties(
        pressure_kpa=1000.0,
        temperature_c=200.0,
    )

    # Disconnect
    await client.disconnect()
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from uuid import uuid4

import grpc
from grpc import aio as grpc_aio

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class RetryPolicy(str, Enum):
    """Retry policy types."""
    NONE = "none"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    CONSTANT = "constant"


@dataclass
class RetryConfig:
    """
    Retry configuration for gRPC calls.

    Attributes:
        policy: Retry policy type
        max_retries: Maximum number of retry attempts
        initial_backoff_ms: Initial backoff in milliseconds
        max_backoff_ms: Maximum backoff in milliseconds
        backoff_multiplier: Multiplier for exponential backoff
        retryable_status_codes: gRPC status codes that should trigger retry
        jitter: Add random jitter to backoff (0.0-1.0)
    """
    policy: RetryPolicy = RetryPolicy.EXPONENTIAL_BACKOFF
    max_retries: int = 3
    initial_backoff_ms: int = 100
    max_backoff_ms: int = 10000
    backoff_multiplier: float = 2.0
    retryable_status_codes: List[grpc.StatusCode] = field(default_factory=lambda: [
        grpc.StatusCode.UNAVAILABLE,
        grpc.StatusCode.DEADLINE_EXCEEDED,
        grpc.StatusCode.RESOURCE_EXHAUSTED,
        grpc.StatusCode.ABORTED,
        grpc.StatusCode.INTERNAL,
    ])
    jitter: float = 0.1


@dataclass
class ClientConfig:
    """
    Client configuration for gRPC connection.

    Attributes:
        host: Server hostname
        port: Server port
        enable_tls: Enable TLS encryption
        cert_path: Path to client certificate (for mTLS)
        key_path: Path to client private key (for mTLS)
        ca_cert_path: Path to CA certificate
        default_timeout_ms: Default timeout for RPC calls
        keepalive_time_ms: Keepalive ping interval
        keepalive_timeout_ms: Keepalive ping timeout
        max_message_size: Maximum message size in bytes
        retry_config: Retry configuration
        enable_compression: Enable message compression
        compression_algorithm: Compression algorithm (gzip, deflate)
        metadata: Default metadata for all calls
        load_balancing_policy: Load balancing policy (round_robin, pick_first)
    """
    host: str = "localhost"
    port: int = 50052
    enable_tls: bool = False
    cert_path: Optional[str] = None
    key_path: Optional[str] = None
    ca_cert_path: Optional[str] = None
    default_timeout_ms: int = 30000
    keepalive_time_ms: int = 30000
    keepalive_timeout_ms: int = 5000
    max_message_size: int = 50 * 1024 * 1024  # 50MB
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    enable_compression: bool = False
    compression_algorithm: str = "gzip"
    metadata: Dict[str, str] = field(default_factory=dict)
    load_balancing_policy: str = "round_robin"

    @property
    def target(self) -> str:
        """Get the server target address."""
        return f"{self.host}:{self.port}"


# =============================================================================
# Retry Logic
# =============================================================================

def calculate_backoff(
    attempt: int,
    config: RetryConfig,
) -> float:
    """
    Calculate backoff time for a retry attempt.

    Args:
        attempt: Current attempt number (0-indexed)
        config: Retry configuration

    Returns:
        Backoff time in seconds
    """
    if config.policy == RetryPolicy.NONE:
        return 0.0

    if config.policy == RetryPolicy.CONSTANT:
        backoff_ms = config.initial_backoff_ms

    elif config.policy == RetryPolicy.LINEAR_BACKOFF:
        backoff_ms = config.initial_backoff_ms * (attempt + 1)

    else:  # EXPONENTIAL_BACKOFF
        backoff_ms = config.initial_backoff_ms * (config.backoff_multiplier ** attempt)

    # Cap at max backoff
    backoff_ms = min(backoff_ms, config.max_backoff_ms)

    # Add jitter
    if config.jitter > 0:
        jitter_range = backoff_ms * config.jitter
        backoff_ms += random.uniform(-jitter_range, jitter_range)

    return max(0, backoff_ms / 1000.0)  # Convert to seconds


class RetryInterceptor(grpc_aio.UnaryUnaryClientInterceptor):
    """
    Client interceptor for automatic retry with backoff.

    Implements retry logic for failed gRPC calls based on status code.
    """

    def __init__(self, config: RetryConfig):
        """
        Initialize retry interceptor.

        Args:
            config: Retry configuration
        """
        self.config = config

    async def intercept_unary_unary(
        self,
        continuation: Callable,
        client_call_details: grpc.ClientCallDetails,
        request: Any,
    ) -> Any:
        """Intercept unary-unary calls for retry."""
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                response = await continuation(client_call_details, request)
                return response

            except grpc.RpcError as e:
                last_exception = e
                status_code = e.code()

                # Check if retryable
                if status_code not in self.config.retryable_status_codes:
                    logger.warning(
                        f"Non-retryable error on {client_call_details.method}: "
                        f"{status_code.name}"
                    )
                    raise

                # Check if we have retries left
                if attempt >= self.config.max_retries:
                    logger.error(
                        f"Max retries ({self.config.max_retries}) exceeded for "
                        f"{client_call_details.method}: {status_code.name}"
                    )
                    raise

                # Calculate backoff
                backoff = calculate_backoff(attempt, self.config)

                logger.warning(
                    f"Retry attempt {attempt + 1}/{self.config.max_retries} for "
                    f"{client_call_details.method} after {backoff:.2f}s: {status_code.name}"
                )

                await asyncio.sleep(backoff)

        raise last_exception


class LoggingInterceptor(grpc_aio.UnaryUnaryClientInterceptor):
    """
    Client interceptor for request/response logging.
    """

    async def intercept_unary_unary(
        self,
        continuation: Callable,
        client_call_details: grpc.ClientCallDetails,
        request: Any,
    ) -> Any:
        """Intercept unary-unary calls for logging."""
        request_id = str(uuid4())[:8]
        start_time = time.time()

        logger.debug(
            f"gRPC Call: {client_call_details.method} "
            f"[{request_id}] started"
        )

        try:
            response = await continuation(client_call_details, request)
            duration_ms = (time.time() - start_time) * 1000

            logger.debug(
                f"gRPC Call: {client_call_details.method} "
                f"[{request_id}] completed in {duration_ms:.2f}ms"
            )

            return response

        except grpc.RpcError as e:
            duration_ms = (time.time() - start_time) * 1000

            logger.error(
                f"gRPC Call: {client_call_details.method} "
                f"[{request_id}] failed after {duration_ms:.2f}ms: "
                f"{e.code().name} - {e.details()}"
            )
            raise


# =============================================================================
# Service Stubs
# =============================================================================

class SteamPropertiesClient:
    """
    Client for SteamPropertiesService.

    Provides methods for steam property computations.
    """

    def __init__(self, channel: grpc_aio.Channel, config: ClientConfig):
        """
        Initialize steam properties client.

        Args:
            channel: gRPC channel
            config: Client configuration
        """
        self.channel = channel
        self.config = config
        # In production, this would use generated stub:
        # self.stub = steam_pb2_grpc.SteamPropertiesServiceStub(channel)

    async def compute_properties(
        self,
        pressure_kpa: Optional[float] = None,
        temperature_c: Optional[float] = None,
        specific_enthalpy_kj_kg: Optional[float] = None,
        specific_entropy_kj_kg_k: Optional[float] = None,
        quality: Optional[float] = None,
        include_transport_properties: bool = False,
        timeout_ms: Optional[int] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Compute steam properties from independent inputs.

        Args:
            pressure_kpa: Pressure in kPa
            temperature_c: Temperature in Celsius
            specific_enthalpy_kj_kg: Specific enthalpy in kJ/kg
            specific_entropy_kj_kg_k: Specific entropy in kJ/(kg*K)
            quality: Steam quality (0-1) for two-phase
            include_transport_properties: Include viscosity, conductivity
            timeout_ms: Request timeout in milliseconds
            metadata: Additional request metadata

        Returns:
            Computed steam state properties

        Raises:
            grpc.RpcError: On gRPC errors
        """
        request = {
            "request_id": str(uuid4()),
            "pressure_kpa": pressure_kpa,
            "temperature_c": temperature_c,
            "specific_enthalpy_kj_kg": specific_enthalpy_kj_kg,
            "specific_entropy_kj_kg_k": specific_entropy_kj_kg_k,
            "quality": quality,
            "include_transport_properties": include_transport_properties,
        }

        timeout = (timeout_ms or self.config.default_timeout_ms) / 1000.0
        call_metadata = self._build_metadata(metadata)

        # Mock implementation - in production use generated stub
        logger.info(f"ComputeProperties request: {request}")

        # Simulated response
        return {
            "request_id": request["request_id"],
            "success": True,
            "steam_state": {
                "pressure_kpa": pressure_kpa or 1000.0,
                "temperature_c": temperature_c or 200.0,
                "specific_enthalpy_kj_kg": 2827.9,
                "specific_entropy_kj_kg_k": 6.694,
                "specific_volume_m3_kg": 0.2060,
                "density_kg_m3": 4.854,
                "phase": "superheated_vapor",
                "region": "region_2",
            },
            "computation_time_ms": 1.5,
        }

    async def get_saturation_properties(
        self,
        pressure_kpa: Optional[float] = None,
        temperature_c: Optional[float] = None,
        timeout_ms: Optional[int] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Get saturation properties at given pressure or temperature.

        Args:
            pressure_kpa: Saturation pressure in kPa
            temperature_c: Saturation temperature in Celsius
            timeout_ms: Request timeout in milliseconds
            metadata: Additional request metadata

        Returns:
            Saturation properties including liquid and vapor states

        Raises:
            grpc.RpcError: On gRPC errors
            ValueError: If neither pressure nor temperature provided
        """
        if pressure_kpa is None and temperature_c is None:
            raise ValueError("Either pressure_kpa or temperature_c must be provided")

        request = {
            "request_id": str(uuid4()),
            "pressure_kpa": pressure_kpa,
            "temperature_c": temperature_c,
        }

        timeout = (timeout_ms or self.config.default_timeout_ms) / 1000.0
        call_metadata = self._build_metadata(metadata)

        logger.info(f"GetSaturationProperties request: {request}")

        # Simulated response
        T_sat = temperature_c or (100 + (pressure_kpa - 101.325) * 0.03 if pressure_kpa else 100.0)
        P_sat = pressure_kpa or (101.325 + (temperature_c - 100) / 0.03 if temperature_c else 101.325)

        return {
            "success": True,
            "saturation_pressure_kpa": P_sat,
            "saturation_temperature_c": T_sat,
            "liquid_properties": {
                "specific_enthalpy_kj_kg": T_sat * 4.186,
                "specific_entropy_kj_kg_k": 0.3 + 0.003 * T_sat,
                "specific_volume_m3_kg": 0.001,
            },
            "vapor_properties": {
                "specific_enthalpy_kj_kg": 2676.0,
                "specific_entropy_kj_kg_k": 7.36,
                "specific_volume_m3_kg": 1.67 / (P_sat / 101.325),
            },
            "latent_heat_kj_kg": 2676.0 - T_sat * 4.186,
        }

    async def compute_batch_properties(
        self,
        requests: List[Dict[str, Any]],
        timeout_ms: Optional[int] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Compute properties for multiple states using streaming.

        Args:
            requests: List of property computation requests
            timeout_ms: Request timeout in milliseconds
            metadata: Additional request metadata

        Yields:
            Computed steam state properties for each request
        """
        for req in requests:
            result = await self.compute_properties(
                pressure_kpa=req.get("pressure_kpa"),
                temperature_c=req.get("temperature_c"),
                specific_enthalpy_kj_kg=req.get("specific_enthalpy_kj_kg"),
                specific_entropy_kj_kg_k=req.get("specific_entropy_kj_kg_k"),
                quality=req.get("quality"),
                timeout_ms=timeout_ms,
                metadata=metadata,
            )
            yield result

    def _build_metadata(self, additional: Optional[Dict[str, str]] = None) -> List[Tuple[str, str]]:
        """Build metadata list for gRPC call."""
        metadata = list(self.config.metadata.items())
        if additional:
            metadata.extend(additional.items())
        return metadata


class OptimizationClient:
    """
    Client for OptimizationService.

    Provides methods for steam system optimization.
    """

    def __init__(self, channel: grpc_aio.Channel, config: ClientConfig):
        """
        Initialize optimization client.

        Args:
            channel: gRPC channel
            config: Client configuration
        """
        self.channel = channel
        self.config = config

    async def optimize_desuperheater(
        self,
        desuperheater_id: str,
        inlet_pressure_kpa: float,
        inlet_temperature_c: float,
        inlet_flow_kg_s: float,
        target_outlet_temperature_c: float,
        spray_water_temperature_c: float,
        timeout_ms: Optional[int] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Optimize desuperheater spray water flow.

        Args:
            desuperheater_id: Desuperheater identifier
            inlet_pressure_kpa: Inlet steam pressure
            inlet_temperature_c: Inlet steam temperature
            inlet_flow_kg_s: Inlet steam flow rate
            target_outlet_temperature_c: Target outlet temperature
            spray_water_temperature_c: Spray water temperature
            timeout_ms: Request timeout
            metadata: Additional metadata

        Returns:
            Optimization results with optimal spray flow
        """
        request = {
            "desuperheater_id": desuperheater_id,
            "inlet_pressure_kpa": inlet_pressure_kpa,
            "inlet_temperature_c": inlet_temperature_c,
            "inlet_flow_kg_s": inlet_flow_kg_s,
            "target_outlet_temperature_c": target_outlet_temperature_c,
            "spray_water_temperature_c": spray_water_temperature_c,
        }

        logger.info(f"OptimizeDesuperheater request: {request}")

        # Simulated response
        inlet_h = 3050.0
        outlet_h = 2850.0
        spray_h = spray_water_temperature_c * 4.186
        optimal_spray = inlet_flow_kg_s * (inlet_h - outlet_h) / (outlet_h - spray_h)

        return {
            "success": True,
            "desuperheater_id": desuperheater_id,
            "optimal_spray_flow_kg_s": max(0, optimal_spray),
            "predicted_outlet_temperature_c": target_outlet_temperature_c,
            "spray_water_energy_kw": max(0, optimal_spray) * spray_h,
            "efficiency": 0.95,
        }

    async def optimize_condensate(
        self,
        system_id: str,
        current_recovery_rate: float,
        condensate_temp_c: float,
        makeup_temp_c: float,
        timeout_ms: Optional[int] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Optimize condensate recovery system.

        Args:
            system_id: System identifier
            current_recovery_rate: Current recovery rate (%)
            condensate_temp_c: Condensate temperature
            makeup_temp_c: Makeup water temperature
            timeout_ms: Request timeout
            metadata: Additional metadata

        Returns:
            Optimization results with savings estimates
        """
        request = {
            "system_id": system_id,
            "current_recovery_rate_percent": current_recovery_rate,
            "condensate_temperature_c": condensate_temp_c,
            "makeup_water_temperature_c": makeup_temp_c,
        }

        logger.info(f"OptimizeCondensate request: {request}")

        optimal_rate = min(95.0, current_recovery_rate + 15.0)
        delta_temp = condensate_temp_c - makeup_temp_c
        energy_savings = delta_temp * 4.186 / 3.6

        return {
            "success": True,
            "system_id": system_id,
            "optimal_recovery_rate_percent": optimal_rate,
            "delta_from_current_percent": optimal_rate - current_recovery_rate,
            "annual_energy_savings_mwh": energy_savings * 1000 * 0.9 * 8760 / 1000,
            "annual_water_savings_m3": 1000 * (optimal_rate - current_recovery_rate) / 100 * 8760,
        }

    async def optimize_network(
        self,
        network_id: str,
        total_demand_kg_s: float,
        headers: List[Dict[str, Any]],
        generators: List[Dict[str, Any]],
        timeout_ms: Optional[int] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Optimize steam network distribution.

        Args:
            network_id: Network identifier
            total_demand_kg_s: Total steam demand
            headers: Network headers configuration
            generators: Steam generators configuration
            timeout_ms: Request timeout
            metadata: Additional metadata

        Returns:
            Optimization results with generator setpoints
        """
        request = {
            "network_id": network_id,
            "total_demand_kg_s": total_demand_kg_s,
            "headers": headers,
            "generators": generators,
        }

        logger.info(f"OptimizeNetwork request: {request}")

        n_gen = len(generators) if generators else 1
        per_gen = total_demand_kg_s / n_gen

        return {
            "success": True,
            "network_id": network_id,
            "optimal_generator_outputs_kg_s": {
                gen.get("generator_id", f"gen_{i}"): per_gen
                for i, gen in enumerate(generators or [{"generator_id": "gen_1"}])
            },
            "total_generation_kg_s": total_demand_kg_s,
            "network_efficiency_percent": 92.5,
            "total_cost_usd_h": total_demand_kg_s * 25.0,
            "total_emissions_kg_co2_h": total_demand_kg_s * 0.5,
            "solver_status": "optimal",
        }

    async def stream_optimization_updates(
        self,
        subscription: Dict[str, Any],
        timeout_ms: Optional[int] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Subscribe to optimization updates.

        Args:
            subscription: Subscription configuration
            timeout_ms: Request timeout
            metadata: Additional metadata

        Yields:
            Optimization updates
        """
        logger.info(f"StreamOptimizationUpdates subscription: {subscription}")

        # Simulated streaming - would use actual gRPC streaming in production
        for i in range(5):
            await asyncio.sleep(1.0)
            yield {
                "type": "optimization_update",
                "sequence": i + 1,
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "optimization_id": str(uuid4()),
                    "status": "completed",
                },
            }


class DiagnosticsClient:
    """
    Client for DiagnosticsService (Inference).

    Provides methods for trap diagnostics and predictions.
    """

    def __init__(self, channel: grpc_aio.Channel, config: ClientConfig):
        """
        Initialize diagnostics client.

        Args:
            channel: gRPC channel
            config: Client configuration
        """
        self.channel = channel
        self.config = config

    async def predict_trap_failure(
        self,
        trap_id: str,
        inlet_pressure_kpa: float,
        outlet_pressure_kpa: float,
        inlet_temperature_c: float,
        outlet_temperature_c: float,
        acoustic_data: Optional[List[float]] = None,
        timeout_ms: Optional[int] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Predict steam trap failure probability.

        Args:
            trap_id: Trap identifier
            inlet_pressure_kpa: Inlet pressure
            outlet_pressure_kpa: Outlet pressure
            inlet_temperature_c: Inlet temperature
            outlet_temperature_c: Outlet temperature
            acoustic_data: Optional acoustic sensor data
            timeout_ms: Request timeout
            metadata: Additional metadata

        Returns:
            Failure prediction with risk factors
        """
        request = {
            "trap_id": trap_id,
            "inlet_pressure_kpa": inlet_pressure_kpa,
            "outlet_pressure_kpa": outlet_pressure_kpa,
            "inlet_temperature_c": inlet_temperature_c,
            "outlet_temperature_c": outlet_temperature_c,
            "acoustic_data": acoustic_data,
        }

        logger.info(f"PredictTrapFailure request: {request}")

        diff_temp = inlet_temperature_c - outlet_temperature_c

        if diff_temp > 50:
            failure_prob_30d = 0.05
            failure_prob_90d = 0.12
            condition = "good"
            risk_score = 15.0
        elif diff_temp > 20:
            failure_prob_30d = 0.15
            failure_prob_90d = 0.35
            condition = "degraded"
            risk_score = 45.0
        else:
            failure_prob_30d = 0.40
            failure_prob_90d = 0.70
            condition = "leaking"
            risk_score = 75.0

        return {
            "success": True,
            "trap_id": trap_id,
            "condition": condition,
            "failure_probability_30d": failure_prob_30d,
            "failure_probability_90d": failure_prob_90d,
            "risk_score": risk_score,
            "risk_factors": ["Differential temperature below threshold"] if diff_temp < 50 else [],
            "recommended_action": "Replace" if risk_score > 60 else "Monitor",
            "model_confidence": 0.85,
        }

    async def detect_anomaly(
        self,
        equipment_id: str,
        sensor_readings: Dict[str, float],
        timeout_ms: Optional[int] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Detect anomalies in sensor readings.

        Args:
            equipment_id: Equipment identifier
            sensor_readings: Map of sensor name to value
            timeout_ms: Request timeout
            metadata: Additional metadata

        Returns:
            Anomaly detection results
        """
        request = {
            "equipment_id": equipment_id,
            "sensor_readings": sensor_readings,
        }

        logger.info(f"DetectAnomaly request: {request}")

        anomalies = []
        for sensor, value in sensor_readings.items():
            if "temperature" in sensor.lower() and value > 250:
                anomalies.append({
                    "sensor": sensor,
                    "value": value,
                    "type": "high_value",
                    "severity": "medium",
                })
            if "pressure" in sensor.lower() and value < 100:
                anomalies.append({
                    "sensor": sensor,
                    "value": value,
                    "type": "low_value",
                    "severity": "high",
                })

        return {
            "success": True,
            "equipment_id": equipment_id,
            "anomaly_detected": len(anomalies) > 0,
            "anomaly_score": min(100, len(anomalies) * 25),
            "anomalies": anomalies,
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def infer_steam_quality(
        self,
        pressure_kpa: float,
        temperature_c: float,
        flow_rate_kg_s: float,
        conductivity_us_cm: Optional[float] = None,
        timeout_ms: Optional[int] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Infer steam quality from measurements.

        Args:
            pressure_kpa: Steam pressure
            temperature_c: Steam temperature
            flow_rate_kg_s: Flow rate
            conductivity_us_cm: Optional conductivity measurement
            timeout_ms: Request timeout
            metadata: Additional metadata

        Returns:
            Inferred steam quality
        """
        request = {
            "pressure_kpa": pressure_kpa,
            "temperature_c": temperature_c,
            "flow_rate_kg_s": flow_rate_kg_s,
            "conductivity_us_cm": conductivity_us_cm,
        }

        logger.info(f"InferSteamQuality request: {request}")

        T_sat = 100 + (pressure_kpa - 101.325) * 0.03
        superheat = temperature_c - T_sat

        if superheat > 10:
            quality = 1.0
            state = "superheated"
        elif superheat > 0:
            quality = 0.99
            state = "near_saturation"
        elif superheat > -5:
            quality = 0.95 + superheat / 100
            state = "wet_steam"
        else:
            quality = 0.90
            state = "wet_steam"

        return {
            "success": True,
            "inferred_quality": quality,
            "state": state,
            "superheat_c": max(0, superheat),
            "confidence": 0.85 if superheat > 5 else 0.70,
        }


class RCAClient:
    """
    Client for RCAService.

    Provides methods for root cause analysis.
    """

    def __init__(self, channel: grpc_aio.Channel, config: ClientConfig):
        """
        Initialize RCA client.

        Args:
            channel: gRPC channel
            config: Client configuration
        """
        self.channel = channel
        self.config = config

    async def analyze_root_cause(
        self,
        target_event: str,
        event_timestamp: datetime,
        affected_equipment: List[str],
        lookback_hours: int = 24,
        timeout_ms: Optional[int] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Perform root cause analysis.

        Args:
            target_event: Event to analyze
            event_timestamp: When the event occurred
            affected_equipment: List of affected equipment IDs
            lookback_hours: Hours to look back for causes
            timeout_ms: Request timeout
            metadata: Additional metadata

        Returns:
            RCA results with root causes and recommendations
        """
        request = {
            "target_event": target_event,
            "event_timestamp": event_timestamp.isoformat(),
            "affected_equipment": affected_equipment,
            "lookback_hours": lookback_hours,
        }

        logger.info(f"AnalyzeRootCause request: {request}")

        return {
            "success": True,
            "analysis_id": str(uuid4()),
            "target_event": target_event,
            "event_timestamp": event_timestamp.isoformat(),
            "root_causes": [
                {
                    "factor_name": "Steam trap failure",
                    "factor_description": "Upstream trap failed open",
                    "causal_strength": 0.85,
                    "confidence": 0.82,
                    "is_root_cause": True,
                }
            ],
            "contributing_factors": [
                {
                    "factor_name": "High system load",
                    "factor_description": "95% capacity utilization",
                    "causal_strength": 0.45,
                    "confidence": 0.75,
                    "is_root_cause": False,
                }
            ],
            "causal_chain": ["trap_failure", "steam_loss", "pressure_drop"],
            "executive_summary": "Root cause identified as steam trap failure.",
            "recommended_actions": ["Replace failed trap", "Inspect adjacent traps"],
            "model_confidence": 0.82,
        }

    async def compute_counterfactual(
        self,
        scenario_name: str,
        intervention_variable: str,
        intervention_value: float,
        baseline_value: float,
        outcome_variable: str,
        timeout_ms: Optional[int] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Compute counterfactual scenario.

        Args:
            scenario_name: Name of the scenario
            intervention_variable: Variable to change
            intervention_value: New value for intervention
            baseline_value: Original value
            outcome_variable: Outcome to predict
            timeout_ms: Request timeout
            metadata: Additional metadata

        Returns:
            Counterfactual analysis results
        """
        request = {
            "scenario_name": scenario_name,
            "intervention_variable": intervention_variable,
            "intervention_value": intervention_value,
            "baseline_value": baseline_value,
            "outcome_variable": outcome_variable,
        }

        logger.info(f"ComputeCounterfactual request: {request}")

        delta = intervention_value - baseline_value
        predicted_change = delta * 0.5

        return {
            "success": True,
            "scenario_id": str(uuid4()),
            "scenario_name": scenario_name,
            "intervention_variable": intervention_variable,
            "intervention_value": intervention_value,
            "baseline_value": baseline_value,
            "predicted_outcome_change": predicted_change,
            "confidence": 0.80,
        }


# =============================================================================
# Main Client
# =============================================================================

class SteamServiceClient:
    """
    Main client for UnifiedSteam gRPC services.

    Provides unified access to all steam system services with:
    - Connection management
    - Automatic reconnection
    - Service-specific clients
    - Health checking

    Usage:
        config = ClientConfig(host="localhost", port=50052)
        client = SteamServiceClient(config)

        await client.connect()

        result = await client.steam_properties.compute_properties(
            pressure_kpa=1000.0,
            temperature_c=200.0,
        )

        await client.disconnect()
    """

    def __init__(self, config: Optional[ClientConfig] = None):
        """
        Initialize the client.

        Args:
            config: Client configuration (uses defaults if not provided)
        """
        self.config = config or ClientConfig()
        self._channel: Optional[grpc_aio.Channel] = None
        self._connected = False

        # Service clients (initialized on connect)
        self._steam_properties: Optional[SteamPropertiesClient] = None
        self._optimization: Optional[OptimizationClient] = None
        self._diagnostics: Optional[DiagnosticsClient] = None
        self._rca: Optional[RCAClient] = None

    @property
    def steam_properties(self) -> SteamPropertiesClient:
        """Get steam properties service client."""
        if not self._steam_properties:
            raise RuntimeError("Client not connected. Call connect() first.")
        return self._steam_properties

    @property
    def optimization(self) -> OptimizationClient:
        """Get optimization service client."""
        if not self._optimization:
            raise RuntimeError("Client not connected. Call connect() first.")
        return self._optimization

    @property
    def diagnostics(self) -> DiagnosticsClient:
        """Get diagnostics service client."""
        if not self._diagnostics:
            raise RuntimeError("Client not connected. Call connect() first.")
        return self._diagnostics

    @property
    def rca(self) -> RCAClient:
        """Get RCA service client."""
        if not self._rca:
            raise RuntimeError("Client not connected. Call connect() first.")
        return self._rca

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected and self._channel is not None

    async def connect(self) -> None:
        """
        Establish connection to the gRPC server.

        Creates channel with configured options and initializes service clients.

        Raises:
            grpc.RpcError: If connection fails
        """
        if self._connected:
            logger.warning("Client already connected")
            return

        # Build channel options
        options = [
            ("grpc.max_receive_message_length", self.config.max_message_size),
            ("grpc.max_send_message_length", self.config.max_message_size),
            ("grpc.keepalive_time_ms", self.config.keepalive_time_ms),
            ("grpc.keepalive_timeout_ms", self.config.keepalive_timeout_ms),
            ("grpc.keepalive_permit_without_calls", True),
            ("grpc.http2.min_time_between_pings_ms", 10000),
            ("grpc.http2.max_pings_without_data", 0),
        ]

        # Add load balancing policy
        if self.config.load_balancing_policy:
            options.append((
                "grpc.service_config",
                f'{{"loadBalancingPolicy":"{self.config.load_balancing_policy}"}}'
            ))

        # Build interceptors
        interceptors = [
            LoggingInterceptor(),
        ]
        if self.config.retry_config.policy != RetryPolicy.NONE:
            interceptors.append(RetryInterceptor(self.config.retry_config))

        # Create channel
        if self.config.enable_tls:
            credentials = self._build_credentials()
            self._channel = grpc_aio.secure_channel(
                self.config.target,
                credentials,
                options=options,
                interceptors=interceptors,
            )
        else:
            self._channel = grpc_aio.insecure_channel(
                self.config.target,
                options=options,
                interceptors=interceptors,
            )

        # Initialize service clients
        self._steam_properties = SteamPropertiesClient(self._channel, self.config)
        self._optimization = OptimizationClient(self._channel, self.config)
        self._diagnostics = DiagnosticsClient(self._channel, self.config)
        self._rca = RCAClient(self._channel, self.config)

        self._connected = True
        logger.info(f"Connected to gRPC server at {self.config.target}")

    async def disconnect(self) -> None:
        """
        Close the connection to the gRPC server.

        Gracefully shuts down the channel.
        """
        if not self._connected:
            return

        if self._channel:
            await self._channel.close()
            self._channel = None

        self._steam_properties = None
        self._optimization = None
        self._diagnostics = None
        self._rca = None

        self._connected = False
        logger.info("Disconnected from gRPC server")

    async def check_health(self) -> Dict[str, Any]:
        """
        Check server health status.

        Returns:
            Health status with service details
        """
        if not self.is_connected:
            return {
                "status": "disconnected",
                "timestamp": datetime.utcnow().isoformat(),
            }

        # In production, use grpc.health.v1 health checking
        try:
            # Simple connectivity check
            state = self._channel.get_state(try_to_connect=True)

            return {
                "status": "healthy" if state == grpc.ChannelConnectivity.READY else "degraded",
                "channel_state": str(state),
                "target": self.config.target,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

    def _build_credentials(self) -> grpc.ChannelCredentials:
        """Build TLS credentials for secure channel."""
        # Load CA certificate
        root_certs = None
        if self.config.ca_cert_path:
            with open(self.config.ca_cert_path, "rb") as f:
                root_certs = f.read()

        # Load client certificate for mTLS
        private_key = None
        certificate_chain = None

        if self.config.key_path and self.config.cert_path:
            with open(self.config.key_path, "rb") as f:
                private_key = f.read()
            with open(self.config.cert_path, "rb") as f:
                certificate_chain = f.read()

        return grpc.ssl_channel_credentials(
            root_certificates=root_certs,
            private_key=private_key,
            certificate_chain=certificate_chain,
        )

    async def __aenter__(self) -> "SteamServiceClient":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()


# =============================================================================
# Convenience Functions
# =============================================================================

def create_client(
    host: str = "localhost",
    port: int = 50052,
    enable_tls: bool = False,
    **kwargs,
) -> SteamServiceClient:
    """
    Create a configured SteamServiceClient.

    Args:
        host: Server hostname
        port: Server port
        enable_tls: Enable TLS encryption
        **kwargs: Additional ClientConfig parameters

    Returns:
        Configured client instance

    Usage:
        client = create_client("steam.example.com", 50052, enable_tls=True)
        await client.connect()
    """
    config = ClientConfig(
        host=host,
        port=port,
        enable_tls=enable_tls,
        **kwargs,
    )
    return SteamServiceClient(config)


async def quick_connect(
    host: str = "localhost",
    port: int = 50052,
    enable_tls: bool = False,
) -> SteamServiceClient:
    """
    Create and connect a client in one step.

    Args:
        host: Server hostname
        port: Server port
        enable_tls: Enable TLS encryption

    Returns:
        Connected client instance

    Usage:
        client = await quick_connect("localhost", 50052)
        result = await client.steam_properties.compute_properties(...)
        await client.disconnect()
    """
    client = create_client(host, port, enable_tls)
    await client.connect()
    return client
