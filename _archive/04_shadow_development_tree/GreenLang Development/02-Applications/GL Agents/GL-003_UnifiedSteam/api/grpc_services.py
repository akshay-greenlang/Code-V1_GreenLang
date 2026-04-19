"""
GL-003 UnifiedSteam gRPC Services

gRPC service definitions for low-latency internal service-to-service calls.
Provides real-time steam property computation, optimization, and inference services.

This module provides:
- Service implementations (servicers) for all gRPC services
- Server interceptors for authentication, rate limiting, and logging
- Data store with caching and event queues for streaming
- Type conversion utilities between Pydantic and Proto enums

Usage:
    from api.grpc_services import (
        SteamPropertiesServicer,
        OptimizationServicer,
        InferenceServicer,
        RCAServicer,
        get_all_servicers,
        get_interceptors,
    )

    # Get all servicers for registration
    servicers = get_all_servicers()

    # Get interceptors for server
    interceptors = get_interceptors(enable_auth=True, enable_rate_limit=True)
"""

from __future__ import annotations

import asyncio
import logging
import json
from concurrent import futures
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import grpc
from google.protobuf import timestamp_pb2

from .api_auth import (
    Permission,
    SteamSystemUser,
    get_auth_config,
    verify_token,
)
from .api_schemas import (
    SteamPhase,
    SteamRegion,
    TrapCondition,
    OptimizationType,
    RecommendationPriority,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Proto Message Mocks (Would be generated from .proto in production)
# =============================================================================

class ProtoMessage:
    """Base class for mock proto messages."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def SerializeToString(self) -> bytes:
        """Serialize to string (mock)."""
        return json.dumps(self.__dict__, default=str).encode()

    @classmethod
    def FromString(cls, data: bytes) -> "ProtoMessage":
        """Deserialize from string (mock)."""
        obj = cls()
        obj.__dict__.update(json.loads(data.decode()))
        return obj

    def HasField(self, field_name: str) -> bool:
        """Check if field is set."""
        return hasattr(self, field_name) and getattr(self, field_name) is not None


# Proto Enums
class ProtoSteamPhase:
    STEAM_PHASE_UNSPECIFIED = 0
    STEAM_PHASE_SUBCOOLED_LIQUID = 1
    STEAM_PHASE_SATURATED_LIQUID = 2
    STEAM_PHASE_TWO_PHASE = 3
    STEAM_PHASE_SATURATED_VAPOR = 4
    STEAM_PHASE_SUPERHEATED_VAPOR = 5
    STEAM_PHASE_SUPERCRITICAL = 6


class ProtoSteamRegion:
    STEAM_REGION_UNSPECIFIED = 0
    STEAM_REGION_1 = 1  # Compressed liquid
    STEAM_REGION_2 = 2  # Superheated vapor
    STEAM_REGION_3 = 3  # Near-critical
    STEAM_REGION_4 = 4  # Two-phase
    STEAM_REGION_5 = 5  # High-temperature


class ProtoTrapCondition:
    TRAP_CONDITION_UNSPECIFIED = 0
    TRAP_CONDITION_GOOD = 1
    TRAP_CONDITION_LEAKING = 2
    TRAP_CONDITION_BLOCKED = 3
    TRAP_CONDITION_BLOW_THROUGH = 4
    TRAP_CONDITION_FAILED = 5


class ProtoAlarmSeverity:
    ALARM_SEVERITY_UNSPECIFIED = 0
    ALARM_SEVERITY_CRITICAL = 1
    ALARM_SEVERITY_HIGH = 2
    ALARM_SEVERITY_MEDIUM = 3
    ALARM_SEVERITY_LOW = 4
    ALARM_SEVERITY_INFO = 5


# =============================================================================
# Type Conversion Utilities
# =============================================================================

def datetime_to_timestamp(dt: datetime) -> timestamp_pb2.Timestamp:
    """Convert datetime to protobuf Timestamp."""
    ts = timestamp_pb2.Timestamp()
    ts.FromDatetime(dt)
    return ts


def timestamp_to_datetime(ts: timestamp_pb2.Timestamp) -> datetime:
    """Convert protobuf Timestamp to datetime."""
    return ts.ToDatetime()


def steam_phase_to_proto(phase: SteamPhase) -> int:
    """Convert SteamPhase enum to proto enum value."""
    mapping = {
        SteamPhase.SUBCOOLED_LIQUID: ProtoSteamPhase.STEAM_PHASE_SUBCOOLED_LIQUID,
        SteamPhase.SATURATED_LIQUID: ProtoSteamPhase.STEAM_PHASE_SATURATED_LIQUID,
        SteamPhase.TWO_PHASE: ProtoSteamPhase.STEAM_PHASE_TWO_PHASE,
        SteamPhase.SATURATED_VAPOR: ProtoSteamPhase.STEAM_PHASE_SATURATED_VAPOR,
        SteamPhase.SUPERHEATED_VAPOR: ProtoSteamPhase.STEAM_PHASE_SUPERHEATED_VAPOR,
        SteamPhase.SUPERCRITICAL: ProtoSteamPhase.STEAM_PHASE_SUPERCRITICAL,
    }
    return mapping.get(phase, ProtoSteamPhase.STEAM_PHASE_UNSPECIFIED)


def proto_to_steam_phase(proto_phase: int) -> SteamPhase:
    """Convert proto enum value to SteamPhase enum."""
    mapping = {
        ProtoSteamPhase.STEAM_PHASE_SUBCOOLED_LIQUID: SteamPhase.SUBCOOLED_LIQUID,
        ProtoSteamPhase.STEAM_PHASE_SATURATED_LIQUID: SteamPhase.SATURATED_LIQUID,
        ProtoSteamPhase.STEAM_PHASE_TWO_PHASE: SteamPhase.TWO_PHASE,
        ProtoSteamPhase.STEAM_PHASE_SATURATED_VAPOR: SteamPhase.SATURATED_VAPOR,
        ProtoSteamPhase.STEAM_PHASE_SUPERHEATED_VAPOR: SteamPhase.SUPERHEATED_VAPOR,
        ProtoSteamPhase.STEAM_PHASE_SUPERCRITICAL: SteamPhase.SUPERCRITICAL,
    }
    return mapping.get(proto_phase, SteamPhase.SUPERHEATED_VAPOR)


def steam_region_to_proto(region: SteamRegion) -> int:
    """Convert SteamRegion enum to proto enum value."""
    mapping = {
        SteamRegion.REGION_1: ProtoSteamRegion.STEAM_REGION_1,
        SteamRegion.REGION_2: ProtoSteamRegion.STEAM_REGION_2,
        SteamRegion.REGION_3: ProtoSteamRegion.STEAM_REGION_3,
        SteamRegion.REGION_4: ProtoSteamRegion.STEAM_REGION_4,
        SteamRegion.REGION_5: ProtoSteamRegion.STEAM_REGION_5,
    }
    return mapping.get(region, ProtoSteamRegion.STEAM_REGION_UNSPECIFIED)


def trap_condition_to_proto(condition: TrapCondition) -> int:
    """Convert TrapCondition to proto enum."""
    mapping = {
        TrapCondition.GOOD: ProtoTrapCondition.TRAP_CONDITION_GOOD,
        TrapCondition.LEAKING: ProtoTrapCondition.TRAP_CONDITION_LEAKING,
        TrapCondition.BLOCKED: ProtoTrapCondition.TRAP_CONDITION_BLOCKED,
        TrapCondition.BLOW_THROUGH: ProtoTrapCondition.TRAP_CONDITION_BLOW_THROUGH,
        TrapCondition.FAILED_OPEN: ProtoTrapCondition.TRAP_CONDITION_FAILED,
        TrapCondition.FAILED_CLOSED: ProtoTrapCondition.TRAP_CONDITION_FAILED,
    }
    return mapping.get(condition, ProtoTrapCondition.TRAP_CONDITION_UNSPECIFIED)


# =============================================================================
# gRPC Authentication Interceptor
# =============================================================================

class AuthInterceptor(grpc.ServerInterceptor):
    """
    gRPC server interceptor for authentication.
    Validates JWT tokens from metadata.
    """

    def __init__(self):
        self.config = get_auth_config()

    def intercept_service(self, continuation, handler_call_details):
        """Intercept incoming RPC calls for authentication."""
        metadata = dict(handler_call_details.invocation_metadata)

        # Check for authorization header
        auth_header = metadata.get("authorization", "")

        if not auth_header:
            # Check for x-api-key
            api_key = metadata.get("x-api-key", "")
            if not api_key:
                return self._abort_with_unauthenticated()

        # Validate JWT token
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            try:
                token_data = verify_token(token, self.config)
                return continuation(handler_call_details)
            except Exception as e:
                logger.warning(f"Token validation failed: {e}")
                return self._abort_with_unauthenticated()

        return continuation(handler_call_details)

    def _abort_with_unauthenticated(self):
        """Return unauthenticated error handler."""

        def abort_handler(request, context):
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid or missing authentication")

        return grpc.unary_unary_rpc_method_handler(abort_handler)


# =============================================================================
# Rate Limiting Interceptor
# =============================================================================

class TokenBucket:
    """
    Token bucket algorithm for rate limiting.

    Provides smooth rate limiting with burst capacity.
    """

    def __init__(
        self,
        rate: float,
        capacity: float,
    ):
        """
        Initialize token bucket.

        Args:
            rate: Tokens per second to add
            capacity: Maximum tokens (burst capacity)
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = datetime.utcnow()
        self._lock = asyncio.Lock()

    async def consume(self, tokens: float = 1.0) -> bool:
        """
        Attempt to consume tokens.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens consumed, False if rate limited
        """
        async with self._lock:
            now = datetime.utcnow()
            elapsed = (now - self.last_update).total_seconds()

            # Add tokens based on elapsed time
            self.tokens = min(
                self.capacity,
                self.tokens + elapsed * self.rate
            )
            self.last_update = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def get_wait_time(self, tokens: float = 1.0) -> float:
        """Get wait time in seconds until tokens available."""
        if self.tokens >= tokens:
            return 0.0
        return (tokens - self.tokens) / self.rate


class RateLimitInterceptor(grpc.ServerInterceptor):
    """
    gRPC server interceptor for rate limiting.

    Implements per-client and per-method rate limits using token bucket algorithm.
    """

    def __init__(
        self,
        requests_per_second: float = 100.0,
        burst_capacity: float = 200.0,
        per_client: bool = True,
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_second: Sustained request rate
            burst_capacity: Maximum burst size
            per_client: Whether to rate limit per client or globally
        """
        self.requests_per_second = requests_per_second
        self.burst_capacity = burst_capacity
        self.per_client = per_client

        # Global bucket
        self._global_bucket = TokenBucket(requests_per_second, burst_capacity)

        # Per-client buckets
        self._client_buckets: Dict[str, TokenBucket] = {}
        self._bucket_lock = asyncio.Lock()

    def _get_client_key(self, metadata: Dict[str, str]) -> str:
        """Extract client key from metadata."""
        # Try to get client identifier from metadata
        return (
            metadata.get("x-client-id") or
            metadata.get("x-forwarded-for") or
            metadata.get("authorization", "")[:20] or
            "anonymous"
        )

    async def _get_bucket(self, client_key: str) -> TokenBucket:
        """Get or create bucket for client."""
        if not self.per_client:
            return self._global_bucket

        async with self._bucket_lock:
            if client_key not in self._client_buckets:
                self._client_buckets[client_key] = TokenBucket(
                    self.requests_per_second,
                    self.burst_capacity,
                )
            return self._client_buckets[client_key]

    def intercept_service(self, continuation, handler_call_details):
        """Intercept incoming RPC calls for rate limiting."""
        metadata = dict(handler_call_details.invocation_metadata)
        client_key = self._get_client_key(metadata)

        # For synchronous interceptor, we can't use async
        # In production, use grpc.aio interceptors for proper async support
        bucket = self._client_buckets.get(client_key, self._global_bucket)

        # Synchronous check (simplified)
        if bucket.tokens < 1.0:
            return self._abort_with_rate_limit(bucket.get_wait_time())

        bucket.tokens -= 1.0
        return continuation(handler_call_details)

    def _abort_with_rate_limit(self, retry_after: float):
        """Return rate limit exceeded error handler."""

        def abort_handler(request, context):
            context.set_trailing_metadata([
                ("retry-after", str(int(retry_after) + 1)),
                ("x-ratelimit-reset", str(int(datetime.utcnow().timestamp()) + int(retry_after) + 1)),
            ])
            context.abort(
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                f"Rate limit exceeded. Retry after {int(retry_after) + 1} seconds"
            )

        return grpc.unary_unary_rpc_method_handler(abort_handler)


# =============================================================================
# Logging Interceptor
# =============================================================================

class LoggingInterceptor(grpc.ServerInterceptor):
    """
    gRPC server interceptor for request/response logging.

    Logs all RPC calls with timing, status, and metadata.
    """

    def __init__(
        self,
        log_requests: bool = True,
        log_responses: bool = True,
        log_metadata: bool = False,
        sensitive_methods: Optional[List[str]] = None,
    ):
        """
        Initialize logging interceptor.

        Args:
            log_requests: Log incoming requests
            log_responses: Log outgoing responses
            log_metadata: Include metadata in logs
            sensitive_methods: Methods to redact request/response bodies
        """
        self.log_requests = log_requests
        self.log_responses = log_responses
        self.log_metadata = log_metadata
        self.sensitive_methods = sensitive_methods or []

    def intercept_service(self, continuation, handler_call_details):
        """Intercept incoming RPC calls for logging."""
        method = handler_call_details.method
        metadata = dict(handler_call_details.invocation_metadata)

        # Extract request ID for correlation
        request_id = metadata.get("x-request-id", str(uuid4())[:8])

        # Log request
        if self.log_requests:
            log_data = {
                "event": "grpc_request",
                "request_id": request_id,
                "method": method,
                "timestamp": datetime.utcnow().isoformat(),
            }
            if self.log_metadata:
                # Filter sensitive metadata
                safe_metadata = {
                    k: v for k, v in metadata.items()
                    if k not in ["authorization", "x-api-key"]
                }
                log_data["metadata"] = safe_metadata

            logger.info(f"gRPC Request: {json.dumps(log_data)}")

        start_time = datetime.utcnow()

        # Continue to actual handler
        handler = continuation(handler_call_details)

        if handler is None:
            return None

        # Wrap handler to log response
        return self._wrap_handler(handler, method, request_id, start_time)

    def _wrap_handler(self, handler, method: str, request_id: str, start_time: datetime):
        """Wrap handler to log responses."""

        # Get the appropriate handler type
        if handler.unary_unary:
            original = handler.unary_unary

            def logged_unary_unary(request, context):
                try:
                    response = original(request, context)
                    self._log_response(method, request_id, start_time, "OK")
                    return response
                except Exception as e:
                    self._log_response(method, request_id, start_time, "ERROR", str(e))
                    raise

            return grpc.unary_unary_rpc_method_handler(
                logged_unary_unary,
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )

        elif handler.unary_stream:
            original = handler.unary_stream

            def logged_unary_stream(request, context):
                try:
                    for response in original(request, context):
                        yield response
                    self._log_response(method, request_id, start_time, "OK")
                except Exception as e:
                    self._log_response(method, request_id, start_time, "ERROR", str(e))
                    raise

            return grpc.unary_stream_rpc_method_handler(
                logged_unary_stream,
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )

        # Return original for other handler types
        return handler

    def _log_response(
        self,
        method: str,
        request_id: str,
        start_time: datetime,
        status: str,
        error: Optional[str] = None,
    ):
        """Log response details."""
        if not self.log_responses:
            return

        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        log_data = {
            "event": "grpc_response",
            "request_id": request_id,
            "method": method,
            "status": status,
            "duration_ms": round(duration_ms, 2),
            "timestamp": datetime.utcnow().isoformat(),
        }

        if error:
            log_data["error"] = error
            logger.error(f"gRPC Response: {json.dumps(log_data)}")
        else:
            logger.info(f"gRPC Response: {json.dumps(log_data)}")


# =============================================================================
# gRPC Data Store
# =============================================================================

class GRPCDataStore:
    """
    Data store for gRPC services.
    Provides caching and shared state for service implementations.
    """

    def __init__(self):
        # Cache for steam property calculations
        self._property_cache: Dict[str, Dict[str, Any]] = {}

        # Event queues for streaming
        self._optimization_update_queue: asyncio.Queue = asyncio.Queue()
        self._prediction_update_queue: asyncio.Queue = asyncio.Queue()
        self._alarm_queue: asyncio.Queue = asyncio.Queue()

    def _compute_steam_state(
        self,
        pressure_kpa: Optional[float] = None,
        temperature_c: Optional[float] = None,
        specific_enthalpy_kj_kg: Optional[float] = None,
        specific_entropy_kj_kg_k: Optional[float] = None,
        quality: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Compute steam state from inputs.

        Mock implementation - use CoolProp/IAPWS in production.
        """
        P = pressure_kpa or 1000.0
        T = temperature_c

        # Simplified saturation temperature
        T_sat = 100 + (P - 101.325) * 0.03

        # Determine phase
        if T is not None:
            if T < T_sat - 1:
                phase = SteamPhase.SUBCOOLED_LIQUID
                region = SteamRegion.REGION_1
            elif abs(T - T_sat) <= 1:
                phase = SteamPhase.TWO_PHASE if quality and 0 < quality < 1 else SteamPhase.SATURATED_VAPOR
                region = SteamRegion.REGION_4
            else:
                phase = SteamPhase.SUPERHEATED_VAPOR
                region = SteamRegion.REGION_2
        else:
            T = T_sat
            phase = SteamPhase.SATURATED_VAPOR
            region = SteamRegion.REGION_4

        # Mock property values
        if phase == SteamPhase.SUPERHEATED_VAPOR:
            h = 2676 + 2.0 * (T - T_sat)
            s = 7.36 + 0.002 * (T - T_sat)
            v = 0.001 + 0.002 * (T - T_sat) / P
        elif phase == SteamPhase.SUBCOOLED_LIQUID:
            h = 4.186 * T
            s = 0.3 + 0.001 * T
            v = 0.001
        else:
            h = 2676.0
            s = 7.36
            v = 1.67 / (P / 101.325)

        return {
            "pressure_kpa": P,
            "temperature_c": T,
            "specific_enthalpy_kj_kg": h,
            "specific_entropy_kj_kg_k": s,
            "specific_volume_m3_kg": v,
            "density_kg_m3": 1 / v,
            "quality": quality,
            "phase": phase,
            "region": region,
        }

    async def optimize_desuperheater(
        self,
        desuperheater_id: str,
        inlet_pressure_kpa: float,
        inlet_temperature_c: float,
        inlet_flow_kg_s: float,
        target_outlet_temperature_c: float,
        spray_water_temperature_c: float,
    ) -> Dict[str, Any]:
        """Optimize desuperheater spray water flow."""
        # Energy balance calculation (simplified)
        inlet_h = 3050.0  # Superheated
        outlet_h = 2850.0  # Target
        spray_h = spray_water_temperature_c * 4.186

        optimal_spray = inlet_flow_kg_s * (inlet_h - outlet_h) / (outlet_h - spray_h)
        optimal_spray = max(0, optimal_spray)

        return {
            "desuperheater_id": desuperheater_id,
            "optimal_spray_flow_kg_s": optimal_spray,
            "predicted_outlet_temperature_c": target_outlet_temperature_c,
            "spray_water_energy_kw": optimal_spray * spray_h,
            "efficiency": 0.95,
        }

    async def optimize_condensate(
        self,
        system_id: str,
        current_recovery_rate: float,
        condensate_temp_c: float,
        makeup_temp_c: float,
    ) -> Dict[str, Any]:
        """Optimize condensate recovery system."""
        optimal_rate = min(95.0, current_recovery_rate + 15.0)
        delta_temp = condensate_temp_c - makeup_temp_c

        # Estimate savings
        energy_savings_kwh_per_m3 = delta_temp * 4.186 / 3.6
        annual_savings_mwh = energy_savings_kwh_per_m3 * 1000 * 0.9 * 8760 / 1000

        return {
            "system_id": system_id,
            "optimal_recovery_rate_percent": optimal_rate,
            "delta_from_current_percent": optimal_rate - current_recovery_rate,
            "annual_energy_savings_mwh": annual_savings_mwh,
            "annual_water_savings_m3": 1000 * (optimal_rate - current_recovery_rate) / 100 * 8760,
        }

    async def optimize_network(
        self,
        network_id: str,
        total_demand_kg_s: float,
        headers: List[Dict],
        generators: List[Dict],
    ) -> Dict[str, Any]:
        """Optimize steam network distribution."""
        # Distribute demand across generators
        n_gen = len(generators) if generators else 1
        per_gen_output = total_demand_kg_s / n_gen

        optimal_outputs = {}
        for gen in (generators or []):
            gen_id = gen.get("generator_id", "gen_1")
            optimal_outputs[gen_id] = per_gen_output

        return {
            "network_id": network_id,
            "optimal_generator_outputs_kg_s": optimal_outputs,
            "total_generation_kg_s": total_demand_kg_s,
            "network_efficiency_percent": 92.5,
            "total_cost_usd_h": total_demand_kg_s * 25.0,
            "total_emissions_kg_co2_h": total_demand_kg_s * 0.5,
            "solver_status": "optimal",
        }

    async def predict_trap_failure(
        self,
        trap_id: str,
        inlet_pressure_kpa: float,
        outlet_pressure_kpa: float,
        inlet_temperature_c: float,
        outlet_temperature_c: float,
        acoustic_data: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Predict trap failure probability."""
        # Mock prediction based on differential temperature
        diff_temp = inlet_temperature_c - outlet_temperature_c

        # High diff_temp suggests good operation, low suggests potential issue
        if diff_temp > 50:
            failure_prob_30d = 0.05
            failure_prob_90d = 0.12
            condition = TrapCondition.GOOD
            risk_score = 15.0
        elif diff_temp > 20:
            failure_prob_30d = 0.15
            failure_prob_90d = 0.35
            condition = TrapCondition.DEGRADED
            risk_score = 45.0
        else:
            failure_prob_30d = 0.40
            failure_prob_90d = 0.70
            condition = TrapCondition.LEAKING
            risk_score = 75.0

        return {
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
        historical_baseline: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Detect anomalies in sensor readings."""
        anomalies = []

        # Simple threshold-based detection (mock)
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
    ) -> Dict[str, Any]:
        """Infer steam quality from measurements."""
        # Simplified - in production use calorimetric or other methods
        T_sat = 100 + (pressure_kpa - 101.325) * 0.03
        superheat = temperature_c - T_sat

        if superheat > 10:
            quality = 1.0  # Superheated
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
            "inferred_quality": quality,
            "state": state,
            "superheat_c": max(0, superheat),
            "confidence": 0.85 if superheat > 5 else 0.70,
        }

    async def analyze_root_cause(
        self,
        target_event: str,
        event_timestamp: datetime,
        affected_equipment: List[str],
        lookback_hours: int,
    ) -> Dict[str, Any]:
        """Perform root cause analysis."""
        return {
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
    ) -> Dict[str, Any]:
        """Compute counterfactual scenario."""
        # Mock counterfactual computation
        delta = intervention_value - baseline_value
        predicted_change = delta * 0.5  # Assumed coefficient

        return {
            "scenario_id": str(uuid4()),
            "scenario_name": scenario_name,
            "intervention_variable": intervention_variable,
            "intervention_value": intervention_value,
            "baseline_value": baseline_value,
            "predicted_outcome_change": predicted_change,
            "confidence": 0.80,
        }


# Global data store
grpc_data_store = GRPCDataStore()


# =============================================================================
# SteamPropertiesService Implementation
# =============================================================================

class SteamPropertiesServicer:
    """
    gRPC servicer for steam properties computations.
    Provides low-latency thermodynamic property calculations.
    """

    def __init__(self, data_store: Optional[GRPCDataStore] = None):
        self.data_store = data_store or grpc_data_store

    async def ComputeProperties(
        self,
        request: Any,
        context: grpc.aio.ServicerContext,
    ) -> Dict[str, Any]:
        """
        Compute steam properties from inputs.

        Args:
            request: SteamPropertiesRequest proto message
            context: gRPC context

        Returns:
            SteamPropertiesResponse proto message
        """
        try:
            start_time = datetime.utcnow()

            state = self.data_store._compute_steam_state(
                pressure_kpa=getattr(request, 'pressure_kpa', None),
                temperature_c=getattr(request, 'temperature_c', None),
                specific_enthalpy_kj_kg=getattr(request, 'specific_enthalpy_kj_kg', None),
                specific_entropy_kj_kg_k=getattr(request, 'specific_entropy_kj_kg_k', None),
                quality=getattr(request, 'quality', None),
            )

            computation_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            return {
                "request_id": getattr(request, 'request_id', str(uuid4())),
                "success": True,
                "steam_state": {
                    "pressure_kpa": state["pressure_kpa"],
                    "temperature_c": state["temperature_c"],
                    "specific_enthalpy_kj_kg": state["specific_enthalpy_kj_kg"],
                    "specific_entropy_kj_kg_k": state["specific_entropy_kj_kg_k"],
                    "specific_volume_m3_kg": state["specific_volume_m3_kg"],
                    "density_kg_m3": state["density_kg_m3"],
                    "quality": state.get("quality"),
                    "phase": steam_phase_to_proto(state["phase"]),
                    "region": steam_region_to_proto(state["region"]),
                },
                "computation_time_ms": computation_time,
            }

        except Exception as e:
            logger.error(f"ComputeProperties failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return {
                "request_id": getattr(request, 'request_id', ''),
                "success": False,
                "error_message": str(e),
            }

    async def GetSaturationProperties(
        self,
        request: Any,
        context: grpc.aio.ServicerContext,
    ) -> Dict[str, Any]:
        """
        Get saturation properties at given pressure or temperature.

        Args:
            request: SaturationRequest proto message
            context: gRPC context

        Returns:
            SaturationResponse proto message
        """
        try:
            pressure_kpa = getattr(request, 'pressure_kpa', None)
            temperature_c = getattr(request, 'temperature_c', None)

            if pressure_kpa:
                T_sat = 100 + (pressure_kpa - 101.325) * 0.03
            elif temperature_c:
                pressure_kpa = 101.325 + (temperature_c - 100) / 0.03
                T_sat = temperature_c
            else:
                raise ValueError("Either pressure_kpa or temperature_c must be provided")

            # Saturation properties (mock)
            return {
                "success": True,
                "saturation_pressure_kpa": pressure_kpa,
                "saturation_temperature_c": T_sat,
                "liquid_properties": {
                    "specific_enthalpy_kj_kg": T_sat * 4.186,
                    "specific_entropy_kj_kg_k": 0.3 + 0.003 * T_sat,
                    "specific_volume_m3_kg": 0.001,
                },
                "vapor_properties": {
                    "specific_enthalpy_kj_kg": 2676.0,
                    "specific_entropy_kj_kg_k": 7.36,
                    "specific_volume_m3_kg": 1.67 / (pressure_kpa / 101.325),
                },
                "latent_heat_kj_kg": 2676.0 - T_sat * 4.186,
            }

        except Exception as e:
            logger.error(f"GetSaturationProperties failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return {"success": False, "error_message": str(e)}


# =============================================================================
# OptimizationService Implementation
# =============================================================================

class OptimizationServicer:
    """
    gRPC servicer for optimization computations.
    Provides desuperheater, condensate, and network optimization.
    """

    def __init__(self, data_store: Optional[GRPCDataStore] = None):
        self.data_store = data_store or grpc_data_store

    async def OptimizeDesuperheater(
        self,
        request: Any,
        context: grpc.aio.ServicerContext,
    ) -> Dict[str, Any]:
        """Optimize desuperheater spray water flow."""
        try:
            result = await self.data_store.optimize_desuperheater(
                desuperheater_id=request.desuperheater_id,
                inlet_pressure_kpa=request.inlet_pressure_kpa,
                inlet_temperature_c=request.inlet_temperature_c,
                inlet_flow_kg_s=request.inlet_flow_kg_s,
                target_outlet_temperature_c=request.target_outlet_temperature_c,
                spray_water_temperature_c=request.spray_water_temperature_c,
            )
            return {"success": True, **result}

        except Exception as e:
            logger.error(f"OptimizeDesuperheater failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return {"success": False, "error_message": str(e)}

    async def OptimizeCondensate(
        self,
        request: Any,
        context: grpc.aio.ServicerContext,
    ) -> Dict[str, Any]:
        """Optimize condensate recovery system."""
        try:
            result = await self.data_store.optimize_condensate(
                system_id=request.system_id,
                current_recovery_rate=request.current_recovery_rate_percent,
                condensate_temp_c=request.condensate_temperature_c,
                makeup_temp_c=request.makeup_water_temperature_c,
            )
            return {"success": True, **result}

        except Exception as e:
            logger.error(f"OptimizeCondensate failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return {"success": False, "error_message": str(e)}

    async def OptimizeNetwork(
        self,
        request: Any,
        context: grpc.aio.ServicerContext,
    ) -> Dict[str, Any]:
        """Optimize steam network distribution."""
        try:
            result = await self.data_store.optimize_network(
                network_id=request.network_id,
                total_demand_kg_s=request.total_demand_kg_s,
                headers=list(getattr(request, 'headers', [])),
                generators=list(getattr(request, 'generators', [])),
            )
            return {"success": True, **result}

        except Exception as e:
            logger.error(f"OptimizeNetwork failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return {"success": False, "error_message": str(e)}

    async def StreamOptimizationUpdates(
        self,
        request: Any,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream optimization updates."""
        try:
            while not context.cancelled():
                try:
                    update = await asyncio.wait_for(
                        self.data_store._optimization_update_queue.get(),
                        timeout=60.0,
                    )
                    yield update
                except asyncio.TimeoutError:
                    # Send heartbeat
                    yield {
                        "heartbeat": True,
                        "timestamp": datetime.utcnow().isoformat(),
                    }

        except Exception as e:
            logger.error(f"StreamOptimizationUpdates failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))


# =============================================================================
# InferenceService Implementation
# =============================================================================

class InferenceServicer:
    """
    gRPC servicer for ML inference operations.
    Provides trap failure prediction, anomaly detection, and quality inference.
    """

    def __init__(self, data_store: Optional[GRPCDataStore] = None):
        self.data_store = data_store or grpc_data_store

    async def PredictTrapFailure(
        self,
        request: Any,
        context: grpc.aio.ServicerContext,
    ) -> Dict[str, Any]:
        """Predict trap failure probability."""
        try:
            result = await self.data_store.predict_trap_failure(
                trap_id=request.trap_id,
                inlet_pressure_kpa=request.inlet_pressure_kpa,
                outlet_pressure_kpa=request.outlet_pressure_kpa,
                inlet_temperature_c=request.inlet_temperature_c,
                outlet_temperature_c=request.outlet_temperature_c,
                acoustic_data=list(getattr(request, 'acoustic_data', [])) or None,
            )

            return {
                "success": True,
                "trap_id": result["trap_id"],
                "condition": trap_condition_to_proto(result["condition"]),
                "failure_probability_30d": result["failure_probability_30d"],
                "failure_probability_90d": result["failure_probability_90d"],
                "risk_score": result["risk_score"],
                "risk_factors": result["risk_factors"],
                "recommended_action": result["recommended_action"],
                "model_confidence": result["model_confidence"],
            }

        except Exception as e:
            logger.error(f"PredictTrapFailure failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return {"success": False, "error_message": str(e)}

    async def DetectAnomaly(
        self,
        request: Any,
        context: grpc.aio.ServicerContext,
    ) -> Dict[str, Any]:
        """Detect anomalies in sensor readings."""
        try:
            sensor_readings = dict(getattr(request, 'sensor_readings', {}))
            result = await self.data_store.detect_anomaly(
                equipment_id=request.equipment_id,
                sensor_readings=sensor_readings,
            )
            return {"success": True, **result}

        except Exception as e:
            logger.error(f"DetectAnomaly failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return {"success": False, "error_message": str(e)}

    async def InferSteamQuality(
        self,
        request: Any,
        context: grpc.aio.ServicerContext,
    ) -> Dict[str, Any]:
        """Infer steam quality from measurements."""
        try:
            result = await self.data_store.infer_steam_quality(
                pressure_kpa=request.pressure_kpa,
                temperature_c=request.temperature_c,
                flow_rate_kg_s=request.flow_rate_kg_s,
                conductivity_us_cm=getattr(request, 'conductivity_us_cm', None),
            )
            return {"success": True, **result}

        except Exception as e:
            logger.error(f"InferSteamQuality failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return {"success": False, "error_message": str(e)}

    async def StreamPredictions(
        self,
        request: Any,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream prediction updates."""
        try:
            while not context.cancelled():
                try:
                    update = await asyncio.wait_for(
                        self.data_store._prediction_update_queue.get(),
                        timeout=60.0,
                    )
                    yield update
                except asyncio.TimeoutError:
                    yield {"heartbeat": True, "timestamp": datetime.utcnow().isoformat()}

        except Exception as e:
            logger.error(f"StreamPredictions failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))


# =============================================================================
# RCAService Implementation
# =============================================================================

class RCAServicer:
    """
    gRPC servicer for root cause analysis.
    Provides causal inference and counterfactual analysis.
    """

    def __init__(self, data_store: Optional[GRPCDataStore] = None):
        self.data_store = data_store or grpc_data_store

    async def AnalyzeRootCause(
        self,
        request: Any,
        context: grpc.aio.ServicerContext,
    ) -> Dict[str, Any]:
        """Perform root cause analysis."""
        try:
            result = await self.data_store.analyze_root_cause(
                target_event=request.target_event,
                event_timestamp=timestamp_to_datetime(request.event_timestamp)
                    if hasattr(request, 'event_timestamp') else datetime.utcnow(),
                affected_equipment=list(getattr(request, 'affected_equipment', [])),
                lookback_hours=getattr(request, 'lookback_hours', 24),
            )
            return {"success": True, **result}

        except Exception as e:
            logger.error(f"AnalyzeRootCause failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return {"success": False, "error_message": str(e)}

    async def ComputeCounterfactual(
        self,
        request: Any,
        context: grpc.aio.ServicerContext,
    ) -> Dict[str, Any]:
        """Compute counterfactual scenario."""
        try:
            result = await self.data_store.compute_counterfactual(
                scenario_name=request.scenario_name,
                intervention_variable=request.intervention_variable,
                intervention_value=request.intervention_value,
                baseline_value=request.baseline_value,
                outcome_variable=request.outcome_variable,
            )
            return {"success": True, **result}

        except Exception as e:
            logger.error(f"ComputeCounterfactual failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return {"success": False, "error_message": str(e)}


# =============================================================================
# gRPC Server Configuration
# =============================================================================

async def serve_grpc(
    host: str = "0.0.0.0",
    port: int = 50052,
    max_workers: int = 10,
    enable_reflection: bool = True,
    enable_auth: bool = True,
) -> grpc.aio.Server:
    """
    Start the gRPC server.

    Args:
        host: Host to bind to
        port: Port to listen on
        max_workers: Maximum number of worker threads
        enable_reflection: Enable gRPC reflection for debugging
        enable_auth: Enable authentication interceptor

    Returns:
        Running gRPC server instance
    """
    interceptors = []
    if enable_auth:
        interceptors.append(AuthInterceptor())

    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        interceptors=interceptors,
    )

    # NOTE: In production, add generated service stubs:
    # steam_pb2_grpc.add_SteamPropertiesServiceServicer_to_server(
    #     SteamPropertiesServicer(), server
    # )
    # steam_pb2_grpc.add_OptimizationServiceServicer_to_server(
    #     OptimizationServicer(), server
    # )
    # steam_pb2_grpc.add_InferenceServiceServicer_to_server(
    #     InferenceServicer(), server
    # )
    # steam_pb2_grpc.add_RCAServiceServicer_to_server(
    #     RCAServicer(), server
    # )

    listen_addr = f"{host}:{port}"
    server.add_insecure_port(listen_addr)

    await server.start()
    logger.info(f"gRPC server started on {listen_addr}")

    return server


async def serve_grpc_with_tls(
    host: str = "0.0.0.0",
    port: int = 50052,
    server_cert_path: str = "",
    server_key_path: str = "",
    ca_cert_path: Optional[str] = None,
    max_workers: int = 10,
    enable_auth: bool = True,
) -> grpc.aio.Server:
    """
    Start the gRPC server with TLS/mTLS.

    Args:
        host: Host to bind to
        port: Port to listen on
        server_cert_path: Path to server certificate
        server_key_path: Path to server private key
        ca_cert_path: Path to CA certificate for mTLS (optional)
        max_workers: Maximum number of worker threads
        enable_auth: Enable authentication interceptor

    Returns:
        Running gRPC server instance with TLS
    """
    with open(server_key_path, "rb") as f:
        server_key = f.read()
    with open(server_cert_path, "rb") as f:
        server_cert = f.read()

    ca_cert = None
    if ca_cert_path:
        with open(ca_cert_path, "rb") as f:
            ca_cert = f.read()

    if ca_cert:
        # mTLS - require client certificate
        credentials = grpc.ssl_server_credentials(
            [(server_key, server_cert)],
            root_certificates=ca_cert,
            require_client_auth=True,
        )
    else:
        # TLS only
        credentials = grpc.ssl_server_credentials(
            [(server_key, server_cert)],
        )

    interceptors = []
    if enable_auth:
        interceptors.append(AuthInterceptor())

    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        interceptors=interceptors,
    )

    listen_addr = f"{host}:{port}"
    server.add_secure_port(listen_addr, credentials)

    await server.start()
    logger.info(f"gRPC server started with TLS on {listen_addr}")

    return server


# =============================================================================
# Utility Functions
# =============================================================================

def get_all_servicers(data_store: Optional[GRPCDataStore] = None) -> Dict[str, Any]:
    """
    Get all gRPC servicers for registration.

    Args:
        data_store: Optional shared data store instance

    Returns:
        Dictionary mapping service names to servicer instances

    Usage:
        servicers = get_all_servicers()
        for name, servicer in servicers.items():
            # Register with server using generated _pb2_grpc module
            add_servicer_func = getattr(steam_pb2_grpc, f"add_{name}Servicer_to_server")
            add_servicer_func(servicer, server)
    """
    store = data_store or grpc_data_store

    return {
        "SteamPropertiesService": SteamPropertiesServicer(store),
        "OptimizationService": OptimizationServicer(store),
        "InferenceService": InferenceServicer(store),
        "RCAService": RCAServicer(store),
    }


def get_interceptors(
    enable_auth: bool = True,
    enable_rate_limit: bool = True,
    enable_logging: bool = True,
    rate_limit_rps: float = 100.0,
    rate_limit_burst: float = 200.0,
) -> List[grpc.ServerInterceptor]:
    """
    Get list of server interceptors.

    Args:
        enable_auth: Enable authentication interceptor
        enable_rate_limit: Enable rate limiting interceptor
        enable_logging: Enable logging interceptor
        rate_limit_rps: Rate limit requests per second
        rate_limit_burst: Rate limit burst capacity

    Returns:
        List of interceptors in recommended order

    Usage:
        interceptors = get_interceptors(enable_auth=True, enable_rate_limit=True)
        server = grpc.aio.server(interceptors=interceptors)
    """
    interceptors = []

    # Logging first to capture all requests
    if enable_logging:
        interceptors.append(LoggingInterceptor())

    # Rate limiting before auth to prevent auth DoS
    if enable_rate_limit:
        interceptors.append(RateLimitInterceptor(
            requests_per_second=rate_limit_rps,
            burst_capacity=rate_limit_burst,
        ))

    # Authentication last
    if enable_auth:
        interceptors.append(AuthInterceptor())

    return interceptors


# =============================================================================
# Streaming Data Service (Additional)
# =============================================================================

class StreamingServicer:
    """
    gRPC servicer for real-time data streaming.

    Provides bidirectional streaming for:
    - Real-time steam state updates
    - Alarm notifications
    - KPI updates
    - Sensor data ingestion
    """

    def __init__(self, data_store: Optional[GRPCDataStore] = None):
        self.data_store = data_store or grpc_data_store
        self._subscribers: Dict[str, List[asyncio.Queue]] = {
            "steam_states": [],
            "alarms": [],
            "kpis": [],
        }

    async def SubscribeSteamStates(
        self,
        request: Any,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Subscribe to real-time steam state updates.

        Args:
            request: Subscription request with equipment IDs
            context: gRPC context

        Yields:
            Steam state updates for subscribed equipment
        """
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers["steam_states"].append(queue)

        try:
            equipment_ids = set(getattr(request, "equipment_ids", []))

            while not context.cancelled():
                try:
                    update = await asyncio.wait_for(queue.get(), timeout=30.0)

                    # Filter by equipment if specified
                    if equipment_ids:
                        if update.get("equipment_id") not in equipment_ids:
                            continue

                    yield update

                except asyncio.TimeoutError:
                    # Send heartbeat
                    yield {
                        "type": "heartbeat",
                        "timestamp": datetime.utcnow().isoformat(),
                    }

        finally:
            self._subscribers["steam_states"].remove(queue)

    async def SubscribeAlarms(
        self,
        request: Any,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Subscribe to alarm notifications.

        Args:
            request: Subscription request with severity filter
            context: gRPC context

        Yields:
            Alarm notifications matching filter
        """
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers["alarms"].append(queue)

        try:
            min_severity = getattr(request, "min_severity", None)

            while not context.cancelled():
                try:
                    alarm = await asyncio.wait_for(queue.get(), timeout=60.0)

                    # Filter by severity if specified
                    if min_severity:
                        alarm_severity = alarm.get("severity", "INFO")
                        # Simple severity comparison
                        severity_order = ["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
                        if severity_order.index(alarm_severity) < severity_order.index(min_severity):
                            continue

                    yield alarm

                except asyncio.TimeoutError:
                    yield {
                        "type": "heartbeat",
                        "timestamp": datetime.utcnow().isoformat(),
                    }

        finally:
            self._subscribers["alarms"].remove(queue)

    async def SubscribeKPIs(
        self,
        request: Any,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Subscribe to KPI updates.

        Args:
            request: Subscription request with KPI names
            context: gRPC context

        Yields:
            KPI updates for subscribed metrics
        """
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers["kpis"].append(queue)

        try:
            kpi_names = set(getattr(request, "kpi_names", []))

            while not context.cancelled():
                try:
                    kpi_update = await asyncio.wait_for(queue.get(), timeout=30.0)

                    # Filter by KPI name if specified
                    if kpi_names:
                        if kpi_update.get("kpi_name") not in kpi_names:
                            continue

                    yield kpi_update

                except asyncio.TimeoutError:
                    yield {
                        "type": "heartbeat",
                        "timestamp": datetime.utcnow().isoformat(),
                    }

        finally:
            self._subscribers["kpis"].remove(queue)

    async def PushSensorData(
        self,
        request_iterator: AsyncIterator[Any],
        context: grpc.aio.ServicerContext,
    ) -> Dict[str, Any]:
        """
        Receive streaming sensor data from clients.

        Args:
            request_iterator: Stream of sensor data messages
            context: gRPC context

        Returns:
            Summary of received data
        """
        records_received = 0
        records_processed = 0
        errors = []

        try:
            async for sensor_data in request_iterator:
                records_received += 1

                try:
                    # Process sensor data
                    equipment_id = sensor_data.equipment_id
                    timestamp = timestamp_to_datetime(sensor_data.timestamp)
                    readings = dict(sensor_data.readings)

                    # Store or process the data
                    # In production, this would write to a time-series database
                    logger.debug(f"Received sensor data: {equipment_id} @ {timestamp}")

                    records_processed += 1

                except Exception as e:
                    errors.append(str(e))
                    logger.warning(f"Error processing sensor data: {e}")

        except Exception as e:
            logger.error(f"PushSensorData stream error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))

        return {
            "success": len(errors) == 0,
            "records_received": records_received,
            "records_processed": records_processed,
            "errors": errors[:10] if errors else [],  # Limit error list
        }

    async def publish_steam_state(self, state: Dict[str, Any]) -> None:
        """Publish steam state update to all subscribers."""
        for queue in self._subscribers["steam_states"]:
            await queue.put(state)

    async def publish_alarm(self, alarm: Dict[str, Any]) -> None:
        """Publish alarm to all subscribers."""
        for queue in self._subscribers["alarms"]:
            await queue.put(alarm)

    async def publish_kpi(self, kpi: Dict[str, Any]) -> None:
        """Publish KPI update to all subscribers."""
        for queue in self._subscribers["kpis"]:
            await queue.put(kpi)


# Global streaming servicer instance
streaming_servicer = StreamingServicer()


def get_streaming_servicer() -> StreamingServicer:
    """Get the global streaming servicer instance."""
    return streaming_servicer
