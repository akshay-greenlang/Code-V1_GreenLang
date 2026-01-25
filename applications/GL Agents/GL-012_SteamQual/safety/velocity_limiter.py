# GL-012 STEAMQUAL - Velocity Limiter
# Rate limiting for steam quality setpoint changes
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field
import hashlib, json, logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class VelocityLimitStatus(str, Enum):
    WITHIN_LIMITS = "within_limits"
    RATE_LIMITED = "rate_limited"
    CLAMPED = "clamped"
    EMERGENCY_OVERRIDE = "emergency_override"


class SetpointType(str, Enum):
    DRYNESS_FRACTION = "dryness_fraction"
    SUPERHEAT = "superheat"
    PRESSURE = "pressure"
    TEMPERATURE = "temperature"
    FLOW_RATE = "flow_rate"


class VelocityLimitConfig(BaseModel):
    max_delta_dryness_fraction: float = Field(default=0.02, ge=0.001, le=0.1, description="Max change in dryness fraction per cycle")
    max_delta_superheat_c: float = Field(default=5.0, ge=0.5, le=20.0, description="Max change in superheat per cycle (C)")
    max_delta_pressure_mpa: float = Field(default=0.5, ge=0.1, le=2.0, description="Max change in pressure per cycle (MPa)")
    max_delta_temperature_c: float = Field(default=10.0, ge=1.0, le=50.0, description="Max change in temperature per cycle (C)")
    max_delta_flow_rate_pct: float = Field(default=5.0, ge=1.0, le=20.0, description="Max change in flow rate per cycle (%)")
    cycle_time_seconds: float = Field(default=60.0, ge=10.0, le=600.0, description="Control cycle time")
    ramp_smoothing: bool = Field(default=True, description="Enable smooth ramping")
    emergency_override_enabled: bool = Field(default=True, description="Allow emergency overrides")


class VelocityLimitResult(BaseModel):
    original_setpoint: float = Field(..., description="Original requested setpoint")
    limited_setpoint: float = Field(..., description="Rate-limited setpoint")
    previous_setpoint: float = Field(..., description="Previous setpoint value")
    delta_requested: float = Field(..., description="Requested change")
    delta_allowed: float = Field(..., description="Allowed change after limiting")
    max_delta: float = Field(..., description="Maximum allowed delta")
    status: VelocityLimitStatus = Field(..., description="Limiting status")
    setpoint_type: SetpointType = Field(..., description="Type of setpoint")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_hash: str = Field(..., description="SHA-256 hash for audit")
    is_rate_limited: bool = Field(..., description="True if rate limiting was applied")


class VelocityLimiter:
    VERSION = "1.0.0"

    def __init__(self, config: Optional[VelocityLimitConfig] = None, agent_id: str = "GL-012"):
        self.config = config or VelocityLimitConfig()
        self.agent_id = agent_id
        self._previous_setpoints: Dict[str, float] = {}
        self._last_update_times: Dict[str, datetime] = {}
        logger.info(f"VelocityLimiter initialized: agent={agent_id}")

    def limit_setpoint(self, setpoint_type: SetpointType, requested_value: float, 
                        header_id: str, emergency_override: bool = False) -> VelocityLimitResult:
        key = f"{header_id}_{setpoint_type.value}"
        previous_value = self._previous_setpoints.get(key, requested_value)
        max_delta = self._get_max_delta(setpoint_type)
        delta_requested = requested_value - previous_value
        if emergency_override and self.config.emergency_override_enabled:
            limited_value = requested_value
            status = VelocityLimitStatus.EMERGENCY_OVERRIDE
            is_limited = False
        elif abs(delta_requested) <= max_delta:
            limited_value = requested_value
            status = VelocityLimitStatus.WITHIN_LIMITS
            is_limited = False
        else:
            if delta_requested > 0:
                limited_value = previous_value + max_delta
            else:
                limited_value = previous_value - max_delta
            status = VelocityLimitStatus.RATE_LIMITED
            is_limited = True
        self._previous_setpoints[key] = limited_value
        self._last_update_times[key] = datetime.now(timezone.utc)
        provenance_hash = self._compute_provenance_hash(setpoint_type, requested_value, limited_value, header_id)
        return VelocityLimitResult(
            original_setpoint=requested_value, limited_setpoint=limited_value, previous_setpoint=previous_value,
            delta_requested=delta_requested, delta_allowed=limited_value - previous_value, max_delta=max_delta,
            status=status, setpoint_type=setpoint_type, provenance_hash=provenance_hash, is_rate_limited=is_limited
        )

    def limit_dryness_fraction(self, requested_value: float, header_id: str,
                                 emergency_override: bool = False) -> VelocityLimitResult:
        return self.limit_setpoint(SetpointType.DRYNESS_FRACTION, requested_value, header_id, emergency_override)

    def limit_superheat(self, requested_value: float, header_id: str,
                          emergency_override: bool = False) -> VelocityLimitResult:
        return self.limit_setpoint(SetpointType.SUPERHEAT, requested_value, header_id, emergency_override)

    def _get_max_delta(self, setpoint_type: SetpointType) -> float:
        delta_map = {
            SetpointType.DRYNESS_FRACTION: self.config.max_delta_dryness_fraction,
            SetpointType.SUPERHEAT: self.config.max_delta_superheat_c,
            SetpointType.PRESSURE: self.config.max_delta_pressure_mpa,
            SetpointType.TEMPERATURE: self.config.max_delta_temperature_c,
            SetpointType.FLOW_RATE: self.config.max_delta_flow_rate_pct,
        }
        return delta_map.get(setpoint_type, 0.01)

    def _compute_provenance_hash(self, setpoint_type: SetpointType, requested: float,
                                   limited: float, header_id: str) -> str:
        provenance_data = {"setpoint_type": setpoint_type.value, "requested": requested,
                           "limited": limited, "header_id": header_id, "agent_id": self.agent_id}
        return hashlib.sha256(json.dumps(provenance_data, sort_keys=True).encode()).hexdigest()

    def reset_state(self, header_id: Optional[str] = None) -> None:
        if header_id:
            keys_to_remove = [k for k in self._previous_setpoints.keys() if k.startswith(header_id)]
            for key in keys_to_remove:
                del self._previous_setpoints[key]
                if key in self._last_update_times:
                    del self._last_update_times[key]
        else:
            self._previous_setpoints.clear()
            self._last_update_times.clear()

    def get_state(self, header_id: str) -> Dict[str, float]:
        return {k: v for k, v in self._previous_setpoints.items() if k.startswith(header_id)}


def create_velocity_limiter(max_delta_dryness_fraction: float = 0.02, max_delta_superheat_c: float = 5.0,
                             agent_id: str = "GL-012", **kwargs) -> VelocityLimiter:
    config = VelocityLimitConfig(max_delta_dryness_fraction=max_delta_dryness_fraction,
                                  max_delta_superheat_c=max_delta_superheat_c, **kwargs)
    return VelocityLimiter(config=config, agent_id=agent_id)


__all__ = [
    "VelocityLimitStatus", "SetpointType", "VelocityLimitConfig", "VelocityLimitResult",
    "VelocityLimiter", "create_velocity_limiter",
]
