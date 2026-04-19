# -*- coding: utf-8 -*-
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field
from enum import Enum
from dataclasses import dataclass
from collections import deque
import hashlib
import logging

logger = logging.getLogger(__name__)

class ParameterType(str, Enum):
    AIR_FUEL_RATIO = "air_fuel_ratio"
    TEMPERATURE = "temperature"
    O2_SETPOINT = "o2_setpoint"
    CO_SETPOINT = "co_setpoint"
    FUEL_FLOW = "fuel_flow"
    AIR_FLOW = "air_flow"
    DAMPER_POSITION = "damper_position"
    VALVE_POSITION = "valve_position"
    PRESSURE = "pressure"
    LOAD = "load"

class VelocityLimitStatus(str, Enum):
    NORMAL = "normal"
    RATE_LIMITED = "rate_limited"
    SAFETY_LIMITED = "safety_limited"
    EMERGENCY_BYPASS = "emergency_bypass"
    FROZEN = "frozen"

class RampDirection(str, Enum):
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"

class SafetyBoundType(str, Enum):
    ABSOLUTE_MIN = "absolute_min"
    ABSOLUTE_MAX = "absolute_max"
    RATE_MAX = "rate_max"
    OPERATIONAL = "operational"

@dataclass
class VelocityLimit:
    parameter_type: ParameterType
    max_increase_rate: float
    max_decrease_rate: float
    absolute_min: float
    absolute_max: float
    unit: str = ""
    description: str = ""
    safety_critical: bool = False

@dataclass
class VelocityViolation:
    parameter_type: ParameterType
    timestamp: float
    requested_value: float
    allowed_value: float
    violation_magnitude: float
    violation_rate: float
    bound_type: SafetyBoundType
    description: str

@dataclass
class ParameterState:
    parameter_type: ParameterType
    current_value: float
    previous_value: float
    target_value: float
    timestamp: float
    previous_timestamp: float
    rate_of_change: float
    status: VelocityLimitStatus
    direction: RampDirection
    time_to_target: float
    is_at_target: bool
    violations_count: int = 0

class VelocityLimiterConfig(BaseModel):
    afr_max_increase_rate: float = Field(default=0.1, ge=0.01, le=1.0)
    afr_max_decrease_rate: float = Field(default=0.05, ge=0.01, le=1.0)
    afr_min: float = Field(default=0.8)
    afr_max: float = Field(default=1.5)
    temp_max_increase_rate: float = Field(default=50.0, ge=1.0, le=200.0)
    temp_max_decrease_rate: float = Field(default=30.0, ge=1.0, le=200.0)
    temp_min: float = Field(default=0.0)
    temp_max: float = Field(default=1500.0)
    o2_max_rate: float = Field(default=0.5, ge=0.1, le=5.0)
    o2_min: float = Field(default=1.0)
    o2_max: float = Field(default=15.0)
    co_max_rate: float = Field(default=50.0, ge=10.0, le=500.0)
    co_min: float = Field(default=0.0)
    co_max: float = Field(default=400.0)
    deadband: float = Field(default=0.001)
    target_tolerance: float = Field(default=0.01)
    history_size: int = Field(default=100)
    enable_safety_bounds: bool = Field(default=True)
    log_violations: bool = Field(default=True)

class VelocityLimiterInput(BaseModel):
    parameter_type: ParameterType
    current_value: float
    target_value: float
    timestamp: float = Field(ge=0)
    max_rate_override: Optional[float] = Field(None, ge=0)
    emergency_bypass: bool = Field(default=False)
    freeze_output: bool = Field(default=False)

class VelocityLimiterOutput(BaseModel):
    limited_value: float
    requested_value: float
    previous_value: float
    actual_rate: float
    allowed_rate: float
    rate_limited: bool
    status: VelocityLimitStatus
    direction: RampDirection
    progress_percent: float = Field(ge=0, le=100)
    time_to_target_sec: float = Field(ge=0)
    is_at_target: bool
    safety_bounds_applied: bool
    violation_detected: bool
    violation_details: Optional[str] = None
    provenance_hash: str

class VelocityLimiterSummary(BaseModel):
    total_parameters: int
    actively_limited: int
    at_target: int
    violations_total: int
    violations_by_type: Dict[str, int] = Field(default_factory=dict)
    status_by_parameter: Dict[str, str] = Field(default_factory=dict)
    last_update_timestamp: float
    provenance_hash: str


class VelocityLimiter:
    DEFAULT_LIMITS = {
        ParameterType.AIR_FUEL_RATIO: {"max_increase_rate": 0.1, "max_decrease_rate": 0.05, "absolute_min": 0.8, "absolute_max": 1.5, "unit": "ratio", "description": "AFR", "safety_critical": True},
        ParameterType.TEMPERATURE: {"max_increase_rate": 0.833, "max_decrease_rate": 0.5, "absolute_min": 0.0, "absolute_max": 1500.0, "unit": "C", "description": "Temp", "safety_critical": True},
        ParameterType.O2_SETPOINT: {"max_increase_rate": 0.5, "max_decrease_rate": 0.5, "absolute_min": 1.0, "absolute_max": 15.0, "unit": "%", "description": "O2", "safety_critical": True},
        ParameterType.CO_SETPOINT: {"max_increase_rate": 50.0, "max_decrease_rate": 50.0, "absolute_min": 0.0, "absolute_max": 400.0, "unit": "ppm", "description": "CO", "safety_critical": False},
        ParameterType.FUEL_FLOW: {"max_increase_rate": 5.0, "max_decrease_rate": 5.0, "absolute_min": 0.0, "absolute_max": 100.0, "unit": "%", "description": "Fuel", "safety_critical": True},
        ParameterType.AIR_FLOW: {"max_increase_rate": 10.0, "max_decrease_rate": 10.0, "absolute_min": 0.0, "absolute_max": 100.0, "unit": "%", "description": "Air", "safety_critical": True},
        ParameterType.DAMPER_POSITION: {"max_increase_rate": 5.0, "max_decrease_rate": 5.0, "absolute_min": 0.0, "absolute_max": 100.0, "unit": "%", "description": "Damper", "safety_critical": False},
        ParameterType.VALVE_POSITION: {"max_increase_rate": 10.0, "max_decrease_rate": 10.0, "absolute_min": 0.0, "absolute_max": 100.0, "unit": "%", "description": "Valve", "safety_critical": False},
        ParameterType.PRESSURE: {"max_increase_rate": 1000.0, "max_decrease_rate": 1000.0, "absolute_min": -5000.0, "absolute_max": 50000.0, "unit": "Pa", "description": "Pressure", "safety_critical": True},
        ParameterType.LOAD: {"max_increase_rate": 2.0, "max_decrease_rate": 2.0, "absolute_min": 0.0, "absolute_max": 100.0, "unit": "%", "description": "Load", "safety_critical": False}
    }

    def __init__(self, config: Optional[VelocityLimiterConfig] = None):
        self.config = config or VelocityLimiterConfig()
        self._limits: Dict[ParameterType, VelocityLimit] = {}
        self._states: Dict[ParameterType, ParameterState] = {}
        self._history = {pt: deque(maxlen=self.config.history_size) for pt in ParameterType}
        self._violations: List[VelocityViolation] = []
        self._violations_count = {pt: 0 for pt in ParameterType}
        self._initialize_limits()

    def _initialize_limits(self):
        for pt, d in self.DEFAULT_LIMITS.items():
            mir, mdr = d["max_increase_rate"], d["max_decrease_rate"]
            amin, amax = d["absolute_min"], d["absolute_max"]
            if pt == ParameterType.AIR_FUEL_RATIO:
                mir, mdr = self.config.afr_max_increase_rate, self.config.afr_max_decrease_rate
                amin, amax = self.config.afr_min, self.config.afr_max
            elif pt == ParameterType.TEMPERATURE:
                mir, mdr = self.config.temp_max_increase_rate / 60.0, self.config.temp_max_decrease_rate / 60.0
                amin, amax = self.config.temp_min, self.config.temp_max
            elif pt == ParameterType.O2_SETPOINT:
                mir = mdr = self.config.o2_max_rate
                amin, amax = self.config.o2_min, self.config.o2_max
            elif pt == ParameterType.CO_SETPOINT:
                mir = mdr = self.config.co_max_rate
                amin, amax = self.config.co_min, self.config.co_max
            self._limits[pt] = VelocityLimit(pt, mir, mdr, amin, amax, d["unit"], d["description"], d["safety_critical"])


    def limit_velocity(self, inp: VelocityLimiterInput) -> VelocityLimiterOutput:
        pt, limit = inp.parameter_type, self._limits[inp.parameter_type]
        ps = self._states.get(pt)
        pv = ps.current_value if ps else inp.current_value
        pts = ps.timestamp if ps else inp.timestamp
        dt, tv = max(inp.timestamp - pts, 0.001), inp.target_value

        if inp.freeze_output:
            lv, st, rl, ar = pv, VelocityLimitStatus.FROZEN, False, 0.0
        elif inp.emergency_bypass:
            lv = max(limit.absolute_min, min(tv, limit.absolute_max))
            st, rl, ar = VelocityLimitStatus.EMERGENCY_BYPASS, False, (lv - pv) / dt
        else:
            lv, st, rl, ar = self._apply_rate_limit(pv, tv, dt, limit, inp.max_rate_override)

        sa = False
        if self.config.enable_safety_bounds:
            cl = max(limit.absolute_min, min(lv, limit.absolute_max))
            if cl != lv:
                sa, st, lv = True, VelocityLimitStatus.SAFETY_LIMITED, cl

        d = tv - pv
        dr = RampDirection.STABLE if abs(d) < self.config.deadband else (RampDirection.INCREASING if d > 0 else RampDirection.DECREASING)
        tc = abs(tv - pv)
        prog = min(100.0, (abs(lv - pv) / tc) * 100) if tc > self.config.target_tolerance else 100.0
        alr = inp.max_rate_override if inp.max_rate_override else (limit.max_increase_rate if dr == RampDirection.INCREASING else limit.max_decrease_rate if dr == RampDirection.DECREASING else 0.0)
        rem = abs(tv - lv)
        ttt = rem / alr if alr > 0 and rem > self.config.target_tolerance else 0.0
        iat = abs(lv - tv) <= self.config.target_tolerance

        vd, vdt = False, None
        if rl or sa:
            bt = SafetyBoundType.ABSOLUTE_MAX if tv > limit.absolute_max else SafetyBoundType.ABSOLUTE_MIN if tv < limit.absolute_min else SafetyBoundType.RATE_MAX
            v = self._record_violation(pt, inp.timestamp, tv, lv, ar, alr, bt)
            vd, vdt = True, v.description

        self._states[pt] = ParameterState(pt, lv, pv, tv, inp.timestamp, pts, ar, st, dr, ttt, iat, self._violations_count[pt])
        self._history[pt].append({"timestamp": inp.timestamp, "value": lv, "target": tv, "rate": ar, "status": st.value})
        ph = hashlib.sha256(str({"pt": pt.value, "lv": lv, "ar": ar, "ts": inp.timestamp}).encode()).hexdigest()

        return VelocityLimiterOutput(limited_value=round(lv, 6), requested_value=tv, previous_value=pv, actual_rate=round(ar, 6), allowed_rate=round(alr, 6), rate_limited=rl, status=st, direction=dr, progress_percent=round(prog, 2), time_to_target_sec=round(ttt, 2), is_at_target=iat, safety_bounds_applied=sa, violation_detected=vd, violation_details=vdt, provenance_hash=ph)


    def _apply_rate_limit(self, pv, tv, dt, lim, ro):
        d = tv - pv
        if abs(d) < self.config.deadband:
            return pv, VelocityLimitStatus.NORMAL, False, 0.0
        mr = ro if ro else (lim.max_increase_rate if d > 0 else lim.max_decrease_rate)
        mc = mr * dt
        if abs(d) <= mc:
            return tv, VelocityLimitStatus.NORMAL, False, d / dt
        lv = pv + mc * (1 if d > 0 else -1)
        return lv, VelocityLimitStatus.RATE_LIMITED, True, (lv - pv) / dt

    def _record_violation(self, pt, ts, rv, av, ar, alr, bt):
        desc = f"{pt.value}: Requested {rv:.4f}, allowed {av:.4f}"
        v = VelocityViolation(pt, ts, rv, av, abs(rv - av), abs(ar) - alr, bt, desc)
        self._violations.append(v)
        self._violations_count[pt] += 1
        return v

    def limit_multiple(self, inputs):
        return [self.limit_velocity(i) for i in inputs]

    def get_parameter_state(self, pt):
        return self._states.get(pt)

    def get_all_states(self):
        return dict(self._states)

    def get_summary(self):
        t = len(self._states)
        l = sum(1 for s in self._states.values() if s.status == VelocityLimitStatus.RATE_LIMITED)
        a = sum(1 for s in self._states.values() if s.is_at_target)
        vbt = {pt.value: c for pt, c in self._violations_count.items() if c > 0}
        sbp = {pt.value: s.status.value for pt, s in self._states.items()}
        lts = max((s.timestamp for s in self._states.values()), default=0.0)
        ph = hashlib.sha256(f"{t}{l}{a}{vbt}".encode()).hexdigest()
        return VelocityLimiterSummary(total_parameters=t, actively_limited=l, at_target=a, violations_total=sum(self._violations_count.values()), violations_by_type=vbt, status_by_parameter=sbp, last_update_timestamp=lts, provenance_hash=ph)

    def get_violations(self, pt=None, since=None):
        v = self._violations
        if pt: v = [x for x in v if x.parameter_type == pt]
        if since: v = [x for x in v if x.timestamp >= since]
        return v

    def get_history(self, pt, limit=None):
        h = list(self._history.get(pt, []))
        return h[-limit:] if limit else h

    def reset_all(self):
        self._states.clear()
        for pt in ParameterType: self._history[pt].clear()
        self._violations.clear()
        self._violations_count = {pt: 0 for pt in ParameterType}

    def set_limit(self, pt, mir=None, mdr=None, amin=None, amax=None):
        lim = self._limits[pt]
        if mir is not None: lim.max_increase_rate = mir
        if mdr is not None: lim.max_decrease_rate = mdr
        if amin is not None: lim.absolute_min = amin
        if amax is not None: lim.absolute_max = amax

    def get_limit(self, pt):
        return self._limits.get(pt)


def create_default_limiter():
    return VelocityLimiter()

def create_conservative_limiter():
    return VelocityLimiter(VelocityLimiterConfig(afr_max_increase_rate=0.05, afr_max_decrease_rate=0.025, temp_max_increase_rate=25.0, temp_max_decrease_rate=15.0, o2_max_rate=0.25, co_max_rate=25.0))

def create_aggressive_limiter():
    return VelocityLimiter(VelocityLimiterConfig(afr_max_increase_rate=0.2, afr_max_decrease_rate=0.1, temp_max_increase_rate=100.0, temp_max_decrease_rate=60.0, o2_max_rate=1.0, co_max_rate=100.0))
