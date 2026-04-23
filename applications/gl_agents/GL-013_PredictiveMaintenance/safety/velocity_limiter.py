# -*- coding: utf-8 -*-
import hashlib, logging, math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Deque, Tuple

logger = logging.getLogger(__name__)

class VelocityViolationType(Enum):
    NONE = "none"
    MAX_DELTA_EXCEEDED = "max_delta_exceeded"
    RAMP_RATE_EXCEEDED = "ramp_rate_exceeded"
    SMOOTHING_APPLIED = "smoothing_applied"
    COMBINED = "combined"

class ConstraintAction(Enum):
    PASSTHROUGH = "passthrough"
    CLAMPED = "clamped"
    SMOOTHED = "smoothed"
    RAMPED = "ramped"
    HELD = "held"

@dataclass
class VelocityConfig:
    max_delta_per_second: float = 0.1
    smoothing_window: float = 5.0
    ramp_rate: float = 0.05
    enable_clamping: bool = True
    enable_smoothing: bool = True
    enable_ramping: bool = True
    min_value: float = 0.0
    max_value: float = 1.0
    history_size: int = 100
    violation_threshold: int = 5

@dataclass
class VelocityCheckResult:
    original_value: float
    constrained_value: float
    was_constrained: bool
    violation_type: VelocityViolationType
    action_taken: ConstraintAction
    delta_requested: float
    delta_allowed: float
    delta_per_second: float
    smoothed_value: Optional[float] = None
    ramp_target: Optional[float] = None
    constraint_ratio: float = 1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    elapsed_seconds: float = 0.0
    provenance_hash: str = ""

@dataclass
class VelocityState:
    asset_id: str
    metric_name: str
    current_value: float = 0.0
    previous_value: float = 0.0
    ema_value: float = 0.0
    ramp_target: float = 0.0
    last_update: Optional[datetime] = None
    history: Deque[Tuple[datetime, float, float]] = field(default_factory=lambda: deque(maxlen=100))
    violation_count: int = 0

class VelocityLimiter:
    def __init__(self, config=None):
        self.config = config or VelocityConfig()
        self._states = {}
        self._violation_callbacks = []

    def _get_state_key(self, asset_id, metric_name):
        return f"{asset_id}::{metric_name}"

    def _get_or_create_state(self, asset_id, metric_name, initial_value=0.0):
        key = self._get_state_key(asset_id, metric_name)
        if key not in self._states:
            self._states[key] = VelocityState(asset_id=asset_id, metric_name=metric_name, current_value=initial_value, previous_value=initial_value, ema_value=initial_value, ramp_target=initial_value, history=deque(maxlen=self.config.history_size))
        return self._states[key]

    def apply(self, asset_id, metric_name, new_value, timestamp=None):
        timestamp = timestamp or datetime.now(timezone.utc)
        state = self._get_or_create_state(asset_id, metric_name, new_value)
        if state.last_update is None:
            state.current_value = state.previous_value = state.ema_value = state.ramp_target = new_value
            state.last_update = timestamp
            state.history.append((timestamp, new_value, new_value))
            return VelocityCheckResult(original_value=new_value, constrained_value=new_value, was_constrained=False, violation_type=VelocityViolationType.NONE, action_taken=ConstraintAction.PASSTHROUGH, delta_requested=0.0, delta_allowed=0.0, delta_per_second=0.0, smoothed_value=new_value, ramp_target=new_value, timestamp=timestamp, elapsed_seconds=0.0)
        elapsed = max((timestamp - state.last_update).total_seconds(), 0.001)
        previous_value = state.current_value
        delta_requested = new_value - previous_value
        delta_per_second = delta_requested / elapsed
        constrained_value, was_constrained = new_value, False
        violation_type, action_taken = VelocityViolationType.NONE, ConstraintAction.PASSTHROUGH
        smoothed_value, ramp_target = None, state.ramp_target
        if self.config.enable_clamping:
            max_delta = self.config.max_delta_per_second * elapsed
            if abs(delta_requested) > max_delta:
                constrained_value = previous_value + (max_delta if delta_requested > 0 else -max_delta)
                was_constrained, violation_type, action_taken = True, VelocityViolationType.MAX_DELTA_EXCEEDED, ConstraintAction.CLAMPED
        if self.config.enable_smoothing:
            alpha = min(max(1.0 - math.exp(-elapsed / self.config.smoothing_window), 0.01), 1.0)
            smoothed_value = alpha * constrained_value + (1 - alpha) * state.ema_value
            if abs(smoothed_value - previous_value) < abs(constrained_value - previous_value):
                if not was_constrained: action_taken, violation_type = ConstraintAction.SMOOTHED, VelocityViolationType.SMOOTHING_APPLIED
                constrained_value, was_constrained = smoothed_value, True
        if self.config.enable_ramping:
            ramp_target = new_value
            max_ramp_step = self.config.ramp_rate * elapsed
            current_ramp_delta = constrained_value - previous_value
            if abs(current_ramp_delta) > max_ramp_step:
                constrained_value = previous_value + (max_ramp_step if current_ramp_delta > 0 else -max_ramp_step)
                if not was_constrained: action_taken, violation_type = ConstraintAction.RAMPED, VelocityViolationType.RAMP_RATE_EXCEEDED
                elif violation_type != VelocityViolationType.NONE: violation_type = VelocityViolationType.COMBINED
                was_constrained = True
        constrained_value = max(self.config.min_value, min(self.config.max_value, constrained_value))
        delta_allowed = constrained_value - previous_value
        constraint_ratio = min(1.0, abs(delta_allowed) / abs(delta_requested)) if abs(delta_requested) > 1e-10 else 1.0
        state.previous_value, state.current_value = previous_value, constrained_value
        state.ema_value = smoothed_value if smoothed_value else constrained_value
        state.ramp_target, state.last_update = ramp_target, timestamp
        state.history.append((timestamp, new_value, constrained_value))
        if was_constrained and violation_type in (VelocityViolationType.MAX_DELTA_EXCEEDED, VelocityViolationType.RAMP_RATE_EXCEEDED):
            state.violation_count += 1
            if state.violation_count >= self.config.violation_threshold: self._trigger_violation_alert(asset_id, metric_name, state)
        else: state.violation_count = 0
        return VelocityCheckResult(original_value=new_value, constrained_value=constrained_value, was_constrained=was_constrained, violation_type=violation_type, action_taken=action_taken, delta_requested=delta_requested, delta_allowed=delta_allowed, delta_per_second=delta_per_second, smoothed_value=smoothed_value, ramp_target=ramp_target, constraint_ratio=constraint_ratio, timestamp=timestamp, elapsed_seconds=elapsed)

    def apply_batch(self, asset_id, values, timestamp=None):
        timestamp = timestamp or datetime.now(timezone.utc)
        return {m: self.apply(asset_id, m, v, timestamp) for m, v in values.items()}

    def get_state(self, asset_id, metric_name):
        return self._states.get(self._get_state_key(asset_id, metric_name))

    def get_history(self, asset_id, metric_name, limit=None):
        state = self.get_state(asset_id, metric_name)
        return list(state.history)[-limit:] if state and limit else list(state.history) if state else []

    def reset(self, asset_id=None, metric_name=None):
        if asset_id is None:
            count = len(self._states)
            self._states.clear()
            logger.info(f'Reset all {count} velocity states')
            return count
        if metric_name is None:
            prefix = f'{asset_id}::'
            keys_to_remove = [k for k in self._states if k.startswith(prefix)]
            for k in keys_to_remove:
                del self._states[k]
            logger.info(f'Reset {len(keys_to_remove)} states for asset {asset_id}')
            return len(keys_to_remove)
        key = self._get_state_key(asset_id, metric_name)
        if key in self._states:
            del self._states[key]
            logger.info(f'Reset state for {asset_id}::{metric_name}')
            return 1
        return 0

    def register_violation_callback(self, callback):
        if callable(callback):
            self._violation_callbacks.append(callback)
            logger.debug(f'Registered violation callback')

    def _trigger_violation_alert(self, asset_id, metric_name, state):
        alert_data = {'asset_id': asset_id, 'metric_name': metric_name, 'violation_count': state.violation_count, 'current_value': state.current_value, 'previous_value': state.previous_value, 'threshold': self.config.violation_threshold, 'timestamp': datetime.now(timezone.utc).isoformat()}
        logger.warning(f'Velocity violation threshold reached: {asset_id}::{metric_name} ({state.violation_count} violations)')
        for callback in self._violation_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                logger.error(f'Violation callback failed: {e}')

    def get_statistics(self, asset_id=None):
        if asset_id:
            prefix = f'{asset_id}::'
            states = {k: v for k, v in self._states.items() if k.startswith(prefix)}
        else:
            states = self._states
        total_constraints = sum(1 for s in states.values() if s.violation_count > 0)
        total_history = sum(len(s.history) for s in states.values())
        return {'total_tracked_metrics': len(states), 'metrics_with_violations': total_constraints, 'total_history_entries': total_history, 'config': {'max_delta_per_second': self.config.max_delta_per_second, 'smoothing_window': self.config.smoothing_window, 'ramp_rate': self.config.ramp_rate, 'violation_threshold': self.config.violation_threshold}}


def check_velocity(current_value, previous_value, elapsed_seconds, max_delta_per_second=0.1):
    if elapsed_seconds <= 0:
        elapsed_seconds = 0.001
    delta = current_value - previous_value
    delta_per_second = delta / elapsed_seconds
    max_allowed_delta = max_delta_per_second * elapsed_seconds
    is_violation = abs(delta) > max_allowed_delta
    return is_violation, delta_per_second, max_allowed_delta
