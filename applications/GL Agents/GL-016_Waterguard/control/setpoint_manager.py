# -*- coding: utf-8 -*-
"""
GL-016 Waterguard Setpoint Manager Module

Optimal setpoint calculation with constraints, validation, and history tracking.

Author: GreenLang Control Systems Team
Version: 1.0.0
"""

from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Tuple
from pydantic import BaseModel, Field, validator
from datetime import datetime, timedelta
import hashlib
import logging
import threading

logger = logging.getLogger(__name__)


class SetpointConstraint(BaseModel):
    """Constraint on setpoint values."""
    parameter_name: str = Field(...)
    min_value: float = Field(...)
    max_value: float = Field(...)
    rate_limit_per_minute: float = Field(default=10.0, ge=0)
    unit: str = Field(default="")
    description: str = Field(default="")


class SetpointHistory(BaseModel):
    """Historical setpoint record."""
    parameter_name: str = Field(...)
    value: float = Field(...)
    timestamp: datetime = Field(default_factory=datetime.now)
    source: str = Field(default="calculated")
    reason: str = Field(default="")
    provenance_hash: str = Field(...)


class OptimizationResult(BaseModel):
    """Result of setpoint optimization."""
    parameter_name: str = Field(...)
    current_value: float = Field(...)
    recommended_value: float = Field(...)
    applied_value: float = Field(...)
    constrained: bool = Field(default=False)
    rate_limited: bool = Field(default=False)
    optimization_score: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=datetime.now)
    provenance_hash: str = Field(...)


class SetpointConfig(BaseModel):
    """Setpoint manager configuration."""
    constraints: List[SetpointConstraint] = Field(default_factory=list)
    optimization_interval_seconds: float = Field(default=60.0)
    history_retention_hours: float = Field(default=168.0)
    enable_auto_optimization: bool = Field(default=True)


class SetpointOutput(BaseModel):
    """Setpoint manager output."""
    setpoints: Dict[str, float] = Field(default_factory=dict)
    optimizations: List[OptimizationResult] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)
    provenance_hash: str = Field(...)


class SetpointManager:
    """Setpoint calculation and management with optimization."""

    def __init__(self, config: SetpointConfig, alert_callback: Optional[Callable] = None):
        self.config = config
        self._lock = threading.RLock()
        self._current_setpoints: Dict[str, float] = {}
        self._history: List[SetpointHistory] = []
        self._constraints = {c.parameter_name: c for c in config.constraints}
        self._last_optimization = datetime.now()
        self._alert_callback = alert_callback
        logger.info("SetpointManager initialized")

    def calculate_optimal_setpoint(self, parameter: str, process_data: Dict[str, float]) -> OptimizationResult:
        """Calculate optimal setpoint for a parameter."""
        with self._lock:
            constraint = self._constraints.get(parameter)
            current = self._current_setpoints.get(parameter, 0.0)

            # Simple optimization: use process data to determine optimal value
            # In production, this would use more sophisticated algorithms
            recommended = self._compute_recommendation(parameter, process_data, constraint)

            # Apply constraints
            constrained = False
            if constraint:
                if recommended < constraint.min_value:
                    recommended = constraint.min_value
                    constrained = True
                elif recommended > constraint.max_value:
                    recommended = constraint.max_value
                    constrained = True

            # Apply rate limiting
            rate_limited = False
            applied = recommended
            if constraint and abs(recommended - current) > constraint.rate_limit_per_minute:
                direction = 1 if recommended > current else -1
                applied = current + direction * constraint.rate_limit_per_minute
                rate_limited = True

            result = OptimizationResult(
                parameter_name=parameter,
                current_value=current,
                recommended_value=recommended,
                applied_value=applied,
                constrained=constrained,
                rate_limited=rate_limited,
                optimization_score=self._compute_score(parameter, applied, process_data),
                provenance_hash=hashlib.sha256(f"{parameter}:{applied}:{datetime.now()}".encode()).hexdigest()
            )

            return result

    def validate_setpoint(self, parameter: str, value: float) -> Tuple[bool, str]:
        """Validate a setpoint value against constraints."""
        constraint = self._constraints.get(parameter)
        if not constraint:
            return True, "No constraints defined"

        if value < constraint.min_value:
            return False, f"Value {value} below minimum {constraint.min_value}"
        if value > constraint.max_value:
            return False, f"Value {value} above maximum {constraint.max_value}"

        return True, "Valid"

    def apply_setpoint(self, parameter: str, value: float, source: str = "manual", reason: str = "") -> bool:
        """Apply a setpoint value."""
        with self._lock:
            valid, msg = self.validate_setpoint(parameter, value)
            if not valid:
                logger.warning(f"Setpoint validation failed: {msg}")
                return False

            # Apply rate limiting
            constraint = self._constraints.get(parameter)
            current = self._current_setpoints.get(parameter)
            if current is not None and constraint:
                max_change = constraint.rate_limit_per_minute
                if abs(value - current) > max_change:
                    direction = 1 if value > current else -1
                    value = current + direction * max_change
                    logger.info(f"Rate limited setpoint change for {parameter}")

            self._current_setpoints[parameter] = value

            # Record history
            history = SetpointHistory(
                parameter_name=parameter,
                value=value,
                source=source,
                reason=reason,
                provenance_hash=hashlib.sha256(f"{parameter}:{value}:{datetime.now()}".encode()).hexdigest()
            )
            self._history.append(history)
            self._trim_history()

            logger.info(f"Setpoint applied: {parameter}={value} ({source})")
            return True

    def get_setpoint(self, parameter: str) -> Optional[float]:
        """Get current setpoint value."""
        return self._current_setpoints.get(parameter)

    def get_all_setpoints(self) -> Dict[str, float]:
        """Get all current setpoints."""
        with self._lock:
            return dict(self._current_setpoints)

    def get_history(self, parameter: Optional[str] = None, hours: float = 24.0) -> List[SetpointHistory]:
        """Get setpoint history."""
        with self._lock:
            cutoff = datetime.now() - timedelta(hours=hours)
            history = [h for h in self._history if h.timestamp > cutoff]
            if parameter:
                history = [h for h in history if h.parameter_name == parameter]
            return history

    def apply_ramp_rate_limiting(self, parameter: str, target: float) -> float:
        """Apply ramp rate limiting to reach target value."""
        with self._lock:
            current = self._current_setpoints.get(parameter, target)
            constraint = self._constraints.get(parameter)

            if not constraint:
                return target

            max_change = constraint.rate_limit_per_minute
            if abs(target - current) <= max_change:
                return target

            direction = 1 if target > current else -1
            return current + direction * max_change

    def _compute_recommendation(self, parameter: str, process_data: Dict[str, float], constraint: Optional[SetpointConstraint]) -> float:
        """Compute recommended setpoint based on process data."""
        # Simple logic - in production, this would use optimization algorithms
        current = self._current_setpoints.get(parameter, 0.0)

        # Example: adjust based on efficiency metric
        efficiency = process_data.get("efficiency", 1.0)
        if efficiency < 0.95:
            adjustment = (0.95 - efficiency) * 10
            recommended = current + adjustment
        else:
            recommended = current

        return recommended

    def _compute_score(self, parameter: str, value: float, process_data: Dict[str, float]) -> float:
        """Compute optimization score."""
        # Simple scoring - higher is better
        constraint = self._constraints.get(parameter)
        if not constraint:
            return 0.5

        # Score based on position within range
        range_size = constraint.max_value - constraint.min_value
        if range_size <= 0:
            return 0.5

        normalized = (value - constraint.min_value) / range_size
        return normalized

    def _trim_history(self) -> None:
        """Trim old history entries."""
        cutoff = datetime.now() - timedelta(hours=self.config.history_retention_hours)
        self._history = [h for h in self._history if h.timestamp > cutoff]
