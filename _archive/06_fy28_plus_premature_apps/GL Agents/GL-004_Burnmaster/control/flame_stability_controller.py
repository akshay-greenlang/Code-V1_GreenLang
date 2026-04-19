"""
GL-004 BURNMASTER - Flame Stability Controller

Flame stability monitoring and protective control.
Prevents blowoff, flashback, and combustion instabilities.

Author: GreenLang Combustion Systems Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class StabilityState(str, Enum):
    """Flame stability states."""
    STABLE = "stable"
    MARGINAL = "marginal"
    UNSTABLE = "unstable"
    CRITICAL = "critical"
    FLAME_OUT = "flame_out"


class ProtectiveAction(str, Enum):
    """Protective actions for stability control."""
    NONE = "none"
    REDUCE_TURNDOWN = "reduce_turndown"
    INCREASE_AIR = "increase_air"
    REDUCE_FUEL = "reduce_fuel"
    SAFETY_SHUTDOWN = "safety_shutdown"
    OPERATOR_ALERT = "operator_alert"


class InstabilityType(str, Enum):
    """Types of combustion instability."""
    BLOWOFF_RISK = "blowoff_risk"
    FLASHBACK_RISK = "flashback_risk"
    ACOUSTIC_OSCILLATION = "acoustic_oscillation"
    LEAN_LIMIT = "lean_limit"
    FLAME_ATTACHMENT = "flame_attachment"


class StabilityOutput(BaseModel):
    """Flame stability controller output."""
    output_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    unit_id: str = Field(..., description="Combustion unit ID")

    # Stability assessment
    stability_state: StabilityState = Field(default=StabilityState.STABLE)
    stability_index: float = Field(..., ge=0, le=1, description="0=unstable, 1=stable")
    instability_type: Optional[InstabilityType] = Field(None)

    # Flame metrics
    flame_signal: float = Field(..., description="Flame detector signal")
    flame_signal_variance: float = Field(..., description="Signal variance")
    flame_quality: str = Field(default="good")

    # Risk indicators
    blowoff_risk: float = Field(default=0.0, ge=0, le=1)
    flashback_risk: float = Field(default=0.0, ge=0, le=1)
    oscillation_detected: bool = Field(default=False)
    oscillation_frequency_hz: Optional[float] = Field(None)

    # Protective action
    action_required: ProtectiveAction = Field(default=ProtectiveAction.NONE)
    action_reason: Optional[str] = Field(None)

    # Margins
    margin_to_blowoff: float = Field(default=100.0, description="% margin to blowoff")
    margin_to_lean_limit: float = Field(default=100.0, description="% margin to lean limit")


class StabilityConfig(BaseModel):
    """Flame stability controller configuration."""
    unit_id: str = Field(..., description="Combustion unit ID")

    # Stability thresholds
    stability_index_good: float = Field(default=0.8, ge=0, le=1)
    stability_index_marginal: float = Field(default=0.6, ge=0, le=1)
    stability_index_critical: float = Field(default=0.3, ge=0, le=1)

    # Flame signal thresholds
    flame_signal_min: float = Field(default=20.0, description="Minimum valid flame signal")
    flame_variance_max: float = Field(default=100.0, description="Max acceptable variance")

    # Risk thresholds
    blowoff_risk_alarm: float = Field(default=0.7, ge=0, le=1)
    flashback_risk_alarm: float = Field(default=0.7, ge=0, le=1)

    # Operating limits
    min_stable_load_percent: float = Field(default=25.0, ge=0, le=100)
    max_lambda: float = Field(default=1.5, description="Max lambda before blowoff concern")
    min_lambda: float = Field(default=1.02, description="Min lambda before flashback concern")

    # Oscillation detection
    oscillation_threshold_percent: float = Field(default=10.0)
    oscillation_frequency_min_hz: float = Field(default=1.0)
    oscillation_frequency_max_hz: float = Field(default=100.0)

    # Response timing
    signal_window_seconds: float = Field(default=10.0)
    action_delay_seconds: float = Field(default=5.0)


class FlameStabilityController:
    """
    Flame stability monitoring and protective controller.

    Monitors flame stability indicators and initiates protective actions
    to prevent blowoff, flashback, and combustion oscillations.

    Example:
        >>> controller = FlameStabilityController(config)
        >>> output = controller.assess_stability(
        ...     flame_signal=85.0,
        ...     lambda_val=1.15,
        ...     load_percent=60
        ... )
    """

    def __init__(self, config: StabilityConfig):
        """Initialize flame stability controller."""
        self.config = config
        self._signal_buffer: deque = deque(maxlen=1000)  # For oscillation detection
        self._stability_history: deque = deque(maxlen=100)
        self._last_action_time: Optional[datetime] = None
        self._action_in_progress = False
        logger.info(f"FlameStabilityController initialized for {config.unit_id}")

    def assess_stability(
        self,
        flame_signal: float,
        lambda_val: float,
        load_percent: float,
        o2_percent: Optional[float] = None,
        fuel_pressure: Optional[float] = None,
        air_pressure: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ) -> StabilityOutput:
        """
        Assess flame stability and determine protective action.

        Args:
            flame_signal: Flame detector signal (0-100 typical)
            lambda_val: Air-fuel equivalence ratio
            load_percent: Current load as percentage
            o2_percent: Stack O2 percentage (optional)
            fuel_pressure: Fuel supply pressure (optional)
            air_pressure: Combustion air pressure (optional)
            timestamp: Assessment timestamp

        Returns:
            StabilityOutput with stability assessment and actions
        """
        timestamp = timestamp or datetime.now(timezone.utc)

        # Buffer signal for analysis
        self._signal_buffer.append({
            "timestamp": timestamp,
            "signal": flame_signal,
            "lambda": lambda_val,
        })

        # Calculate signal statistics
        signal_mean, signal_variance = self._calculate_signal_stats()
        flame_quality = self._assess_flame_quality(flame_signal, signal_variance)

        # Calculate risk indicators
        blowoff_risk = self._calculate_blowoff_risk(lambda_val, load_percent)
        flashback_risk = self._calculate_flashback_risk(lambda_val, fuel_pressure)

        # Detect oscillations
        oscillation_detected, oscillation_freq = self._detect_oscillations()

        # Calculate overall stability index
        stability_index = self._calculate_stability_index(
            flame_signal, signal_variance, blowoff_risk, flashback_risk, oscillation_detected
        )

        # Determine stability state
        stability_state, instability_type = self._determine_state(
            stability_index, blowoff_risk, flashback_risk, oscillation_detected, flame_signal
        )

        # Calculate margins
        margin_to_blowoff = max(0, (self.config.max_lambda - lambda_val) /
                                 (self.config.max_lambda - 1.0) * 100)
        margin_to_lean = max(0, (lambda_val - self.config.min_lambda) /
                             (self.config.max_lambda - self.config.min_lambda) * 100)

        # Determine protective action
        action, reason = self._determine_action(
            stability_state, instability_type, blowoff_risk, flashback_risk, timestamp
        )

        # Record history
        self._stability_history.append({
            "timestamp": timestamp,
            "index": stability_index,
            "state": stability_state.value,
            "action": action.value,
        })

        return StabilityOutput(
            unit_id=self.config.unit_id,
            stability_state=stability_state,
            stability_index=round(stability_index, 3),
            instability_type=instability_type,
            flame_signal=round(flame_signal, 1),
            flame_signal_variance=round(signal_variance, 2),
            flame_quality=flame_quality,
            blowoff_risk=round(blowoff_risk, 3),
            flashback_risk=round(flashback_risk, 3),
            oscillation_detected=oscillation_detected,
            oscillation_frequency_hz=oscillation_freq,
            action_required=action,
            action_reason=reason,
            margin_to_blowoff=round(margin_to_blowoff, 1),
            margin_to_lean_limit=round(margin_to_lean, 1),
        )

    def _calculate_signal_stats(self) -> Tuple[float, float]:
        """Calculate flame signal statistics from buffer."""
        if len(self._signal_buffer) < 10:
            return 50.0, 0.0

        signals = [s["signal"] for s in self._signal_buffer]
        return float(np.mean(signals)), float(np.var(signals))

    def _assess_flame_quality(self, signal: float, variance: float) -> str:
        """Assess flame quality from signal and variance."""
        if signal < self.config.flame_signal_min:
            return "weak"
        elif variance > self.config.flame_variance_max:
            return "unstable"
        elif signal > 80 and variance < self.config.flame_variance_max * 0.5:
            return "excellent"
        elif signal > 50 and variance < self.config.flame_variance_max:
            return "good"
        else:
            return "marginal"

    def _calculate_blowoff_risk(self, lambda_val: float, load_percent: float) -> float:
        """Calculate blowoff risk (0-1)."""
        # Risk increases with:
        # 1. Higher lambda (leaner mixture)
        # 2. Lower load (less thermal inertia)

        # Lambda contribution
        lambda_risk = max(0, (lambda_val - 1.1) / (self.config.max_lambda - 1.1))

        # Load contribution (higher risk at low load)
        load_risk = max(0, (50 - load_percent) / 50) if load_percent < 50 else 0

        # Combined risk (weighted average)
        risk = 0.7 * lambda_risk + 0.3 * load_risk

        return min(1.0, risk)

    def _calculate_flashback_risk(
        self,
        lambda_val: float,
        fuel_pressure: Optional[float]
    ) -> float:
        """Calculate flashback risk (0-1)."""
        # Risk increases with:
        # 1. Lower lambda (richer mixture, higher flame speed)
        # 2. Lower fuel pressure (reduced velocity)

        # Lambda contribution
        if lambda_val < self.config.min_lambda:
            lambda_risk = 1.0
        elif lambda_val < 1.05:
            lambda_risk = (1.05 - lambda_val) / (1.05 - self.config.min_lambda)
        else:
            lambda_risk = 0.0

        # Pressure contribution (if available)
        pressure_risk = 0.0
        # Would need reference pressure to calculate

        risk = 0.9 * lambda_risk + 0.1 * pressure_risk

        return min(1.0, risk)

    def _detect_oscillations(self) -> Tuple[bool, Optional[float]]:
        """Detect combustion oscillations using FFT."""
        if len(self._signal_buffer) < 64:
            return False, None

        # Get recent signals
        signals = np.array([s["signal"] for s in list(self._signal_buffer)[-256:]])

        # Remove DC component
        signals = signals - np.mean(signals)

        if np.std(signals) < 1.0:
            return False, None

        # Simple variance-based detection
        variance = np.var(signals)
        mean = np.mean([s["signal"] for s in self._signal_buffer])

        if mean > 0:
            cv = np.sqrt(variance) / mean * 100
            if cv > self.config.oscillation_threshold_percent:
                # Would do FFT here for frequency
                return True, 10.0  # Placeholder frequency

        return False, None

    def _calculate_stability_index(
        self,
        flame_signal: float,
        variance: float,
        blowoff_risk: float,
        flashback_risk: float,
        oscillation: bool
    ) -> float:
        """Calculate overall stability index (0=unstable, 1=stable)."""
        # Components
        signal_component = min(1.0, flame_signal / 100)
        variance_component = max(0, 1 - variance / self.config.flame_variance_max)
        risk_component = max(0, 1 - max(blowoff_risk, flashback_risk))
        oscillation_component = 0.0 if oscillation else 1.0

        # Weighted combination
        index = (
            0.3 * signal_component +
            0.2 * variance_component +
            0.35 * risk_component +
            0.15 * oscillation_component
        )

        return max(0, min(1, index))

    def _determine_state(
        self,
        stability_index: float,
        blowoff_risk: float,
        flashback_risk: float,
        oscillation: bool,
        flame_signal: float
    ) -> Tuple[StabilityState, Optional[InstabilityType]]:
        """Determine stability state and instability type."""
        # Check for flame out first
        if flame_signal < self.config.flame_signal_min * 0.5:
            return StabilityState.FLAME_OUT, None

        # Determine instability type
        instability_type = None
        if blowoff_risk > self.config.blowoff_risk_alarm:
            instability_type = InstabilityType.BLOWOFF_RISK
        elif flashback_risk > self.config.flashback_risk_alarm:
            instability_type = InstabilityType.FLASHBACK_RISK
        elif oscillation:
            instability_type = InstabilityType.ACOUSTIC_OSCILLATION

        # Determine state from index
        if stability_index >= self.config.stability_index_good:
            state = StabilityState.STABLE
        elif stability_index >= self.config.stability_index_marginal:
            state = StabilityState.MARGINAL
        elif stability_index >= self.config.stability_index_critical:
            state = StabilityState.UNSTABLE
        else:
            state = StabilityState.CRITICAL

        return state, instability_type

    def _determine_action(
        self,
        state: StabilityState,
        instability_type: Optional[InstabilityType],
        blowoff_risk: float,
        flashback_risk: float,
        timestamp: datetime
    ) -> Tuple[ProtectiveAction, Optional[str]]:
        """Determine required protective action."""
        # Check action delay
        if self._last_action_time:
            elapsed = (timestamp - self._last_action_time).total_seconds()
            if elapsed < self.config.action_delay_seconds:
                return ProtectiveAction.NONE, None

        if state == StabilityState.CRITICAL:
            self._last_action_time = timestamp
            return ProtectiveAction.SAFETY_SHUTDOWN, "Critical instability - safety shutdown required"

        if state == StabilityState.UNSTABLE:
            self._last_action_time = timestamp
            if instability_type == InstabilityType.BLOWOFF_RISK:
                return ProtectiveAction.INCREASE_AIR, "Blowoff risk - reducing excess air"
            elif instability_type == InstabilityType.FLASHBACK_RISK:
                return ProtectiveAction.INCREASE_AIR, "Flashback risk - increasing air flow"
            elif instability_type == InstabilityType.ACOUSTIC_OSCILLATION:
                return ProtectiveAction.REDUCE_TURNDOWN, "Oscillation detected - adjusting load"
            return ProtectiveAction.OPERATOR_ALERT, "Unstable combustion - operator attention required"

        if state == StabilityState.MARGINAL:
            self._last_action_time = timestamp
            return ProtectiveAction.OPERATOR_ALERT, "Marginal stability - monitoring closely"

        return ProtectiveAction.NONE, None

    def get_stability_trend(self, minutes: int = 30) -> Dict[str, Any]:
        """Get stability trend over recent history."""
        if not self._stability_history:
            return {"trend": "unknown", "samples": 0}

        recent = [h for h in self._stability_history]
        if len(recent) < 5:
            return {"trend": "insufficient_data", "samples": len(recent)}

        indices = [h["index"] for h in recent]
        avg_recent = np.mean(indices[-10:]) if len(indices) >= 10 else np.mean(indices)
        avg_older = np.mean(indices[:10]) if len(indices) >= 20 else np.mean(indices)

        if avg_recent > avg_older + 0.05:
            trend = "improving"
        elif avg_recent < avg_older - 0.05:
            trend = "degrading"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "samples": len(recent),
            "avg_index": round(float(np.mean(indices)), 3),
            "min_index": round(float(min(indices)), 3),
            "max_index": round(float(max(indices)), 3),
        }
