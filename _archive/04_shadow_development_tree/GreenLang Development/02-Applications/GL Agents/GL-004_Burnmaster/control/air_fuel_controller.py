"""
GL-004 BURNMASTER - Air-Fuel Ratio Controller

Supervisory air-fuel ratio control for combustion optimization.
Provides setpoint recommendations to existing DCS air/fuel ratio controls.

Author: GreenLang Combustion Systems Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ControlMode(str, Enum):
    """Air-fuel controller operating modes."""
    MANUAL = "manual"           # No automatic adjustments
    ADVISORY = "advisory"       # Recommendations only
    SUPERVISORY = "supervisory" # Setpoint adjustments via DCS
    AUTO = "auto"               # Full automatic control


class AFRStrategy(str, Enum):
    """Air-fuel ratio control strategies."""
    FIXED_RATIO = "fixed_ratio"       # Constant A/F ratio
    O2_TRIM = "o2_trim"               # Trim based on stack O2
    CROSS_LIMITING = "cross_limiting" # Air/fuel cross-limiting
    PARALLEL = "parallel"             # Parallel positioning
    CHARACTERIZATION = "characterization"  # Load-based characterization


class SetpointAdjustment(BaseModel):
    """Setpoint adjustment recommendation."""
    adjustment_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tag_name: str = Field(..., description="DCS tag to adjust")
    current_value: float = Field(..., description="Current setpoint value")
    recommended_value: float = Field(..., description="Recommended value")
    change: float = Field(..., description="Change amount")
    change_percent: float = Field(..., description="Change as percentage")
    reason: str = Field(..., description="Reason for adjustment")
    confidence: float = Field(default=0.9, ge=0, le=1)
    expected_impact: Dict[str, float] = Field(default_factory=dict)
    constraints_checked: List[str] = Field(default_factory=list)
    approved: bool = Field(default=False)
    applied: bool = Field(default=False)


class AFRControlOutput(BaseModel):
    """Air-fuel ratio controller output."""
    output_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    unit_id: str = Field(..., description="Combustion unit ID")

    # Current state
    current_afr: float = Field(..., description="Current air-fuel ratio")
    current_lambda: float = Field(..., description="Current lambda (equivalence)")
    current_excess_air_pct: float = Field(..., description="Current excess air %")
    measured_o2_pct: float = Field(..., description="Measured stack O2 %")

    # Targets
    target_o2_pct: float = Field(..., description="Target O2 %")
    target_lambda: float = Field(..., description="Target lambda")

    # Adjustments
    adjustments: List[SetpointAdjustment] = Field(default_factory=list)

    # Status
    mode: ControlMode = Field(default=ControlMode.ADVISORY)
    strategy: AFRStrategy = Field(default=AFRStrategy.O2_TRIM)
    is_at_target: bool = Field(default=False)
    deviation_from_target: float = Field(default=0.0)

    # Quality
    data_quality: str = Field(default="good")
    constraints_active: List[str] = Field(default_factory=list)


class AFRConfig(BaseModel):
    """Air-fuel ratio controller configuration."""
    unit_id: str = Field(..., description="Combustion unit ID")

    # Target settings
    target_o2_min: float = Field(default=2.0, ge=0, le=21)
    target_o2_max: float = Field(default=4.0, ge=0, le=21)
    target_o2_setpoint: float = Field(default=3.0, ge=0, le=21)

    # Lambda targets
    target_lambda_min: float = Field(default=1.05, ge=1.0)
    target_lambda_max: float = Field(default=1.25, le=2.0)

    # Deadband and rate limits
    o2_deadband: float = Field(default=0.2, ge=0)
    adjustment_rate_limit_per_min: float = Field(default=1.0, ge=0)
    max_single_adjustment: float = Field(default=0.5, ge=0)

    # Constraints
    co_limit_ppm: float = Field(default=100.0, ge=0)
    min_air_flow_pct: float = Field(default=20.0, ge=0, le=100)
    max_air_flow_pct: float = Field(default=100.0, ge=0, le=100)

    # Strategy
    strategy: AFRStrategy = Field(default=AFRStrategy.O2_TRIM)
    mode: ControlMode = Field(default=ControlMode.ADVISORY)


class AirFuelController:
    """
    Air-fuel ratio supervisory controller.

    Provides setpoint recommendations for DCS-level air/fuel ratio controls
    to optimize combustion efficiency while maintaining emissions compliance.

    Example:
        >>> controller = AirFuelController(config)
        >>> output = controller.calculate_adjustments(
        ...     current_o2=4.5,
        ...     current_air_flow=10000,
        ...     current_fuel_flow=500,
        ...     load_percent=75
        ... )
    """

    # Stoichiometric A/F ratios by fuel type (mass basis)
    STOICH_AFR: Dict[str, float] = {
        "natural_gas": 17.2,
        "methane": 17.2,
        "propane": 15.7,
        "fuel_oil": 14.7,
        "diesel": 14.5,
        "hydrogen": 34.3,
    }

    def __init__(self, config: AFRConfig):
        """Initialize air-fuel controller."""
        self.config = config
        self._last_adjustment_time: Optional[datetime] = None
        self._adjustment_history: List[SetpointAdjustment] = []
        self._integral_error: float = 0.0
        logger.info(f"AirFuelController initialized for {config.unit_id}")

    def calculate_lambda(
        self,
        air_flow: float,
        fuel_flow: float,
        fuel_type: str = "natural_gas"
    ) -> float:
        """Calculate lambda (equivalence ratio) from air and fuel flows."""
        stoich_afr = self.STOICH_AFR.get(fuel_type, 17.2)
        if fuel_flow <= 0:
            return 1.0
        actual_afr = air_flow / fuel_flow
        return actual_afr / stoich_afr

    def calculate_excess_air(self, lambda_val: float) -> float:
        """Calculate excess air percentage from lambda."""
        return (lambda_val - 1.0) * 100

    def calculate_o2_from_lambda(self, lambda_val: float) -> float:
        """Estimate O2 percentage from lambda (simplified)."""
        # O2 ~ 21 * (lambda - 1) / lambda for air
        if lambda_val <= 1.0:
            return 0.0
        return 21 * (lambda_val - 1) / lambda_val

    def calculate_lambda_from_o2(self, o2_percent: float) -> float:
        """Calculate lambda from measured O2 percentage."""
        # Lambda = 21 / (21 - O2)
        if o2_percent >= 21:
            return 10.0  # Very lean
        return 21 / (21 - o2_percent)

    def calculate_adjustments(
        self,
        current_o2: float,
        current_air_flow: float,
        current_fuel_flow: float,
        load_percent: float,
        measured_co_ppm: float = 0.0,
        fuel_type: str = "natural_gas",
    ) -> AFRControlOutput:
        """
        Calculate air-fuel ratio adjustments.

        Args:
            current_o2: Measured stack O2 percentage
            current_air_flow: Current air flow (units consistent with fuel)
            current_fuel_flow: Current fuel flow
            load_percent: Current load as percentage of max
            measured_co_ppm: Measured CO (for constraint checking)
            fuel_type: Type of fuel being burned

        Returns:
            AFRControlOutput with adjustments and status
        """
        now = datetime.now(timezone.utc)
        adjustments: List[SetpointAdjustment] = []
        constraints_active: List[str] = []

        # Calculate current operating point
        current_lambda = self.calculate_lambda(current_air_flow, current_fuel_flow, fuel_type)
        current_excess_air = self.calculate_excess_air(current_lambda)
        current_afr = current_air_flow / current_fuel_flow if current_fuel_flow > 0 else 0

        # Determine target O2 based on load (characterization curve)
        target_o2 = self._get_target_o2_for_load(load_percent)

        # Calculate deviation
        o2_deviation = current_o2 - target_o2
        is_at_target = abs(o2_deviation) <= self.config.o2_deadband

        # Check constraints
        if measured_co_ppm > self.config.co_limit_ppm * 0.8:
            constraints_active.append(f"CO approaching limit ({measured_co_ppm:.0f} ppm)")
            # Don't reduce air if CO is high
            if o2_deviation < 0:
                o2_deviation = 0

        if current_o2 < 1.5:
            constraints_active.append("Low O2 - safety concern")
            # Force air increase
            o2_deviation = -1.0

        # Generate adjustment if outside deadband
        if not is_at_target and self.config.mode != ControlMode.MANUAL:
            # Check rate limiting
            can_adjust = self._check_rate_limit(now)

            if can_adjust:
                # Calculate air flow adjustment
                air_adjustment = self._calculate_air_adjustment(
                    o2_deviation, current_air_flow, load_percent
                )

                if abs(air_adjustment) > 0.1:  # Minimum adjustment threshold
                    adj = SetpointAdjustment(
                        tag_name=f"{self.config.unit_id}.AIR_FLOW_SP",
                        current_value=current_air_flow,
                        recommended_value=current_air_flow + air_adjustment,
                        change=air_adjustment,
                        change_percent=(air_adjustment / current_air_flow * 100) if current_air_flow > 0 else 0,
                        reason=f"O2 deviation: {o2_deviation:+.2f}% from target {target_o2:.1f}%",
                        confidence=0.9 if len(constraints_active) == 0 else 0.7,
                        expected_impact={
                            "o2_change": -o2_deviation * 0.5,
                            "efficiency_change": o2_deviation * 0.1 if o2_deviation > 0 else 0,
                        },
                        constraints_checked=[
                            "co_limit",
                            "min_air",
                            "max_air",
                            "rate_limit",
                        ],
                    )
                    adjustments.append(adj)
                    self._adjustment_history.append(adj)

        target_lambda = self.calculate_lambda_from_o2(target_o2)

        return AFRControlOutput(
            unit_id=self.config.unit_id,
            current_afr=round(current_afr, 2),
            current_lambda=round(current_lambda, 3),
            current_excess_air_pct=round(current_excess_air, 1),
            measured_o2_pct=round(current_o2, 2),
            target_o2_pct=round(target_o2, 2),
            target_lambda=round(target_lambda, 3),
            adjustments=adjustments,
            mode=self.config.mode,
            strategy=self.config.strategy,
            is_at_target=is_at_target,
            deviation_from_target=round(o2_deviation, 2),
            constraints_active=constraints_active,
        )

    def _get_target_o2_for_load(self, load_percent: float) -> float:
        """Get target O2 based on load (characterization curve)."""
        # Higher excess air at low loads for stability
        if load_percent < 30:
            return self.config.target_o2_max
        elif load_percent < 60:
            # Linear interpolation
            factor = (load_percent - 30) / 30
            return self.config.target_o2_max - factor * (
                self.config.target_o2_max - self.config.target_o2_setpoint
            )
        elif load_percent < 90:
            return self.config.target_o2_setpoint
        else:
            # Slight increase at very high loads for safety margin
            return self.config.target_o2_setpoint + 0.3

    def _check_rate_limit(self, now: datetime) -> bool:
        """Check if adjustment is allowed based on rate limiting."""
        if self._last_adjustment_time is None:
            self._last_adjustment_time = now
            return True

        elapsed_seconds = (now - self._last_adjustment_time).total_seconds()
        min_interval = 60 / self.config.adjustment_rate_limit_per_min

        if elapsed_seconds >= min_interval:
            self._last_adjustment_time = now
            return True
        return False

    def _calculate_air_adjustment(
        self,
        o2_deviation: float,
        current_air_flow: float,
        load_percent: float
    ) -> float:
        """Calculate air flow adjustment to correct O2 deviation."""
        # Proportional gain (tuned for typical combustion systems)
        kp = 0.5  # % air change per % O2 deviation

        # Adjustment based on deviation
        adjustment_pct = -o2_deviation * kp

        # Apply limits
        adjustment_pct = max(
            -self.config.max_single_adjustment,
            min(self.config.max_single_adjustment, adjustment_pct)
        )

        # Convert to absolute
        adjustment = current_air_flow * adjustment_pct / 100

        # Check min/max air constraints
        new_air_flow = current_air_flow + adjustment
        min_air = current_air_flow * self.config.min_air_flow_pct / 100
        max_air = current_air_flow * self.config.max_air_flow_pct / 100

        if new_air_flow < min_air:
            adjustment = min_air - current_air_flow
        elif new_air_flow > max_air:
            adjustment = max_air - current_air_flow

        return adjustment

    def apply_adjustment(self, adjustment_id: str) -> bool:
        """Mark an adjustment as applied (for tracking)."""
        for adj in self._adjustment_history:
            if adj.adjustment_id == adjustment_id:
                adj.applied = True
                logger.info(f"Adjustment {adjustment_id} marked as applied")
                return True
        return False

    def get_adjustment_history(
        self,
        limit: int = 100
    ) -> List[SetpointAdjustment]:
        """Get recent adjustment history."""
        return self._adjustment_history[-limit:]

    def update_config(self, new_config: AFRConfig) -> None:
        """Update controller configuration."""
        self.config = new_config
        logger.info(f"AirFuelController config updated for {new_config.unit_id}")
