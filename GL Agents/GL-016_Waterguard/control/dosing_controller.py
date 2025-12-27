# -*- coding: utf-8 -*-
"""
GL-016 Waterguard Dosing Controller Module

Chemical injection control with feed-forward and feedback control strategies.
Supports oxygen scavenger, alkalinity builder, dispersant, and phosphate programs.

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


class ChemicalType(str, Enum):
    """Chemical type enumeration."""
    OXYGEN_SCAVENGER = "oxygen_scavenger"
    ALKALINITY_BUILDER = "alkalinity_builder"
    DISPERSANT = "dispersant"
    PHOSPHATE = "phosphate"
    SCALE_INHIBITOR = "scale_inhibitor"
    BIOCIDE = "biocide"


class DosingMode(str, Enum):
    """Dosing control mode."""
    FEED_FORWARD = "feed_forward"
    FEEDBACK = "feedback"
    COMBINED = "combined"
    MANUAL = "manual"
    DISABLED = "disabled"


class DosingReasonCode(str, Enum):
    """Reason code for dosing decisions."""
    NORMAL = "normal"
    HIGH_DEMAND = "high_demand"
    LOW_RESIDUAL = "low_residual"
    STARTUP = "startup"
    TANK_LOW = "tank_low"
    PUMP_FAULT = "pump_fault"
    MANUAL_OVERRIDE = "manual_override"
    DISABLED = "disabled"


class PumpConstraints(BaseModel):
    """Pump operational constraints."""
    min_speed_percent: float = Field(default=10.0, ge=0, le=100)
    max_speed_percent: float = Field(default=100.0, ge=0, le=100)
    min_stroke_percent: float = Field(default=10.0, ge=0, le=100)
    max_stroke_percent: float = Field(default=100.0, ge=0, le=100)
    min_on_time_seconds: float = Field(default=5.0, ge=0)
    min_off_time_seconds: float = Field(default=5.0, ge=0)
    max_starts_per_hour: int = Field(default=60, ge=1)


class TankLevelInterlock(BaseModel):
    """Tank level interlock configuration."""
    tank_id: str = Field(..., description="Tank identifier")
    low_level_percent: float = Field(default=10.0, ge=0, le=100)
    low_low_level_percent: float = Field(default=5.0, ge=0, le=100)
    current_level_percent: float = Field(default=50.0, ge=0, le=100)
    interlock_active: bool = Field(default=False)


class ChemicalProgram(BaseModel):
    """Chemical dosing program configuration."""
    chemical_type: ChemicalType
    target_residual: float = Field(..., ge=0, description="Target residual in ppm")
    residual_deadband: float = Field(default=0.5, ge=0)
    feed_forward_ratio: float = Field(default=1.0, ge=0, description="ppm per unit flow")
    feedback_gain: float = Field(default=1.0, ge=0)
    max_dose_rate: float = Field(default=100.0, ge=0, description="Max mL/min")
    min_dose_rate: float = Field(default=0.0, ge=0, description="Min mL/min")


class DosingConfig(BaseModel):
    """Dosing controller configuration."""
    chemical_programs: List[ChemicalProgram] = Field(default_factory=list)
    pump_constraints: PumpConstraints = Field(default_factory=PumpConstraints)
    tank_interlocks: List[TankLevelInterlock] = Field(default_factory=list)
    control_interval_seconds: float = Field(default=1.0, ge=0.1, le=60)
    feedwater_flow_tag: str = Field(default="FT_FEEDWATER")
    makeup_flow_tag: str = Field(default="FT_MAKEUP")


class DosingOutput(BaseModel):
    """Dosing controller output."""
    chemical_type: ChemicalType
    dose_rate_ml_per_min: float = Field(..., ge=0)
    pump_speed_percent: float = Field(..., ge=0, le=100)
    pump_running: bool = Field(default=False)
    reason_code: DosingReasonCode = Field(default=DosingReasonCode.NORMAL)
    feed_forward_component: float = Field(default=0.0)
    feedback_component: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=datetime.now)
    provenance_hash: str = Field(...)


class DosingController:
    """Chemical dosing controller with feed-forward and feedback."""

    def __init__(self, config: DosingConfig, alert_callback: Optional[Callable] = None):
        self.config = config
        self._alert_callback = alert_callback
        self._lock = threading.RLock()
        self._mode = DosingMode.COMBINED
        self._pump_states: Dict[ChemicalType, bool] = {}
        self._last_outputs: Dict[ChemicalType, DosingOutput] = {}
        logger.info("DosingController initialized")

    def calculate(self, feedwater_flow: float, makeup_flow: float, residuals: Dict[ChemicalType, float]) -> List[DosingOutput]:
        """Calculate dosing outputs for all chemical programs."""
        outputs = []
        with self._lock:
            for program in self.config.chemical_programs:
                output = self._calculate_single(program, feedwater_flow, makeup_flow, residuals.get(program.chemical_type))
                outputs.append(output)
                self._last_outputs[program.chemical_type] = output
        return outputs

    def _calculate_single(self, program: ChemicalProgram, feedwater_flow: float, makeup_flow: float, residual: Optional[float]) -> DosingOutput:
        """Calculate dosing for a single chemical program."""
        if self._mode == DosingMode.DISABLED:
            return self._create_disabled_output(program.chemical_type)

        # Check tank interlocks
        for tank in self.config.tank_interlocks:
            if tank.interlock_active and tank.current_level_percent <= tank.low_low_level_percent:
                return self._create_interlock_output(program.chemical_type, "TANK_LOW")

        # Feed-forward component based on flow
        total_flow = feedwater_flow + makeup_flow
        ff_component = total_flow * program.feed_forward_ratio

        # Feedback component based on residual
        fb_component = 0.0
        if residual is not None and self._mode in [DosingMode.FEEDBACK, DosingMode.COMBINED]:
            error = program.target_residual - residual
            if abs(error) > program.residual_deadband:
                fb_component = error * program.feedback_gain

        # Combine components
        if self._mode == DosingMode.FEED_FORWARD:
            dose_rate = ff_component
        elif self._mode == DosingMode.FEEDBACK:
            dose_rate = fb_component
        else:
            dose_rate = ff_component + fb_component

        # Apply limits
        dose_rate = max(program.min_dose_rate, min(program.max_dose_rate, dose_rate))

        # Calculate pump speed
        pump_speed = (dose_rate / program.max_dose_rate) * 100 if program.max_dose_rate > 0 else 0

        return DosingOutput(
            chemical_type=program.chemical_type,
            dose_rate_ml_per_min=dose_rate,
            pump_speed_percent=pump_speed,
            pump_running=dose_rate > 0,
            reason_code=DosingReasonCode.NORMAL,
            feed_forward_component=ff_component,
            feedback_component=fb_component,
            timestamp=datetime.now(),
            provenance_hash=hashlib.sha256(f"{program.chemical_type}:{dose_rate}:{datetime.now()}".encode()).hexdigest()
        )

    def set_mode(self, mode: DosingMode) -> bool:
        with self._lock:
            self._mode = mode
            return True

    def get_mode(self) -> DosingMode:
        return self._mode

    def _create_disabled_output(self, chemical_type: ChemicalType) -> DosingOutput:
        return DosingOutput(
            chemical_type=chemical_type,
            dose_rate_ml_per_min=0.0,
            pump_speed_percent=0.0,
            pump_running=False,
            reason_code=DosingReasonCode.DISABLED,
            timestamp=datetime.now(),
            provenance_hash=hashlib.sha256(f"DISABLED:{chemical_type}".encode()).hexdigest()
        )

    def _create_interlock_output(self, chemical_type: ChemicalType, reason: str) -> DosingOutput:
        return DosingOutput(
            chemical_type=chemical_type,
            dose_rate_ml_per_min=0.0,
            pump_speed_percent=0.0,
            pump_running=False,
            reason_code=DosingReasonCode.TANK_LOW,
            timestamp=datetime.now(),
            provenance_hash=hashlib.sha256(f"INTERLOCK:{chemical_type}:{reason}".encode()).hexdigest()
        )
