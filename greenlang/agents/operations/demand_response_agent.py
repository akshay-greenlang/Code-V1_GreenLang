# -*- coding: utf-8 -*-
"""
GL-OPS-X-004: Demand Response Agent
====================================

Manages demand response operations, coordinating load reduction during
grid stress events and optimizing energy consumption patterns.

Capabilities:
    - Grid signal processing and event detection
    - Load curtailment strategy optimization
    - Demand forecasting and prediction
    - Response strategy execution
    - Performance tracking and verification
    - Grid service participation (frequency response, capacity markets)

Zero-Hallucination Guarantees:
    - All load calculations use deterministic formulas
    - Complete provenance tracking with SHA-256 hashes
    - No LLM calls in the calculation path
    - All response actions traceable to grid signals

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class GridSignal(str, Enum):
    """Types of grid signals."""
    EMERGENCY = "emergency"  # Critical grid emergency
    WARNING = "warning"  # Grid stress warning
    ECONOMIC = "economic"  # Price-based signal
    RENEWABLE = "renewable"  # Excess renewable available
    NORMAL = "normal"  # Normal operations
    TEST = "test"  # Test signal


class ResponseStrategy(str, Enum):
    """Demand response strategies."""
    FULL_CURTAILMENT = "full_curtailment"  # Maximum load reduction
    PARTIAL_CURTAILMENT = "partial_curtailment"  # Moderate load reduction
    LOAD_SHIFTING = "load_shifting"  # Shift to later time
    THERMAL_MASS = "thermal_mass"  # Use building thermal mass
    STORAGE_DISCHARGE = "storage_discharge"  # Discharge stored energy
    GENERATION_DISPATCH = "generation_dispatch"  # Start on-site generation
    NO_ACTION = "no_action"  # No response needed


class EventStatus(str, Enum):
    """Status of a demand response event."""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class LoadType(str, Enum):
    """Types of controllable loads."""
    HVAC = "hvac"
    LIGHTING = "lighting"
    PROCESS = "process"
    EQUIPMENT = "equipment"
    CHARGING = "charging"
    STORAGE = "storage"
    DISCRETIONARY = "discretionary"


# =============================================================================
# Pydantic Models
# =============================================================================

class DemandForecast(BaseModel):
    """Demand forecast data point."""
    timestamp: datetime = Field(..., description="Forecast timestamp")
    predicted_load_kw: float = Field(..., ge=0, description="Predicted load in kW")
    confidence: float = Field(default=0.8, ge=0, le=1, description="Forecast confidence")
    peak_probability: float = Field(default=0.0, ge=0, le=1, description="Peak event probability")
    temperature_f: Optional[float] = Field(None, description="Temperature forecast")
    renewable_availability: Optional[float] = Field(None, ge=0, le=100, description="Renewable % forecast")


class ControllableLoad(BaseModel):
    """A controllable load that can participate in demand response."""
    load_id: str = Field(..., description="Load identifier")
    load_type: LoadType = Field(..., description="Type of load")
    facility_id: str = Field(..., description="Facility identifier")
    name: str = Field(..., description="Load name")

    # Capacity
    rated_power_kw: float = Field(..., ge=0, description="Rated power in kW")
    min_power_kw: float = Field(default=0.0, ge=0, description="Minimum power")
    max_curtailment_kw: float = Field(..., ge=0, description="Maximum curtailable load")

    # Current state
    current_power_kw: float = Field(default=0.0, ge=0, description="Current power consumption")
    available: bool = Field(default=True, description="Whether load is available for DR")

    # Response characteristics
    ramp_rate_kw_per_min: float = Field(default=0.0, ge=0, description="Ramp rate")
    minimum_runtime_minutes: int = Field(default=0, ge=0, description="Minimum runtime")
    recovery_time_minutes: int = Field(default=0, ge=0, description="Recovery time after curtailment")

    # Priority
    priority: int = Field(default=5, ge=1, le=10, description="Curtailment priority (1=first to curtail)")

    # Cost
    curtailment_cost_per_kwh: float = Field(default=0.0, ge=0, description="Cost per kWh curtailed")


class LoadCurtailment(BaseModel):
    """A load curtailment action."""
    curtailment_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    load_id: str = Field(..., description="Load being curtailed")
    facility_id: str = Field(..., description="Facility identifier")

    # Curtailment details
    target_reduction_kw: float = Field(..., ge=0, description="Target reduction")
    actual_reduction_kw: float = Field(default=0.0, ge=0, description="Actual reduction achieved")

    # Timing
    start_time: datetime = Field(..., description="Curtailment start")
    end_time: datetime = Field(..., description="Curtailment end")
    duration_minutes: int = Field(..., ge=1, description="Duration in minutes")

    # Status
    executed: bool = Field(default=False, description="Whether executed")
    verified: bool = Field(default=False, description="Whether verified")

    # Metrics
    energy_saved_kwh: float = Field(default=0.0, ge=0, description="Energy saved")
    emissions_avoided_kg: float = Field(default=0.0, ge=0, description="Emissions avoided")
    cost_savings: float = Field(default=0.0, description="Cost savings")


class DemandEvent(BaseModel):
    """A demand response event."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    facility_id: str = Field(..., description="Facility identifier")

    # Event details
    signal_type: GridSignal = Field(..., description="Grid signal type")
    strategy: ResponseStrategy = Field(..., description="Response strategy")
    status: EventStatus = Field(default=EventStatus.PENDING)

    # Timing
    notification_time: datetime = Field(..., description="When notified")
    event_start: datetime = Field(..., description="Event start time")
    event_end: datetime = Field(..., description="Event end time")
    duration_minutes: int = Field(..., ge=1, description="Event duration")

    # Targets
    target_reduction_kw: float = Field(..., ge=0, description="Target reduction")
    baseline_load_kw: float = Field(..., ge=0, description="Baseline load")

    # Results
    actual_reduction_kw: float = Field(default=0.0, ge=0, description="Actual reduction")
    performance_ratio: float = Field(default=0.0, ge=0, description="Actual/Target ratio")

    # Curtailments
    curtailments: List[LoadCurtailment] = Field(default_factory=list)

    # Incentives
    incentive_rate: float = Field(default=0.0, ge=0, description="$/kW incentive rate")
    earned_incentive: float = Field(default=0.0, ge=0, description="Earned incentive")

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DemandResponseInput(BaseModel):
    """Input for the Demand Response Agent."""
    operation: str = Field(..., description="Operation to perform")
    grid_signal: Optional[GridSignal] = Field(None, description="Grid signal received")
    facility_id: Optional[str] = Field(None, description="Facility identifier")
    target_reduction_kw: Optional[float] = Field(None, description="Target reduction")
    event_start: Optional[datetime] = Field(None, description="Event start time")
    event_duration_minutes: Optional[int] = Field(None, description="Event duration")
    forecast_data: List[DemandForecast] = Field(default_factory=list, description="Demand forecast")
    loads: List[ControllableLoad] = Field(default_factory=list, description="Controllable loads")
    event_id: Optional[str] = Field(None, description="Event ID for updates")

    @field_validator('operation')
    @classmethod
    def validate_operation(cls, v: str) -> str:
        """Validate operation is supported."""
        valid_ops = {
            'process_signal', 'create_event', 'execute_event',
            'complete_event', 'get_active_events', 'get_event_history',
            'register_load', 'get_loads', 'forecast_demand',
            'calculate_capacity', 'get_performance', 'get_statistics'
        }
        if v not in valid_ops:
            raise ValueError(f"Operation must be one of: {valid_ops}")
        return v


class DemandResponseOutput(BaseModel):
    """Output from the Demand Response Agent."""
    success: bool = Field(..., description="Whether operation succeeded")
    operation: str = Field(..., description="Operation performed")
    data: Dict[str, Any] = Field(default_factory=dict, description="Result data")
    provenance_hash: str = Field(default="", description="SHA-256 hash for audit")
    processing_time_ms: float = Field(default=0.0, description="Processing duration")
    timestamp: datetime = Field(default_factory=DeterministicClock.now)


# =============================================================================
# Demand Response Agent Implementation
# =============================================================================

class DemandResponseAgent(BaseAgent):
    """
    GL-OPS-X-004: Demand Response Agent

    Manages demand response operations, coordinating load reduction during
    grid stress events and optimizing energy consumption patterns.

    Zero-Hallucination Guarantees:
        - All load calculations use deterministic formulas
        - Complete provenance tracking with SHA-256 hashes
        - No LLM calls in the calculation path
        - All response actions traceable to grid signals

    Usage:
        agent = DemandResponseAgent()

        # Process grid signal
        result = agent.run({
            "operation": "process_signal",
            "grid_signal": "emergency",
            "facility_id": "FAC-001",
            "target_reduction_kw": 500
        })

        # Get active events
        result = agent.run({
            "operation": "get_active_events",
            "facility_id": "FAC-001"
        })
    """

    AGENT_ID = "GL-OPS-X-004"
    AGENT_NAME = "Demand Response Agent"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the Demand Response Agent."""
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Demand response management and optimization",
                version=self.VERSION,
                parameters={
                    "default_baseline_hours": 10,
                    "performance_threshold": 0.9,
                    "max_event_duration_hours": 4,
                }
            )
        super().__init__(config)

        # Controllable loads by facility
        self._loads: Dict[str, Dict[str, ControllableLoad]] = defaultdict(dict)

        # Active events
        self._active_events: Dict[str, DemandEvent] = {}

        # Event history
        self._event_history: List[DemandEvent] = []

        # Demand forecast cache
        self._forecasts: Dict[str, List[DemandForecast]] = defaultdict(list)

        # Baseline data (historical load)
        self._baselines: Dict[str, deque] = defaultdict(lambda: deque(maxlen=720))  # 10 days hourly

        # Grid carbon intensity for avoided emissions
        self._grid_carbon_intensity = 400.0  # gCO2/kWh default

        # Statistics
        self._total_events = 0
        self._total_curtailment_kwh = 0.0
        self._total_incentives = 0.0

        self.logger.info(f"Initialized {self.AGENT_ID}: {self.AGENT_NAME}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute demand response operations."""
        start_time = time.time()

        try:
            dr_input = DemandResponseInput(**input_data)
            operation = dr_input.operation

            result_data = self._route_operation(dr_input)

            provenance_hash = self._compute_provenance_hash(input_data, result_data)
            processing_time_ms = (time.time() - start_time) * 1000

            output = DemandResponseOutput(
                success=True,
                operation=operation,
                data=result_data,
                provenance_hash=provenance_hash,
                processing_time_ms=processing_time_ms,
            )

            return AgentResult(
                success=True,
                data=output.model_dump(),
            )

        except Exception as e:
            self.logger.error(f"Demand response operation failed: {e}", exc_info=True)
            processing_time_ms = (time.time() - start_time) * 1000

            return AgentResult(
                success=False,
                error=str(e),
                data={
                    "operation": input_data.get("operation", "unknown"),
                    "processing_time_ms": processing_time_ms,
                },
            )

    def _route_operation(self, dr_input: DemandResponseInput) -> Dict[str, Any]:
        """Route to appropriate operation handler."""
        operation = dr_input.operation

        if operation == "process_signal":
            return self._handle_process_signal(
                dr_input.grid_signal,
                dr_input.facility_id,
                dr_input.target_reduction_kw,
                dr_input.event_start,
                dr_input.event_duration_minutes,
            )
        elif operation == "create_event":
            return self._handle_create_event(
                dr_input.facility_id,
                dr_input.grid_signal,
                dr_input.target_reduction_kw,
                dr_input.event_start,
                dr_input.event_duration_minutes,
            )
        elif operation == "execute_event":
            return self._handle_execute_event(dr_input.event_id)
        elif operation == "complete_event":
            return self._handle_complete_event(dr_input.event_id)
        elif operation == "get_active_events":
            return self._handle_get_active_events(dr_input.facility_id)
        elif operation == "get_event_history":
            return self._handle_get_event_history(dr_input.facility_id)
        elif operation == "register_load":
            return self._handle_register_loads(dr_input.loads)
        elif operation == "get_loads":
            return self._handle_get_loads(dr_input.facility_id)
        elif operation == "forecast_demand":
            return self._handle_forecast_demand(dr_input.facility_id, dr_input.forecast_data)
        elif operation == "calculate_capacity":
            return self._handle_calculate_capacity(dr_input.facility_id)
        elif operation == "get_performance":
            return self._handle_get_performance(dr_input.facility_id)
        elif operation == "get_statistics":
            return self._handle_get_statistics()
        else:
            raise ValueError(f"Unknown operation: {operation}")

    # =========================================================================
    # Signal Processing
    # =========================================================================

    def _handle_process_signal(
        self,
        grid_signal: Optional[GridSignal],
        facility_id: Optional[str],
        target_reduction_kw: Optional[float],
        event_start: Optional[datetime],
        event_duration_minutes: Optional[int],
    ) -> Dict[str, Any]:
        """Process incoming grid signal and determine response."""
        if not grid_signal:
            return {"error": "grid_signal is required"}
        if not facility_id:
            return {"error": "facility_id is required"}

        now = DeterministicClock.now()

        # Determine appropriate response strategy
        strategy = self._determine_strategy(grid_signal, facility_id)

        if strategy == ResponseStrategy.NO_ACTION:
            return {
                "signal_type": grid_signal.value,
                "response": "no_action",
                "reason": "Signal does not require demand response",
            }

        # Calculate available capacity
        capacity = self._calculate_available_capacity(facility_id)

        # Use provided values or calculate defaults
        if event_start is None:
            event_start = now + timedelta(minutes=15)  # 15 min lead time

        if event_duration_minutes is None:
            event_duration_minutes = self._default_duration(grid_signal)

        if target_reduction_kw is None:
            target_reduction_kw = self._calculate_target_reduction(
                grid_signal, capacity["total_curtailable_kw"]
            )

        # Create event
        event = self._create_demand_event(
            facility_id, grid_signal, strategy,
            target_reduction_kw, event_start, event_duration_minutes
        )

        return {
            "signal_type": grid_signal.value,
            "strategy": strategy.value,
            "event_id": event.event_id,
            "event_start": event_start.isoformat(),
            "event_duration_minutes": event_duration_minutes,
            "target_reduction_kw": target_reduction_kw,
            "available_capacity_kw": capacity["total_curtailable_kw"],
            "curtailments_planned": len(event.curtailments),
        }

    def _determine_strategy(
        self, signal: GridSignal, facility_id: str
    ) -> ResponseStrategy:
        """Determine response strategy based on signal type."""
        if signal == GridSignal.EMERGENCY:
            return ResponseStrategy.FULL_CURTAILMENT
        elif signal == GridSignal.WARNING:
            return ResponseStrategy.PARTIAL_CURTAILMENT
        elif signal == GridSignal.ECONOMIC:
            return ResponseStrategy.LOAD_SHIFTING
        elif signal == GridSignal.RENEWABLE:
            return ResponseStrategy.NO_ACTION  # Increase load
        elif signal == GridSignal.TEST:
            return ResponseStrategy.PARTIAL_CURTAILMENT
        else:
            return ResponseStrategy.NO_ACTION

    def _default_duration(self, signal: GridSignal) -> int:
        """Get default event duration for signal type."""
        durations = {
            GridSignal.EMERGENCY: 60,
            GridSignal.WARNING: 120,
            GridSignal.ECONOMIC: 180,
            GridSignal.RENEWABLE: 60,
            GridSignal.TEST: 30,
        }
        return durations.get(signal, 60)

    def _calculate_target_reduction(
        self, signal: GridSignal, available_capacity: float
    ) -> float:
        """Calculate target reduction based on signal and capacity."""
        ratios = {
            GridSignal.EMERGENCY: 1.0,  # Full capacity
            GridSignal.WARNING: 0.75,
            GridSignal.ECONOMIC: 0.5,
            GridSignal.TEST: 0.25,
        }
        ratio = ratios.get(signal, 0.5)
        return round(available_capacity * ratio, 2)

    # =========================================================================
    # Event Management
    # =========================================================================

    def _handle_create_event(
        self,
        facility_id: Optional[str],
        grid_signal: Optional[GridSignal],
        target_reduction_kw: Optional[float],
        event_start: Optional[datetime],
        event_duration_minutes: Optional[int],
    ) -> Dict[str, Any]:
        """Create a demand response event."""
        if not facility_id:
            return {"error": "facility_id is required"}

        signal = grid_signal or GridSignal.ECONOMIC
        strategy = self._determine_strategy(signal, facility_id)

        now = DeterministicClock.now()
        if event_start is None:
            event_start = now + timedelta(minutes=15)
        if event_duration_minutes is None:
            event_duration_minutes = 60
        if target_reduction_kw is None:
            capacity = self._calculate_available_capacity(facility_id)
            target_reduction_kw = capacity["total_curtailable_kw"] * 0.5

        event = self._create_demand_event(
            facility_id, signal, strategy,
            target_reduction_kw, event_start, event_duration_minutes
        )

        return {
            "event_id": event.event_id,
            "facility_id": facility_id,
            "signal_type": signal.value,
            "strategy": strategy.value,
            "event_start": event_start.isoformat(),
            "event_duration_minutes": event_duration_minutes,
            "target_reduction_kw": target_reduction_kw,
            "curtailments": [c.model_dump() for c in event.curtailments],
            "created": True,
        }

    def _create_demand_event(
        self,
        facility_id: str,
        signal: GridSignal,
        strategy: ResponseStrategy,
        target_reduction_kw: float,
        event_start: datetime,
        event_duration_minutes: int,
    ) -> DemandEvent:
        """Create a demand event with curtailment plan."""
        now = DeterministicClock.now()
        event_end = event_start + timedelta(minutes=event_duration_minutes)

        # Calculate baseline
        baseline = self._calculate_baseline(facility_id)

        # Create curtailment plan
        curtailments = self._create_curtailment_plan(
            facility_id, target_reduction_kw, event_start, event_duration_minutes
        )

        event = DemandEvent(
            facility_id=facility_id,
            signal_type=signal,
            strategy=strategy,
            notification_time=now,
            event_start=event_start,
            event_end=event_end,
            duration_minutes=event_duration_minutes,
            target_reduction_kw=target_reduction_kw,
            baseline_load_kw=baseline,
            curtailments=curtailments,
            incentive_rate=self._get_incentive_rate(signal),
        )

        self._active_events[event.event_id] = event
        self._total_events += 1

        return event

    def _create_curtailment_plan(
        self,
        facility_id: str,
        target_kw: float,
        start_time: datetime,
        duration_minutes: int,
    ) -> List[LoadCurtailment]:
        """Create curtailment plan to achieve target reduction."""
        loads = self._loads.get(facility_id, {})

        # Sort loads by priority (lower = curtail first)
        sorted_loads = sorted(
            [l for l in loads.values() if l.available],
            key=lambda l: l.priority,
        )

        curtailments = []
        remaining_target = target_kw

        for load in sorted_loads:
            if remaining_target <= 0:
                break

            # Calculate curtailment for this load
            curtail_amount = min(load.max_curtailment_kw, remaining_target)

            if curtail_amount > 0:
                energy_saved = (curtail_amount * duration_minutes) / 60  # kWh
                emissions_avoided = (energy_saved * self._grid_carbon_intensity) / 1000  # kg

                curtailment = LoadCurtailment(
                    load_id=load.load_id,
                    facility_id=facility_id,
                    target_reduction_kw=curtail_amount,
                    start_time=start_time,
                    end_time=start_time + timedelta(minutes=duration_minutes),
                    duration_minutes=duration_minutes,
                    energy_saved_kwh=round(energy_saved, 4),
                    emissions_avoided_kg=round(emissions_avoided, 4),
                )

                curtailments.append(curtailment)
                remaining_target -= curtail_amount

        return curtailments

    def _handle_execute_event(self, event_id: Optional[str]) -> Dict[str, Any]:
        """Execute a demand response event."""
        if not event_id:
            return {"error": "event_id is required"}

        if event_id not in self._active_events:
            return {"error": f"Event not found: {event_id}"}

        event = self._active_events[event_id]

        # Mark event as active
        event.status = EventStatus.ACTIVE

        # Execute curtailments
        for curtailment in event.curtailments:
            curtailment.executed = True
            # Simulate actual reduction (in real system, would send commands)
            curtailment.actual_reduction_kw = curtailment.target_reduction_kw * 0.95

        # Calculate actual reduction
        event.actual_reduction_kw = sum(c.actual_reduction_kw for c in event.curtailments)
        event.performance_ratio = (
            event.actual_reduction_kw / event.target_reduction_kw
            if event.target_reduction_kw > 0 else 0
        )

        return {
            "event_id": event_id,
            "status": EventStatus.ACTIVE.value,
            "actual_reduction_kw": event.actual_reduction_kw,
            "performance_ratio": round(event.performance_ratio, 4),
            "curtailments_executed": len(event.curtailments),
        }

    def _handle_complete_event(self, event_id: Optional[str]) -> Dict[str, Any]:
        """Complete a demand response event."""
        if not event_id:
            return {"error": "event_id is required"}

        if event_id not in self._active_events:
            return {"error": f"Event not found: {event_id}"}

        event = self._active_events[event_id]

        # Mark event as completed
        event.status = EventStatus.COMPLETED

        # Verify curtailments
        for curtailment in event.curtailments:
            curtailment.verified = True

        # Calculate final metrics
        total_energy_saved = sum(c.energy_saved_kwh for c in event.curtailments)
        total_emissions_avoided = sum(c.emissions_avoided_kg for c in event.curtailments)

        # Calculate incentive earned
        event.earned_incentive = (
            event.actual_reduction_kw * event.incentive_rate * event.performance_ratio
        )

        # Update statistics
        self._total_curtailment_kwh += total_energy_saved
        self._total_incentives += event.earned_incentive

        # Move to history
        self._event_history.append(event)
        del self._active_events[event_id]

        return {
            "event_id": event_id,
            "status": EventStatus.COMPLETED.value,
            "final_reduction_kw": event.actual_reduction_kw,
            "performance_ratio": round(event.performance_ratio, 4),
            "energy_saved_kwh": round(total_energy_saved, 4),
            "emissions_avoided_kg": round(total_emissions_avoided, 4),
            "earned_incentive": round(event.earned_incentive, 2),
        }

    def _handle_get_active_events(self, facility_id: Optional[str]) -> Dict[str, Any]:
        """Get active demand response events."""
        events = list(self._active_events.values())

        if facility_id:
            events = [e for e in events if e.facility_id == facility_id]

        return {
            "active_events": [e.model_dump() for e in events],
            "count": len(events),
        }

    def _handle_get_event_history(self, facility_id: Optional[str]) -> Dict[str, Any]:
        """Get event history."""
        events = self._event_history.copy()

        if facility_id:
            events = [e for e in events if e.facility_id == facility_id]

        return {
            "event_history": [e.model_dump() for e in events],
            "count": len(events),
        }

    # =========================================================================
    # Load Management
    # =========================================================================

    def _handle_register_loads(self, loads: List[ControllableLoad]) -> Dict[str, Any]:
        """Register controllable loads."""
        registered = 0

        for load in loads:
            self._loads[load.facility_id][load.load_id] = load
            registered += 1

        return {
            "registered": registered,
            "total_loads": sum(len(l) for l in self._loads.values()),
        }

    def _handle_get_loads(self, facility_id: Optional[str]) -> Dict[str, Any]:
        """Get registered loads."""
        if facility_id:
            loads = list(self._loads.get(facility_id, {}).values())
        else:
            loads = []
            for facility_loads in self._loads.values():
                loads.extend(facility_loads.values())

        return {
            "loads": [l.model_dump() for l in loads],
            "count": len(loads),
        }

    # =========================================================================
    # Capacity Calculation
    # =========================================================================

    def _handle_calculate_capacity(self, facility_id: Optional[str]) -> Dict[str, Any]:
        """Calculate available demand response capacity."""
        if not facility_id:
            return {"error": "facility_id is required"}

        capacity = self._calculate_available_capacity(facility_id)

        return {
            "facility_id": facility_id,
            **capacity,
        }

    def _calculate_available_capacity(self, facility_id: str) -> Dict[str, float]:
        """Calculate total available capacity for a facility."""
        loads = self._loads.get(facility_id, {})

        total_rated = sum(l.rated_power_kw for l in loads.values())
        total_current = sum(l.current_power_kw for l in loads.values() if l.available)
        total_curtailable = sum(l.max_curtailment_kw for l in loads.values() if l.available)

        by_type = defaultdict(float)
        for load in loads.values():
            if load.available:
                by_type[load.load_type.value] += load.max_curtailment_kw

        return {
            "total_rated_kw": round(total_rated, 2),
            "total_current_kw": round(total_current, 2),
            "total_curtailable_kw": round(total_curtailable, 2),
            "curtailable_by_type": dict(by_type),
            "available_loads": sum(1 for l in loads.values() if l.available),
            "total_loads": len(loads),
        }

    def _calculate_baseline(self, facility_id: str) -> float:
        """Calculate baseline load for a facility."""
        baselines = self._baselines.get(facility_id, [])

        if not baselines:
            # Use current load from registered loads
            loads = self._loads.get(facility_id, {})
            return sum(l.current_power_kw for l in loads.values())

        # Average of historical data
        return sum(baselines) / len(baselines) if baselines else 0.0

    # =========================================================================
    # Forecasting
    # =========================================================================

    def _handle_forecast_demand(
        self, facility_id: Optional[str], forecast_data: List[DemandForecast]
    ) -> Dict[str, Any]:
        """Store and analyze demand forecast."""
        if not facility_id:
            return {"error": "facility_id is required"}

        # Store forecast
        self._forecasts[facility_id] = forecast_data

        # Analyze forecast
        if forecast_data:
            peak_load = max(f.predicted_load_kw for f in forecast_data)
            avg_load = sum(f.predicted_load_kw for f in forecast_data) / len(forecast_data)
            peak_events = [f for f in forecast_data if f.peak_probability > 0.7]

            return {
                "facility_id": facility_id,
                "forecast_points": len(forecast_data),
                "peak_load_kw": round(peak_load, 2),
                "average_load_kw": round(avg_load, 2),
                "high_peak_probability_periods": len(peak_events),
            }

        return {
            "facility_id": facility_id,
            "forecast_points": 0,
        }

    # =========================================================================
    # Performance
    # =========================================================================

    def _handle_get_performance(self, facility_id: Optional[str]) -> Dict[str, Any]:
        """Get demand response performance metrics."""
        events = self._event_history.copy()

        if facility_id:
            events = [e for e in events if e.facility_id == facility_id]

        if not events:
            return {
                "total_events": 0,
                "message": "No events in history",
            }

        total_target = sum(e.target_reduction_kw for e in events)
        total_actual = sum(e.actual_reduction_kw for e in events)
        avg_performance = sum(e.performance_ratio for e in events) / len(events)
        total_incentives = sum(e.earned_incentive for e in events)

        total_energy = sum(
            sum(c.energy_saved_kwh for c in e.curtailments)
            for e in events
        )
        total_emissions = sum(
            sum(c.emissions_avoided_kg for c in e.curtailments)
            for e in events
        )

        return {
            "total_events": len(events),
            "total_target_kw": round(total_target, 2),
            "total_actual_kw": round(total_actual, 2),
            "average_performance_ratio": round(avg_performance, 4),
            "total_energy_saved_kwh": round(total_energy, 4),
            "total_emissions_avoided_kg": round(total_emissions, 4),
            "total_incentives_earned": round(total_incentives, 2),
        }

    def _get_incentive_rate(self, signal: GridSignal) -> float:
        """Get incentive rate based on signal type."""
        rates = {
            GridSignal.EMERGENCY: 1.00,  # $1.00/kW
            GridSignal.WARNING: 0.50,
            GridSignal.ECONOMIC: 0.25,
            GridSignal.TEST: 0.10,
        }
        return rates.get(signal, 0.25)

    # =========================================================================
    # Statistics
    # =========================================================================

    def _handle_get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "total_events": self._total_events,
            "active_events": len(self._active_events),
            "completed_events": len(self._event_history),
            "total_curtailment_kwh": round(self._total_curtailment_kwh, 4),
            "total_incentives": round(self._total_incentives, 2),
            "registered_loads": sum(len(l) for l in self._loads.values()),
            "monitored_facilities": len(self._loads),
        }

    # =========================================================================
    # Provenance
    # =========================================================================

    def _compute_provenance_hash(
        self, input_data: Dict[str, Any], output_data: Dict[str, Any]
    ) -> str:
        """Compute SHA-256 hash for audit trail."""
        provenance_str = json.dumps(
            {"input": input_data, "output": output_data},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()[:16]
