# -*- coding: utf-8 -*-
"""
GL-OPS-X-003: Optimization Scheduler
=====================================

Schedules operations for maximum efficiency, balancing emissions reduction,
cost optimization, and operational constraints.

Capabilities:
    - Multi-objective optimization (emissions, cost, reliability)
    - Constraint-based scheduling
    - Resource allocation optimization
    - Time-of-use optimization for grid carbon intensity
    - Load balancing across equipment
    - Schedule conflict resolution

Zero-Hallucination Guarantees:
    - All optimization uses deterministic algorithms
    - Complete provenance tracking with SHA-256 hashes
    - No LLM calls in the optimization path
    - All schedules traceable to input constraints

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import time
import uuid
from collections import defaultdict
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

class OptimizationGoal(str, Enum):
    """Goals for schedule optimization."""
    MINIMIZE_EMISSIONS = "minimize_emissions"
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_EFFICIENCY = "maximize_efficiency"
    BALANCE_LOAD = "balance_load"
    MINIMIZE_PEAK = "minimize_peak"
    MAXIMIZE_RENEWABLE = "maximize_renewable"


class SchedulePeriod(str, Enum):
    """Schedule planning periods."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class ResourceType(str, Enum):
    """Types of resources to schedule."""
    EQUIPMENT = "equipment"
    ENERGY = "energy"
    LABOR = "labor"
    MATERIAL = "material"
    CAPACITY = "capacity"


class ConstraintType(str, Enum):
    """Types of scheduling constraints."""
    TIME_WINDOW = "time_window"
    CAPACITY = "capacity"
    DEPENDENCY = "dependency"
    EXCLUSION = "exclusion"
    MINIMUM_GAP = "minimum_gap"
    MAXIMUM_DURATION = "maximum_duration"
    RESOURCE_LIMIT = "resource_limit"


class ScheduleStatus(str, Enum):
    """Status of a scheduled entry."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    DELAYED = "delayed"


# =============================================================================
# Pydantic Models
# =============================================================================

class ScheduleConstraint(BaseModel):
    """A constraint for scheduling."""
    constraint_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    constraint_type: ConstraintType = Field(..., description="Type of constraint")
    name: str = Field(..., description="Constraint name")

    # Time constraints
    start_time: Optional[datetime] = Field(None, description="Earliest start time")
    end_time: Optional[datetime] = Field(None, description="Latest end time")

    # Resource constraints
    resource_type: Optional[ResourceType] = Field(None, description="Resource type")
    resource_limit: Optional[float] = Field(None, description="Resource limit")

    # Dependencies
    depends_on: List[str] = Field(default_factory=list, description="Tasks this depends on")
    excluded_with: List[str] = Field(default_factory=list, description="Tasks that cannot run together")

    # Additional parameters
    minimum_gap_minutes: Optional[int] = Field(None, description="Minimum gap after previous task")
    maximum_duration_minutes: Optional[int] = Field(None, description="Maximum task duration")

    # Priority
    priority: int = Field(default=1, ge=1, le=10, description="Constraint priority")
    mandatory: bool = Field(default=True, description="Whether constraint is mandatory")


class ResourceAllocation(BaseModel):
    """Allocation of resources to a schedule entry."""
    resource_id: str = Field(..., description="Resource identifier")
    resource_type: ResourceType = Field(..., description="Type of resource")
    quantity: float = Field(..., ge=0, description="Quantity allocated")
    unit: str = Field(default="units", description="Unit of measurement")
    cost_per_unit: Optional[float] = Field(None, description="Cost per unit")
    emissions_per_unit: Optional[float] = Field(None, description="Emissions per unit")


class ScheduleEntry(BaseModel):
    """A scheduled operation entry."""
    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_id: str = Field(..., description="Task identifier")
    task_name: str = Field(..., description="Task name")
    facility_id: str = Field(..., description="Facility identifier")

    # Timing
    scheduled_start: datetime = Field(..., description="Scheduled start time")
    scheduled_end: datetime = Field(..., description="Scheduled end time")
    duration_minutes: int = Field(..., ge=1, description="Duration in minutes")

    # Status
    status: ScheduleStatus = Field(default=ScheduleStatus.SCHEDULED)

    # Resources
    resources: List[ResourceAllocation] = Field(default_factory=list)

    # Optimization results
    estimated_emissions_kg: float = Field(default=0.0, description="Estimated emissions")
    estimated_cost: float = Field(default=0.0, description="Estimated cost")
    grid_carbon_intensity: Optional[float] = Field(None, description="Grid carbon intensity at scheduled time")

    # Constraints satisfied
    constraints_satisfied: List[str] = Field(default_factory=list)
    constraints_violated: List[str] = Field(default_factory=list)

    # Metadata
    priority: int = Field(default=5, ge=1, le=10)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OperationTask(BaseModel):
    """A task to be scheduled."""
    task_id: str = Field(..., description="Task identifier")
    task_name: str = Field(..., description="Task name")
    facility_id: str = Field(..., description="Facility identifier")

    # Requirements
    duration_minutes: int = Field(..., ge=1, description="Task duration")
    required_resources: List[ResourceAllocation] = Field(default_factory=list)

    # Timing preferences
    preferred_start: Optional[datetime] = Field(None, description="Preferred start time")
    deadline: Optional[datetime] = Field(None, description="Task deadline")
    flexible: bool = Field(default=True, description="Whether timing is flexible")

    # Emissions profile
    emissions_rate_kg_per_hour: float = Field(default=0.0, description="Emissions rate")
    energy_consumption_kwh: float = Field(default=0.0, description="Energy consumption")

    # Cost profile
    base_cost: float = Field(default=0.0, description="Base cost")
    time_of_use_sensitive: bool = Field(default=False, description="Sensitive to TOU pricing")

    # Constraints
    constraints: List[ScheduleConstraint] = Field(default_factory=list)

    # Priority
    priority: int = Field(default=5, ge=1, le=10)


class GridCarbonForecast(BaseModel):
    """Grid carbon intensity forecast."""
    timestamp: datetime = Field(..., description="Forecast timestamp")
    carbon_intensity: float = Field(..., ge=0, description="gCO2/kWh")
    renewable_percent: float = Field(default=0.0, ge=0, le=100, description="Renewable percentage")
    price_per_kwh: Optional[float] = Field(None, description="Energy price")


class SchedulerInput(BaseModel):
    """Input for the Optimization Scheduler."""
    operation: str = Field(..., description="Operation to perform")
    tasks: List[OperationTask] = Field(default_factory=list, description="Tasks to schedule")
    optimization_goal: OptimizationGoal = Field(
        default=OptimizationGoal.MINIMIZE_EMISSIONS, description="Primary optimization goal"
    )
    schedule_period: SchedulePeriod = Field(default=SchedulePeriod.DAILY)
    start_time: Optional[datetime] = Field(None, description="Schedule start time")
    end_time: Optional[datetime] = Field(None, description="Schedule end time")
    carbon_forecast: List[GridCarbonForecast] = Field(
        default_factory=list, description="Grid carbon intensity forecast"
    )
    facility_id: Optional[str] = Field(None, description="Facility filter")
    entry_id: Optional[str] = Field(None, description="Entry ID for updates")
    new_status: Optional[ScheduleStatus] = Field(None, description="New entry status")

    @field_validator('operation')
    @classmethod
    def validate_operation(cls, v: str) -> str:
        """Validate operation is supported."""
        valid_ops = {
            'create_schedule', 'optimize_schedule', 'get_schedule',
            'update_entry_status', 'reschedule', 'get_conflicts',
            'get_resource_utilization', 'get_statistics'
        }
        if v not in valid_ops:
            raise ValueError(f"Operation must be one of: {valid_ops}")
        return v


class SchedulerOutput(BaseModel):
    """Output from the Optimization Scheduler."""
    success: bool = Field(..., description="Whether operation succeeded")
    operation: str = Field(..., description="Operation performed")
    data: Dict[str, Any] = Field(default_factory=dict, description="Result data")
    provenance_hash: str = Field(default="", description="SHA-256 hash for audit")
    processing_time_ms: float = Field(default=0.0, description="Processing duration")
    timestamp: datetime = Field(default_factory=DeterministicClock.now)


# =============================================================================
# Optimization Scheduler Implementation
# =============================================================================

class OptimizationScheduler(BaseAgent):
    """
    GL-OPS-X-003: Optimization Scheduler

    Schedules operations for maximum efficiency, balancing emissions reduction,
    cost optimization, and operational constraints.

    Zero-Hallucination Guarantees:
        - All optimization uses deterministic algorithms
        - Complete provenance tracking with SHA-256 hashes
        - No LLM calls in the optimization path
        - All schedules traceable to input constraints

    Usage:
        scheduler = OptimizationScheduler()

        # Create optimized schedule
        result = scheduler.run({
            "operation": "create_schedule",
            "tasks": [...],
            "optimization_goal": "minimize_emissions",
            "carbon_forecast": [...]
        })
    """

    AGENT_ID = "GL-OPS-X-003"
    AGENT_NAME = "Optimization Scheduler"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the Optimization Scheduler."""
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Operational scheduling with emissions optimization",
                version=self.VERSION,
                parameters={
                    "max_iterations": 1000,
                    "convergence_threshold": 0.01,
                    "time_slot_minutes": 15,
                }
            )
        super().__init__(config)

        # Schedule storage by facility
        self._schedules: Dict[str, List[ScheduleEntry]] = defaultdict(list)

        # Carbon forecast cache
        self._carbon_forecast: Dict[datetime, GridCarbonForecast] = {}

        # Statistics
        self._total_schedules_created = 0
        self._total_optimizations = 0

        self.logger.info(f"Initialized {self.AGENT_ID}: {self.AGENT_NAME}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute scheduler operations."""
        start_time = time.time()

        try:
            scheduler_input = SchedulerInput(**input_data)
            operation = scheduler_input.operation

            result_data = self._route_operation(scheduler_input)

            provenance_hash = self._compute_provenance_hash(input_data, result_data)
            processing_time_ms = (time.time() - start_time) * 1000

            output = SchedulerOutput(
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
            self.logger.error(f"Scheduler operation failed: {e}", exc_info=True)
            processing_time_ms = (time.time() - start_time) * 1000

            return AgentResult(
                success=False,
                error=str(e),
                data={
                    "operation": input_data.get("operation", "unknown"),
                    "processing_time_ms": processing_time_ms,
                },
            )

    def _route_operation(self, scheduler_input: SchedulerInput) -> Dict[str, Any]:
        """Route to appropriate operation handler."""
        operation = scheduler_input.operation

        if operation == "create_schedule":
            return self._handle_create_schedule(
                scheduler_input.tasks,
                scheduler_input.optimization_goal,
                scheduler_input.start_time,
                scheduler_input.end_time,
                scheduler_input.carbon_forecast,
            )
        elif operation == "optimize_schedule":
            return self._handle_optimize_schedule(
                scheduler_input.facility_id,
                scheduler_input.optimization_goal,
                scheduler_input.carbon_forecast,
            )
        elif operation == "get_schedule":
            return self._handle_get_schedule(
                scheduler_input.facility_id,
                scheduler_input.start_time,
                scheduler_input.end_time,
            )
        elif operation == "update_entry_status":
            return self._handle_update_entry_status(
                scheduler_input.entry_id,
                scheduler_input.new_status,
            )
        elif operation == "reschedule":
            return self._handle_reschedule(
                scheduler_input.entry_id,
                scheduler_input.start_time,
            )
        elif operation == "get_conflicts":
            return self._handle_get_conflicts(scheduler_input.facility_id)
        elif operation == "get_resource_utilization":
            return self._handle_get_resource_utilization(
                scheduler_input.facility_id,
                scheduler_input.start_time,
                scheduler_input.end_time,
            )
        elif operation == "get_statistics":
            return self._handle_get_statistics()
        else:
            raise ValueError(f"Unknown operation: {operation}")

    # =========================================================================
    # Schedule Creation
    # =========================================================================

    def _handle_create_schedule(
        self,
        tasks: List[OperationTask],
        optimization_goal: OptimizationGoal,
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        carbon_forecast: List[GridCarbonForecast],
    ) -> Dict[str, Any]:
        """Create optimized schedule for tasks."""
        now = DeterministicClock.now()

        if start_time is None:
            start_time = now
        if end_time is None:
            end_time = start_time + timedelta(days=1)

        # Store carbon forecast
        for forecast in carbon_forecast:
            self._carbon_forecast[forecast.timestamp] = forecast

        # Sort tasks by priority and deadline
        sorted_tasks = sorted(
            tasks,
            key=lambda t: (-t.priority, t.deadline or datetime.max),
        )

        scheduled_entries = []
        conflicts = []

        for task in sorted_tasks:
            entry = self._schedule_task(
                task, start_time, end_time, optimization_goal, scheduled_entries
            )

            if entry:
                scheduled_entries.append(entry)
                self._schedules[task.facility_id].append(entry)
            else:
                conflicts.append({
                    "task_id": task.task_id,
                    "reason": "Could not find valid time slot",
                })

        self._total_schedules_created += len(scheduled_entries)

        # Calculate totals
        total_emissions = sum(e.estimated_emissions_kg for e in scheduled_entries)
        total_cost = sum(e.estimated_cost for e in scheduled_entries)

        return {
            "scheduled_entries": [e.model_dump() for e in scheduled_entries],
            "scheduled_count": len(scheduled_entries),
            "conflicts": conflicts,
            "conflict_count": len(conflicts),
            "optimization_goal": optimization_goal.value,
            "total_estimated_emissions_kg": round(total_emissions, 4),
            "total_estimated_cost": round(total_cost, 2),
            "schedule_start": start_time.isoformat(),
            "schedule_end": end_time.isoformat(),
        }

    def _schedule_task(
        self,
        task: OperationTask,
        window_start: datetime,
        window_end: datetime,
        optimization_goal: OptimizationGoal,
        existing_entries: List[ScheduleEntry],
    ) -> Optional[ScheduleEntry]:
        """Schedule a single task within the time window."""
        # Determine effective time window
        effective_start = task.preferred_start or window_start
        if effective_start < window_start:
            effective_start = window_start

        effective_end = task.deadline or window_end
        if effective_end > window_end:
            effective_end = window_end

        # Check if task can fit in window
        task_duration = timedelta(minutes=task.duration_minutes)
        if effective_end - effective_start < task_duration:
            return None

        # Find optimal time slot based on optimization goal
        best_slot = self._find_optimal_slot(
            task, effective_start, effective_end, optimization_goal, existing_entries
        )

        if best_slot is None:
            return None

        slot_start, slot_end = best_slot

        # Calculate metrics for scheduled slot
        carbon_intensity = self._get_carbon_intensity(slot_start)
        emissions = self._calculate_emissions(task, carbon_intensity)
        cost = self._calculate_cost(task, slot_start)

        # Check constraints
        satisfied, violated = self._check_constraints(
            task, slot_start, slot_end, existing_entries
        )

        entry = ScheduleEntry(
            task_id=task.task_id,
            task_name=task.task_name,
            facility_id=task.facility_id,
            scheduled_start=slot_start,
            scheduled_end=slot_end,
            duration_minutes=task.duration_minutes,
            resources=task.required_resources,
            estimated_emissions_kg=emissions,
            estimated_cost=cost,
            grid_carbon_intensity=carbon_intensity,
            constraints_satisfied=satisfied,
            constraints_violated=violated,
            priority=task.priority,
        )

        return entry

    def _find_optimal_slot(
        self,
        task: OperationTask,
        window_start: datetime,
        window_end: datetime,
        optimization_goal: OptimizationGoal,
        existing_entries: List[ScheduleEntry],
    ) -> Optional[Tuple[datetime, datetime]]:
        """Find optimal time slot based on optimization goal."""
        slot_duration = timedelta(minutes=self.config.parameters.get("time_slot_minutes", 15))
        task_duration = timedelta(minutes=task.duration_minutes)

        best_slot = None
        best_score = float('inf') if optimization_goal != OptimizationGoal.MAXIMIZE_RENEWABLE else float('-inf')

        current = window_start

        while current + task_duration <= window_end:
            slot_end = current + task_duration

            # Check for conflicts
            if not self._has_conflict(current, slot_end, existing_entries, task.facility_id):
                score = self._calculate_slot_score(
                    task, current, slot_end, optimization_goal
                )

                if optimization_goal == OptimizationGoal.MAXIMIZE_RENEWABLE:
                    if score > best_score:
                        best_score = score
                        best_slot = (current, slot_end)
                else:
                    if score < best_score:
                        best_score = score
                        best_slot = (current, slot_end)

            current += slot_duration

        return best_slot

    def _calculate_slot_score(
        self,
        task: OperationTask,
        slot_start: datetime,
        slot_end: datetime,
        optimization_goal: OptimizationGoal,
    ) -> float:
        """Calculate score for a time slot based on optimization goal."""
        carbon_intensity = self._get_carbon_intensity(slot_start)
        renewable_percent = self._get_renewable_percent(slot_start)
        energy_price = self._get_energy_price(slot_start)

        if optimization_goal == OptimizationGoal.MINIMIZE_EMISSIONS:
            # Score based on carbon intensity
            emissions = self._calculate_emissions(task, carbon_intensity)
            return emissions

        elif optimization_goal == OptimizationGoal.MINIMIZE_COST:
            # Score based on energy cost
            cost = self._calculate_cost(task, slot_start)
            return cost

        elif optimization_goal == OptimizationGoal.MAXIMIZE_RENEWABLE:
            # Score based on renewable percentage (higher is better)
            return renewable_percent

        elif optimization_goal == OptimizationGoal.MAXIMIZE_EFFICIENCY:
            # Combined score
            emissions = self._calculate_emissions(task, carbon_intensity)
            cost = self._calculate_cost(task, slot_start)
            return emissions * 0.5 + cost * 0.5

        elif optimization_goal == OptimizationGoal.MINIMIZE_PEAK:
            # Score based on time of day (avoid peak hours)
            hour = slot_start.hour
            if 9 <= hour <= 17:  # Peak hours
                return 100.0
            elif 6 <= hour <= 22:  # Shoulder
                return 50.0
            else:  # Off-peak
                return 10.0

        else:
            return 0.0

    def _has_conflict(
        self,
        slot_start: datetime,
        slot_end: datetime,
        existing_entries: List[ScheduleEntry],
        facility_id: str,
    ) -> bool:
        """Check if time slot conflicts with existing entries."""
        for entry in existing_entries:
            if entry.facility_id != facility_id:
                continue

            # Check for overlap
            if slot_start < entry.scheduled_end and slot_end > entry.scheduled_start:
                return True

        return False

    def _calculate_emissions(
        self, task: OperationTask, carbon_intensity: float
    ) -> float:
        """Calculate estimated emissions for a task."""
        # Direct emissions from task
        direct_emissions = task.emissions_rate_kg_per_hour * (task.duration_minutes / 60)

        # Grid emissions from electricity use
        grid_emissions = (task.energy_consumption_kwh * carbon_intensity) / 1000  # gCO2/kWh to kgCO2

        return round(direct_emissions + grid_emissions, 6)

    def _calculate_cost(self, task: OperationTask, slot_start: datetime) -> float:
        """Calculate estimated cost for a task."""
        base_cost = task.base_cost
        energy_price = self._get_energy_price(slot_start)

        energy_cost = task.energy_consumption_kwh * energy_price

        if task.time_of_use_sensitive:
            hour = slot_start.hour
            if 9 <= hour <= 17:  # Peak
                energy_cost *= 1.5
            elif 6 <= hour <= 22:  # Shoulder
                energy_cost *= 1.0
            else:  # Off-peak
                energy_cost *= 0.7

        return round(base_cost + energy_cost, 2)

    def _get_carbon_intensity(self, timestamp: datetime) -> float:
        """Get carbon intensity for a timestamp."""
        # Find nearest forecast
        if timestamp in self._carbon_forecast:
            return self._carbon_forecast[timestamp].carbon_intensity

        # Find closest forecast
        closest = None
        min_diff = timedelta.max

        for ts, forecast in self._carbon_forecast.items():
            diff = abs(ts - timestamp)
            if diff < min_diff:
                min_diff = diff
                closest = forecast

        if closest:
            return closest.carbon_intensity

        # Default based on time of day (simplified)
        hour = timestamp.hour
        if 6 <= hour <= 9:  # Morning ramp
            return 400.0
        elif 9 <= hour <= 17:  # Daytime
            return 350.0
        elif 17 <= hour <= 21:  # Evening peak
            return 450.0
        else:  # Night
            return 300.0

    def _get_renewable_percent(self, timestamp: datetime) -> float:
        """Get renewable percentage for a timestamp."""
        if timestamp in self._carbon_forecast:
            return self._carbon_forecast[timestamp].renewable_percent

        # Default: higher during day
        hour = timestamp.hour
        if 10 <= hour <= 16:
            return 45.0
        else:
            return 25.0

    def _get_energy_price(self, timestamp: datetime) -> float:
        """Get energy price for a timestamp."""
        if timestamp in self._carbon_forecast:
            price = self._carbon_forecast[timestamp].price_per_kwh
            if price is not None:
                return price

        # Default TOU pricing
        hour = timestamp.hour
        if 9 <= hour <= 17:  # Peak
            return 0.25
        elif 6 <= hour <= 22:  # Shoulder
            return 0.15
        else:  # Off-peak
            return 0.08

    def _check_constraints(
        self,
        task: OperationTask,
        slot_start: datetime,
        slot_end: datetime,
        existing_entries: List[ScheduleEntry],
    ) -> Tuple[List[str], List[str]]:
        """Check task constraints against scheduled slot."""
        satisfied = []
        violated = []

        for constraint in task.constraints:
            if constraint.constraint_type == ConstraintType.TIME_WINDOW:
                if constraint.start_time and slot_start < constraint.start_time:
                    violated.append(constraint.constraint_id)
                elif constraint.end_time and slot_end > constraint.end_time:
                    violated.append(constraint.constraint_id)
                else:
                    satisfied.append(constraint.constraint_id)

            elif constraint.constraint_type == ConstraintType.DEPENDENCY:
                # Check if dependent tasks are completed
                all_deps_done = True
                for dep_task_id in constraint.depends_on:
                    dep_entries = [
                        e for e in existing_entries
                        if e.task_id == dep_task_id and e.scheduled_end <= slot_start
                    ]
                    if not dep_entries:
                        all_deps_done = False
                        break

                if all_deps_done:
                    satisfied.append(constraint.constraint_id)
                else:
                    violated.append(constraint.constraint_id)

            elif constraint.constraint_type == ConstraintType.EXCLUSION:
                # Check if excluded tasks are running at same time
                has_exclusion = False
                for excluded_id in constraint.excluded_with:
                    for entry in existing_entries:
                        if entry.task_id == excluded_id:
                            if slot_start < entry.scheduled_end and slot_end > entry.scheduled_start:
                                has_exclusion = True
                                break

                if has_exclusion:
                    violated.append(constraint.constraint_id)
                else:
                    satisfied.append(constraint.constraint_id)

        return satisfied, violated

    # =========================================================================
    # Schedule Optimization
    # =========================================================================

    def _handle_optimize_schedule(
        self,
        facility_id: Optional[str],
        optimization_goal: OptimizationGoal,
        carbon_forecast: List[GridCarbonForecast],
    ) -> Dict[str, Any]:
        """Optimize existing schedule."""
        # Update carbon forecast
        for forecast in carbon_forecast:
            self._carbon_forecast[forecast.timestamp] = forecast

        facilities = [facility_id] if facility_id else list(self._schedules.keys())

        optimizations_made = 0
        total_emissions_saved = 0.0
        total_cost_saved = 0.0

        for fac_id in facilities:
            entries = self._schedules.get(fac_id, [])

            # Get pending/scheduled entries that can be moved
            movable = [
                e for e in entries
                if e.status in [ScheduleStatus.PENDING, ScheduleStatus.SCHEDULED]
            ]

            for entry in movable:
                original_emissions = entry.estimated_emissions_kg
                original_cost = entry.estimated_cost

                # Try to find better slot
                better_slot = self._find_better_slot(entry, optimization_goal, entries)

                if better_slot:
                    slot_start, slot_end = better_slot
                    carbon_intensity = self._get_carbon_intensity(slot_start)

                    # Update entry
                    entry.scheduled_start = slot_start
                    entry.scheduled_end = slot_end
                    entry.grid_carbon_intensity = carbon_intensity

                    # Recalculate metrics
                    # Create a temporary task for calculation
                    temp_task = OperationTask(
                        task_id=entry.task_id,
                        task_name=entry.task_name,
                        facility_id=entry.facility_id,
                        duration_minutes=entry.duration_minutes,
                        emissions_rate_kg_per_hour=original_emissions / (entry.duration_minutes / 60) if entry.duration_minutes > 0 else 0,
                        energy_consumption_kwh=0,  # Simplified
                        base_cost=original_cost,
                    )

                    new_emissions = self._calculate_emissions(temp_task, carbon_intensity)
                    new_cost = self._calculate_cost(temp_task, slot_start)

                    entry.estimated_emissions_kg = new_emissions
                    entry.estimated_cost = new_cost

                    total_emissions_saved += original_emissions - new_emissions
                    total_cost_saved += original_cost - new_cost
                    optimizations_made += 1

        self._total_optimizations += 1

        return {
            "optimizations_made": optimizations_made,
            "total_emissions_saved_kg": round(total_emissions_saved, 4),
            "total_cost_saved": round(total_cost_saved, 2),
            "optimization_goal": optimization_goal.value,
        }

    def _find_better_slot(
        self,
        entry: ScheduleEntry,
        optimization_goal: OptimizationGoal,
        all_entries: List[ScheduleEntry],
    ) -> Optional[Tuple[datetime, datetime]]:
        """Find a better time slot for an entry."""
        current_score = self._calculate_entry_score(entry, optimization_goal)

        # Define search window (same day)
        window_start = entry.scheduled_start.replace(hour=0, minute=0, second=0)
        window_end = window_start + timedelta(days=1)

        slot_duration = timedelta(minutes=self.config.parameters.get("time_slot_minutes", 15))
        task_duration = timedelta(minutes=entry.duration_minutes)

        best_slot = None
        best_score = current_score

        current = window_start
        while current + task_duration <= window_end:
            slot_end = current + task_duration

            # Skip current slot
            if current == entry.scheduled_start:
                current += slot_duration
                continue

            # Check for conflicts with other entries
            other_entries = [e for e in all_entries if e.entry_id != entry.entry_id]
            if not self._has_conflict(current, slot_end, other_entries, entry.facility_id):
                # Create temp entry for scoring
                temp_entry = entry.model_copy()
                temp_entry.scheduled_start = current
                temp_entry.scheduled_end = slot_end
                temp_entry.grid_carbon_intensity = self._get_carbon_intensity(current)

                score = self._calculate_entry_score(temp_entry, optimization_goal)

                if optimization_goal == OptimizationGoal.MAXIMIZE_RENEWABLE:
                    if score > best_score:
                        best_score = score
                        best_slot = (current, slot_end)
                else:
                    if score < best_score * 0.95:  # 5% improvement threshold
                        best_score = score
                        best_slot = (current, slot_end)

            current += slot_duration

        return best_slot

    def _calculate_entry_score(
        self, entry: ScheduleEntry, optimization_goal: OptimizationGoal
    ) -> float:
        """Calculate score for an entry."""
        if optimization_goal == OptimizationGoal.MINIMIZE_EMISSIONS:
            return entry.estimated_emissions_kg
        elif optimization_goal == OptimizationGoal.MINIMIZE_COST:
            return entry.estimated_cost
        elif optimization_goal == OptimizationGoal.MAXIMIZE_RENEWABLE:
            return self._get_renewable_percent(entry.scheduled_start)
        else:
            return entry.estimated_emissions_kg * 0.5 + entry.estimated_cost * 0.5

    # =========================================================================
    # Schedule Query
    # =========================================================================

    def _handle_get_schedule(
        self,
        facility_id: Optional[str],
        start_time: Optional[datetime],
        end_time: Optional[datetime],
    ) -> Dict[str, Any]:
        """Get schedule entries."""
        now = DeterministicClock.now()

        if end_time is None:
            end_time = now + timedelta(days=7)
        if start_time is None:
            start_time = now

        facilities = [facility_id] if facility_id else list(self._schedules.keys())

        all_entries = []
        for fac_id in facilities:
            entries = self._schedules.get(fac_id, [])
            filtered = [
                e for e in entries
                if e.scheduled_start >= start_time and e.scheduled_end <= end_time
            ]
            all_entries.extend(filtered)

        # Sort by start time
        all_entries.sort(key=lambda e: e.scheduled_start)

        return {
            "entries": [e.model_dump() for e in all_entries],
            "count": len(all_entries),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
        }

    def _handle_update_entry_status(
        self,
        entry_id: Optional[str],
        new_status: Optional[ScheduleStatus],
    ) -> Dict[str, Any]:
        """Update schedule entry status."""
        if not entry_id:
            return {"error": "entry_id is required"}
        if not new_status:
            return {"error": "new_status is required"}

        for entries in self._schedules.values():
            for entry in entries:
                if entry.entry_id == entry_id:
                    entry.status = new_status
                    return {
                        "entry_id": entry_id,
                        "new_status": new_status.value,
                        "updated": True,
                    }

        return {"entry_id": entry_id, "updated": False, "error": "Entry not found"}

    def _handle_reschedule(
        self,
        entry_id: Optional[str],
        new_start: Optional[datetime],
    ) -> Dict[str, Any]:
        """Reschedule an entry to a new time."""
        if not entry_id:
            return {"error": "entry_id is required"}
        if not new_start:
            return {"error": "start_time is required for reschedule"}

        for entries in self._schedules.values():
            for entry in entries:
                if entry.entry_id == entry_id:
                    old_start = entry.scheduled_start
                    entry.scheduled_start = new_start
                    entry.scheduled_end = new_start + timedelta(minutes=entry.duration_minutes)
                    entry.grid_carbon_intensity = self._get_carbon_intensity(new_start)

                    return {
                        "entry_id": entry_id,
                        "old_start": old_start.isoformat(),
                        "new_start": new_start.isoformat(),
                        "rescheduled": True,
                    }

        return {"entry_id": entry_id, "rescheduled": False, "error": "Entry not found"}

    # =========================================================================
    # Conflict Detection
    # =========================================================================

    def _handle_get_conflicts(self, facility_id: Optional[str]) -> Dict[str, Any]:
        """Get scheduling conflicts."""
        facilities = [facility_id] if facility_id else list(self._schedules.keys())

        conflicts = []

        for fac_id in facilities:
            entries = self._schedules.get(fac_id, [])

            # Check each pair of entries
            for i, entry1 in enumerate(entries):
                for entry2 in entries[i + 1:]:
                    if entry1.scheduled_start < entry2.scheduled_end and entry1.scheduled_end > entry2.scheduled_start:
                        conflicts.append({
                            "entry1_id": entry1.entry_id,
                            "entry2_id": entry2.entry_id,
                            "facility_id": fac_id,
                            "overlap_start": max(entry1.scheduled_start, entry2.scheduled_start).isoformat(),
                            "overlap_end": min(entry1.scheduled_end, entry2.scheduled_end).isoformat(),
                        })

        return {
            "conflicts": conflicts,
            "conflict_count": len(conflicts),
        }

    # =========================================================================
    # Resource Utilization
    # =========================================================================

    def _handle_get_resource_utilization(
        self,
        facility_id: Optional[str],
        start_time: Optional[datetime],
        end_time: Optional[datetime],
    ) -> Dict[str, Any]:
        """Get resource utilization statistics."""
        now = DeterministicClock.now()

        if end_time is None:
            end_time = now + timedelta(days=1)
        if start_time is None:
            start_time = now

        facilities = [facility_id] if facility_id else list(self._schedules.keys())

        utilization = {}

        for fac_id in facilities:
            entries = self._schedules.get(fac_id, [])
            filtered = [
                e for e in entries
                if e.scheduled_start >= start_time and e.scheduled_end <= end_time
            ]

            total_window = (end_time - start_time).total_seconds() / 3600  # hours
            scheduled_hours = sum(e.duration_minutes / 60 for e in filtered)

            utilization[fac_id] = {
                "scheduled_hours": round(scheduled_hours, 2),
                "available_hours": round(total_window, 2),
                "utilization_percent": round((scheduled_hours / total_window) * 100, 2) if total_window > 0 else 0,
                "entry_count": len(filtered),
            }

        return {
            "utilization": utilization,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
        }

    # =========================================================================
    # Statistics
    # =========================================================================

    def _handle_get_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        total_entries = sum(len(entries) for entries in self._schedules.values())

        return {
            "total_schedules_created": self._total_schedules_created,
            "total_optimizations": self._total_optimizations,
            "total_entries": total_entries,
            "facilities_scheduled": len(self._schedules),
            "carbon_forecasts_cached": len(self._carbon_forecast),
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
