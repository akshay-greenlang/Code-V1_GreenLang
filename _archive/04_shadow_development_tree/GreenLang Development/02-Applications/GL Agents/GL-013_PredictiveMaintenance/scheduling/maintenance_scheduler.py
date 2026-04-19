# -*- coding: utf-8 -*-
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np

class SchedulePriority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    PLANNED = 5

class MaintenanceType(Enum):
    PREVENTIVE = 'preventive'
    PREDICTIVE = 'predictive'
    CORRECTIVE = 'corrective'
    CONDITION_BASED = 'condition_based'
    TIME_BASED = 'time_based'

class ScheduleStatus(Enum):
    DRAFT = 'draft'
    SCHEDULED = 'scheduled'
    IN_PROGRESS = 'in_progress'
    COMPLETED = 'completed'
    CANCELLED = 'cancelled'
    DEFERRED = 'deferred'

@dataclass
class MaintenanceWindow:
    start_time: datetime
    end_time: datetime
    is_production_critical: bool = False
    max_concurrent_jobs: int = 5
    allowed_maintenance_types: List[MaintenanceType] = field(default_factory=lambda: list(MaintenanceType))

@dataclass
class ScheduledTask:
    task_id: str
    asset_id: str
    maintenance_type: MaintenanceType
    priority: SchedulePriority
    scheduled_start: datetime
    estimated_duration_hours: float
    required_skills: List[str]
    required_parts: List[str]
    predicted_failure_date: Optional[datetime] = None
    rul_hours: Optional[float] = None
    health_index: Optional[float] = None
    status: ScheduleStatus = ScheduleStatus.DRAFT

@dataclass
class ScheduleOptimizationResult:
    schedule: List[ScheduledTask]
    total_cost: float
    total_downtime_hours: float
    resource_utilization: float
    risk_reduction: float
    optimization_score: float
    provenance_hash: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class SchedulerConfig:
    planning_horizon_days: int = 30
    min_buffer_hours: float = 24.0
    max_concurrent_maintenance: int = 3
    risk_tolerance: float = 0.1
    cost_weight: float = 0.3
    downtime_weight: float = 0.4
    risk_weight: float = 0.3

class MaintenanceScheduler:
    def __init__(self, config: Optional[SchedulerConfig] = None):
        self.config = config or SchedulerConfig()
        self._maintenance_windows: List[MaintenanceWindow] = []
        self._scheduled_tasks: List[ScheduledTask] = []
        
    def add_maintenance_window(self, window: MaintenanceWindow) -> None:
        self._maintenance_windows.append(window)
        
    def get_available_windows(self, start: datetime, end: datetime) -> List[MaintenanceWindow]:
        return [w for w in self._maintenance_windows if w.start_time >= start and w.end_time <= end]
    
    def schedule_maintenance(self, task: ScheduledTask, preferred_window: Optional[MaintenanceWindow] = None) -> ScheduledTask:
        if preferred_window:
            task.scheduled_start = preferred_window.start_time
        task.status = ScheduleStatus.SCHEDULED
        self._scheduled_tasks.append(task)
        return task
    
    def optimize_schedule(self, tasks: List[ScheduledTask]) -> ScheduleOptimizationResult:
        # Sort by priority and predicted failure date
        sorted_tasks = sorted(tasks, key=lambda t: (t.priority.value, t.predicted_failure_date or datetime.max))
        
        # Simple greedy scheduling
        scheduled = []
        current_time = datetime.utcnow()
        total_cost = 0.0
        total_downtime = 0.0
        
        for task in sorted_tasks:
            # Find next available slot
            slot_start = current_time + timedelta(hours=self.config.min_buffer_hours)
            task.scheduled_start = slot_start
            task.status = ScheduleStatus.SCHEDULED
            scheduled.append(task)
            total_downtime += task.estimated_duration_hours
            total_cost += task.estimated_duration_hours * 100  # Base cost estimation
            current_time = slot_start + timedelta(hours=task.estimated_duration_hours)
        
        # Calculate metrics
        resource_util = min(1.0, len(scheduled) / max(1, self.config.max_concurrent_maintenance * self.config.planning_horizon_days))
        risk_reduction = sum(1.0 - (t.health_index or 0.5) for t in scheduled) / max(1, len(scheduled))
        
        optimization_score = (
            self.config.cost_weight * (1 - total_cost / 100000) +
            self.config.downtime_weight * (1 - total_downtime / 1000) +
            self.config.risk_weight * risk_reduction
        )
        
        # Calculate provenance
        hash_input = f'{len(scheduled)}:{total_cost}:{total_downtime}'
        provenance = hashlib.sha256(hash_input.encode()).hexdigest()
        
        return ScheduleOptimizationResult(
            schedule=scheduled,
            total_cost=total_cost,
            total_downtime_hours=total_downtime,
            resource_utilization=resource_util,
            risk_reduction=risk_reduction,
            optimization_score=optimization_score,
            provenance_hash=provenance,
        )
    
    def defer_task(self, task_id: str, new_date: datetime, reason: str) -> Optional[ScheduledTask]:
        for task in self._scheduled_tasks:
            if task.task_id == task_id:
                task.scheduled_start = new_date
                task.status = ScheduleStatus.DEFERRED
                return task
        return None
    
    def get_schedule(self, start: datetime, end: datetime) -> List[ScheduledTask]:
        return [t for t in self._scheduled_tasks if start <= t.scheduled_start <= end]
    
    def calculate_risk_of_deferral(self, task: ScheduledTask, defer_days: int) -> float:
        if task.rul_hours is None:
            return 0.5
        defer_hours = defer_days * 24
        remaining_after_defer = task.rul_hours - defer_hours
        if remaining_after_defer <= 0:
            return 1.0
        return max(0.0, 1.0 - remaining_after_defer / task.rul_hours)
