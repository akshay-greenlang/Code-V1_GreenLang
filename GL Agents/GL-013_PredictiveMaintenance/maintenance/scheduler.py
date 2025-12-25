"""GL-013 PredictiveMaintenance - Maintenance Scheduler Module"""

from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import hashlib, logging
from collections import defaultdict
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class MaintenanceType(str, Enum):
    INSPECTION = "inspection"
    PREVENTIVE = "preventive"
    PREDICTIVE = "predictive"
    CORRECTIVE = "corrective"
    CONDITION_BASED = "condition_based"

class MaintenancePriority(str, Enum):
    EMERGENCY = "emergency"
    URGENT = "urgent"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    DEFERRED = "deferred"

class TechnicianSkill(str, Enum):
    MECHANICAL = "mechanical"
    ELECTRICAL = "electrical"
    INSTRUMENTATION = "instrumentation"
    VIBRATION = "vibration"
    LUBRICATION = "lubrication"
    ALIGNMENT = "alignment"

class ProductionWindow(BaseModel):
    window_id: str
    start_time: datetime
    end_time: datetime
    production_line: str
    available_for_maintenance: bool = True
    capacity_reduction_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    notes: Optional[str] = None

class TechnicianAvailability(BaseModel):
    technician_id: str
    name: str
    skills: List[TechnicianSkill]
    available_from: datetime
    available_until: datetime
    location: Optional[str] = None
    current_workload_hours: float = Field(default=0.0, ge=0.0)
    max_daily_hours: float = Field(default=8.0, ge=0.0, le=24.0)

class MaintenanceRequest(BaseModel):
    request_id: str
    asset_id: str
    maintenance_type: MaintenanceType
    priority: MaintenancePriority
    estimated_duration_hours: float = Field(default=2.0, ge=0.1)
    required_skills: List[TechnicianSkill] = Field(default_factory=list)
    earliest_start: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    latest_completion: datetime = Field(default_factory=lambda: datetime.now(timezone.utc) + timedelta(days=7))
    production_line: Optional[str] = None
    group_id: Optional[str] = None
    uncertainty_level: str = Field(default="medium")
    risk_score: float = Field(default=50.0, ge=0.0, le=100.0)

class ScheduledMaintenance(BaseModel):
    schedule_id: str
    request_id: str
    asset_id: str
    scheduled_start: datetime
    scheduled_end: datetime
    assigned_technician: Optional[str] = None
    production_window_id: Optional[str] = None
    maintenance_type: MaintenanceType
    priority: MaintenancePriority
    status: str = Field(default="scheduled")
    group_id: Optional[str] = None
    conflict_resolved: bool = False
    provenance_hash: str

class MaintenanceSchedulerConfig(BaseModel):
    max_daily_maintenance_hours: float = Field(default=16.0)
    min_gap_between_maintenance_minutes: int = Field(default=30, ge=0)
    group_work_radius_hours: float = Field(default=4.0)
    uncertainty_gating_enabled: bool = Field(default=True)
    production_priority_weight: float = Field(default=0.6)
    risk_priority_weight: float = Field(default=0.4)

class MaintenanceScheduler:
    """Production-aware maintenance scheduler with multi-asset optimization."""

    def __init__(self, config: Optional[MaintenanceSchedulerConfig] = None):
        self.config = config or MaintenanceSchedulerConfig()
        self._schedule_count = 0
        self._schedules: List[ScheduledMaintenance] = []
        logger.info("MaintenanceScheduler initialized")

    def schedule_maintenance(self, requests: List[MaintenanceRequest], production_windows: List[ProductionWindow], technicians: List[TechnicianAvailability]) -> List[ScheduledMaintenance]:
        """Schedule maintenance requests considering production and technician constraints."""
        schedules = []
        if self.config.uncertainty_gating_enabled:
            requests = self._apply_uncertainty_gating(requests)
        sorted_requests = self._prioritize_requests(requests)
        available_slots = self._find_available_slots(production_windows)
        for request in sorted_requests:
            schedule = self._schedule_single_request(request, available_slots, technicians)
            if schedule:
                schedules.append(schedule)
                self._update_slot_availability(available_slots, schedule)
        self._schedules.extend(schedules)
        return schedules

    def _apply_uncertainty_gating(self, requests: List[MaintenanceRequest]) -> List[MaintenanceRequest]:
        """High uncertainty -> recommend inspection instead of replacement."""
        gated = []
        for req in requests:
            if req.uncertainty_level == "high" and req.maintenance_type != MaintenanceType.INSPECTION:
                gated_req = req.copy()
                gated_req.maintenance_type = MaintenanceType.INSPECTION
                gated_req.estimated_duration_hours = min(req.estimated_duration_hours, 2.0)
                gated.append(gated_req)
            else:
                gated.append(req)
        return gated

    def _prioritize_requests(self, requests: List[MaintenanceRequest]) -> List[MaintenanceRequest]:
        priority_order = {MaintenancePriority.EMERGENCY: 0, MaintenancePriority.URGENT: 1, MaintenancePriority.HIGH: 2, MaintenancePriority.MEDIUM: 3, MaintenancePriority.LOW: 4, MaintenancePriority.DEFERRED: 5}
        return sorted(requests, key=lambda r: (priority_order.get(r.priority, 5), -r.risk_score))

    def _find_available_slots(self, windows: List[ProductionWindow]) -> List[Dict]:
        slots = []
        for w in windows:
            if w.available_for_maintenance:
                slots.append({"window_id": w.window_id, "start": w.start_time, "end": w.end_time, "production_line": w.production_line, "remaining_hours": (w.end_time - w.start_time).total_seconds() / 3600})
        return sorted(slots, key=lambda s: s["start"])

    def _schedule_single_request(self, request: MaintenanceRequest, slots: List[Dict], technicians: List[TechnicianAvailability]) -> Optional[ScheduledMaintenance]:
        for slot in slots:
            if slot["remaining_hours"] >= request.estimated_duration_hours:
                if request.earliest_start <= slot["end"] and request.latest_completion >= slot["start"]:
                    start = max(slot["start"], request.earliest_start)
                    end = start + timedelta(hours=request.estimated_duration_hours)
                    tech = self._find_available_technician(technicians, request.required_skills, start, end)
                    self._schedule_count += 1
                    prov = hashlib.sha256(f"{request.request_id}|{start.isoformat()}".encode()).hexdigest()
                    return ScheduledMaintenance(schedule_id=f"SCHED-{self._schedule_count:06d}", request_id=request.request_id, asset_id=request.asset_id, scheduled_start=start, scheduled_end=end, assigned_technician=tech, production_window_id=slot["window_id"], maintenance_type=request.maintenance_type, priority=request.priority, group_id=request.group_id, provenance_hash=prov)
        return None

    def _find_available_technician(self, technicians: List[TechnicianAvailability], required_skills: List[TechnicianSkill], start: datetime, end: datetime) -> Optional[str]:
        for tech in technicians:
            if start >= tech.available_from and end <= tech.available_until:
                if not required_skills or any(s in tech.skills for s in required_skills):
                    return tech.technician_id
        return None

    def _update_slot_availability(self, slots: List[Dict], schedule: ScheduledMaintenance) -> None:
        for slot in slots:
            if slot["window_id"] == schedule.production_window_id:
                used_hours = (schedule.scheduled_end - schedule.scheduled_start).total_seconds() / 3600
                slot["remaining_hours"] -= used_hours
                slot["start"] = schedule.scheduled_end + timedelta(minutes=self.config.min_gap_between_maintenance_minutes)
                break

    def optimize_schedule(self, schedules: List[ScheduledMaintenance]) -> List[ScheduledMaintenance]:
        """Optimize schedule by grouping nearby maintenance tasks."""
        if not schedules: return schedules
        return sorted(schedules, key=lambda s: (s.group_id or "", s.scheduled_start))
