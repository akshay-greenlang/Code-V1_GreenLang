"""
GL-073: Maintenance Schedule Optimizer Agent (MAINT-SCHEDULER)

This module implements the MaintenanceSchedulerAgent for optimal maintenance
scheduling using reliability-centered maintenance (RCM) principles.

Standards Reference:
    - ISO 55000 (Asset Management)
    - SMRP Best Practices
    - ASME PTC (Performance Test Codes)
    - Reliability-Centered Maintenance (RCM)

Example:
    >>> agent = MaintenanceSchedulerAgent()
    >>> result = agent.run(input_data)
    >>> print(f"Optimized Schedule Efficiency: {result.schedule_efficiency_score}")
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class MaintenanceType(str, Enum):
    PREVENTIVE = "PREVENTIVE"
    PREDICTIVE = "PREDICTIVE"
    CORRECTIVE = "CORRECTIVE"
    CONDITION_BASED = "CONDITION_BASED"
    TIME_BASED = "TIME_BASED"


class AssetCriticality(str, Enum):
    CRITICAL = "CRITICAL"
    IMPORTANT = "IMPORTANT"
    STANDARD = "STANDARD"
    LOW = "LOW"


class MaintenanceStrategy(str, Enum):
    RUN_TO_FAILURE = "RUN_TO_FAILURE"
    TIME_BASED = "TIME_BASED"
    CONDITION_BASED = "CONDITION_BASED"
    PREDICTIVE = "PREDICTIVE"
    RELIABILITY_CENTERED = "RELIABILITY_CENTERED"


# =============================================================================
# INPUT MODELS
# =============================================================================

class Asset(BaseModel):
    asset_id: str = Field(..., description="Unique asset identifier")
    asset_name: str = Field(..., description="Asset name/description")
    asset_type: str = Field(..., description="Asset type/category")
    criticality: AssetCriticality = Field(..., description="Asset criticality level")
    commissioning_date: datetime = Field(..., description="Asset commissioning date")
    design_life_years: float = Field(default=20.0, gt=0, description="Design life in years")
    current_age_years: float = Field(default=0.0, ge=0, description="Current age in years")
    location: Optional[str] = Field(None, description="Asset location")


class MaintenanceTask(BaseModel):
    task_id: str = Field(..., description="Unique task identifier")
    asset_id: str = Field(..., description="Associated asset ID")
    task_description: str = Field(..., description="Task description")
    maintenance_type: MaintenanceType = Field(..., description="Type of maintenance")
    interval_days: int = Field(..., gt=0, description="Maintenance interval in days")
    duration_hours: float = Field(..., gt=0, description="Task duration in hours")
    last_performed: Optional[datetime] = Field(None, description="Last performance date")
    required_skills: List[str] = Field(default_factory=list, description="Required technician skills")
    estimated_cost: float = Field(default=1000.0, ge=0, description="Estimated cost in USD")
    priority: int = Field(default=3, ge=1, le=5, description="Priority (1=highest, 5=lowest)")


class ResourceConstraint(BaseModel):
    constraint_type: str = Field(..., description="Constraint type (e.g., TECHNICIANS, BUDGET, DOWNTIME)")
    available_quantity: float = Field(..., ge=0, description="Available quantity")
    unit: str = Field(..., description="Unit of measure")
    time_period: str = Field(default="WEEKLY", description="Time period for constraint")


class OperationalWindow(BaseModel):
    window_id: str = Field(..., description="Window identifier")
    start_date: datetime = Field(..., description="Window start date")
    end_date: datetime = Field(..., description="Window end date")
    available_downtime_hours: float = Field(..., ge=0, description="Available downtime hours")
    production_impact_cost_per_hour: float = Field(default=10000.0, ge=0, description="Production loss cost (USD/hr)")


class MaintenanceSchedulerInput(BaseModel):
    analysis_id: Optional[str] = Field(None, description="Analysis identifier")
    facility_name: str = Field(default="Industrial Facility", description="Facility name")
    assets: List[Asset] = Field(default_factory=list, description="Asset inventory")
    maintenance_tasks: List[MaintenanceTask] = Field(default_factory=list, description="Maintenance task list")
    resource_constraints: List[ResourceConstraint] = Field(default_factory=list, description="Resource constraints")
    operational_windows: List[OperationalWindow] = Field(default_factory=list, description="Planned downtime windows")
    planning_horizon_days: int = Field(default=365, gt=0, description="Planning horizon in days")
    current_date: datetime = Field(default_factory=datetime.utcnow, description="Current date for scheduling")
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# OUTPUT MODELS
# =============================================================================

class ScheduledTask(BaseModel):
    task_id: str
    asset_id: str
    asset_name: str
    task_description: str
    maintenance_type: str
    scheduled_date: datetime
    scheduled_window_id: Optional[str]
    duration_hours: float
    estimated_cost: float
    criticality: str
    days_until_due: int
    priority_score: float


class MaintenanceRecommendation(BaseModel):
    recommendation_id: str
    category: str
    priority: str
    description: str
    affected_assets: List[str]
    estimated_savings_usd: Optional[float] = None
    implementation_timeframe_days: Optional[int] = None


class MaintenanceWarning(BaseModel):
    warning_id: str
    warning_type: str
    asset_id: str
    asset_name: str
    description: str
    days_until_critical: Optional[int] = None
    recommended_action: str


class ResourceUtilization(BaseModel):
    resource_type: str
    total_available: float
    total_scheduled: float
    utilization_percent: float
    unit: str


class CostForecast(BaseModel):
    period: str
    preventive_cost_usd: float
    predictive_cost_usd: float
    corrective_cost_usd: float
    total_cost_usd: float
    downtime_cost_usd: float


class ProvenanceRecord(BaseModel):
    operation: str
    timestamp: datetime
    input_hash: str
    output_hash: str
    tool_name: str


class MaintenanceSchedulerOutput(BaseModel):
    analysis_id: str
    facility_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    planning_horizon_days: int

    # Scheduled Tasks
    scheduled_tasks: List[ScheduledTask] = Field(default_factory=list)
    total_tasks_scheduled: int
    critical_tasks_count: int
    overdue_tasks_count: int

    # Optimization Metrics
    schedule_efficiency_score: float = Field(..., ge=0, le=100, description="Schedule optimization score 0-100")
    resource_utilization_score: float = Field(..., ge=0, le=100)
    cost_optimization_score: float = Field(..., ge=0, le=100)

    # Resource Analysis
    resource_utilization: List[ResourceUtilization] = Field(default_factory=list)

    # Cost Forecast
    cost_forecasts: List[CostForecast] = Field(default_factory=list)
    total_forecast_cost_usd: float

    # Recommendations and Warnings
    recommendations: List[MaintenanceRecommendation] = Field(default_factory=list)
    warnings: List[MaintenanceWarning] = Field(default_factory=list)

    # Asset Health
    assets_requiring_attention: int
    average_asset_age_years: float

    # Provenance
    provenance_chain: List[ProvenanceRecord] = Field(default_factory=list)
    provenance_hash: str

    # Processing Metadata
    processing_time_ms: float
    validation_status: str
    validation_errors: List[str] = Field(default_factory=list)


# =============================================================================
# MAINTENANCE SCHEDULER AGENT
# =============================================================================

class MaintenanceSchedulerAgent:
    """GL-073: Maintenance Schedule Optimizer Agent - RCM-based scheduling."""

    AGENT_ID = "GL-073"
    AGENT_NAME = "MAINT-SCHEDULER"
    VERSION = "1.0.0"

    # Scoring weights
    CRITICALITY_WEIGHTS = {
        AssetCriticality.CRITICAL: 1.0,
        AssetCriticality.IMPORTANT: 0.7,
        AssetCriticality.STANDARD: 0.4,
        AssetCriticality.LOW: 0.2
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._provenance_steps: List[Dict[str, Any]] = []
        self._validation_errors: List[str] = []
        logger.info(f"MaintenanceSchedulerAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: MaintenanceSchedulerInput) -> MaintenanceSchedulerOutput:
        start_time = datetime.utcnow()
        self._provenance_steps = []
        self._validation_errors = []

        logger.info(f"Starting maintenance schedule optimization for {input_data.facility_name}")

        # Step 1: Build asset lookup
        asset_lookup = {asset.asset_id: asset for asset in input_data.assets}

        # Step 2: Calculate task priorities and due dates
        prioritized_tasks = self._prioritize_tasks(
            input_data.maintenance_tasks,
            asset_lookup,
            input_data.current_date
        )
        self._track_provenance(
            "task_prioritization",
            {"tasks_count": len(input_data.maintenance_tasks)},
            {"prioritized_count": len(prioritized_tasks)},
            "priority_calculator"
        )

        # Step 3: Schedule tasks within operational windows
        scheduled_tasks = self._schedule_tasks(
            prioritized_tasks,
            input_data.operational_windows,
            input_data.current_date,
            input_data.planning_horizon_days,
            asset_lookup
        )
        self._track_provenance(
            "task_scheduling",
            {"tasks_to_schedule": len(prioritized_tasks), "windows": len(input_data.operational_windows)},
            {"scheduled": len(scheduled_tasks)},
            "schedule_optimizer"
        )

        # Step 4: Calculate resource utilization
        resource_util = self._calculate_resource_utilization(
            scheduled_tasks,
            input_data.resource_constraints
        )
        self._track_provenance(
            "resource_utilization",
            {"constraints": len(input_data.resource_constraints)},
            {"utilization_count": len(resource_util)},
            "resource_analyzer"
        )

        # Step 5: Generate cost forecasts
        cost_forecasts = self._generate_cost_forecasts(
            scheduled_tasks,
            input_data.planning_horizon_days
        )
        total_cost = sum(cf.total_cost_usd for cf in cost_forecasts)
        self._track_provenance(
            "cost_forecasting",
            {"scheduled_tasks": len(scheduled_tasks)},
            {"total_cost_usd": total_cost},
            "cost_forecaster"
        )

        # Step 6: Generate recommendations and warnings
        recommendations = self._generate_recommendations(
            input_data.assets,
            scheduled_tasks,
            input_data.maintenance_tasks
        )
        warnings = self._generate_warnings(
            prioritized_tasks,
            asset_lookup,
            input_data.current_date
        )

        # Step 7: Calculate metrics
        critical_count = sum(1 for t in scheduled_tasks if t.criticality == AssetCriticality.CRITICAL.value)
        overdue_count = sum(1 for t in scheduled_tasks if t.days_until_due < 0)

        schedule_efficiency = self._calculate_schedule_efficiency(scheduled_tasks, prioritized_tasks)
        resource_util_score = self._calculate_resource_score(resource_util)
        cost_opt_score = self._calculate_cost_optimization_score(scheduled_tasks)

        # Asset health metrics
        assets_needing_attention = len(warnings)
        avg_age = sum(a.current_age_years for a in input_data.assets) / len(input_data.assets) if input_data.assets else 0.0

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance_hash()
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return MaintenanceSchedulerOutput(
            analysis_id=input_data.analysis_id or f"MAINT-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            facility_name=input_data.facility_name,
            planning_horizon_days=input_data.planning_horizon_days,
            scheduled_tasks=scheduled_tasks,
            total_tasks_scheduled=len(scheduled_tasks),
            critical_tasks_count=critical_count,
            overdue_tasks_count=overdue_count,
            schedule_efficiency_score=round(schedule_efficiency, 2),
            resource_utilization_score=round(resource_util_score, 2),
            cost_optimization_score=round(cost_opt_score, 2),
            resource_utilization=resource_util,
            cost_forecasts=cost_forecasts,
            total_forecast_cost_usd=round(total_cost, 2),
            recommendations=recommendations,
            warnings=warnings,
            assets_requiring_attention=assets_needing_attention,
            average_asset_age_years=round(avg_age, 2),
            provenance_chain=[ProvenanceRecord(**s) for s in self._provenance_steps],
            provenance_hash=provenance_hash,
            processing_time_ms=round(processing_time, 2),
            validation_status="PASS" if not self._validation_errors else "FAIL",
            validation_errors=self._validation_errors
        )

    def _prioritize_tasks(
        self,
        tasks: List[MaintenanceTask],
        asset_lookup: Dict[str, Asset],
        current_date: datetime
    ) -> List[Dict[str, Any]]:
        prioritized = []

        for task in tasks:
            asset = asset_lookup.get(task.asset_id)
            if not asset:
                continue

            # Calculate days until due
            if task.last_performed:
                next_due = task.last_performed + timedelta(days=task.interval_days)
                days_until_due = (next_due - current_date).days
            else:
                days_until_due = 0  # Overdue if never performed

            # Calculate priority score (higher = more urgent)
            criticality_weight = self.CRITICALITY_WEIGHTS.get(asset.criticality, 0.5)
            urgency_factor = max(0.1, 1.0 - (days_until_due / task.interval_days))
            priority_score = (criticality_weight * 50.0) + (urgency_factor * 50.0) + ((6 - task.priority) * 10.0)

            prioritized.append({
                "task": task,
                "asset": asset,
                "days_until_due": days_until_due,
                "priority_score": priority_score
            })

        # Sort by priority score (descending)
        return sorted(prioritized, key=lambda x: -x["priority_score"])

    def _schedule_tasks(
        self,
        prioritized_tasks: List[Dict[str, Any]],
        windows: List[OperationalWindow],
        current_date: datetime,
        horizon_days: int,
        asset_lookup: Dict[str, Asset]
    ) -> List[ScheduledTask]:
        scheduled = []
        window_usage = {w.window_id: 0.0 for w in windows}

        for item in prioritized_tasks:
            task = item["task"]
            asset = item["asset"]
            days_until_due = item["days_until_due"]
            priority_score = item["priority_score"]

            # Find suitable window
            best_window = None
            scheduled_date = None

            for window in windows:
                if window_usage[window.window_id] + task.duration_hours <= window.available_downtime_hours:
                    # Check if window is within acceptable timeframe
                    if days_until_due < 0 or window.start_date <= current_date + timedelta(days=days_until_due):
                        best_window = window
                        scheduled_date = window.start_date
                        window_usage[window.window_id] += task.duration_hours
                        break

            # If no window found, schedule at earliest opportunity
            if not scheduled_date:
                if days_until_due < 0:
                    scheduled_date = current_date  # Immediate
                else:
                    scheduled_date = current_date + timedelta(days=max(0, days_until_due - 7))

            scheduled.append(ScheduledTask(
                task_id=task.task_id,
                asset_id=task.asset_id,
                asset_name=asset.asset_name,
                task_description=task.task_description,
                maintenance_type=task.maintenance_type.value,
                scheduled_date=scheduled_date,
                scheduled_window_id=best_window.window_id if best_window else None,
                duration_hours=task.duration_hours,
                estimated_cost=task.estimated_cost,
                criticality=asset.criticality.value,
                days_until_due=days_until_due,
                priority_score=round(priority_score, 2)
            ))

        return scheduled

    def _calculate_resource_utilization(
        self,
        scheduled_tasks: List[ScheduledTask],
        constraints: List[ResourceConstraint]
    ) -> List[ResourceUtilization]:
        utilization = []

        # Calculate total hours scheduled
        total_hours = sum(t.duration_hours for t in scheduled_tasks)

        for constraint in constraints:
            if constraint.constraint_type == "TECHNICIANS":
                # Assume technicians work 40 hours per week
                available_hours = constraint.available_quantity * 40.0 * 52.0  # Annual
                util_percent = (total_hours / available_hours * 100.0) if available_hours > 0 else 0.0

                utilization.append(ResourceUtilization(
                    resource_type=constraint.constraint_type,
                    total_available=constraint.available_quantity,
                    total_scheduled=round(total_hours / 2080.0, 2),  # Convert to FTE
                    utilization_percent=round(util_percent, 2),
                    unit="FTE"
                ))
            elif constraint.constraint_type == "BUDGET":
                total_cost = sum(t.estimated_cost for t in scheduled_tasks)
                util_percent = (total_cost / constraint.available_quantity * 100.0) if constraint.available_quantity > 0 else 0.0

                utilization.append(ResourceUtilization(
                    resource_type=constraint.constraint_type,
                    total_available=constraint.available_quantity,
                    total_scheduled=round(total_cost, 2),
                    utilization_percent=round(util_percent, 2),
                    unit="USD"
                ))

        return utilization

    def _generate_cost_forecasts(
        self,
        scheduled_tasks: List[ScheduledTask],
        horizon_days: int
    ) -> List[CostForecast]:
        # Group tasks by quarter
        forecasts = []
        quarters = ["Q1", "Q2", "Q3", "Q4"]

        for i, quarter in enumerate(quarters):
            preventive = sum(
                t.estimated_cost for t in scheduled_tasks
                if t.maintenance_type == MaintenanceType.PREVENTIVE.value
            ) / 4.0

            predictive = sum(
                t.estimated_cost for t in scheduled_tasks
                if t.maintenance_type == MaintenanceType.PREDICTIVE.value
            ) / 4.0

            corrective = sum(
                t.estimated_cost for t in scheduled_tasks
                if t.maintenance_type == MaintenanceType.CORRECTIVE.value
            ) / 4.0

            downtime = sum(t.duration_hours * 1000.0 for t in scheduled_tasks) / 4.0  # Assume $1000/hr

            total = preventive + predictive + corrective

            forecasts.append(CostForecast(
                period=quarter,
                preventive_cost_usd=round(preventive, 2),
                predictive_cost_usd=round(predictive, 2),
                corrective_cost_usd=round(corrective, 2),
                total_cost_usd=round(total, 2),
                downtime_cost_usd=round(downtime, 2)
            ))

        return forecasts

    def _generate_recommendations(
        self,
        assets: List[Asset],
        scheduled_tasks: List[ScheduledTask],
        all_tasks: List[MaintenanceTask]
    ) -> List[MaintenanceRecommendation]:
        recommendations = []
        rec_id = 0

        # Recommend condition-based monitoring for aging assets
        aging_assets = [a for a in assets if a.current_age_years > a.design_life_years * 0.7]
        if aging_assets:
            rec_id += 1
            recommendations.append(MaintenanceRecommendation(
                recommendation_id=f"REC-{rec_id:03d}",
                category="PREDICTIVE_MONITORING",
                priority="HIGH",
                description=f"Implement condition-based monitoring for {len(aging_assets)} aging assets",
                affected_assets=[a.asset_id for a in aging_assets],
                estimated_savings_usd=50000.0 * len(aging_assets),
                implementation_timeframe_days=90
            ))

        # Recommend task optimization
        high_cost_tasks = [t for t in scheduled_tasks if t.estimated_cost > 10000.0]
        if len(high_cost_tasks) > 5:
            rec_id += 1
            recommendations.append(MaintenanceRecommendation(
                recommendation_id=f"REC-{rec_id:03d}",
                category="COST_OPTIMIZATION",
                priority="MEDIUM",
                description="Review high-cost maintenance tasks for potential optimization opportunities",
                affected_assets=list(set([t.asset_id for t in high_cost_tasks])),
                estimated_savings_usd=100000.0,
                implementation_timeframe_days=120
            ))

        return recommendations

    def _generate_warnings(
        self,
        prioritized_tasks: List[Dict[str, Any]],
        asset_lookup: Dict[str, Asset],
        current_date: datetime
    ) -> List[MaintenanceWarning]:
        warnings = []
        warn_id = 0

        for item in prioritized_tasks:
            task = item["task"]
            asset = item["asset"]
            days_until_due = item["days_until_due"]

            # Warning for overdue tasks
            if days_until_due < 0:
                warn_id += 1
                warnings.append(MaintenanceWarning(
                    warning_id=f"WARN-{warn_id:03d}",
                    warning_type="OVERDUE_MAINTENANCE",
                    asset_id=asset.asset_id,
                    asset_name=asset.asset_name,
                    description=f"Task '{task.task_description}' is {abs(days_until_due)} days overdue",
                    days_until_critical=0,
                    recommended_action="Schedule immediately to prevent equipment failure"
                ))

            # Warning for upcoming critical tasks
            elif days_until_due <= 30 and asset.criticality == AssetCriticality.CRITICAL:
                warn_id += 1
                warnings.append(MaintenanceWarning(
                    warning_id=f"WARN-{warn_id:03d}",
                    warning_type="CRITICAL_MAINTENANCE_DUE",
                    asset_id=asset.asset_id,
                    asset_name=asset.asset_name,
                    description=f"Critical asset maintenance due in {days_until_due} days",
                    days_until_critical=days_until_due,
                    recommended_action="Ensure resources are allocated for timely completion"
                ))

        return warnings

    def _calculate_schedule_efficiency(
        self,
        scheduled: List[ScheduledTask],
        prioritized: List[Dict[str, Any]]
    ) -> float:
        if not prioritized:
            return 100.0

        # Efficiency based on how many high-priority tasks are scheduled on time
        on_time = sum(1 for t in scheduled if t.days_until_due >= 0)
        efficiency = (on_time / len(prioritized) * 100.0) if prioritized else 100.0

        return min(100.0, efficiency)

    def _calculate_resource_score(self, utilization: List[ResourceUtilization]) -> float:
        if not utilization:
            return 100.0

        # Optimal utilization is 70-85%
        scores = []
        for util in utilization:
            if 70.0 <= util.utilization_percent <= 85.0:
                scores.append(100.0)
            elif util.utilization_percent < 70.0:
                scores.append(util.utilization_percent / 70.0 * 100.0)
            else:
                scores.append(max(0.0, 100.0 - (util.utilization_percent - 85.0) * 2.0))

        return sum(scores) / len(scores) if scores else 100.0

    def _calculate_cost_optimization_score(self, scheduled: List[ScheduledTask]) -> float:
        if not scheduled:
            return 100.0

        # Higher ratio of preventive/predictive vs corrective = better score
        preventive_count = sum(1 for t in scheduled if t.maintenance_type in [MaintenanceType.PREVENTIVE.value, MaintenanceType.PREDICTIVE.value])
        total_count = len(scheduled)

        ratio = (preventive_count / total_count * 100.0) if total_count > 0 else 0.0
        return min(100.0, ratio)

    def _track_provenance(self, operation: str, inputs: Dict, outputs: Dict, tool_name: str) -> None:
        self._provenance_steps.append({
            "operation": operation,
            "timestamp": datetime.utcnow(),
            "input_hash": hashlib.sha256(json.dumps(inputs, sort_keys=True, default=str).encode()).hexdigest(),
            "output_hash": hashlib.sha256(json.dumps(outputs, sort_keys=True, default=str).encode()).hexdigest(),
            "tool_name": tool_name
        })

    def _calculate_provenance_hash(self) -> str:
        data = {
            "agent_id": self.AGENT_ID,
            "version": self.VERSION,
            "steps": [
                {
                    "operation": s["operation"],
                    "input_hash": s["input_hash"],
                    "output_hash": s["output_hash"]
                }
                for s in self._provenance_steps
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "GL-073",
    "name": "MAINT-SCHEDULER",
    "version": "1.0.0",
    "summary": "Reliability-centered maintenance schedule optimization",
    "tags": ["maintenance", "scheduling", "RCM", "asset-management", "reliability"],
    "standards": [
        {"ref": "ISO 55000", "description": "Asset Management"},
        {"ref": "SMRP Best Practices", "description": "Society for Maintenance & Reliability Professionals"},
        {"ref": "RCM", "description": "Reliability-Centered Maintenance methodology"}
    ],
    "provenance": {
        "calculation_verified": True,
        "enable_audit": True
    }
}
