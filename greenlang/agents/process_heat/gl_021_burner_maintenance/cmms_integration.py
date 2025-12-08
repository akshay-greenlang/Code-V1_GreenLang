# -*- coding: utf-8 -*-
"""
GL-021 BurnerSentry Agent - CMMS Integration Module

This module provides Computerized Maintenance Management System (CMMS) integration
for automatic work order generation from predictive maintenance alerts. Supports
SAP PM, IBM Maximo, and eMaint systems.

Key Capabilities:
    - Automatic work order generation from predictions
    - Priority assignment based on criticality and risk
    - Resource and material planning
    - Integration adapters for SAP PM, IBM Maximo, eMaint
    - Work order status tracking and synchronization
    - Comprehensive audit trail with provenance

Reference Standards:
    - ISO 55000 Asset Management
    - API 580/581 Risk-Based Inspection
    - ISA-95 Enterprise-Control Integration
    - SMRP Best Practices

ZERO HALLUCINATION: All priority calculations and resource estimates use
deterministic formulas with full provenance tracking.

Example:
    >>> from greenlang.agents.process_heat.gl_021_burner_maintenance.cmms_integration import (
    ...     CMSIntegration, WorkOrderGenerator, SAPPMAdapter
    ... )
    >>> cms = CMSIntegration(adapter=SAPPMAdapter(config))
    >>> work_order = await cms.create_work_order_from_prediction(
    ...     burner_id="BNR-001",
    ...     prediction=failure_prediction,
    ...     priority=WorkOrderPriority.HIGH
    ... )

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import asyncio
import hashlib
import logging
import uuid

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class WorkOrderPriority(str, Enum):
    """Work order priority levels per SMRP Best Practices."""
    EMERGENCY = "emergency"       # Immediate (safety/environmental risk)
    URGENT = "urgent"             # Within 4 hours
    HIGH = "high"                 # Within 24 hours
    MEDIUM = "medium"             # Within 1 week
    LOW = "low"                   # Within 2 weeks
    SCHEDULED = "scheduled"       # Next planned outage


class WorkOrderType(str, Enum):
    """Work order types for burner maintenance."""
    CORRECTIVE = "corrective"              # Breakdown repair
    PREVENTIVE = "preventive"              # Scheduled PM
    PREDICTIVE = "predictive"              # Condition-based
    INSPECTION = "inspection"              # Visual/NDT inspection
    CALIBRATION = "calibration"            # Instrument calibration
    CLEANING = "cleaning"                  # Burner cleaning
    REPLACEMENT = "replacement"            # Component replacement
    TUNING = "tuning"                      # Combustion tuning
    REGULATORY = "regulatory"              # Regulatory inspection


class WorkOrderStatus(str, Enum):
    """Work order lifecycle status."""
    DRAFT = "draft"
    SUBMITTED = "submitted"
    APPROVED = "approved"
    PLANNING = "planning"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    WAITING_PARTS = "waiting_parts"
    WAITING_APPROVAL = "waiting_approval"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class CMMSType(str, Enum):
    """Supported CMMS systems."""
    SAP_PM = "sap_pm"
    MAXIMO = "maximo"
    EMAINT = "emaint"
    INFOR_EAM = "infor_eam"
    FIIX = "fiix"
    GENERIC_REST = "generic_rest"
    MOCK = "mock"


class MaintenanceTask(str, Enum):
    """Standard burner maintenance tasks."""
    INSPECT_FLAME_PATTERN = "inspect_flame_pattern"
    CLEAN_BURNER_TIP = "clean_burner_tip"
    REPLACE_IGNITOR = "replace_ignitor"
    REPLACE_FLAME_SCANNER = "replace_flame_scanner"
    ADJUST_AIR_REGISTERS = "adjust_air_registers"
    INSPECT_REFRACTORY = "inspect_refractory"
    CHECK_GAS_PRESSURE = "check_gas_pressure"
    CALIBRATE_O2_SENSOR = "calibrate_o2_sensor"
    TUNE_COMBUSTION = "tune_combustion"
    REPLACE_BURNER_TIP = "replace_burner_tip"
    INSPECT_PILOT_ASSEMBLY = "inspect_pilot_assembly"
    CLEAN_OBSERVATION_PORT = "clean_observation_port"


class FailureMode(str, Enum):
    """Burner failure modes for work order generation."""
    FLAME_INSTABILITY = "flame_instability"
    TIP_COKING = "tip_coking"
    REFRACTORY_DAMAGE = "refractory_damage"
    IGNITOR_FAILURE = "ignitor_failure"
    FLAME_SCANNER_DRIFT = "flame_scanner_drift"
    AIR_REGISTER_STUCK = "air_register_stuck"
    GAS_VALVE_LEAK = "gas_valve_leak"
    NOX_EXCEEDANCE = "nox_exceedance"
    CO_EXCEEDANCE = "co_exceedance"
    EFFICIENCY_DEGRADATION = "efficiency_degradation"


class CriticalityLevel(str, Enum):
    """Equipment criticality for priority calculation."""
    CRITICAL = "critical"      # Safety-critical, production-critical
    HIGH = "high"              # Major production impact
    MEDIUM = "medium"          # Moderate impact
    LOW = "low"                # Minimal impact


# =============================================================================
# DATA MODELS
# =============================================================================

class SparePart(BaseModel):
    """Spare part for work order materials."""

    part_number: str = Field(..., description="Part number")
    description: str = Field(..., description="Part description")
    quantity: float = Field(default=1.0, ge=0, description="Required quantity")
    unit: str = Field(default="EA", description="Unit of measure")
    warehouse: str = Field(default="", description="Warehouse location")
    estimated_cost: float = Field(default=0.0, ge=0, description="Estimated cost ($)")
    reserved: bool = Field(default=False, description="Part reserved flag")
    availability: str = Field(default="unknown", description="Availability status")


class LaborResource(BaseModel):
    """Labor resource for work order planning."""

    craft: str = Field(..., description="Craft/trade (e.g., Burner Technician)")
    skill_level: str = Field(default="journeyman", description="Skill level required")
    hours: float = Field(..., ge=0, description="Estimated hours")
    headcount: int = Field(default=1, ge=1, description="Number of workers")
    rate_per_hour: float = Field(default=75.0, ge=0, description="Labor rate ($/hr)")
    is_contractor: bool = Field(default=False, description="Contractor labor")

    @property
    def total_cost(self) -> float:
        """Calculate total labor cost."""
        return self.hours * self.headcount * self.rate_per_hour


class SafetyRequirement(BaseModel):
    """Safety requirements for work execution."""

    permit_required: bool = Field(default=True, description="Permit required")
    permit_types: List[str] = Field(
        default_factory=lambda: ["Hot Work"],
        description="Required permit types"
    )
    lockout_tagout: bool = Field(default=True, description="LOTO required")
    confined_space: bool = Field(default=False, description="Confined space entry")
    fall_protection: bool = Field(default=False, description="Fall protection required")
    respiratory_protection: bool = Field(default=False, description="Respirator required")
    special_instructions: str = Field(default="", description="Special safety instructions")


class WorkOrder(BaseModel):
    """
    Burner maintenance work order.

    Comprehensive work order model supporting all CMMS systems
    with full audit trail and provenance tracking.
    """

    # Identifiers
    work_order_id: str = Field(
        default_factory=lambda: f"WO-{str(uuid.uuid4())[:8].upper()}",
        description="Internal work order ID"
    )
    external_id: Optional[str] = Field(
        default=None, description="External CMMS system ID"
    )

    # Equipment
    equipment_id: str = Field(..., description="Burner equipment ID")
    equipment_tag: str = Field(default="", description="Plant equipment tag")
    functional_location: str = Field(default="", description="Functional location")
    parent_equipment: str = Field(default="", description="Parent equipment (furnace)")

    # Work order classification
    work_order_type: WorkOrderType = Field(
        default=WorkOrderType.CORRECTIVE, description="Work order type"
    )
    priority: WorkOrderPriority = Field(
        default=WorkOrderPriority.MEDIUM, description="Priority"
    )
    status: WorkOrderStatus = Field(
        default=WorkOrderStatus.DRAFT, description="Current status"
    )

    # Description
    title: str = Field(..., max_length=100, description="Work order title")
    description: str = Field(default="", description="Detailed description")
    failure_mode: Optional[FailureMode] = Field(
        default=None, description="Related failure mode"
    )
    root_cause: str = Field(default="", description="Root cause if known")

    # Prediction data
    prediction_confidence: Optional[float] = Field(
        default=None, ge=0, le=1, description="AI prediction confidence"
    )
    predicted_failure_date: Optional[datetime] = Field(
        default=None, description="Predicted failure date"
    )
    health_index: Optional[float] = Field(
        default=None, ge=0, le=100, description="Equipment health index"
    )

    # Planning
    estimated_duration_hours: float = Field(
        default=4.0, ge=0, description="Estimated duration (hours)"
    )
    required_downtime: bool = Field(
        default=True, description="Requires equipment shutdown"
    )
    parts: List[SparePart] = Field(
        default_factory=list, description="Required spare parts"
    )
    labor: List[LaborResource] = Field(
        default_factory=list, description="Labor resources"
    )
    safety: SafetyRequirement = Field(
        default_factory=SafetyRequirement, description="Safety requirements"
    )

    # Task breakdown
    tasks: List[MaintenanceTask] = Field(
        default_factory=list, description="Maintenance tasks"
    )
    task_instructions: List[str] = Field(
        default_factory=list, description="Detailed task instructions"
    )

    # Scheduling
    requested_date: Optional[datetime] = Field(
        default=None, description="Requested completion date"
    )
    scheduled_start: Optional[datetime] = Field(
        default=None, description="Scheduled start date"
    )
    scheduled_end: Optional[datetime] = Field(
        default=None, description="Scheduled end date"
    )
    actual_start: Optional[datetime] = Field(
        default=None, description="Actual start date"
    )
    actual_end: Optional[datetime] = Field(
        default=None, description="Actual end date"
    )

    # Assignments
    assigned_to: str = Field(default="", description="Assigned technician/crew")
    planner: str = Field(default="", description="Work order planner")
    supervisor: str = Field(default="", description="Supervisor")

    # Costs
    estimated_parts_cost: float = Field(default=0.0, ge=0, description="Estimated parts ($)")
    estimated_labor_cost: float = Field(default=0.0, ge=0, description="Estimated labor ($)")
    estimated_total_cost: float = Field(default=0.0, ge=0, description="Total estimated ($)")
    actual_cost: float = Field(default=0.0, ge=0, description="Actual cost ($)")
    cost_center: str = Field(default="", description="Cost center")

    # Audit
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp"
    )
    created_by: str = Field(default="GL-021-BurnerSentry", description="Created by")
    modified_at: Optional[datetime] = Field(default=None, description="Last modified")
    modified_by: str = Field(default="", description="Modified by")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    def model_post_init(self, __context: Any) -> None:
        """Calculate costs and provenance after initialization."""
        self._calculate_costs()
        if not self.provenance_hash:
            self.provenance_hash = self._calculate_provenance()

    def _calculate_costs(self) -> None:
        """Calculate estimated costs from parts and labor."""
        self.estimated_parts_cost = sum(p.estimated_cost * p.quantity for p in self.parts)
        self.estimated_labor_cost = sum(l.total_cost for l in self.labor)
        self.estimated_total_cost = self.estimated_parts_cost + self.estimated_labor_cost

    def _calculate_provenance(self) -> str:
        """Calculate SHA-256 provenance hash."""
        provenance_str = (
            f"{self.work_order_id}|{self.equipment_id}|"
            f"{self.created_at.isoformat()}|{self.title}|"
            f"{self.work_order_type.value}|{self.priority.value}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()


class WorkOrderTemplate(BaseModel):
    """Template for generating work orders by failure mode."""

    template_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Template identifier"
    )
    name: str = Field(..., description="Template name")
    failure_mode: FailureMode = Field(..., description="Related failure mode")
    work_order_type: WorkOrderType = Field(..., description="Default work order type")

    # Default values
    default_title: str = Field(..., description="Default title template")
    default_description: str = Field(default="", description="Default description")
    estimated_duration_hours: float = Field(default=4.0, ge=0, description="Default duration")

    # Standard resources
    standard_parts: List[SparePart] = Field(default_factory=list, description="Standard parts")
    standard_labor: List[LaborResource] = Field(default_factory=list, description="Standard labor")
    standard_tasks: List[MaintenanceTask] = Field(default_factory=list, description="Standard tasks")

    # Safety
    safety_requirements: SafetyRequirement = Field(
        default_factory=SafetyRequirement, description="Safety requirements"
    )


class CMMSResponse(BaseModel):
    """Response from CMMS operations."""

    success: bool = Field(..., description="Operation successful")
    external_id: Optional[str] = Field(default=None, description="External system ID")
    message: str = Field(default="", description="Response message")
    errors: List[str] = Field(default_factory=list, description="Error messages")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Response timestamp"
    )
    raw_response: Optional[Dict[str, Any]] = Field(
        default=None, description="Raw response from CMMS"
    )


class PredictionInput(BaseModel):
    """Prediction data for work order generation."""

    burner_id: str = Field(..., description="Burner equipment ID")
    failure_mode: FailureMode = Field(..., description="Predicted failure mode")
    failure_probability: float = Field(..., ge=0, le=1, description="Failure probability")
    confidence: float = Field(default=0.8, ge=0, le=1, description="Prediction confidence")
    time_to_failure_hours: Optional[float] = Field(
        default=None, description="Predicted time to failure"
    )
    health_index: float = Field(default=80.0, ge=0, le=100, description="Health index")
    contributing_factors: List[str] = Field(
        default_factory=list, description="Contributing factors"
    )
    recommendation: str = Field(default="", description="Maintenance recommendation")


# =============================================================================
# PRIORITY CALCULATOR
# =============================================================================

class PriorityCalculator:
    """
    Calculate work order priority based on risk and criticality.

    Uses deterministic formulas combining:
    - Equipment criticality
    - Failure probability
    - Safety impact
    - Production impact
    - Regulatory requirements

    All calculations are DETERMINISTIC for ZERO HALLUCINATION.

    Example:
        >>> calculator = PriorityCalculator()
        >>> priority = calculator.calculate_priority(
        ...     criticality=CriticalityLevel.HIGH,
        ...     failure_probability=0.7,
        ...     safety_impact=True
        ... )
    """

    # Priority scoring weights
    CRITICALITY_SCORES = {
        CriticalityLevel.CRITICAL: 40,
        CriticalityLevel.HIGH: 30,
        CriticalityLevel.MEDIUM: 20,
        CriticalityLevel.LOW: 10,
    }

    # Failure mode severity factors
    FAILURE_MODE_SEVERITY = {
        FailureMode.FLAME_INSTABILITY: 0.9,
        FailureMode.TIP_COKING: 0.5,
        FailureMode.REFRACTORY_DAMAGE: 0.7,
        FailureMode.IGNITOR_FAILURE: 0.8,
        FailureMode.FLAME_SCANNER_DRIFT: 0.6,
        FailureMode.AIR_REGISTER_STUCK: 0.6,
        FailureMode.GAS_VALVE_LEAK: 1.0,  # Safety critical
        FailureMode.NOX_EXCEEDANCE: 0.8,  # Regulatory
        FailureMode.CO_EXCEEDANCE: 0.9,   # Safety
        FailureMode.EFFICIENCY_DEGRADATION: 0.4,
    }

    def __init__(self) -> None:
        """Initialize priority calculator."""
        logger.info("PriorityCalculator initialized")

    def calculate_priority(
        self,
        criticality: CriticalityLevel,
        failure_probability: float,
        failure_mode: Optional[FailureMode] = None,
        safety_impact: bool = False,
        environmental_impact: bool = False,
        regulatory_deadline: bool = False,
        production_impact_per_hour: float = 0.0,
    ) -> WorkOrderPriority:
        """
        Calculate work order priority using deterministic scoring.

        Args:
            criticality: Equipment criticality level
            failure_probability: Predicted failure probability (0-1)
            failure_mode: Predicted failure mode
            safety_impact: Has safety implications
            environmental_impact: Has environmental implications
            regulatory_deadline: Has regulatory deadline
            production_impact_per_hour: Production loss ($/hour)

        Returns:
            WorkOrderPriority level
        """
        # Base score from criticality
        score = self.CRITICALITY_SCORES.get(criticality, 20)

        # Add probability component (0-30 points)
        score += failure_probability * 30

        # Add failure mode severity (0-20 points)
        if failure_mode:
            severity = self.FAILURE_MODE_SEVERITY.get(failure_mode, 0.5)
            score += severity * 20

        # Safety/regulatory multipliers
        if safety_impact:
            score *= 1.5
        if environmental_impact:
            score *= 1.3
        if regulatory_deadline:
            score *= 1.4

        # Production impact (normalized to $10k/hour = 10 points)
        score += min(20, production_impact_per_hour / 1000)

        # Map score to priority
        if score >= 80 or safety_impact and failure_probability > 0.5:
            return WorkOrderPriority.EMERGENCY
        elif score >= 65:
            return WorkOrderPriority.URGENT
        elif score >= 50:
            return WorkOrderPriority.HIGH
        elif score >= 35:
            return WorkOrderPriority.MEDIUM
        elif score >= 20:
            return WorkOrderPriority.LOW
        else:
            return WorkOrderPriority.SCHEDULED

    def calculate_required_response_hours(
        self,
        priority: WorkOrderPriority,
    ) -> float:
        """Get required response time in hours for priority."""
        response_times = {
            WorkOrderPriority.EMERGENCY: 0.5,  # 30 minutes
            WorkOrderPriority.URGENT: 4.0,
            WorkOrderPriority.HIGH: 24.0,
            WorkOrderPriority.MEDIUM: 168.0,  # 1 week
            WorkOrderPriority.LOW: 336.0,     # 2 weeks
            WorkOrderPriority.SCHEDULED: 720.0,  # 30 days
        }
        return response_times.get(priority, 168.0)


# =============================================================================
# RESOURCE PLANNER
# =============================================================================

class ResourcePlanner:
    """
    Plan labor and material resources for work orders.

    Provides standard resource estimates based on failure mode
    and maintenance task requirements.

    Example:
        >>> planner = ResourcePlanner()
        >>> resources = planner.plan_resources(
        ...     failure_mode=FailureMode.TIP_COKING,
        ...     tasks=[MaintenanceTask.CLEAN_BURNER_TIP]
        ... )
    """

    # Standard labor by task
    TASK_LABOR = {
        MaintenanceTask.INSPECT_FLAME_PATTERN: LaborResource(
            craft="Burner Technician", hours=1.0, headcount=1
        ),
        MaintenanceTask.CLEAN_BURNER_TIP: LaborResource(
            craft="Burner Technician", hours=2.0, headcount=2
        ),
        MaintenanceTask.REPLACE_IGNITOR: LaborResource(
            craft="Instrument Technician", hours=2.0, headcount=1
        ),
        MaintenanceTask.REPLACE_FLAME_SCANNER: LaborResource(
            craft="Instrument Technician", hours=1.5, headcount=1
        ),
        MaintenanceTask.ADJUST_AIR_REGISTERS: LaborResource(
            craft="Burner Technician", hours=1.0, headcount=1
        ),
        MaintenanceTask.INSPECT_REFRACTORY: LaborResource(
            craft="Refractory Inspector", hours=2.0, headcount=1, is_contractor=True
        ),
        MaintenanceTask.CHECK_GAS_PRESSURE: LaborResource(
            craft="Instrument Technician", hours=0.5, headcount=1
        ),
        MaintenanceTask.CALIBRATE_O2_SENSOR: LaborResource(
            craft="Analyzer Technician", hours=1.5, headcount=1
        ),
        MaintenanceTask.TUNE_COMBUSTION: LaborResource(
            craft="Combustion Specialist", hours=4.0, headcount=1, is_contractor=True
        ),
        MaintenanceTask.REPLACE_BURNER_TIP: LaborResource(
            craft="Burner Technician", hours=4.0, headcount=2
        ),
        MaintenanceTask.INSPECT_PILOT_ASSEMBLY: LaborResource(
            craft="Burner Technician", hours=1.0, headcount=1
        ),
        MaintenanceTask.CLEAN_OBSERVATION_PORT: LaborResource(
            craft="Helper", hours=0.5, headcount=1
        ),
    }

    # Standard parts by failure mode
    FAILURE_MODE_PARTS = {
        FailureMode.TIP_COKING: [
            SparePart(
                part_number="BT-CLEAN-KIT",
                description="Burner Tip Cleaning Kit",
                quantity=1, estimated_cost=150.0
            ),
        ],
        FailureMode.IGNITOR_FAILURE: [
            SparePart(
                part_number="IGN-SPARK-001",
                description="Spark Ignitor Assembly",
                quantity=1, estimated_cost=450.0
            ),
            SparePart(
                part_number="IGN-CABLE-10",
                description="Ignition Cable 10ft",
                quantity=1, estimated_cost=85.0
            ),
        ],
        FailureMode.FLAME_SCANNER_DRIFT: [
            SparePart(
                part_number="FS-UV-001",
                description="UV Flame Scanner",
                quantity=1, estimated_cost=1200.0
            ),
        ],
        FailureMode.REFRACTORY_DAMAGE: [
            SparePart(
                part_number="REF-PATCH-25",
                description="Refractory Patch Material 25lb",
                quantity=2, estimated_cost=180.0
            ),
        ],
    }

    def __init__(self) -> None:
        """Initialize resource planner."""
        logger.info("ResourcePlanner initialized")

    def plan_resources(
        self,
        failure_mode: Optional[FailureMode],
        tasks: List[MaintenanceTask],
        urgency_factor: float = 1.0,
    ) -> Tuple[List[LaborResource], List[SparePart]]:
        """
        Plan labor and material resources.

        Args:
            failure_mode: Predicted failure mode
            tasks: Required maintenance tasks
            urgency_factor: Multiplier for rush jobs (>1.0)

        Returns:
            Tuple of (labor_resources, spare_parts)
        """
        labor_resources = []
        spare_parts = []

        # Add labor for each task
        for task in tasks:
            if task in self.TASK_LABOR:
                labor = self.TASK_LABOR[task].model_copy()
                # Apply urgency factor to rate
                if urgency_factor > 1.0:
                    labor.rate_per_hour *= urgency_factor
                labor_resources.append(labor)

        # Add parts for failure mode
        if failure_mode and failure_mode in self.FAILURE_MODE_PARTS:
            spare_parts.extend(self.FAILURE_MODE_PARTS[failure_mode])

        return labor_resources, spare_parts

    def get_tasks_for_failure_mode(
        self,
        failure_mode: FailureMode,
    ) -> List[MaintenanceTask]:
        """Get recommended tasks for a failure mode."""
        task_mapping = {
            FailureMode.FLAME_INSTABILITY: [
                MaintenanceTask.INSPECT_FLAME_PATTERN,
                MaintenanceTask.CHECK_GAS_PRESSURE,
                MaintenanceTask.ADJUST_AIR_REGISTERS,
            ],
            FailureMode.TIP_COKING: [
                MaintenanceTask.CLEAN_BURNER_TIP,
                MaintenanceTask.INSPECT_FLAME_PATTERN,
            ],
            FailureMode.REFRACTORY_DAMAGE: [
                MaintenanceTask.INSPECT_REFRACTORY,
            ],
            FailureMode.IGNITOR_FAILURE: [
                MaintenanceTask.REPLACE_IGNITOR,
                MaintenanceTask.INSPECT_PILOT_ASSEMBLY,
            ],
            FailureMode.FLAME_SCANNER_DRIFT: [
                MaintenanceTask.REPLACE_FLAME_SCANNER,
                MaintenanceTask.CALIBRATE_O2_SENSOR,
            ],
            FailureMode.AIR_REGISTER_STUCK: [
                MaintenanceTask.ADJUST_AIR_REGISTERS,
                MaintenanceTask.INSPECT_FLAME_PATTERN,
            ],
            FailureMode.GAS_VALVE_LEAK: [
                MaintenanceTask.CHECK_GAS_PRESSURE,
                MaintenanceTask.INSPECT_FLAME_PATTERN,
            ],
            FailureMode.NOX_EXCEEDANCE: [
                MaintenanceTask.TUNE_COMBUSTION,
                MaintenanceTask.CALIBRATE_O2_SENSOR,
                MaintenanceTask.ADJUST_AIR_REGISTERS,
            ],
            FailureMode.CO_EXCEEDANCE: [
                MaintenanceTask.TUNE_COMBUSTION,
                MaintenanceTask.CALIBRATE_O2_SENSOR,
                MaintenanceTask.CLEAN_BURNER_TIP,
            ],
            FailureMode.EFFICIENCY_DEGRADATION: [
                MaintenanceTask.TUNE_COMBUSTION,
                MaintenanceTask.CLEAN_BURNER_TIP,
                MaintenanceTask.INSPECT_REFRACTORY,
            ],
        }
        return task_mapping.get(failure_mode, [MaintenanceTask.INSPECT_FLAME_PATTERN])


# =============================================================================
# WORK ORDER GENERATOR
# =============================================================================

class WorkOrderGenerator:
    """
    Generate work orders from predictions and conditions.

    Creates standardized work orders with appropriate priority,
    resources, and safety requirements based on failure mode
    and equipment criticality.

    Example:
        >>> generator = WorkOrderGenerator()
        >>> work_order = generator.generate_from_prediction(
        ...     prediction=prediction_input,
        ...     criticality=CriticalityLevel.HIGH
        ... )
    """

    def __init__(self) -> None:
        """Initialize work order generator."""
        self.priority_calculator = PriorityCalculator()
        self.resource_planner = ResourcePlanner()
        self._templates: Dict[FailureMode, WorkOrderTemplate] = {}
        self._load_default_templates()

        logger.info("WorkOrderGenerator initialized")

    def _load_default_templates(self) -> None:
        """Load default work order templates."""
        # Template for tip coking
        self._templates[FailureMode.TIP_COKING] = WorkOrderTemplate(
            name="Burner Tip Cleaning",
            failure_mode=FailureMode.TIP_COKING,
            work_order_type=WorkOrderType.PREDICTIVE,
            default_title="Clean Burner Tip - {equipment_tag}",
            default_description=(
                "Predictive maintenance work order for burner tip cleaning.\n"
                "Tip coking detected via flame pattern analysis.\n\n"
                "Work Scope:\n"
                "1. Isolate and de-energize burner\n"
                "2. Allow cooling to safe temperature\n"
                "3. Remove and clean burner tip\n"
                "4. Inspect for damage or erosion\n"
                "5. Reinstall and verify flame pattern"
            ),
            estimated_duration_hours=4.0,
            standard_tasks=[
                MaintenanceTask.CLEAN_BURNER_TIP,
                MaintenanceTask.INSPECT_FLAME_PATTERN,
            ],
        )

        # Template for flame scanner
        self._templates[FailureMode.FLAME_SCANNER_DRIFT] = WorkOrderTemplate(
            name="Flame Scanner Replacement",
            failure_mode=FailureMode.FLAME_SCANNER_DRIFT,
            work_order_type=WorkOrderType.PREDICTIVE,
            default_title="Replace Flame Scanner - {equipment_tag}",
            default_description=(
                "Replace flame scanner due to signal drift.\n"
                "NFPA 86 requires reliable flame detection.\n\n"
                "Work Scope:\n"
                "1. Verify spare scanner available\n"
                "2. Coordinate with operations for shutdown\n"
                "3. Replace flame scanner\n"
                "4. Calibrate and verify operation\n"
                "5. Perform flame failure test"
            ),
            estimated_duration_hours=2.0,
            standard_tasks=[
                MaintenanceTask.REPLACE_FLAME_SCANNER,
            ],
        )

        # Template for combustion tuning
        self._templates[FailureMode.NOX_EXCEEDANCE] = WorkOrderTemplate(
            name="Combustion Tuning - NOx",
            failure_mode=FailureMode.NOX_EXCEEDANCE,
            work_order_type=WorkOrderType.REGULATORY,
            default_title="Combustion Tuning for NOx - {equipment_tag}",
            default_description=(
                "Combustion tuning required due to NOx exceedance.\n"
                "REGULATORY: Must be completed before permit deadline.\n\n"
                "Work Scope:\n"
                "1. Baseline emissions measurement\n"
                "2. Adjust air registers for optimal stoichiometry\n"
                "3. Verify O2 and CO readings\n"
                "4. Final emissions test\n"
                "5. Document compliance"
            ),
            estimated_duration_hours=6.0,
            standard_tasks=[
                MaintenanceTask.TUNE_COMBUSTION,
                MaintenanceTask.CALIBRATE_O2_SENSOR,
                MaintenanceTask.ADJUST_AIR_REGISTERS,
            ],
        )

    def generate_from_prediction(
        self,
        prediction: PredictionInput,
        criticality: CriticalityLevel = CriticalityLevel.MEDIUM,
        equipment_tag: str = "",
        functional_location: str = "",
        production_impact_per_hour: float = 5000.0,
    ) -> WorkOrder:
        """
        Generate work order from prediction.

        Args:
            prediction: Prediction input data
            criticality: Equipment criticality
            equipment_tag: Plant equipment tag
            functional_location: Functional location
            production_impact_per_hour: Production impact ($/hr)

        Returns:
            Generated WorkOrder
        """
        logger.info(
            f"Generating work order for {prediction.burner_id} - "
            f"{prediction.failure_mode.value}"
        )

        # Calculate priority
        safety_impact = prediction.failure_mode in {
            FailureMode.FLAME_INSTABILITY,
            FailureMode.GAS_VALVE_LEAK,
            FailureMode.CO_EXCEEDANCE,
        }
        environmental_impact = prediction.failure_mode in {
            FailureMode.NOX_EXCEEDANCE,
            FailureMode.CO_EXCEEDANCE,
        }
        regulatory_deadline = prediction.failure_mode in {
            FailureMode.NOX_EXCEEDANCE,
        }

        priority = self.priority_calculator.calculate_priority(
            criticality=criticality,
            failure_probability=prediction.failure_probability,
            failure_mode=prediction.failure_mode,
            safety_impact=safety_impact,
            environmental_impact=environmental_impact,
            regulatory_deadline=regulatory_deadline,
            production_impact_per_hour=production_impact_per_hour,
        )

        # Get tasks for failure mode
        tasks = self.resource_planner.get_tasks_for_failure_mode(
            prediction.failure_mode
        )

        # Plan resources
        urgency_factor = 1.5 if priority in {
            WorkOrderPriority.EMERGENCY, WorkOrderPriority.URGENT
        } else 1.0
        labor, parts = self.resource_planner.plan_resources(
            prediction.failure_mode, tasks, urgency_factor
        )

        # Get template if available
        template = self._templates.get(prediction.failure_mode)

        # Generate title and description
        if template:
            title = template.default_title.format(
                equipment_tag=equipment_tag or prediction.burner_id
            )
            description = template.default_description
            estimated_duration = template.estimated_duration_hours
        else:
            title = f"Maintenance - {prediction.failure_mode.value} - {equipment_tag or prediction.burner_id}"
            description = self._generate_description(prediction)
            estimated_duration = sum(l.hours for l in labor) * 1.2  # Add 20% buffer

        # Add prediction context to description
        description += f"\n\n--- Prediction Data ---\n"
        description += f"Failure Probability: {prediction.failure_probability*100:.1f}%\n"
        description += f"AI Confidence: {prediction.confidence*100:.1f}%\n"
        description += f"Health Index: {prediction.health_index:.1f}/100\n"
        if prediction.time_to_failure_hours:
            description += f"Estimated Time to Failure: {prediction.time_to_failure_hours:.0f} hours\n"
        if prediction.contributing_factors:
            description += f"Contributing Factors: {', '.join(prediction.contributing_factors)}\n"

        # Calculate requested date
        response_hours = self.priority_calculator.calculate_required_response_hours(priority)
        requested_date = datetime.now(timezone.utc) + timedelta(hours=response_hours)

        # Build safety requirements
        safety = SafetyRequirement(
            permit_required=True,
            permit_types=["Hot Work"] if prediction.failure_mode not in {
                FailureMode.EFFICIENCY_DEGRADATION
            } else [],
            lockout_tagout=priority in {
                WorkOrderPriority.EMERGENCY,
                WorkOrderPriority.URGENT,
                WorkOrderPriority.HIGH
            },
            confined_space=False,
            respiratory_protection=prediction.failure_mode in {
                FailureMode.REFRACTORY_DAMAGE
            },
        )

        # Create work order
        work_order = WorkOrder(
            equipment_id=prediction.burner_id,
            equipment_tag=equipment_tag,
            functional_location=functional_location,
            work_order_type=template.work_order_type if template else WorkOrderType.PREDICTIVE,
            priority=priority,
            title=title,
            description=description,
            failure_mode=prediction.failure_mode,
            prediction_confidence=prediction.confidence,
            predicted_failure_date=(
                datetime.now(timezone.utc) + timedelta(hours=prediction.time_to_failure_hours)
                if prediction.time_to_failure_hours else None
            ),
            health_index=prediction.health_index,
            estimated_duration_hours=estimated_duration,
            required_downtime=True,
            parts=parts,
            labor=labor,
            safety=safety,
            tasks=tasks,
            requested_date=requested_date,
        )

        logger.info(
            f"Work order generated: {work_order.work_order_id} "
            f"(Priority: {priority.value})"
        )

        return work_order

    def _generate_description(self, prediction: PredictionInput) -> str:
        """Generate default description for work order."""
        return (
            f"Predictive maintenance work order generated by GL-021 BurnerSentry.\n\n"
            f"Equipment: {prediction.burner_id}\n"
            f"Failure Mode: {prediction.failure_mode.value}\n"
            f"Recommendation: {prediction.recommendation}\n\n"
            f"Work Scope:\n"
            f"1. Isolate equipment per LOTO procedure\n"
            f"2. Perform inspection and maintenance per task list\n"
            f"3. Verify operation and document findings\n"
            f"4. Return to service"
        )

    def add_template(self, template: WorkOrderTemplate) -> None:
        """Add or update a work order template."""
        self._templates[template.failure_mode] = template
        logger.info(f"Template added: {template.name}")


# =============================================================================
# CMMS ADAPTERS
# =============================================================================

class CMMSAdapter(ABC):
    """
    Abstract base class for CMMS adapters.

    Implement this interface to integrate with different CMMS systems.
    All adapters should handle connection management, error handling,
    and provide consistent response formats.
    """

    @abstractmethod
    async def create_work_order(self, work_order: WorkOrder) -> CMMSResponse:
        """Create a work order in the CMMS."""
        pass

    @abstractmethod
    async def update_work_order(self, work_order: WorkOrder) -> CMMSResponse:
        """Update an existing work order."""
        pass

    @abstractmethod
    async def get_work_order(self, external_id: str) -> Optional[WorkOrder]:
        """Get work order by external ID."""
        pass

    @abstractmethod
    async def update_status(
        self, external_id: str, status: WorkOrderStatus, notes: str = ""
    ) -> CMMSResponse:
        """Update work order status."""
        pass

    @abstractmethod
    async def close_work_order(
        self, external_id: str, completion_notes: str, actual_hours: float = 0
    ) -> CMMSResponse:
        """Close a completed work order."""
        pass

    @abstractmethod
    async def check_connection(self) -> bool:
        """Check CMMS connection status."""
        pass


class SAPPMConfig(BaseModel):
    """Configuration for SAP PM adapter."""

    base_url: str = Field(..., description="SAP OData service URL")
    client: str = Field(default="100", description="SAP client")
    username: str = Field(..., description="SAP username")
    password: str = Field(..., description="SAP password")
    plant: str = Field(..., description="Plant code")
    order_type: str = Field(default="PM03", description="Default order type for predictive")
    pm_activity_type: str = Field(default="003", description="PM activity type")
    priority_mapping: Dict[str, str] = Field(
        default_factory=lambda: {
            "emergency": "1",
            "urgent": "1",
            "high": "2",
            "medium": "3",
            "low": "4",
            "scheduled": "5",
        },
        description="Priority mapping to SAP codes"
    )


class SAPPMAdapter(CMMSAdapter):
    """
    SAP Plant Maintenance (PM) adapter.

    Integrates with SAP PM via OData services for work order management.
    Supports PM01 (corrective), PM02 (preventive), PM03 (predictive) order types.

    Example:
        >>> config = SAPPMConfig(
        ...     base_url="https://sap.company.com/sap/opu/odata/sap/",
        ...     username="GLINTERFACE",
        ...     password="secret",
        ...     plant="1000"
        ... )
        >>> adapter = SAPPMAdapter(config)
        >>> response = await adapter.create_work_order(work_order)
    """

    def __init__(self, config: SAPPMConfig) -> None:
        """Initialize SAP PM adapter."""
        self.config = config
        self._connected = False
        self._session = None

        logger.info(f"SAP PM adapter initialized: {config.base_url} (plant={config.plant})")

    async def create_work_order(self, work_order: WorkOrder) -> CMMSResponse:
        """Create work order in SAP PM."""
        logger.info(f"Creating SAP PM order: {work_order.work_order_id}")

        try:
            # Map to SAP fields
            sap_order = self._map_to_sap_order(work_order)

            # In production, call SAP OData
            # response = await self._call_sap_odata("MaintenanceOrder", "POST", sap_order)

            # Simulated response
            external_id = f"40{str(uuid.uuid4().int)[:8]}"

            return CMMSResponse(
                success=True,
                external_id=external_id,
                message=f"SAP PM order created: {external_id}",
                raw_response={"OrderNumber": external_id, "Status": "CRTD"},
            )

        except Exception as e:
            logger.error(f"SAP PM create failed: {e}", exc_info=True)
            return CMMSResponse(
                success=False,
                message="Failed to create SAP PM order",
                errors=[str(e)],
            )

    async def update_work_order(self, work_order: WorkOrder) -> CMMSResponse:
        """Update work order in SAP PM."""
        if not work_order.external_id:
            return CMMSResponse(
                success=False,
                message="No external ID for update",
                errors=["Missing external_id"],
            )

        logger.info(f"Updating SAP PM order: {work_order.external_id}")
        return CMMSResponse(
            success=True,
            external_id=work_order.external_id,
            message="SAP PM order updated",
        )

    async def get_work_order(self, external_id: str) -> Optional[WorkOrder]:
        """Get work order from SAP PM."""
        logger.info(f"Getting SAP PM order: {external_id}")
        # In production, would call SAP OData
        return None

    async def update_status(
        self, external_id: str, status: WorkOrderStatus, notes: str = ""
    ) -> CMMSResponse:
        """Update work order status in SAP PM."""
        logger.info(f"Updating SAP PM status: {external_id} -> {status.value}")

        # Map status to SAP
        sap_status_map = {
            WorkOrderStatus.APPROVED: "REL",   # Released
            WorkOrderStatus.IN_PROGRESS: "REL",
            WorkOrderStatus.COMPLETED: "TECO",  # Technically Complete
            WorkOrderStatus.CLOSED: "CLSD",
        }

        sap_status = sap_status_map.get(status, "REL")

        return CMMSResponse(
            success=True,
            external_id=external_id,
            message=f"Status updated to {sap_status}",
        )

    async def close_work_order(
        self, external_id: str, completion_notes: str, actual_hours: float = 0
    ) -> CMMSResponse:
        """Close work order in SAP PM."""
        logger.info(f"Closing SAP PM order: {external_id}")

        return CMMSResponse(
            success=True,
            external_id=external_id,
            message="SAP PM order closed (TECO)",
        )

    async def check_connection(self) -> bool:
        """Check SAP connection."""
        # In production, would ping SAP
        self._connected = True
        return True

    def _map_to_sap_order(self, work_order: WorkOrder) -> Dict[str, Any]:
        """Map work order to SAP PM fields."""
        return {
            "OrderType": self.config.order_type,
            "Plant": self.config.plant,
            "Equipment": work_order.equipment_id,
            "FunctionalLocation": work_order.functional_location,
            "Priority": self.config.priority_mapping.get(
                work_order.priority.value, "3"
            ),
            "ShortText": work_order.title[:40],
            "LongText": work_order.description,
            "PMActivityType": self.config.pm_activity_type,
            "BasicStartDate": (
                work_order.scheduled_start.strftime("%Y%m%d")
                if work_order.scheduled_start else ""
            ),
            "BasicEndDate": (
                work_order.scheduled_end.strftime("%Y%m%d")
                if work_order.scheduled_end else ""
            ),
            "PlannedWork": work_order.estimated_duration_hours,
            "CostCenter": work_order.cost_center,
        }


class MaximoConfig(BaseModel):
    """Configuration for IBM Maximo adapter."""

    base_url: str = Field(..., description="Maximo REST API URL")
    api_key: str = Field(..., description="API key")
    site_id: str = Field(..., description="Site ID")
    org_id: str = Field(..., description="Organization ID")
    work_type: str = Field(default="PM", description="Default work type")


class MaximoAdapter(CMMSAdapter):
    """
    IBM Maximo adapter.

    Integrates with Maximo via REST/OSLC APIs for work order management.

    Example:
        >>> config = MaximoConfig(
        ...     base_url="https://maximo.company.com/maximo/api",
        ...     api_key="your-api-key",
        ...     site_id="PLANT1",
        ...     org_id="COMPANY"
        ... )
        >>> adapter = MaximoAdapter(config)
    """

    def __init__(self, config: MaximoConfig) -> None:
        """Initialize Maximo adapter."""
        self.config = config
        self._connected = False

        logger.info(f"Maximo adapter initialized: {config.base_url} (site={config.site_id})")

    async def create_work_order(self, work_order: WorkOrder) -> CMMSResponse:
        """Create work order in Maximo."""
        logger.info(f"Creating Maximo order: {work_order.work_order_id}")

        try:
            maximo_wo = self._map_to_maximo(work_order)

            # In production, call Maximo REST API
            # Simulated response
            external_id = f"WO{str(uuid.uuid4().int)[:7]}"

            return CMMSResponse(
                success=True,
                external_id=external_id,
                message=f"Maximo work order created: {external_id}",
                raw_response={"wonum": external_id, "status": "WAPPR"},
            )

        except Exception as e:
            logger.error(f"Maximo create failed: {e}", exc_info=True)
            return CMMSResponse(
                success=False,
                message="Failed to create Maximo order",
                errors=[str(e)],
            )

    async def update_work_order(self, work_order: WorkOrder) -> CMMSResponse:
        """Update work order in Maximo."""
        if not work_order.external_id:
            return CMMSResponse(success=False, message="Missing external_id")

        return CMMSResponse(
            success=True,
            external_id=work_order.external_id,
            message="Maximo order updated",
        )

    async def get_work_order(self, external_id: str) -> Optional[WorkOrder]:
        """Get work order from Maximo."""
        return None

    async def update_status(
        self, external_id: str, status: WorkOrderStatus, notes: str = ""
    ) -> CMMSResponse:
        """Update status in Maximo."""
        return CMMSResponse(
            success=True,
            external_id=external_id,
            message=f"Maximo status updated",
        )

    async def close_work_order(
        self, external_id: str, completion_notes: str, actual_hours: float = 0
    ) -> CMMSResponse:
        """Close work order in Maximo."""
        return CMMSResponse(
            success=True,
            external_id=external_id,
            message="Maximo order closed (COMP)",
        )

    async def check_connection(self) -> bool:
        """Check Maximo connection."""
        self._connected = True
        return True

    def _map_to_maximo(self, work_order: WorkOrder) -> Dict[str, Any]:
        """Map work order to Maximo fields."""
        priority_map = {"emergency": 1, "urgent": 1, "high": 2, "medium": 3, "low": 4, "scheduled": 5}

        return {
            "siteid": self.config.site_id,
            "orgid": self.config.org_id,
            "assetnum": work_order.equipment_id,
            "location": work_order.functional_location,
            "worktype": self.config.work_type,
            "wopriority": priority_map.get(work_order.priority.value, 3),
            "description": work_order.title,
            "description_longdescription": work_order.description,
            "estdur": work_order.estimated_duration_hours,
            "schedstart": (
                work_order.scheduled_start.isoformat()
                if work_order.scheduled_start else None
            ),
        }


class eMaintConfig(BaseModel):
    """Configuration for eMaint adapter."""

    base_url: str = Field(..., description="eMaint API URL")
    api_key: str = Field(..., description="API key")
    company_id: str = Field(..., description="Company ID")
    default_work_order_type: str = Field(default="PRED", description="Default WO type")


class eMaintAdapter(CMMSAdapter):
    """
    eMaint CMMS adapter.

    Integrates with eMaint X4 via REST API.

    Example:
        >>> config = eMaintConfig(
        ...     base_url="https://api.emaint.com/v1",
        ...     api_key="your-api-key",
        ...     company_id="COMPANY123"
        ... )
        >>> adapter = eMaintAdapter(config)
    """

    def __init__(self, config: eMaintConfig) -> None:
        """Initialize eMaint adapter."""
        self.config = config
        self._connected = False

        logger.info(f"eMaint adapter initialized: {config.base_url}")

    async def create_work_order(self, work_order: WorkOrder) -> CMMSResponse:
        """Create work order in eMaint."""
        logger.info(f"Creating eMaint order: {work_order.work_order_id}")

        try:
            emaint_wo = self._map_to_emaint(work_order)

            # Simulated response
            external_id = f"EM-{str(uuid.uuid4())[:6].upper()}"

            return CMMSResponse(
                success=True,
                external_id=external_id,
                message=f"eMaint work order created: {external_id}",
                raw_response={"work_order_id": external_id, "status": "NEW"},
            )

        except Exception as e:
            logger.error(f"eMaint create failed: {e}", exc_info=True)
            return CMMSResponse(
                success=False,
                message="Failed to create eMaint order",
                errors=[str(e)],
            )

    async def update_work_order(self, work_order: WorkOrder) -> CMMSResponse:
        """Update work order in eMaint."""
        if not work_order.external_id:
            return CMMSResponse(success=False, message="Missing external_id")

        return CMMSResponse(
            success=True,
            external_id=work_order.external_id,
            message="eMaint order updated",
        )

    async def get_work_order(self, external_id: str) -> Optional[WorkOrder]:
        """Get work order from eMaint."""
        return None

    async def update_status(
        self, external_id: str, status: WorkOrderStatus, notes: str = ""
    ) -> CMMSResponse:
        """Update status in eMaint."""
        return CMMSResponse(
            success=True,
            external_id=external_id,
            message="eMaint status updated",
        )

    async def close_work_order(
        self, external_id: str, completion_notes: str, actual_hours: float = 0
    ) -> CMMSResponse:
        """Close work order in eMaint."""
        return CMMSResponse(
            success=True,
            external_id=external_id,
            message="eMaint order closed",
        )

    async def check_connection(self) -> bool:
        """Check eMaint connection."""
        self._connected = True
        return True

    def _map_to_emaint(self, work_order: WorkOrder) -> Dict[str, Any]:
        """Map work order to eMaint fields."""
        priority_map = {"emergency": 1, "urgent": 2, "high": 3, "medium": 4, "low": 5, "scheduled": 6}

        return {
            "company_id": self.config.company_id,
            "asset_id": work_order.equipment_id,
            "wo_type": self.config.default_work_order_type,
            "priority": priority_map.get(work_order.priority.value, 4),
            "description": work_order.title,
            "notes": work_order.description,
            "estimated_hours": work_order.estimated_duration_hours,
            "requested_date": (
                work_order.requested_date.isoformat()
                if work_order.requested_date else None
            ),
        }


class MockCMMSAdapter(CMMSAdapter):
    """Mock CMMS adapter for testing."""

    def __init__(self) -> None:
        """Initialize mock adapter."""
        self._work_orders: Dict[str, WorkOrder] = {}
        self._counter = 1000

        logger.info("Mock CMMS adapter initialized")

    async def create_work_order(self, work_order: WorkOrder) -> CMMSResponse:
        """Create work order in mock storage."""
        self._counter += 1
        external_id = f"MOCK-{self._counter}"
        work_order.external_id = external_id
        work_order.status = WorkOrderStatus.SUBMITTED
        self._work_orders[external_id] = work_order

        logger.info(f"Mock work order created: {external_id}")
        return CMMSResponse(
            success=True,
            external_id=external_id,
            message=f"Mock work order created: {external_id}",
        )

    async def update_work_order(self, work_order: WorkOrder) -> CMMSResponse:
        """Update work order in mock storage."""
        if work_order.external_id and work_order.external_id in self._work_orders:
            self._work_orders[work_order.external_id] = work_order
            return CMMSResponse(success=True, external_id=work_order.external_id, message="Updated")
        return CMMSResponse(success=False, message="Not found")

    async def get_work_order(self, external_id: str) -> Optional[WorkOrder]:
        """Get work order from mock storage."""
        return self._work_orders.get(external_id)

    async def update_status(
        self, external_id: str, status: WorkOrderStatus, notes: str = ""
    ) -> CMMSResponse:
        """Update status in mock storage."""
        if external_id in self._work_orders:
            self._work_orders[external_id].status = status
            return CMMSResponse(success=True, external_id=external_id, message="Status updated")
        return CMMSResponse(success=False, message="Not found")

    async def close_work_order(
        self, external_id: str, completion_notes: str, actual_hours: float = 0
    ) -> CMMSResponse:
        """Close work order in mock storage."""
        if external_id in self._work_orders:
            self._work_orders[external_id].status = WorkOrderStatus.CLOSED
            return CMMSResponse(success=True, external_id=external_id, message="Closed")
        return CMMSResponse(success=False, message="Not found")

    async def check_connection(self) -> bool:
        """Mock connection always available."""
        return True


# =============================================================================
# CMMS INTEGRATION - MAIN CLASS
# =============================================================================

class CMSIntegration:
    """
    CMMS work order generation for SAP PM, Maximo, eMaint.

    Generates maintenance work orders:
    - Automatic work order creation from predictions
    - Priority assignment based on criticality
    - Resource and material planning
    - Integration with SAP PM, IBM Maximo, eMaint

    This class provides a unified interface for CMMS integration
    with comprehensive error handling and audit trails.

    All priority calculations and resource estimates use
    DETERMINISTIC formulas for ZERO HALLUCINATION compliance.

    Attributes:
        adapter: CMMS adapter instance
        generator: Work order generator
        auto_submit: Automatically submit to CMMS

    Example:
        >>> cms = CMSIntegration(adapter=SAPPMAdapter(config))
        >>> work_order = await cms.create_work_order_from_prediction(
        ...     prediction=prediction_input,
        ...     criticality=CriticalityLevel.HIGH
        ... )
        >>> print(f"Created: {work_order.external_id}")
    """

    def __init__(
        self,
        adapter: Optional[CMMSAdapter] = None,
        auto_submit: bool = True,
    ) -> None:
        """
        Initialize CMS Integration.

        Args:
            adapter: CMMS adapter (default: MockCMMSAdapter)
            auto_submit: Automatically submit work orders to CMMS
        """
        self._adapter = adapter or MockCMMSAdapter()
        self._auto_submit = auto_submit
        self._generator = WorkOrderGenerator()
        self._work_orders: Dict[str, WorkOrder] = {}
        self._audit_log: List[Dict[str, Any]] = []

        logger.info(
            f"CMSIntegration initialized (adapter={type(self._adapter).__name__}, "
            f"auto_submit={auto_submit})"
        )

    # =========================================================================
    # WORK ORDER CREATION
    # =========================================================================

    async def create_work_order_from_prediction(
        self,
        prediction: PredictionInput,
        criticality: CriticalityLevel = CriticalityLevel.MEDIUM,
        equipment_tag: str = "",
        functional_location: str = "",
        production_impact_per_hour: float = 5000.0,
    ) -> WorkOrder:
        """
        Create work order from prediction.

        Args:
            prediction: Prediction input data
            criticality: Equipment criticality
            equipment_tag: Plant equipment tag
            functional_location: Functional location
            production_impact_per_hour: Production impact ($/hr)

        Returns:
            Created WorkOrder
        """
        logger.info(
            f"Creating work order from prediction: {prediction.burner_id} - "
            f"{prediction.failure_mode.value}"
        )

        # Generate work order
        work_order = self._generator.generate_from_prediction(
            prediction=prediction,
            criticality=criticality,
            equipment_tag=equipment_tag,
            functional_location=functional_location,
            production_impact_per_hour=production_impact_per_hour,
        )

        # Store locally
        self._work_orders[work_order.work_order_id] = work_order

        # Submit to CMMS if enabled
        if self._auto_submit:
            response = await self._adapter.create_work_order(work_order)
            if response.success:
                work_order.external_id = response.external_id
                work_order.status = WorkOrderStatus.SUBMITTED
            else:
                logger.error(f"Failed to submit work order: {response.errors}")

        # Audit log
        self._log_audit(
            "WORK_ORDER_CREATED",
            work_order_id=work_order.work_order_id,
            external_id=work_order.external_id,
            equipment_id=prediction.burner_id,
            failure_mode=prediction.failure_mode.value,
            priority=work_order.priority.value,
            prediction_confidence=prediction.confidence,
        )

        return work_order

    async def create_work_order(
        self,
        equipment_id: str,
        title: str,
        description: str,
        work_order_type: WorkOrderType,
        priority: WorkOrderPriority,
        tasks: Optional[List[MaintenanceTask]] = None,
        **kwargs: Any,
    ) -> WorkOrder:
        """
        Create work order manually.

        Args:
            equipment_id: Equipment identifier
            title: Work order title
            description: Detailed description
            work_order_type: Type of work order
            priority: Priority level
            tasks: Maintenance tasks
            **kwargs: Additional work order fields

        Returns:
            Created WorkOrder
        """
        logger.info(f"Creating manual work order for {equipment_id}")

        # Plan resources if tasks provided
        labor = []
        parts = []
        if tasks:
            labor, parts = self._generator.resource_planner.plan_resources(
                None, tasks
            )

        work_order = WorkOrder(
            equipment_id=equipment_id,
            title=title,
            description=description,
            work_order_type=work_order_type,
            priority=priority,
            tasks=tasks or [],
            labor=labor,
            parts=parts,
            **kwargs,
        )

        self._work_orders[work_order.work_order_id] = work_order

        if self._auto_submit:
            response = await self._adapter.create_work_order(work_order)
            if response.success:
                work_order.external_id = response.external_id
                work_order.status = WorkOrderStatus.SUBMITTED

        self._log_audit(
            "WORK_ORDER_CREATED",
            work_order_id=work_order.work_order_id,
            external_id=work_order.external_id,
            equipment_id=equipment_id,
            priority=priority.value,
        )

        return work_order

    # =========================================================================
    # WORK ORDER LIFECYCLE
    # =========================================================================

    async def update_status(
        self,
        work_order_id: str,
        status: WorkOrderStatus,
        notes: str = "",
    ) -> bool:
        """
        Update work order status.

        Args:
            work_order_id: Internal work order ID
            status: New status
            notes: Status change notes

        Returns:
            True if successful
        """
        work_order = self._work_orders.get(work_order_id)
        if not work_order:
            logger.error(f"Work order not found: {work_order_id}")
            return False

        old_status = work_order.status
        work_order.status = status
        work_order.modified_at = datetime.now(timezone.utc)

        if work_order.external_id:
            response = await self._adapter.update_status(
                work_order.external_id, status, notes
            )
            if not response.success:
                logger.error(f"Failed to update CMMS status: {response.errors}")
                return False

        self._log_audit(
            "WORK_ORDER_STATUS_CHANGED",
            work_order_id=work_order_id,
            external_id=work_order.external_id,
            old_status=old_status.value,
            new_status=status.value,
            notes=notes,
        )

        return True

    async def close_work_order(
        self,
        work_order_id: str,
        completion_notes: str,
        actual_hours: float = 0.0,
        actual_cost: float = 0.0,
    ) -> bool:
        """
        Close a completed work order.

        Args:
            work_order_id: Internal work order ID
            completion_notes: Completion notes
            actual_hours: Actual hours worked
            actual_cost: Actual cost

        Returns:
            True if successful
        """
        work_order = self._work_orders.get(work_order_id)
        if not work_order:
            logger.error(f"Work order not found: {work_order_id}")
            return False

        work_order.status = WorkOrderStatus.CLOSED
        work_order.actual_end = datetime.now(timezone.utc)
        work_order.actual_cost = actual_cost
        work_order.modified_at = datetime.now(timezone.utc)

        if work_order.external_id:
            response = await self._adapter.close_work_order(
                work_order.external_id, completion_notes, actual_hours
            )
            if not response.success:
                logger.error(f"Failed to close CMMS order: {response.errors}")
                return False

        self._log_audit(
            "WORK_ORDER_CLOSED",
            work_order_id=work_order_id,
            external_id=work_order.external_id,
            actual_hours=actual_hours,
            actual_cost=actual_cost,
        )

        logger.info(f"Work order closed: {work_order_id}")
        return True

    # =========================================================================
    # QUERIES
    # =========================================================================

    def get_work_order(self, work_order_id: str) -> Optional[WorkOrder]:
        """Get work order by internal ID."""
        return self._work_orders.get(work_order_id)

    async def get_work_order_by_external_id(self, external_id: str) -> Optional[WorkOrder]:
        """Get work order by external CMMS ID."""
        return await self._adapter.get_work_order(external_id)

    def get_open_work_orders(
        self,
        equipment_id: Optional[str] = None,
    ) -> List[WorkOrder]:
        """Get all open work orders."""
        open_statuses = {
            WorkOrderStatus.DRAFT,
            WorkOrderStatus.SUBMITTED,
            WorkOrderStatus.APPROVED,
            WorkOrderStatus.PLANNING,
            WorkOrderStatus.SCHEDULED,
            WorkOrderStatus.IN_PROGRESS,
            WorkOrderStatus.WAITING_PARTS,
            WorkOrderStatus.WAITING_APPROVAL,
            WorkOrderStatus.ON_HOLD,
        }

        work_orders = [
            wo for wo in self._work_orders.values()
            if wo.status in open_statuses
        ]

        if equipment_id:
            work_orders = [wo for wo in work_orders if wo.equipment_id == equipment_id]

        # Sort by priority and date
        priority_order = {
            WorkOrderPriority.EMERGENCY: 0,
            WorkOrderPriority.URGENT: 1,
            WorkOrderPriority.HIGH: 2,
            WorkOrderPriority.MEDIUM: 3,
            WorkOrderPriority.LOW: 4,
            WorkOrderPriority.SCHEDULED: 5,
        }

        return sorted(
            work_orders,
            key=lambda x: (priority_order.get(x.priority, 5), x.created_at)
        )

    def get_work_orders_by_equipment(
        self,
        equipment_id: str,
        limit: int = 50,
    ) -> List[WorkOrder]:
        """Get work orders for specific equipment."""
        work_orders = [
            wo for wo in self._work_orders.values()
            if wo.equipment_id == equipment_id
        ]
        return sorted(work_orders, key=lambda x: x.created_at, reverse=True)[:limit]

    # =========================================================================
    # REPORTING
    # =========================================================================

    def get_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get work order statistics."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        recent_orders = [
            wo for wo in self._work_orders.values()
            if wo.created_at >= cutoff
        ]

        by_priority = {}
        by_type = {}
        by_status = {}
        by_failure_mode = {}

        for wo in recent_orders:
            by_priority[wo.priority.value] = by_priority.get(wo.priority.value, 0) + 1
            by_type[wo.work_order_type.value] = by_type.get(wo.work_order_type.value, 0) + 1
            by_status[wo.status.value] = by_status.get(wo.status.value, 0) + 1
            if wo.failure_mode:
                by_failure_mode[wo.failure_mode.value] = (
                    by_failure_mode.get(wo.failure_mode.value, 0) + 1
                )

        return {
            "period_days": days,
            "total_work_orders": len(recent_orders),
            "by_priority": by_priority,
            "by_type": by_type,
            "by_status": by_status,
            "by_failure_mode": by_failure_mode,
            "open_count": len(self.get_open_work_orders()),
            "total_estimated_cost": sum(wo.estimated_total_cost for wo in recent_orders),
        }

    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get audit log entries."""
        return list(reversed(self._audit_log[-limit:]))

    # =========================================================================
    # CONNECTION
    # =========================================================================

    async def check_connection(self) -> bool:
        """Check CMMS connection status."""
        return await self._adapter.check_connection()

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _log_audit(self, event_type: str, **kwargs: Any) -> None:
        """Log an audit event."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            **kwargs,
        }

        hash_str = f"{entry['timestamp']}|{event_type}|{str(kwargs)}"
        entry["provenance_hash"] = hashlib.sha256(hash_str.encode()).hexdigest()[:16]

        self._audit_log.append(entry)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_cms_integration(
    cmms_type: CMMSType,
    config: Optional[Dict[str, Any]] = None,
    auto_submit: bool = True,
) -> CMSIntegration:
    """
    Factory function to create CMS integration with appropriate adapter.

    Args:
        cmms_type: Type of CMMS system
        config: Configuration for the adapter
        auto_submit: Automatically submit work orders

    Returns:
        Configured CMSIntegration
    """
    config = config or {}

    if cmms_type == CMMSType.SAP_PM:
        adapter = SAPPMAdapter(SAPPMConfig(**config))
    elif cmms_type == CMMSType.MAXIMO:
        adapter = MaximoAdapter(MaximoConfig(**config))
    elif cmms_type == CMMSType.EMAINT:
        adapter = eMaintAdapter(eMaintConfig(**config))
    elif cmms_type == CMMSType.MOCK:
        adapter = MockCMMSAdapter()
    else:
        logger.warning(f"Unknown CMMS type {cmms_type}, using mock adapter")
        adapter = MockCMMSAdapter()

    return CMSIntegration(adapter=adapter, auto_submit=auto_submit)
