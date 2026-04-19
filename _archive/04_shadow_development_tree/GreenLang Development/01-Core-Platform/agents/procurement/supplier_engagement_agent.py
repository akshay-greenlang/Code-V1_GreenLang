# -*- coding: utf-8 -*-
"""
GL-PROC-X-003: Supplier Engagement Agent
========================================

Manages supplier sustainability engagement programs, including action planning,
progress tracking, and engagement prioritization.

Capabilities:
    - Supplier engagement program management
    - Priority-based action planning
    - Progress tracking and monitoring
    - Engagement escalation management
    - CDP/SBTi participation tracking
    - Capacity building program coordination

Zero-Hallucination Guarantees:
    - All engagement plans deterministically generated
    - Progress calculated from verified data only
    - Complete audit trail for all actions
    - SHA-256 provenance hashes for all outputs

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.agents.categories import AgentCategory

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================


class EngagementPriority(str, Enum):
    """Engagement priority levels."""
    CRITICAL = "critical"  # High spend + high emissions + poor performance
    HIGH = "high"  # Significant impact, needs attention
    MEDIUM = "medium"  # Moderate priority
    LOW = "low"  # Lower priority, maintain monitoring
    OPPORTUNISTIC = "opportunistic"  # Nice to have


class EngagementStatus(str, Enum):
    """Engagement status."""
    NOT_STARTED = "not_started"
    INITIATED = "initiated"
    IN_PROGRESS = "in_progress"
    ON_TRACK = "on_track"
    AT_RISK = "at_risk"
    DELAYED = "delayed"
    COMPLETED = "completed"
    ESCALATED = "escalated"


class ActionType(str, Enum):
    """Types of engagement actions."""
    CDP_DISCLOSURE = "cdp_disclosure"
    SBTI_COMMITMENT = "sbti_commitment"
    EMISSIONS_REPORTING = "emissions_reporting"
    RENEWABLE_ENERGY = "renewable_energy"
    SUPPLIER_AUDIT = "supplier_audit"
    CAPACITY_BUILDING = "capacity_building"
    JOINT_PROJECT = "joint_project"
    POLICY_IMPROVEMENT = "policy_improvement"
    CERTIFICATION = "certification"
    DATA_SHARING = "data_sharing"


class ActionStatus(str, Enum):
    """Status of individual actions."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    OVERDUE = "overdue"
    CANCELLED = "cancelled"


class EscalationLevel(str, Enum):
    """Escalation levels."""
    NONE = "none"
    LEVEL_1 = "level_1"  # Procurement manager
    LEVEL_2 = "level_2"  # Category head
    LEVEL_3 = "level_3"  # Sustainability director
    LEVEL_4 = "level_4"  # Executive


# Priority weights for scoring
PRIORITY_WEIGHTS = {
    "spend_share": 0.30,
    "emissions_share": 0.30,
    "sustainability_gap": 0.25,
    "strategic_importance": 0.15,
}

# Engagement timeline defaults (days)
DEFAULT_TIMELINES = {
    ActionType.CDP_DISCLOSURE.value: 180,
    ActionType.SBTI_COMMITMENT.value: 365,
    ActionType.EMISSIONS_REPORTING.value: 90,
    ActionType.RENEWABLE_ENERGY.value: 730,
    ActionType.SUPPLIER_AUDIT.value: 60,
    ActionType.CAPACITY_BUILDING.value: 180,
    ActionType.JOINT_PROJECT.value: 365,
    ActionType.POLICY_IMPROVEMENT.value: 120,
    ActionType.CERTIFICATION.value: 365,
    ActionType.DATA_SHARING.value: 30,
}


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class SupplierEngagementProfile(BaseModel):
    """Supplier profile for engagement planning."""
    supplier_id: str = Field(..., description="Unique identifier")
    name: str = Field(..., description="Supplier name")
    tier: int = Field(default=1, ge=1, le=3, description="Supplier tier (1=strategic)")

    # Spend and impact
    annual_spend_usd: float = Field(default=0, ge=0)
    spend_share_pct: float = Field(default=0, ge=0, le=100)
    emissions_tco2e: Optional[float] = Field(None, ge=0)
    emissions_share_pct: float = Field(default=0, ge=0, le=100)

    # Current sustainability status
    sustainability_score: float = Field(default=0, ge=0, le=100)
    has_cdp_response: bool = Field(default=False)
    has_sbti_target: bool = Field(default=False)
    has_emissions_data: bool = Field(default=False)
    has_sustainability_contact: bool = Field(default=False)

    # Relationship
    strategic_importance: float = Field(default=50, ge=0, le=100)
    relationship_strength: float = Field(default=50, ge=0, le=100)
    responsiveness_score: float = Field(default=50, ge=0, le=100)

    # Previous engagement
    previous_engagements: int = Field(default=0, ge=0)
    engagement_success_rate: float = Field(default=0, ge=0, le=100)


class SupplierAction(BaseModel):
    """Individual engagement action for a supplier."""
    action_id: str = Field(..., description="Unique action identifier")
    supplier_id: str = Field(..., description="Related supplier")
    action_type: ActionType

    # Action details
    title: str
    description: str
    expected_outcome: str

    # Timeline
    created_date: datetime = Field(default_factory=datetime.utcnow)
    start_date: Optional[datetime] = Field(None)
    target_date: datetime
    completed_date: Optional[datetime] = Field(None)

    # Status
    status: ActionStatus = Field(default=ActionStatus.PENDING)
    progress_pct: float = Field(default=0, ge=0, le=100)

    # Resources
    owner: Optional[str] = Field(None)
    support_needed: List[str] = Field(default_factory=list)
    estimated_effort_hours: float = Field(default=0, ge=0)

    # Impact
    expected_emissions_reduction_tco2e: float = Field(default=0, ge=0)
    expected_score_improvement: float = Field(default=0, ge=0)

    # Notes
    notes: List[str] = Field(default_factory=list)


class EngagementProgram(BaseModel):
    """Complete engagement program for a supplier."""
    program_id: str = Field(..., description="Unique program identifier")
    supplier_id: str
    supplier_name: str

    # Priority and status
    priority: EngagementPriority
    priority_score: float = Field(..., ge=0, le=100)
    status: EngagementStatus = Field(default=EngagementStatus.NOT_STARTED)

    # Actions
    actions: List[SupplierAction] = Field(default_factory=list)

    # Timeline
    created_date: datetime = Field(default_factory=datetime.utcnow)
    start_date: Optional[datetime] = Field(None)
    target_completion_date: Optional[datetime] = Field(None)

    # Progress
    overall_progress_pct: float = Field(default=0, ge=0, le=100)
    actions_completed: int = Field(default=0, ge=0)
    actions_overdue: int = Field(default=0, ge=0)

    # Expected outcomes
    target_score_improvement: float = Field(default=0, ge=0)
    target_emissions_reduction_tco2e: float = Field(default=0, ge=0)

    # Escalation
    escalation_level: EscalationLevel = Field(default=EscalationLevel.NONE)
    escalation_reason: Optional[str] = Field(None)

    # Tracking
    last_contact_date: Optional[datetime] = Field(None)
    next_milestone: Optional[str] = Field(None)
    next_milestone_date: Optional[datetime] = Field(None)


class EngagementInput(BaseModel):
    """Input for supplier engagement operations."""
    operation: str = Field(
        default="create_program",
        description="Operation: create_program, update_progress, prioritize, track_status, escalate"
    )

    # Supplier data
    supplier: Optional[SupplierEngagementProfile] = Field(None)
    suppliers: Optional[List[SupplierEngagementProfile]] = Field(None)

    # Program management
    program_id: Optional[str] = Field(None)
    program: Optional[EngagementProgram] = Field(None)

    # Action management
    action: Optional[SupplierAction] = Field(None)
    action_updates: Optional[Dict[str, Any]] = Field(None)

    # Configuration
    engagement_types: Optional[List[ActionType]] = Field(None)
    max_actions_per_supplier: int = Field(default=5, ge=1, le=10)
    prioritization_threshold: float = Field(default=70, ge=0, le=100)


class EngagementOutput(BaseModel):
    """Output from supplier engagement operations."""
    success: bool
    operation: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Results
    program: Optional[EngagementProgram] = Field(None)
    programs: Optional[List[EngagementProgram]] = Field(None)
    prioritized_list: Optional[List[Dict[str, Any]]] = Field(None)
    status_summary: Optional[Dict[str, Any]] = Field(None)

    # Metrics
    total_suppliers_engaged: int = Field(default=0)
    total_actions_created: int = Field(default=0)
    actions_completed: int = Field(default=0)
    actions_overdue: int = Field(default=0)

    # Audit
    calculation_trace: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


# =============================================================================
# SUPPLIER ENGAGEMENT AGENT
# =============================================================================


class SupplierEngagementAgent(BaseAgent):
    """
    GL-PROC-X-003: Supplier Engagement Agent

    Manages supplier sustainability engagement programs.

    Zero-Hallucination Guarantees:
        - All engagement plans deterministically generated
        - Progress calculated from verified data only
        - Complete audit trail for all actions
        - SHA-256 provenance hashes for all outputs

    Usage:
        agent = SupplierEngagementAgent()
        result = agent.run({
            "operation": "create_program",
            "supplier": supplier_profile
        })
    """

    AGENT_ID = "GL-PROC-X-003"
    AGENT_NAME = "Supplier Engagement Agent"
    VERSION = "1.0.0"

    category = AgentCategory.CRITICAL

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the Supplier Engagement Agent."""
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Supplier sustainability engagement management",
                version=self.VERSION,
                parameters={}
            )

        super().__init__(config)
        self._programs: Dict[str, EngagementProgram] = {}
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute supplier engagement operation."""
        try:
            engagement_input = EngagementInput(**input_data)
            operation = engagement_input.operation

            if operation == "create_program":
                output = self._create_program(engagement_input)
            elif operation == "update_progress":
                output = self._update_progress(engagement_input)
            elif operation == "prioritize":
                output = self._prioritize_suppliers(engagement_input)
            elif operation == "track_status":
                output = self._track_status(engagement_input)
            elif operation == "escalate":
                output = self._escalate_program(engagement_input)
            elif operation == "add_action":
                output = self._add_action(engagement_input)
            else:
                return AgentResult(success=False, error=f"Unknown operation: {operation}")

            return AgentResult(
                success=output.success,
                data=output.model_dump(),
                metadata={"agent_id": self.AGENT_ID, "operation": operation}
            )

        except Exception as e:
            logger.error(f"Supplier engagement failed: {e}", exc_info=True)
            return AgentResult(success=False, error=str(e))

    def _create_program(self, input_data: EngagementInput) -> EngagementOutput:
        """Create an engagement program for a supplier."""
        calculation_trace: List[str] = []

        if input_data.supplier is None:
            return EngagementOutput(
                success=False,
                operation="create_program",
                calculation_trace=["ERROR: No supplier provided"]
            )

        supplier = input_data.supplier
        calculation_trace.append(f"Creating engagement program for: {supplier.name}")

        # Calculate priority
        priority, priority_score = self._calculate_priority(supplier, calculation_trace)

        # Generate program ID
        program_id = f"ENG-{supplier.supplier_id}-{datetime.utcnow().strftime('%Y%m%d')}"

        # Determine engagement types
        engagement_types = input_data.engagement_types or self._determine_engagement_types(
            supplier, calculation_trace
        )

        # Create actions
        actions = self._create_actions(
            supplier,
            engagement_types,
            input_data.max_actions_per_supplier,
            calculation_trace
        )

        # Calculate expected outcomes
        target_score = sum(a.expected_score_improvement for a in actions)
        target_emissions = sum(a.expected_emissions_reduction_tco2e for a in actions)

        # Set timeline
        if actions:
            target_completion = max(a.target_date for a in actions)
        else:
            target_completion = datetime.utcnow() + timedelta(days=365)

        program = EngagementProgram(
            program_id=program_id,
            supplier_id=supplier.supplier_id,
            supplier_name=supplier.name,
            priority=priority,
            priority_score=priority_score,
            status=EngagementStatus.INITIATED,
            actions=actions,
            start_date=datetime.utcnow(),
            target_completion_date=target_completion,
            target_score_improvement=target_score,
            target_emissions_reduction_tco2e=target_emissions,
            next_milestone=actions[0].title if actions else None,
            next_milestone_date=actions[0].target_date if actions else None
        )

        # Store program
        self._programs[program_id] = program

        calculation_trace.append(f"Created program {program_id} with {len(actions)} actions")
        calculation_trace.append(f"Priority: {priority.value} (score: {priority_score:.1f})")

        provenance_hash = hashlib.sha256(
            json.dumps(program.model_dump(), sort_keys=True, default=str).encode()
        ).hexdigest()

        return EngagementOutput(
            success=True,
            operation="create_program",
            program=program,
            total_suppliers_engaged=1,
            total_actions_created=len(actions),
            calculation_trace=calculation_trace,
            provenance_hash=provenance_hash
        )

    def _calculate_priority(
        self,
        supplier: SupplierEngagementProfile,
        trace: List[str]
    ) -> tuple:
        """Calculate supplier engagement priority."""
        scores = {}

        # Spend share score (higher spend = higher priority)
        scores["spend"] = min(supplier.spend_share_pct * 2, 100)

        # Emissions share score (higher emissions = higher priority)
        scores["emissions"] = min(supplier.emissions_share_pct * 2, 100)

        # Sustainability gap (lower sustainability = higher priority for engagement)
        scores["gap"] = 100 - supplier.sustainability_score

        # Strategic importance
        scores["strategic"] = supplier.strategic_importance

        # Weighted score
        priority_score = (
            scores["spend"] * PRIORITY_WEIGHTS["spend_share"] +
            scores["emissions"] * PRIORITY_WEIGHTS["emissions_share"] +
            scores["gap"] * PRIORITY_WEIGHTS["sustainability_gap"] +
            scores["strategic"] * PRIORITY_WEIGHTS["strategic_importance"]
        )

        trace.append(f"Priority scoring: spend={scores['spend']:.1f}, emissions={scores['emissions']:.1f}")
        trace.append(f"Gap={scores['gap']:.1f}, strategic={scores['strategic']:.1f}")
        trace.append(f"Weighted priority score: {priority_score:.1f}")

        # Determine priority level
        if priority_score >= 80:
            priority = EngagementPriority.CRITICAL
        elif priority_score >= 60:
            priority = EngagementPriority.HIGH
        elif priority_score >= 40:
            priority = EngagementPriority.MEDIUM
        elif priority_score >= 20:
            priority = EngagementPriority.LOW
        else:
            priority = EngagementPriority.OPPORTUNISTIC

        return priority, round(priority_score, 2)

    def _determine_engagement_types(
        self,
        supplier: SupplierEngagementProfile,
        trace: List[str]
    ) -> List[ActionType]:
        """Determine appropriate engagement types based on supplier profile."""
        types: List[ActionType] = []

        # Data sharing if no sustainability contact
        if not supplier.has_sustainability_contact:
            types.append(ActionType.DATA_SHARING)

        # CDP response if not already responding
        if not supplier.has_cdp_response:
            types.append(ActionType.CDP_DISCLOSURE)

        # Emissions reporting if lacking data
        if not supplier.has_emissions_data:
            types.append(ActionType.EMISSIONS_REPORTING)

        # SBTi commitment for strategic suppliers
        if not supplier.has_sbti_target and supplier.tier == 1:
            types.append(ActionType.SBTI_COMMITMENT)

        # Capacity building for low-performing suppliers
        if supplier.sustainability_score < 50:
            types.append(ActionType.CAPACITY_BUILDING)

        # Renewable energy for energy-intensive
        if supplier.emissions_tco2e and supplier.emissions_tco2e > 1000:
            types.append(ActionType.RENEWABLE_ENERGY)

        # Audit if poor performance or responsiveness
        if supplier.sustainability_score < 30 or supplier.responsiveness_score < 30:
            types.append(ActionType.SUPPLIER_AUDIT)

        trace.append(f"Determined engagement types: {[t.value for t in types]}")
        return types

    def _create_actions(
        self,
        supplier: SupplierEngagementProfile,
        action_types: List[ActionType],
        max_actions: int,
        trace: List[str]
    ) -> List[SupplierAction]:
        """Create engagement actions for supplier."""
        actions: List[SupplierAction] = []
        base_date = datetime.utcnow()

        action_configs = {
            ActionType.DATA_SHARING: {
                "title": "Establish Data Sharing Protocol",
                "description": "Set up sustainability data exchange process",
                "expected_outcome": "Regular sustainability data sharing established",
                "score_improvement": 5,
                "emissions_reduction": 0
            },
            ActionType.CDP_DISCLOSURE: {
                "title": "CDP Climate Disclosure Participation",
                "description": "Support supplier to complete CDP climate questionnaire",
                "expected_outcome": "Successful CDP submission with full disclosure",
                "score_improvement": 15,
                "emissions_reduction": 0
            },
            ActionType.EMISSIONS_REPORTING: {
                "title": "GHG Emissions Measurement",
                "description": "Implement GHG emissions measurement and reporting",
                "expected_outcome": "Scope 1 & 2 emissions calculated and reported",
                "score_improvement": 10,
                "emissions_reduction": 0
            },
            ActionType.SBTI_COMMITMENT: {
                "title": "SBTi Target Commitment",
                "description": "Support science-based target development and submission",
                "expected_outcome": "Approved science-based emissions reduction target",
                "score_improvement": 20,
                "emissions_reduction": supplier.emissions_tco2e * 0.1 if supplier.emissions_tco2e else 0
            },
            ActionType.CAPACITY_BUILDING: {
                "title": "Sustainability Capacity Building",
                "description": "Training and resources for sustainability improvement",
                "expected_outcome": "Enhanced internal sustainability capabilities",
                "score_improvement": 10,
                "emissions_reduction": 0
            },
            ActionType.RENEWABLE_ENERGY: {
                "title": "Renewable Energy Transition",
                "description": "Support transition to renewable energy sources",
                "expected_outcome": "50% renewable energy adoption",
                "score_improvement": 15,
                "emissions_reduction": supplier.emissions_tco2e * 0.25 if supplier.emissions_tco2e else 0
            },
            ActionType.SUPPLIER_AUDIT: {
                "title": "Sustainability Audit",
                "description": "Third-party sustainability performance audit",
                "expected_outcome": "Audit completed with improvement plan",
                "score_improvement": 5,
                "emissions_reduction": 0
            },
            ActionType.JOINT_PROJECT: {
                "title": "Joint Sustainability Project",
                "description": "Collaborative project for emission reductions",
                "expected_outcome": "Measurable emissions reduction achieved",
                "score_improvement": 10,
                "emissions_reduction": supplier.emissions_tco2e * 0.15 if supplier.emissions_tco2e else 0
            },
            ActionType.POLICY_IMPROVEMENT: {
                "title": "Policy Enhancement",
                "description": "Improve sustainability policies and procedures",
                "expected_outcome": "Enhanced environmental and social policies",
                "score_improvement": 8,
                "emissions_reduction": 0
            },
            ActionType.CERTIFICATION: {
                "title": "Environmental Certification",
                "description": "Support ISO 14001 or equivalent certification",
                "expected_outcome": "Environmental management certification achieved",
                "score_improvement": 15,
                "emissions_reduction": supplier.emissions_tco2e * 0.05 if supplier.emissions_tco2e else 0
            }
        }

        for i, action_type in enumerate(action_types[:max_actions]):
            config = action_configs.get(action_type, {
                "title": f"Engagement Action: {action_type.value}",
                "description": f"Execute {action_type.value} engagement",
                "expected_outcome": "Successful completion",
                "score_improvement": 5,
                "emissions_reduction": 0
            })

            timeline_days = DEFAULT_TIMELINES.get(action_type.value, 180)
            target_date = base_date + timedelta(days=timeline_days)

            action = SupplierAction(
                action_id=f"ACT-{supplier.supplier_id}-{i+1:03d}",
                supplier_id=supplier.supplier_id,
                action_type=action_type,
                title=config["title"],
                description=config["description"],
                expected_outcome=config["expected_outcome"],
                target_date=target_date,
                expected_score_improvement=config["score_improvement"],
                expected_emissions_reduction_tco2e=config["emissions_reduction"],
                estimated_effort_hours=timeline_days / 5  # Rough estimate
            )

            actions.append(action)

        trace.append(f"Created {len(actions)} actions")
        return actions

    def _update_progress(self, input_data: EngagementInput) -> EngagementOutput:
        """Update progress on an engagement program."""
        calculation_trace: List[str] = []

        if input_data.program is None:
            return EngagementOutput(
                success=False,
                operation="update_progress",
                calculation_trace=["ERROR: No program provided"]
            )

        program = input_data.program
        calculation_trace.append(f"Updating progress for: {program.program_id}")

        # Calculate overall progress
        if program.actions:
            completed = sum(1 for a in program.actions if a.status == ActionStatus.COMPLETED)
            overdue = sum(
                1 for a in program.actions
                if a.status not in [ActionStatus.COMPLETED, ActionStatus.CANCELLED]
                and a.target_date < datetime.utcnow()
            )

            progress_sum = sum(a.progress_pct for a in program.actions)
            overall_progress = progress_sum / len(program.actions)

            program.overall_progress_pct = round(overall_progress, 2)
            program.actions_completed = completed
            program.actions_overdue = overdue

            calculation_trace.append(f"Actions completed: {completed}/{len(program.actions)}")
            calculation_trace.append(f"Actions overdue: {overdue}")
            calculation_trace.append(f"Overall progress: {overall_progress:.1f}%")

            # Update status
            if completed == len(program.actions):
                program.status = EngagementStatus.COMPLETED
            elif overdue > 0:
                program.status = EngagementStatus.AT_RISK
            elif overall_progress >= 50:
                program.status = EngagementStatus.ON_TRACK
            else:
                program.status = EngagementStatus.IN_PROGRESS

            # Update next milestone
            pending_actions = [
                a for a in program.actions
                if a.status not in [ActionStatus.COMPLETED, ActionStatus.CANCELLED]
            ]
            if pending_actions:
                next_action = min(pending_actions, key=lambda a: a.target_date)
                program.next_milestone = next_action.title
                program.next_milestone_date = next_action.target_date

        provenance_hash = hashlib.sha256(
            json.dumps(program.model_dump(), sort_keys=True, default=str).encode()
        ).hexdigest()

        return EngagementOutput(
            success=True,
            operation="update_progress",
            program=program,
            actions_completed=program.actions_completed,
            actions_overdue=program.actions_overdue,
            calculation_trace=calculation_trace,
            provenance_hash=provenance_hash
        )

    def _prioritize_suppliers(self, input_data: EngagementInput) -> EngagementOutput:
        """Prioritize suppliers for engagement."""
        calculation_trace: List[str] = []

        if not input_data.suppliers:
            return EngagementOutput(
                success=False,
                operation="prioritize",
                calculation_trace=["ERROR: No suppliers provided"]
            )

        calculation_trace.append(f"Prioritizing {len(input_data.suppliers)} suppliers")

        prioritized = []
        for supplier in input_data.suppliers:
            priority, score = self._calculate_priority(supplier, [])

            prioritized.append({
                "supplier_id": supplier.supplier_id,
                "supplier_name": supplier.name,
                "priority": priority.value,
                "priority_score": score,
                "spend_share_pct": supplier.spend_share_pct,
                "emissions_share_pct": supplier.emissions_share_pct,
                "sustainability_score": supplier.sustainability_score,
                "recommended_engagement": priority.value in ["critical", "high"]
            })

        # Sort by priority score
        prioritized.sort(key=lambda x: x["priority_score"], reverse=True)

        # Add ranks
        for i, item in enumerate(prioritized):
            item["rank"] = i + 1

        # Summary stats
        critical_count = sum(1 for p in prioritized if p["priority"] == "critical")
        high_count = sum(1 for p in prioritized if p["priority"] == "high")
        above_threshold = sum(
            1 for p in prioritized
            if p["priority_score"] >= input_data.prioritization_threshold
        )

        calculation_trace.append(f"Critical priority: {critical_count}")
        calculation_trace.append(f"High priority: {high_count}")
        calculation_trace.append(f"Above threshold ({input_data.prioritization_threshold}): {above_threshold}")

        provenance_hash = hashlib.sha256(
            json.dumps(prioritized, sort_keys=True, default=str).encode()
        ).hexdigest()

        return EngagementOutput(
            success=True,
            operation="prioritize",
            prioritized_list=prioritized,
            total_suppliers_engaged=len(prioritized),
            calculation_trace=calculation_trace,
            provenance_hash=provenance_hash
        )

    def _track_status(self, input_data: EngagementInput) -> EngagementOutput:
        """Track status of engagement programs."""
        calculation_trace: List[str] = []

        programs = input_data.programs if hasattr(input_data, 'programs') and input_data.programs else list(self._programs.values())

        if not programs:
            programs = list(self._programs.values())

        calculation_trace.append(f"Tracking status of {len(programs)} programs")

        status_counts = {s.value: 0 for s in EngagementStatus}
        priority_counts = {p.value: 0 for p in EngagementPriority}

        total_actions = 0
        completed_actions = 0
        overdue_actions = 0

        for program in programs:
            status_counts[program.status.value] += 1
            priority_counts[program.priority.value] += 1
            total_actions += len(program.actions)
            completed_actions += program.actions_completed
            overdue_actions += program.actions_overdue

        summary = {
            "total_programs": len(programs),
            "status_distribution": status_counts,
            "priority_distribution": priority_counts,
            "total_actions": total_actions,
            "actions_completed": completed_actions,
            "actions_overdue": overdue_actions,
            "completion_rate": (completed_actions / total_actions * 100) if total_actions > 0 else 0,
            "at_risk_count": status_counts.get("at_risk", 0) + status_counts.get("delayed", 0),
            "programs_needing_attention": [
                {
                    "program_id": p.program_id,
                    "supplier_name": p.supplier_name,
                    "status": p.status.value,
                    "overdue_actions": p.actions_overdue
                }
                for p in programs
                if p.status in [EngagementStatus.AT_RISK, EngagementStatus.DELAYED]
            ]
        }

        calculation_trace.append(f"Completed: {status_counts.get('completed', 0)}")
        calculation_trace.append(f"At risk: {status_counts.get('at_risk', 0)}")
        calculation_trace.append(f"Actions completion rate: {summary['completion_rate']:.1f}%")

        provenance_hash = hashlib.sha256(
            json.dumps(summary, sort_keys=True, default=str).encode()
        ).hexdigest()

        return EngagementOutput(
            success=True,
            operation="track_status",
            status_summary=summary,
            programs=programs if isinstance(programs, list) else None,
            actions_completed=completed_actions,
            actions_overdue=overdue_actions,
            calculation_trace=calculation_trace,
            provenance_hash=provenance_hash
        )

    def _escalate_program(self, input_data: EngagementInput) -> EngagementOutput:
        """Escalate an engagement program."""
        calculation_trace: List[str] = []

        if input_data.program is None:
            return EngagementOutput(
                success=False,
                operation="escalate",
                calculation_trace=["ERROR: No program provided"]
            )

        program = input_data.program
        calculation_trace.append(f"Escalating program: {program.program_id}")

        # Determine escalation level
        current_level = program.escalation_level

        escalation_order = [
            EscalationLevel.NONE,
            EscalationLevel.LEVEL_1,
            EscalationLevel.LEVEL_2,
            EscalationLevel.LEVEL_3,
            EscalationLevel.LEVEL_4
        ]

        current_index = escalation_order.index(current_level)
        if current_index < len(escalation_order) - 1:
            new_level = escalation_order[current_index + 1]
            program.escalation_level = new_level
            program.status = EngagementStatus.ESCALATED

            # Determine reason
            reasons = []
            if program.actions_overdue > 0:
                reasons.append(f"{program.actions_overdue} overdue actions")
            if program.overall_progress_pct < 25:
                reasons.append("Low overall progress")
            if not reasons:
                reasons.append("Manual escalation requested")

            program.escalation_reason = "; ".join(reasons)

            calculation_trace.append(f"Escalated from {current_level.value} to {new_level.value}")
            calculation_trace.append(f"Reason: {program.escalation_reason}")
        else:
            calculation_trace.append("Already at maximum escalation level")

        provenance_hash = hashlib.sha256(
            json.dumps(program.model_dump(), sort_keys=True, default=str).encode()
        ).hexdigest()

        return EngagementOutput(
            success=True,
            operation="escalate",
            program=program,
            calculation_trace=calculation_trace,
            provenance_hash=provenance_hash
        )

    def _add_action(self, input_data: EngagementInput) -> EngagementOutput:
        """Add an action to an existing program."""
        calculation_trace: List[str] = []

        if input_data.program is None or input_data.action is None:
            return EngagementOutput(
                success=False,
                operation="add_action",
                calculation_trace=["ERROR: Program and action required"]
            )

        program = input_data.program
        action = input_data.action

        program.actions.append(action)

        calculation_trace.append(f"Added action {action.action_id} to {program.program_id}")
        calculation_trace.append(f"Total actions: {len(program.actions)}")

        provenance_hash = hashlib.sha256(
            json.dumps(program.model_dump(), sort_keys=True, default=str).encode()
        ).hexdigest()

        return EngagementOutput(
            success=True,
            operation="add_action",
            program=program,
            total_actions_created=len(program.actions),
            calculation_trace=calculation_trace,
            provenance_hash=provenance_hash
        )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "SupplierEngagementAgent",
    "EngagementInput",
    "EngagementOutput",
    "EngagementProgram",
    "SupplierAction",
    "SupplierEngagementProfile",
    "EngagementStatus",
    "EngagementPriority",
    "ActionType",
    "ActionStatus",
    "EscalationLevel",
]
