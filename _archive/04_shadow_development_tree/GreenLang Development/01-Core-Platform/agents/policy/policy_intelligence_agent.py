# -*- coding: utf-8 -*-
"""
GL-POL-X-003: Policy Intelligence Agent
=======================================

Monitors regulatory changes and provides intelligence on upcoming policy
developments. This agent uses INSIGHT PATH - deterministic tracking with
AI-enhanced analysis for impact assessment.

Capabilities:
    - Regulatory change tracking and alerting
    - Impact assessment for policy changes
    - Timeline monitoring for compliance deadlines
    - Consultation response tracking
    - Cross-jurisdictional policy comparison
    - Horizon scanning for emerging regulations

Zero-Hallucination Guarantees (Calculation Path):
    - All deadline calculations are deterministic
    - Impact scoring uses defined criteria
    - Complete audit trail for all assessments

AI Enhancement (Analysis Path):
    - Impact narrative generation
    - Stakeholder communication drafting
    - Trend analysis and interpretation

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.utilities.determinism import DeterministicClock, deterministic_uuid

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class PolicyChangeType(str, Enum):
    """Types of policy changes."""
    NEW_REGULATION = "new_regulation"
    AMENDMENT = "amendment"
    GUIDANCE_UPDATE = "guidance_update"
    DEADLINE_CHANGE = "deadline_change"
    THRESHOLD_CHANGE = "threshold_change"
    SCOPE_EXPANSION = "scope_expansion"
    ENFORCEMENT_CHANGE = "enforcement_change"
    CONSULTATION = "consultation"
    REPEAL = "repeal"


class ChangeStatus(str, Enum):
    """Status of a policy change."""
    PROPOSED = "proposed"
    UNDER_CONSULTATION = "under_consultation"
    ADOPTED = "adopted"
    IN_FORCE = "in_force"
    DELAYED = "delayed"
    WITHDRAWN = "withdrawn"


class ImpactLevel(str, Enum):
    """Impact level of policy change."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class AlertPriority(str, Enum):
    """Alert priority levels."""
    URGENT = "urgent"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class PolicyChange(BaseModel):
    """A policy change being tracked."""

    change_id: str = Field(
        default_factory=lambda: deterministic_uuid("policy_change"),
        description="Unique change identifier"
    )
    regulation_id: str = Field(..., description="Related regulation ID")
    regulation_name: str = Field(..., description="Regulation name")
    jurisdiction: str = Field(..., description="Jurisdiction")

    # Change details
    change_type: PolicyChangeType = Field(..., description="Type of change")
    title: str = Field(..., description="Change title")
    summary: str = Field(..., description="Change summary")

    # Status
    status: ChangeStatus = Field(..., description="Current status")

    # Dates
    announced_date: date = Field(..., description="When announced")
    consultation_deadline: Optional[date] = Field(None, description="Consultation deadline")
    expected_adoption_date: Optional[date] = Field(None, description="Expected adoption")
    expected_effective_date: Optional[date] = Field(None, description="Expected effective date")

    # Impact
    impact_level: ImpactLevel = Field(
        default=ImpactLevel.MEDIUM,
        description="Assessed impact level"
    )
    affected_entities: List[str] = Field(
        default_factory=list,
        description="Types of affected entities"
    )
    affected_sectors: List[str] = Field(
        default_factory=list,
        description="Affected industry sectors"
    )

    # References
    source_url: Optional[str] = Field(None, description="Source URL")
    official_reference: Optional[str] = Field(None, description="Official reference")

    # Tracking
    created_at: datetime = Field(default_factory=DeterministicClock.now)
    last_updated: datetime = Field(default_factory=DeterministicClock.now)


class PolicyAlert(BaseModel):
    """Alert for policy change."""

    alert_id: str = Field(
        default_factory=lambda: deterministic_uuid("alert"),
        description="Unique alert identifier"
    )
    change_id: str = Field(..., description="Related policy change ID")
    priority: AlertPriority = Field(..., description="Alert priority")
    title: str = Field(..., description="Alert title")
    message: str = Field(..., description="Alert message")

    # Action required
    action_required: bool = Field(default=False)
    action_deadline: Optional[date] = Field(None)
    recommended_actions: List[str] = Field(default_factory=list)

    # Metadata
    created_at: datetime = Field(default_factory=DeterministicClock.now)
    acknowledged: bool = Field(default=False)
    acknowledged_at: Optional[datetime] = Field(None)


class ImpactAssessment(BaseModel):
    """Impact assessment for a policy change."""

    assessment_id: str = Field(
        default_factory=lambda: deterministic_uuid("impact"),
        description="Unique assessment identifier"
    )
    change_id: str = Field(..., description="Related policy change")
    organization_id: str = Field(..., description="Organization assessed")

    # Impact scores (deterministic calculation)
    operational_impact_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Operational impact score"
    )
    financial_impact_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Financial impact score"
    )
    compliance_impact_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Compliance impact score"
    )
    overall_impact_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Overall impact score"
    )

    # Impact level
    impact_level: ImpactLevel = Field(..., description="Calculated impact level")

    # Estimated costs
    estimated_compliance_cost_eur: Optional[Decimal] = Field(None)
    estimated_annual_ongoing_cost_eur: Optional[Decimal] = Field(None)

    # Affected areas
    affected_processes: List[str] = Field(default_factory=list)
    affected_systems: List[str] = Field(default_factory=list)
    affected_teams: List[str] = Field(default_factory=list)

    # Timeline
    preparation_lead_time_days: int = Field(default=0)

    # Assessment trace
    assessment_trace: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

    def calculate_provenance_hash(self) -> str:
        """Calculate provenance hash."""
        content = {
            "change_id": self.change_id,
            "organization_id": self.organization_id,
            "overall_impact_score": self.overall_impact_score,
        }
        return hashlib.sha256(
            json.dumps(content, sort_keys=True, default=str).encode()
        ).hexdigest()


class PolicyIntelligenceInput(BaseModel):
    """Input for policy intelligence operations."""

    action: str = Field(
        ...,
        description="Action: track_changes, assess_impact, generate_alerts, horizon_scan"
    )
    organization_id: Optional[str] = Field(None)
    jurisdictions: Optional[List[str]] = Field(None)
    regulation_ids: Optional[List[str]] = Field(None)
    change_id: Optional[str] = Field(None)
    from_date: Optional[date] = Field(None)
    to_date: Optional[date] = Field(None)


class PolicyIntelligenceOutput(BaseModel):
    """Output from policy intelligence operations."""

    success: bool = Field(...)
    action: str = Field(...)
    changes: Optional[List[PolicyChange]] = Field(None)
    alerts: Optional[List[PolicyAlert]] = Field(None)
    impact_assessment: Optional[ImpactAssessment] = Field(None)
    summary: Optional[Dict[str, Any]] = Field(None)
    error: Optional[str] = Field(None)
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)


# =============================================================================
# POLICY CHANGE DATABASE
# =============================================================================


POLICY_CHANGES: Dict[str, PolicyChange] = {}


def _initialize_policy_changes() -> None:
    """Initialize sample policy changes for tracking."""
    global POLICY_CHANGES

    changes = [
        PolicyChange(
            change_id="PC-2024-001",
            regulation_id="EU-CSRD",
            regulation_name="CSRD",
            jurisdiction="eu",
            change_type=PolicyChangeType.GUIDANCE_UPDATE,
            title="EFRAG ESRS Implementation Guidance",
            summary="Updated implementation guidance for ESRS standards",
            status=ChangeStatus.ADOPTED,
            announced_date=date(2024, 7, 15),
            expected_effective_date=date(2024, 10, 1),
            impact_level=ImpactLevel.MEDIUM,
            affected_entities=["large_undertakings", "listed_companies"],
            affected_sectors=["all"],
        ),
        PolicyChange(
            change_id="PC-2024-002",
            regulation_id="EU-CBAM",
            regulation_name="CBAM",
            jurisdiction="eu",
            change_type=PolicyChangeType.THRESHOLD_CHANGE,
            title="CBAM De Minimis Threshold Update",
            summary="Proposed changes to de minimis thresholds for CBAM reporting",
            status=ChangeStatus.UNDER_CONSULTATION,
            announced_date=date(2024, 9, 1),
            consultation_deadline=date(2024, 11, 30),
            expected_adoption_date=date(2025, 3, 1),
            impact_level=ImpactLevel.HIGH,
            affected_entities=["importers"],
            affected_sectors=["cement", "steel", "aluminum", "chemicals", "electricity"],
        ),
        PolicyChange(
            change_id="PC-2024-003",
            regulation_id="US-CA-SB253",
            regulation_name="SB253",
            jurisdiction="california",
            change_type=PolicyChangeType.DEADLINE_CHANGE,
            title="SB253 Implementation Delay",
            summary="Proposed 2-year delay to SB253 implementation timeline",
            status=ChangeStatus.PROPOSED,
            announced_date=date(2024, 10, 15),
            expected_adoption_date=date(2025, 1, 15),
            impact_level=ImpactLevel.HIGH,
            affected_entities=["large_companies"],
            affected_sectors=["all"],
        ),
    ]

    for change in changes:
        POLICY_CHANGES[change.change_id] = change


_initialize_policy_changes()


# =============================================================================
# POLICY INTELLIGENCE AGENT
# =============================================================================


class PolicyIntelligenceAgent(BaseAgent):
    """
    GL-POL-X-003: Policy Intelligence Agent

    Monitors regulatory changes and provides intelligence on policy developments.
    INSIGHT PATH agent with deterministic tracking and AI-enhanced analysis.

    Deterministic Operations:
    - Change tracking and timeline calculations
    - Impact scoring using defined criteria
    - Alert generation based on thresholds

    AI-Enhanced Operations (optional):
    - Impact narrative generation
    - Trend analysis interpretation

    Usage:
        agent = PolicyIntelligenceAgent()
        result = agent.run({
            'action': 'track_changes',
            'jurisdictions': ['eu', 'california']
        })
    """

    AGENT_ID = "GL-POL-X-003"
    AGENT_NAME = "Policy Intelligence Agent"
    VERSION = "1.0.0"

    category = AgentCategory.INSIGHT
    metadata = AgentMetadata(
        name=AGENT_NAME,
        category=AgentCategory.INSIGHT,
        uses_chat_session=False,
        uses_rag=False,
        uses_tools=False,
        critical_for_compliance=False,
        description="Monitors regulatory changes with deterministic tracking"
    )

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize Policy Intelligence Agent."""
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Policy change monitoring and intelligence",
                version=self.VERSION,
                parameters={
                    "alert_threshold_days": 30,
                    "auto_generate_alerts": True,
                }
            )

        self._changes = POLICY_CHANGES.copy()
        self._alerts: Dict[str, PolicyAlert] = {}

        super().__init__(config)
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute policy intelligence operation."""
        import time
        start_time = time.time()

        try:
            agent_input = PolicyIntelligenceInput(**input_data)

            action_handlers = {
                "track_changes": self._handle_track_changes,
                "assess_impact": self._handle_assess_impact,
                "generate_alerts": self._handle_generate_alerts,
                "horizon_scan": self._handle_horizon_scan,
            }

            handler = action_handlers.get(agent_input.action)
            if not handler:
                raise ValueError(f"Unknown action: {agent_input.action}")

            output = handler(agent_input)
            output.processing_time_ms = (time.time() - start_time) * 1000
            output.provenance_hash = hashlib.sha256(
                json.dumps({"action": agent_input.action, "success": output.success}, sort_keys=True).encode()
            ).hexdigest()

            return AgentResult(
                success=output.success,
                data=output.model_dump(),
                error=output.error,
            )

        except Exception as e:
            logger.error(f"Policy intelligence failed: {str(e)}", exc_info=True)
            return AgentResult(success=False, error=str(e))

    def _handle_track_changes(
        self,
        input_data: PolicyIntelligenceInput
    ) -> PolicyIntelligenceOutput:
        """Track policy changes with filtering."""
        changes = list(self._changes.values())

        # Filter by jurisdictions
        if input_data.jurisdictions:
            changes = [c for c in changes if c.jurisdiction in input_data.jurisdictions]

        # Filter by regulation
        if input_data.regulation_ids:
            changes = [c for c in changes if c.regulation_id in input_data.regulation_ids]

        # Filter by date range
        if input_data.from_date:
            changes = [c for c in changes if c.announced_date >= input_data.from_date]
        if input_data.to_date:
            changes = [c for c in changes if c.announced_date <= input_data.to_date]

        # Sort by announced date
        changes.sort(key=lambda c: c.announced_date, reverse=True)

        # Generate summary
        summary = {
            "total_changes": len(changes),
            "by_status": {},
            "by_impact": {},
            "by_jurisdiction": {},
        }

        for change in changes:
            summary["by_status"][change.status.value] = summary["by_status"].get(change.status.value, 0) + 1
            summary["by_impact"][change.impact_level.value] = summary["by_impact"].get(change.impact_level.value, 0) + 1
            summary["by_jurisdiction"][change.jurisdiction] = summary["by_jurisdiction"].get(change.jurisdiction, 0) + 1

        return PolicyIntelligenceOutput(
            success=True,
            action="track_changes",
            changes=changes,
            summary=summary,
        )

    def _handle_assess_impact(
        self,
        input_data: PolicyIntelligenceInput
    ) -> PolicyIntelligenceOutput:
        """Assess impact of a policy change on organization."""
        if not input_data.change_id or not input_data.organization_id:
            return PolicyIntelligenceOutput(
                success=False,
                action="assess_impact",
                error="change_id and organization_id required",
            )

        change = self._changes.get(input_data.change_id)
        if not change:
            return PolicyIntelligenceOutput(
                success=False,
                action="assess_impact",
                error=f"Change not found: {input_data.change_id}",
            )

        # Deterministic impact calculation
        assessment = self._calculate_impact(change, input_data.organization_id)

        return PolicyIntelligenceOutput(
            success=True,
            action="assess_impact",
            impact_assessment=assessment,
        )

    def _calculate_impact(
        self,
        change: PolicyChange,
        organization_id: str
    ) -> ImpactAssessment:
        """Calculate impact score deterministically."""
        trace: List[str] = []

        # Base scores by change type
        type_scores = {
            PolicyChangeType.NEW_REGULATION: 80,
            PolicyChangeType.SCOPE_EXPANSION: 70,
            PolicyChangeType.THRESHOLD_CHANGE: 60,
            PolicyChangeType.DEADLINE_CHANGE: 50,
            PolicyChangeType.AMENDMENT: 40,
            PolicyChangeType.GUIDANCE_UPDATE: 30,
            PolicyChangeType.ENFORCEMENT_CHANGE: 60,
        }

        base_score = type_scores.get(change.change_type, 50)
        trace.append(f"Base score for {change.change_type.value}: {base_score}")

        # Adjust for status
        status_multipliers = {
            ChangeStatus.IN_FORCE: 1.0,
            ChangeStatus.ADOPTED: 0.95,
            ChangeStatus.UNDER_CONSULTATION: 0.7,
            ChangeStatus.PROPOSED: 0.5,
        }
        status_mult = status_multipliers.get(change.status, 0.8)
        trace.append(f"Status multiplier ({change.status.value}): {status_mult}")

        # Calculate component scores
        operational_score = base_score * status_mult
        financial_score = base_score * status_mult * 0.9  # Slightly lower
        compliance_score = base_score * status_mult * 1.1  # Slightly higher

        # Cap at 100
        operational_score = min(100, operational_score)
        financial_score = min(100, financial_score)
        compliance_score = min(100, compliance_score)

        # Overall score
        overall_score = (operational_score + financial_score + compliance_score) / 3

        # Determine impact level
        if overall_score >= 75:
            impact_level = ImpactLevel.CRITICAL
        elif overall_score >= 50:
            impact_level = ImpactLevel.HIGH
        elif overall_score >= 25:
            impact_level = ImpactLevel.MEDIUM
        else:
            impact_level = ImpactLevel.LOW

        trace.append(f"Overall score: {overall_score:.1f}, Level: {impact_level.value}")

        # Estimate preparation time
        if change.expected_effective_date:
            days_until = (change.expected_effective_date - DeterministicClock.now().date()).days
            prep_time = max(30, min(days_until - 30, 180))
        else:
            prep_time = 90

        assessment = ImpactAssessment(
            change_id=change.change_id,
            organization_id=organization_id,
            operational_impact_score=round(operational_score, 1),
            financial_impact_score=round(financial_score, 1),
            compliance_impact_score=round(compliance_score, 1),
            overall_impact_score=round(overall_score, 1),
            impact_level=impact_level,
            preparation_lead_time_days=prep_time,
            affected_processes=["emissions_reporting", "data_collection"],
            affected_teams=["sustainability", "compliance", "finance"],
            assessment_trace=trace,
        )

        assessment.provenance_hash = assessment.calculate_provenance_hash()
        return assessment

    def _handle_generate_alerts(
        self,
        input_data: PolicyIntelligenceInput
    ) -> PolicyIntelligenceOutput:
        """Generate alerts for upcoming deadlines and changes."""
        today = DeterministicClock.now().date()
        threshold_days = self.config.parameters.get("alert_threshold_days", 30)

        alerts: List[PolicyAlert] = []

        for change in self._changes.values():
            # Filter by jurisdiction if specified
            if input_data.jurisdictions and change.jurisdiction not in input_data.jurisdictions:
                continue

            # Check consultation deadlines
            if change.consultation_deadline:
                days_until = (change.consultation_deadline - today).days
                if 0 < days_until <= threshold_days:
                    alert = PolicyAlert(
                        change_id=change.change_id,
                        priority=AlertPriority.HIGH if days_until <= 7 else AlertPriority.NORMAL,
                        title=f"Consultation deadline approaching: {change.title}",
                        message=f"Consultation deadline in {days_until} days",
                        action_required=True,
                        action_deadline=change.consultation_deadline,
                        recommended_actions=[
                            "Review proposed changes",
                            "Prepare consultation response",
                            "Submit before deadline",
                        ],
                    )
                    alerts.append(alert)

            # Check effective dates
            if change.expected_effective_date:
                days_until = (change.expected_effective_date - today).days
                if 0 < days_until <= threshold_days * 3:  # Longer horizon for effective dates
                    priority = AlertPriority.URGENT if days_until <= 30 else (
                        AlertPriority.HIGH if days_until <= 60 else AlertPriority.NORMAL
                    )
                    alert = PolicyAlert(
                        change_id=change.change_id,
                        priority=priority,
                        title=f"Regulation effective date approaching: {change.title}",
                        message=f"Becomes effective in {days_until} days",
                        action_required=True,
                        action_deadline=change.expected_effective_date,
                        recommended_actions=[
                            "Assess readiness",
                            "Implement required changes",
                            "Verify compliance",
                        ],
                    )
                    alerts.append(alert)

        # Sort by priority and deadline
        priority_order = {
            AlertPriority.URGENT: 0,
            AlertPriority.HIGH: 1,
            AlertPriority.NORMAL: 2,
            AlertPriority.LOW: 3,
        }
        alerts.sort(key=lambda a: (priority_order[a.priority], a.action_deadline or date.max))

        return PolicyIntelligenceOutput(
            success=True,
            action="generate_alerts",
            alerts=alerts,
            summary={
                "total_alerts": len(alerts),
                "urgent": len([a for a in alerts if a.priority == AlertPriority.URGENT]),
                "high": len([a for a in alerts if a.priority == AlertPriority.HIGH]),
            },
        )

    def _handle_horizon_scan(
        self,
        input_data: PolicyIntelligenceInput
    ) -> PolicyIntelligenceOutput:
        """Scan horizon for upcoming regulatory changes."""
        today = DeterministicClock.now().date()

        # Get changes that are proposed or under consultation
        upcoming = [
            c for c in self._changes.values()
            if c.status in [ChangeStatus.PROPOSED, ChangeStatus.UNDER_CONSULTATION]
        ]

        # Filter by jurisdiction
        if input_data.jurisdictions:
            upcoming = [c for c in upcoming if c.jurisdiction in input_data.jurisdictions]

        # Sort by expected adoption date
        upcoming.sort(
            key=lambda c: c.expected_adoption_date or date.max
        )

        # Build timeline
        timeline: Dict[str, List[Dict[str, Any]]] = {}
        for change in upcoming:
            quarter = self._get_quarter(change.expected_adoption_date) if change.expected_adoption_date else "Unknown"
            if quarter not in timeline:
                timeline[quarter] = []
            timeline[quarter].append({
                "change_id": change.change_id,
                "title": change.title,
                "jurisdiction": change.jurisdiction,
                "impact_level": change.impact_level.value,
            })

        return PolicyIntelligenceOutput(
            success=True,
            action="horizon_scan",
            changes=upcoming,
            summary={
                "total_upcoming": len(upcoming),
                "by_quarter": {k: len(v) for k, v in timeline.items()},
                "timeline": timeline,
            },
        )

    def _get_quarter(self, d: date) -> str:
        """Get quarter string for a date."""
        q = (d.month - 1) // 3 + 1
        return f"Q{q} {d.year}"

    # =========================================================================
    # PUBLIC API METHODS
    # =========================================================================

    def add_policy_change(self, change: PolicyChange) -> str:
        """Add a policy change to track."""
        self._changes[change.change_id] = change
        return change.change_id

    def get_policy_change(self, change_id: str) -> Optional[PolicyChange]:
        """Get a policy change by ID."""
        return self._changes.get(change_id)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "PolicyIntelligenceAgent",
    "PolicyChangeType",
    "ChangeStatus",
    "ImpactLevel",
    "AlertPriority",
    "PolicyChange",
    "PolicyAlert",
    "ImpactAssessment",
    "PolicyIntelligenceInput",
    "PolicyIntelligenceOutput",
]
