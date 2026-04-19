"""
GL-016_Waterguard GraphQL API

Strawberry GraphQL schema for the Waterguard cooling tower optimization system.
Provides Query, Mutation, and Subscription types for real-time water chemistry
monitoring and optimization control.

Author: GL-APIDeveloper
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import AsyncGenerator, List, Optional

import strawberry
from strawberry.types import Info

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

@strawberry.enum
class OperatingModeGQL(Enum):
    """Operating mode for the cooling tower system."""
    NORMAL = "normal"
    CONSERVATION = "conservation"
    HIGH_LOAD = "high_load"
    MAINTENANCE = "maintenance"
    EMERGENCY = "emergency"


@strawberry.enum
class RecommendationPriorityGQL(Enum):
    """Priority level for recommendations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@strawberry.enum
class RecommendationTypeGQL(Enum):
    """Type of optimization recommendation."""
    BLOWDOWN_ADJUSTMENT = "blowdown_adjustment"
    DOSING_RATE_CHANGE = "dosing_rate_change"
    COC_TARGET_UPDATE = "coc_target_update"
    MAINTENANCE_ALERT = "maintenance_alert"
    COMPLIANCE_WARNING = "compliance_warning"
    EFFICIENCY_IMPROVEMENT = "efficiency_improvement"


@strawberry.enum
class RecommendationStatusGQL(Enum):
    """Status of a recommendation."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    IMPLEMENTED = "implemented"
    EXPIRED = "expired"


@strawberry.enum
class ComplianceStatusGQL(Enum):
    """Compliance status for constraints."""
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"
    UNKNOWN = "unknown"


# =============================================================================
# Input Types
# =============================================================================

@strawberry.input
class OptimizationInput:
    """Input for triggering optimization."""
    tower_id: str
    operating_mode: Optional[OperatingModeGQL] = OperatingModeGQL.NORMAL
    force_optimization: bool = False
    target_coc: Optional[float] = None


@strawberry.input
class RecommendationApprovalInput:
    """Input for approving/rejecting a recommendation."""
    recommendation_id: str
    approved: bool
    operator_notes: Optional[str] = None
    modified_value: Optional[float] = None


@strawberry.input
class SetpointUpdateInput:
    """Input for updating a setpoint."""
    tower_id: str
    parameter: str
    value: float
    unit: Optional[str] = None


@strawberry.input
class OperatingModeChangeInput:
    """Input for changing operating mode."""
    tower_id: str
    mode: OperatingModeGQL
    reason: Optional[str] = None


# =============================================================================
# Object Types
# =============================================================================

@strawberry.type
class ChemistryReading:
    """Single chemistry parameter reading."""
    parameter: str
    value: float
    unit: str
    min_limit: Optional[float]
    max_limit: Optional[float]
    target: Optional[float]
    timestamp: datetime

    @strawberry.field
    def is_within_limits(self) -> bool:
        """Check if value is within acceptable limits."""
        if self.min_limit is not None and self.value < self.min_limit:
            return False
        if self.max_limit is not None and self.value > self.max_limit:
            return False
        return True


@strawberry.type
class ChemistryState:
    """Complete water chemistry state."""
    tower_id: str
    timestamp: datetime

    # Primary parameters
    ph: float
    conductivity: float
    tds: float
    cycles_of_concentration: float

    # Secondary parameters
    alkalinity: Optional[float]
    hardness: Optional[float]
    chloride: Optional[float]
    silica: Optional[float]
    temperature: Optional[float]

    # Indices
    langelier_saturation_index: Optional[float]
    ryznar_stability_index: Optional[float]

    # Detailed readings
    readings: List[ChemistryReading]

    # Status
    overall_status: ComplianceStatusGQL
    parameters_out_of_spec: List[str]


@strawberry.type
class OptimizationResult:
    """Result of a single optimization calculation."""
    parameter: str
    current_value: float
    recommended_value: float
    change_percent: float
    impact_score: float
    confidence: float


@strawberry.type
class OptimizationResponse:
    """Response from optimization engine."""
    optimization_id: str
    tower_id: str
    timestamp: datetime
    status: str
    results: List[OptimizationResult]
    recommended_coc: float
    recommended_blowdown_rate: float
    projected_water_savings_percent: float
    projected_energy_savings_percent: float
    execution_time_ms: float
    model_version: str


@strawberry.type
class Recommendation:
    """Optimization recommendation."""
    recommendation_id: str
    tower_id: str
    type: RecommendationTypeGQL
    priority: RecommendationPriorityGQL
    status: RecommendationStatusGQL

    title: str
    description: str
    action_required: str

    current_value: Optional[float]
    recommended_value: Optional[float]
    parameter: Optional[str]
    unit: Optional[str]

    impact_score: float
    confidence: float
    projected_savings: Optional[float]

    created_at: datetime
    expires_at: Optional[datetime]
    approved_at: Optional[datetime]
    approved_by: Optional[str]

    reasoning: Optional[str]


@strawberry.type
class RecommendationApprovalResult:
    """Result of recommendation approval."""
    recommendation_id: str
    status: RecommendationStatusGQL
    approved: bool
    approved_at: datetime
    approved_by: str
    message: str


@strawberry.type
class SetpointUpdateResult:
    """Result of setpoint update."""
    tower_id: str
    parameter: str
    old_value: float
    new_value: float
    updated_at: datetime
    updated_by: str
    success: bool
    message: str


@strawberry.type
class OperatingModeChangeResult:
    """Result of operating mode change."""
    tower_id: str
    old_mode: OperatingModeGQL
    new_mode: OperatingModeGQL
    changed_at: datetime
    changed_by: str
    success: bool
    message: str


@strawberry.type
class ConstraintStatus:
    """Status of a compliance constraint."""
    constraint_id: str
    constraint_name: str
    category: str
    status: ComplianceStatusGQL
    current_value: float
    limit_value: float
    margin_percent: float
    in_violation: bool
    violation_count_24h: int


@strawberry.type
class ComplianceStatus:
    """Overall compliance status."""
    tower_id: str
    timestamp: datetime
    overall_status: ComplianceStatusGQL
    compliance_score: float
    total_constraints: int
    compliant_constraints: int
    warning_constraints: int
    violated_constraints: int
    constraints: List[ConstraintStatus]
    total_violations_24h: int
    critical_violations: List[str]


@strawberry.type
class SavingsMetric:
    """Single savings metric."""
    metric_name: str
    baseline_value: float
    current_value: float
    savings_value: float
    savings_percent: float
    unit: str
    monetary_value: Optional[float]


@strawberry.type
class Savings:
    """Comprehensive savings report."""
    tower_id: str
    timestamp: datetime
    period_start: datetime
    period_end: datetime

    water_savings_gallons: float
    water_savings_percent: float
    water_cost_savings: float

    energy_savings_kwh: float
    energy_savings_percent: float
    energy_cost_savings: float

    chemical_savings: float
    chemical_savings_percent: float

    co2_reduction_kg: float
    co2_reduction_percent: float

    total_cost_savings: float
    projected_annual_savings: float

    metrics: List[SavingsMetric]


@strawberry.type
class ChemistryUpdate:
    """Real-time chemistry update for subscriptions."""
    tower_id: str
    timestamp: datetime
    parameter: str
    value: float
    unit: str
    previous_value: Optional[float]
    change_percent: Optional[float]
    status: ComplianceStatusGQL


@strawberry.type
class RecommendationEvent:
    """Real-time recommendation event for subscriptions."""
    event_type: str  # created, updated, approved, rejected, expired
    recommendation_id: str
    tower_id: str
    timestamp: datetime
    type: RecommendationTypeGQL
    priority: RecommendationPriorityGQL
    status: RecommendationStatusGQL
    title: str
    description: str


# =============================================================================
# Mock Data Providers
# =============================================================================

async def get_chemistry_state(tower_id: str) -> ChemistryState:
    """Get mock chemistry state."""
    return ChemistryState(
        tower_id=tower_id,
        timestamp=datetime.utcnow(),
        ph=7.8,
        conductivity=1500.0,
        tds=1200.0,
        cycles_of_concentration=4.5,
        alkalinity=120.0,
        hardness=200.0,
        chloride=150.0,
        silica=25.0,
        temperature=32.5,
        langelier_saturation_index=0.5,
        ryznar_stability_index=6.5,
        readings=[
            ChemistryReading(
                parameter="pH",
                value=7.8,
                unit="pH",
                min_limit=7.0,
                max_limit=8.5,
                target=7.5,
                timestamp=datetime.utcnow(),
            ),
            ChemistryReading(
                parameter="Conductivity",
                value=1500.0,
                unit="uS/cm",
                min_limit=800.0,
                max_limit=2000.0,
                target=1500.0,
                timestamp=datetime.utcnow(),
            ),
        ],
        overall_status=ComplianceStatusGQL.COMPLIANT,
        parameters_out_of_spec=[],
    )


async def get_recommendations(
    tower_id: str,
    status_filter: Optional[RecommendationStatusGQL] = None,
) -> List[Recommendation]:
    """Get mock recommendations."""
    recommendations = [
        Recommendation(
            recommendation_id="rec-001",
            tower_id=tower_id,
            type=RecommendationTypeGQL.BLOWDOWN_ADJUSTMENT,
            priority=RecommendationPriorityGQL.MEDIUM,
            status=RecommendationStatusGQL.PENDING,
            title="Increase Blowdown Rate",
            description="Conductivity trending high. Consider increasing blowdown.",
            action_required="Increase blowdown rate from 10 gpm to 12.5 gpm",
            current_value=10.0,
            recommended_value=12.5,
            parameter="blowdown_rate",
            unit="gpm",
            impact_score=75.0,
            confidence=0.92,
            projected_savings=500.0,
            created_at=datetime.utcnow() - timedelta(hours=2),
            expires_at=datetime.utcnow() + timedelta(hours=22),
            approved_at=None,
            approved_by=None,
            reasoning="Based on 24-hour conductivity trend analysis.",
        ),
    ]

    if status_filter:
        recommendations = [r for r in recommendations if r.status == status_filter]

    return recommendations


async def get_compliance_status(tower_id: str) -> ComplianceStatus:
    """Get mock compliance status."""
    return ComplianceStatus(
        tower_id=tower_id,
        timestamp=datetime.utcnow(),
        overall_status=ComplianceStatusGQL.COMPLIANT,
        compliance_score=98.5,
        total_constraints=15,
        compliant_constraints=14,
        warning_constraints=1,
        violated_constraints=0,
        constraints=[
            ConstraintStatus(
                constraint_id="const-ph",
                constraint_name="pH Range",
                category="chemistry",
                status=ComplianceStatusGQL.COMPLIANT,
                current_value=7.8,
                limit_value=8.5,
                margin_percent=8.2,
                in_violation=False,
                violation_count_24h=0,
            ),
        ],
        total_violations_24h=0,
        critical_violations=[],
    )


async def get_savings(tower_id: str, period_days: int) -> Savings:
    """Get mock savings data."""
    return Savings(
        tower_id=tower_id,
        timestamp=datetime.utcnow(),
        period_start=datetime.utcnow() - timedelta(days=period_days),
        period_end=datetime.utcnow(),
        water_savings_gallons=15000.0,
        water_savings_percent=15.0,
        water_cost_savings=750.0,
        energy_savings_kwh=4000.0,
        energy_savings_percent=8.0,
        energy_cost_savings=400.0,
        chemical_savings=240.0,
        chemical_savings_percent=12.0,
        co2_reduction_kg=3750.0,
        co2_reduction_percent=15.0,
        total_cost_savings=1390.0,
        projected_annual_savings=33360.0,
        metrics=[
            SavingsMetric(
                metric_name="Makeup Water",
                baseline_value=100000.0,
                current_value=85000.0,
                savings_value=15000.0,
                savings_percent=15.0,
                unit="gallons",
                monetary_value=750.0,
            ),
        ],
    )


# =============================================================================
# Query Type
# =============================================================================

@strawberry.type
class Query:
    """GraphQL Query type for Waterguard API."""

    @strawberry.field
    async def chemistry_state(
        self,
        tower_id: str,
        info: Info,
    ) -> ChemistryState:
        """
        Get current water chemistry state for a tower.

        Args:
            tower_id: Cooling tower identifier

        Returns:
            Complete chemistry state with all parameters
        """
        logger.info(f"GraphQL query: chemistry_state for {tower_id}")
        return await get_chemistry_state(tower_id)

    @strawberry.field
    async def recommendations(
        self,
        tower_id: str,
        status: Optional[RecommendationStatusGQL] = None,
        priority: Optional[RecommendationPriorityGQL] = None,
        limit: int = 20,
        info: Info = None,
    ) -> List[Recommendation]:
        """
        Get active recommendations for a tower.

        Args:
            tower_id: Cooling tower identifier
            status: Filter by status
            priority: Filter by priority
            limit: Maximum results

        Returns:
            List of recommendations
        """
        logger.info(f"GraphQL query: recommendations for {tower_id}")
        recommendations = await get_recommendations(tower_id, status)

        if priority:
            recommendations = [r for r in recommendations if r.priority == priority]

        return recommendations[:limit]

    @strawberry.field
    async def recommendation(
        self,
        recommendation_id: str,
        info: Info = None,
    ) -> Optional[Recommendation]:
        """
        Get a specific recommendation by ID.

        Args:
            recommendation_id: Recommendation identifier

        Returns:
            Recommendation or None if not found
        """
        logger.info(f"GraphQL query: recommendation {recommendation_id}")
        # In production, fetch from database
        recommendations = await get_recommendations("tower-001")
        for rec in recommendations:
            if rec.recommendation_id == recommendation_id:
                return rec
        return None

    @strawberry.field
    async def compliance_status(
        self,
        tower_id: str,
        info: Info = None,
    ) -> ComplianceStatus:
        """
        Get compliance status for a tower.

        Args:
            tower_id: Cooling tower identifier

        Returns:
            Compliance status report
        """
        logger.info(f"GraphQL query: compliance_status for {tower_id}")
        return await get_compliance_status(tower_id)

    @strawberry.field
    async def savings(
        self,
        tower_id: str,
        period_days: int = 30,
        info: Info = None,
    ) -> Savings:
        """
        Get savings report for a tower.

        Args:
            tower_id: Cooling tower identifier
            period_days: Report period in days

        Returns:
            Savings report
        """
        logger.info(f"GraphQL query: savings for {tower_id}")
        return await get_savings(tower_id, period_days)


# =============================================================================
# Mutation Type
# =============================================================================

@strawberry.type
class Mutation:
    """GraphQL Mutation type for Waterguard API."""

    @strawberry.mutation
    async def trigger_optimization(
        self,
        input: OptimizationInput,
        info: Info = None,
    ) -> OptimizationResponse:
        """
        Trigger an optimization cycle for a tower.

        Args:
            input: Optimization parameters

        Returns:
            Optimization results
        """
        logger.info(f"GraphQL mutation: trigger_optimization for {input.tower_id}")

        optimization_id = f"opt-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"

        return OptimizationResponse(
            optimization_id=optimization_id,
            tower_id=input.tower_id,
            timestamp=datetime.utcnow(),
            status="completed",
            results=[
                OptimizationResult(
                    parameter="cycles_of_concentration",
                    current_value=4.5,
                    recommended_value=5.2,
                    change_percent=15.6,
                    impact_score=85.0,
                    confidence=0.94,
                ),
            ],
            recommended_coc=5.2,
            recommended_blowdown_rate=10.5,
            projected_water_savings_percent=15.0,
            projected_energy_savings_percent=8.0,
            execution_time_ms=245.5,
            model_version="v2.1.0",
        )

    @strawberry.mutation
    async def approve_recommendation(
        self,
        input: RecommendationApprovalInput,
        info: Info = None,
    ) -> RecommendationApprovalResult:
        """
        Approve or reject a recommendation.

        Args:
            input: Approval decision

        Returns:
            Approval result
        """
        logger.info(
            f"GraphQL mutation: approve_recommendation {input.recommendation_id} "
            f"{'approved' if input.approved else 'rejected'}"
        )

        return RecommendationApprovalResult(
            recommendation_id=input.recommendation_id,
            status=RecommendationStatusGQL.APPROVED if input.approved else RecommendationStatusGQL.REJECTED,
            approved=input.approved,
            approved_at=datetime.utcnow(),
            approved_by="operator@example.com",  # Get from auth context
            message=f"Recommendation {input.recommendation_id} has been {'approved' if input.approved else 'rejected'}",
        )

    @strawberry.mutation
    async def update_setpoint(
        self,
        input: SetpointUpdateInput,
        info: Info = None,
    ) -> SetpointUpdateResult:
        """
        Update a control setpoint.

        Args:
            input: Setpoint update parameters

        Returns:
            Update result
        """
        logger.info(
            f"GraphQL mutation: update_setpoint {input.parameter} = {input.value} "
            f"for {input.tower_id}"
        )

        return SetpointUpdateResult(
            tower_id=input.tower_id,
            parameter=input.parameter,
            old_value=1500.0,  # Get from current state
            new_value=input.value,
            updated_at=datetime.utcnow(),
            updated_by="engineer@example.com",  # Get from auth context
            success=True,
            message=f"Setpoint {input.parameter} updated to {input.value}",
        )

    @strawberry.mutation
    async def change_operating_mode(
        self,
        input: OperatingModeChangeInput,
        info: Info = None,
    ) -> OperatingModeChangeResult:
        """
        Change the operating mode for a tower.

        Args:
            input: Mode change parameters

        Returns:
            Mode change result
        """
        logger.info(
            f"GraphQL mutation: change_operating_mode to {input.mode.value} "
            f"for {input.tower_id}"
        )

        return OperatingModeChangeResult(
            tower_id=input.tower_id,
            old_mode=OperatingModeGQL.NORMAL,  # Get from current state
            new_mode=input.mode,
            changed_at=datetime.utcnow(),
            changed_by="engineer@example.com",  # Get from auth context
            success=True,
            message=f"Operating mode changed to {input.mode.value}",
        )


# =============================================================================
# Subscription Type
# =============================================================================

@strawberry.type
class Subscription:
    """GraphQL Subscription type for real-time updates."""

    @strawberry.subscription
    async def chemistry_updates(
        self,
        tower_id: str,
        parameters: Optional[List[str]] = None,
        info: Info = None,
    ) -> AsyncGenerator[ChemistryUpdate, None]:
        """
        Subscribe to real-time chemistry updates.

        Args:
            tower_id: Cooling tower identifier
            parameters: Optional list of parameters to monitor

        Yields:
            Chemistry update events
        """
        logger.info(f"GraphQL subscription: chemistry_updates for {tower_id}")

        # Simulate real-time updates
        import random

        base_values = {
            "pH": 7.8,
            "conductivity": 1500.0,
            "tds": 1200.0,
            "temperature": 32.5,
        }

        params_to_monitor = parameters or list(base_values.keys())

        while True:
            await asyncio.sleep(5)  # Update every 5 seconds

            param = random.choice(params_to_monitor)
            base_value = base_values.get(param, 100.0)
            new_value = base_value * (1 + random.uniform(-0.02, 0.02))
            change = ((new_value - base_value) / base_value) * 100

            yield ChemistryUpdate(
                tower_id=tower_id,
                timestamp=datetime.utcnow(),
                parameter=param,
                value=round(new_value, 2),
                unit="pH" if param == "pH" else ("uS/cm" if param == "conductivity" else "ppm"),
                previous_value=base_value,
                change_percent=round(change, 2),
                status=ComplianceStatusGQL.COMPLIANT,
            )

    @strawberry.subscription
    async def recommendation_events(
        self,
        tower_id: str,
        info: Info = None,
    ) -> AsyncGenerator[RecommendationEvent, None]:
        """
        Subscribe to recommendation events.

        Args:
            tower_id: Cooling tower identifier

        Yields:
            Recommendation event notifications
        """
        logger.info(f"GraphQL subscription: recommendation_events for {tower_id}")

        event_types = ["created", "updated", "approved"]

        while True:
            await asyncio.sleep(30)  # Simulate occasional events

            import random

            yield RecommendationEvent(
                event_type=random.choice(event_types),
                recommendation_id=f"rec-{uuid.uuid4().hex[:8]}",
                tower_id=tower_id,
                timestamp=datetime.utcnow(),
                type=RecommendationTypeGQL.BLOWDOWN_ADJUSTMENT,
                priority=RecommendationPriorityGQL.MEDIUM,
                status=RecommendationStatusGQL.PENDING,
                title="Sample Recommendation",
                description="This is a sample recommendation event.",
            )


# =============================================================================
# Schema Definition
# =============================================================================

schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    subscription=Subscription,
)
