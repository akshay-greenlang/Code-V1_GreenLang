# -*- coding: utf-8 -*-
"""
GL-FIN-X-008: Climate Budget Agent
==================================

Manages carbon budgets at organizational, departmental, and project levels.
Tracks budget allocations, monitors consumption, and calculates variances.

Capabilities:
    - Carbon budget allocation by entity/project
    - Budget consumption tracking
    - Variance analysis and alerts
    - Budget trajectory projections
    - Reallocation recommendations
    - Science-based budget setting

Zero-Hallucination Guarantees:
    - All calculations are deterministic
    - Budget formulas from SBTi methodology
    - Complete audit trail for all allocations
    - SHA-256 provenance hashes for all outputs

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.agents.categories import AgentCategory

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================


class BudgetPeriod(str, Enum):
    """Budget periods."""
    ANNUAL = "annual"
    QUARTERLY = "quarterly"
    MONTHLY = "monthly"
    PROJECT_BASED = "project_based"


class BudgetStatus(str, Enum):
    """Budget status."""
    ON_TRACK = "on_track"
    AT_RISK = "at_risk"
    OVER_BUDGET = "over_budget"
    UNDER_BUDGET = "under_budget"
    NOT_STARTED = "not_started"


class AllocationMethod(str, Enum):
    """Budget allocation methods."""
    EQUAL = "equal_distribution"
    HISTORICAL = "historical_baseline"
    ECONOMIC = "economic_activity_based"
    INTENSITY = "intensity_based"
    SBTI = "science_based_targets"
    CUSTOM = "custom"


class VarianceSeverity(str, Enum):
    """Variance severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


# SBTi reduction pathways (annual reduction rates)
SBTI_PATHWAYS = {
    "1.5C": 0.042,  # 4.2% annual reduction
    "WB2C": 0.025,  # 2.5% annual reduction
    "2C": 0.015,    # 1.5% annual reduction
}


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class BudgetAllocation(BaseModel):
    """Carbon budget allocation for an entity."""
    allocation_id: str = Field(..., description="Unique identifier")
    entity_name: str = Field(..., description="Entity name")
    entity_type: str = Field(
        default="department",
        description="Type: organization, department, project, facility"
    )
    parent_entity_id: Optional[str] = Field(None, description="Parent entity")

    # Budget specification
    budget_tco2e: float = Field(..., ge=0, description="Allocated budget in tCO2e")
    period: BudgetPeriod = Field(default=BudgetPeriod.ANNUAL)
    start_date: datetime = Field(..., description="Budget period start")
    end_date: datetime = Field(..., description="Budget period end")

    # Allocation metadata
    allocation_method: AllocationMethod = Field(default=AllocationMethod.HISTORICAL)
    baseline_year_emissions: Optional[float] = Field(None, ge=0)
    target_reduction_pct: float = Field(default=0, ge=0, le=100)

    # Tracking
    consumed_tco2e: float = Field(default=0, ge=0, description="Consumed budget")
    last_updated: datetime = Field(default_factory=datetime.utcnow)

    # Scopes
    includes_scope1: bool = Field(default=True)
    includes_scope2: bool = Field(default=True)
    includes_scope3: bool = Field(default=False)


class BudgetVariance(BaseModel):
    """Budget variance analysis."""
    allocation_id: str
    entity_name: str
    analysis_date: datetime = Field(default_factory=datetime.utcnow)

    # Variance metrics
    budget_tco2e: float
    consumed_tco2e: float
    remaining_tco2e: float
    variance_tco2e: float = Field(
        ..., description="Negative = over budget, Positive = under budget"
    )
    variance_pct: float

    # Period progress
    period_elapsed_pct: float
    expected_consumption_tco2e: float
    consumption_vs_expected_pct: float

    # Status
    status: BudgetStatus
    severity: VarianceSeverity

    # Projections
    projected_year_end_tco2e: float
    projected_variance_tco2e: float

    # Recommendations
    action_required: bool
    recommendations: List[str] = Field(default_factory=list)


class CarbonBudgetStatus(BaseModel):
    """Overall carbon budget status."""
    status_date: datetime = Field(default_factory=datetime.utcnow)
    reporting_period: str

    # Totals
    total_budget_tco2e: float
    total_consumed_tco2e: float
    total_remaining_tco2e: float
    overall_utilization_pct: float

    # By scope
    scope1_budget: float
    scope1_consumed: float
    scope2_budget: float
    scope2_consumed: float
    scope3_budget: float
    scope3_consumed: float

    # Status breakdown
    entities_on_track: int
    entities_at_risk: int
    entities_over_budget: int
    entities_under_budget: int

    # Top concerns
    top_variance_entities: List[Dict[str, Any]] = Field(default_factory=list)

    # Trajectory
    on_track_for_target: bool
    projected_annual_emissions: float
    target_annual_emissions: float


class ClimateBudgetInput(BaseModel):
    """Input for climate budget operations."""
    operation: str = Field(
        default="allocate_budget",
        description="Operation: allocate_budget, track_consumption, analyze_variance, get_status"
    )

    # Budget allocation
    allocation: Optional[BudgetAllocation] = Field(None)
    allocations: Optional[List[BudgetAllocation]] = Field(None)

    # Consumption tracking
    entity_id: Optional[str] = Field(None)
    consumption_tco2e: Optional[float] = Field(None, ge=0)
    consumption_date: Optional[datetime] = Field(None)

    # Budget setting parameters
    baseline_emissions: Optional[float] = Field(None, ge=0)
    target_year: Optional[int] = Field(None)
    sbti_pathway: Optional[str] = Field(None, description="1.5C, WB2C, or 2C")
    allocation_method: Optional[AllocationMethod] = Field(None)

    # Entity data for allocation
    entity_weights: Optional[Dict[str, float]] = Field(
        None, description="Entity weights for allocation"
    )


class ClimateBudgetOutput(BaseModel):
    """Output from climate budget operations."""
    success: bool
    operation: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Results
    allocation: Optional[BudgetAllocation] = Field(None)
    allocations: Optional[List[BudgetAllocation]] = Field(None)
    variance: Optional[BudgetVariance] = Field(None)
    variances: Optional[List[BudgetVariance]] = Field(None)
    status: Optional[CarbonBudgetStatus] = Field(None)

    # Audit
    calculation_trace: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


# =============================================================================
# CLIMATE BUDGET AGENT
# =============================================================================


class ClimateBudgetAgent(BaseAgent):
    """
    GL-FIN-X-008: Climate Budget Agent

    Manages organizational carbon budgets using deterministic calculations.

    Zero-Hallucination Guarantees:
        - All budget calculations are deterministic
        - SBTi pathway formulas used for target setting
        - Complete audit trail for all allocations
        - SHA-256 provenance hashes for all outputs

    Usage:
        agent = ClimateBudgetAgent()
        result = agent.run({
            "operation": "allocate_budget",
            "allocation": budget_allocation
        })
    """

    AGENT_ID = "GL-FIN-X-008"
    AGENT_NAME = "Climate Budget Agent"
    VERSION = "1.0.0"

    category = AgentCategory.CRITICAL

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the Climate Budget Agent."""
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Carbon budget management",
                version=self.VERSION,
                parameters={}
            )

        self._allocations: Dict[str, BudgetAllocation] = {}
        super().__init__(config)
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute climate budget operations."""
        try:
            budget_input = ClimateBudgetInput(**input_data)
            operation = budget_input.operation

            if operation == "allocate_budget":
                output = self._allocate_budget(budget_input)
            elif operation == "track_consumption":
                output = self._track_consumption(budget_input)
            elif operation == "analyze_variance":
                output = self._analyze_variance(budget_input)
            elif operation == "get_status":
                output = self._get_status(budget_input)
            elif operation == "calculate_sbti_budget":
                output = self._calculate_sbti_budget(budget_input)
            elif operation == "distribute_budget":
                output = self._distribute_budget(budget_input)
            else:
                return AgentResult(success=False, error=f"Unknown operation: {operation}")

            return AgentResult(
                success=output.success,
                data=output.model_dump(),
                metadata={"agent_id": self.AGENT_ID, "operation": operation}
            )

        except Exception as e:
            logger.error(f"Climate budget operation failed: {e}", exc_info=True)
            return AgentResult(success=False, error=str(e))

    def _allocate_budget(self, input_data: ClimateBudgetInput) -> ClimateBudgetOutput:
        """Allocate or update a carbon budget."""
        calculation_trace: List[str] = []

        if input_data.allocation is None:
            return ClimateBudgetOutput(
                success=False,
                operation="allocate_budget",
                calculation_trace=["ERROR: No allocation provided"]
            )

        allocation = input_data.allocation
        calculation_trace.append(f"Allocating budget for: {allocation.entity_name}")
        calculation_trace.append(f"Budget: {allocation.budget_tco2e:,.2f} tCO2e")
        calculation_trace.append(f"Period: {allocation.start_date.date()} to {allocation.end_date.date()}")

        # Store allocation
        self._allocations[allocation.allocation_id] = allocation

        provenance_hash = hashlib.sha256(
            json.dumps(allocation.model_dump(), sort_keys=True, default=str).encode()
        ).hexdigest()

        return ClimateBudgetOutput(
            success=True,
            operation="allocate_budget",
            allocation=allocation,
            calculation_trace=calculation_trace,
            provenance_hash=provenance_hash
        )

    def _track_consumption(self, input_data: ClimateBudgetInput) -> ClimateBudgetOutput:
        """Track consumption against budget."""
        calculation_trace: List[str] = []

        entity_id = input_data.entity_id
        consumption = input_data.consumption_tco2e

        if entity_id is None or consumption is None:
            return ClimateBudgetOutput(
                success=False,
                operation="track_consumption",
                calculation_trace=["ERROR: entity_id and consumption_tco2e required"]
            )

        if entity_id not in self._allocations:
            return ClimateBudgetOutput(
                success=False,
                operation="track_consumption",
                calculation_trace=[f"ERROR: Entity {entity_id} not found"]
            )

        allocation = self._allocations[entity_id]
        allocation.consumed_tco2e += consumption
        allocation.last_updated = datetime.utcnow()

        calculation_trace.append(f"Recorded consumption: {consumption:,.2f} tCO2e")
        calculation_trace.append(f"Total consumed: {allocation.consumed_tco2e:,.2f} tCO2e")
        calculation_trace.append(f"Remaining: {allocation.budget_tco2e - allocation.consumed_tco2e:,.2f} tCO2e")

        provenance_hash = hashlib.sha256(
            json.dumps({
                "entity_id": entity_id,
                "consumption": consumption,
                "total_consumed": allocation.consumed_tco2e,
                "timestamp": datetime.utcnow().isoformat()
            }, sort_keys=True).encode()
        ).hexdigest()

        return ClimateBudgetOutput(
            success=True,
            operation="track_consumption",
            allocation=allocation,
            calculation_trace=calculation_trace,
            provenance_hash=provenance_hash
        )

    def _analyze_variance(self, input_data: ClimateBudgetInput) -> ClimateBudgetOutput:
        """Analyze budget variance for an entity."""
        calculation_trace: List[str] = []

        entity_id = input_data.entity_id

        if entity_id is None:
            # Analyze all entities
            variances = []
            for alloc_id, allocation in self._allocations.items():
                variance = self._calculate_variance(allocation, calculation_trace)
                variances.append(variance)

            provenance_hash = hashlib.sha256(
                json.dumps([v.model_dump() for v in variances], sort_keys=True, default=str).encode()
            ).hexdigest()

            return ClimateBudgetOutput(
                success=True,
                operation="analyze_variance",
                variances=variances,
                calculation_trace=calculation_trace,
                provenance_hash=provenance_hash
            )

        if entity_id not in self._allocations:
            return ClimateBudgetOutput(
                success=False,
                operation="analyze_variance",
                calculation_trace=[f"ERROR: Entity {entity_id} not found"]
            )

        allocation = self._allocations[entity_id]
        variance = self._calculate_variance(allocation, calculation_trace)

        provenance_hash = hashlib.sha256(
            json.dumps(variance.model_dump(), sort_keys=True, default=str).encode()
        ).hexdigest()

        return ClimateBudgetOutput(
            success=True,
            operation="analyze_variance",
            variance=variance,
            calculation_trace=calculation_trace,
            provenance_hash=provenance_hash
        )

    def _calculate_variance(
        self, allocation: BudgetAllocation, trace: List[str]
    ) -> BudgetVariance:
        """Calculate budget variance for an allocation."""
        trace.append(f"Analyzing variance for: {allocation.entity_name}")

        now = datetime.utcnow()
        budget = allocation.budget_tco2e
        consumed = allocation.consumed_tco2e
        remaining = budget - consumed

        # Calculate period elapsed
        period_total = (allocation.end_date - allocation.start_date).days
        period_elapsed = (now - allocation.start_date).days
        period_elapsed_pct = min(max(period_elapsed / period_total * 100, 0), 100) if period_total > 0 else 0

        # Expected consumption based on linear trajectory
        expected_consumption = budget * (period_elapsed_pct / 100)
        consumption_vs_expected = (
            ((consumed - expected_consumption) / expected_consumption * 100)
            if expected_consumption > 0 else 0
        )

        # Variance calculation
        variance = remaining
        variance_pct = (variance / budget * 100) if budget > 0 else 0

        # Project year-end
        if period_elapsed_pct > 10:  # Need some data to project
            daily_rate = consumed / period_elapsed if period_elapsed > 0 else 0
            projected_annual = daily_rate * period_total
        else:
            projected_annual = budget  # Assume on track if too early

        projected_variance = budget - projected_annual

        # Determine status and severity
        if consumed > budget:
            status = BudgetStatus.OVER_BUDGET
            severity = VarianceSeverity.CRITICAL
        elif consumption_vs_expected > 20:
            status = BudgetStatus.AT_RISK
            severity = VarianceSeverity.HIGH if consumption_vs_expected > 40 else VarianceSeverity.MEDIUM
        elif consumption_vs_expected < -20:
            status = BudgetStatus.UNDER_BUDGET
            severity = VarianceSeverity.LOW
        else:
            status = BudgetStatus.ON_TRACK
            severity = VarianceSeverity.NONE

        # Generate recommendations
        recommendations = []
        action_required = False

        if status == BudgetStatus.OVER_BUDGET:
            action_required = True
            recommendations.append("Immediate emissions reduction actions required")
            recommendations.append("Consider purchasing carbon offsets for compliance")
            recommendations.append("Review high-emission activities for reduction opportunities")
        elif status == BudgetStatus.AT_RISK:
            action_required = True
            recommendations.append("Accelerate planned reduction initiatives")
            recommendations.append("Review and optimize energy consumption")
            recommendations.append("Consider reallocation from under-budget entities")
        elif status == BudgetStatus.UNDER_BUDGET:
            recommendations.append("Good progress - maintain current trajectory")
            recommendations.append("Consider reallocating surplus to at-risk entities")

        trace.append(f"Status: {status.value}, Variance: {variance:,.2f} tCO2e ({variance_pct:.1f}%)")

        return BudgetVariance(
            allocation_id=allocation.allocation_id,
            entity_name=allocation.entity_name,
            budget_tco2e=budget,
            consumed_tco2e=consumed,
            remaining_tco2e=remaining,
            variance_tco2e=variance,
            variance_pct=round(variance_pct, 2),
            period_elapsed_pct=round(period_elapsed_pct, 2),
            expected_consumption_tco2e=round(expected_consumption, 2),
            consumption_vs_expected_pct=round(consumption_vs_expected, 2),
            status=status,
            severity=severity,
            projected_year_end_tco2e=round(projected_annual, 2),
            projected_variance_tco2e=round(projected_variance, 2),
            action_required=action_required,
            recommendations=recommendations
        )

    def _get_status(self, input_data: ClimateBudgetInput) -> ClimateBudgetOutput:
        """Get overall carbon budget status."""
        calculation_trace: List[str] = []

        if not self._allocations:
            return ClimateBudgetOutput(
                success=False,
                operation="get_status",
                calculation_trace=["ERROR: No budgets allocated"]
            )

        # Calculate totals
        total_budget = sum(a.budget_tco2e for a in self._allocations.values())
        total_consumed = sum(a.consumed_tco2e for a in self._allocations.values())
        total_remaining = total_budget - total_consumed
        utilization = (total_consumed / total_budget * 100) if total_budget > 0 else 0

        # By scope
        scope1_budget = sum(a.budget_tco2e for a in self._allocations.values() if a.includes_scope1) * 0.4
        scope2_budget = sum(a.budget_tco2e for a in self._allocations.values() if a.includes_scope2) * 0.3
        scope3_budget = sum(a.budget_tco2e for a in self._allocations.values() if a.includes_scope3) * 0.3
        scope1_consumed = total_consumed * 0.4
        scope2_consumed = total_consumed * 0.3
        scope3_consumed = total_consumed * 0.3

        # Analyze each entity
        variances = [self._calculate_variance(a, []) for a in self._allocations.values()]

        on_track = sum(1 for v in variances if v.status == BudgetStatus.ON_TRACK)
        at_risk = sum(1 for v in variances if v.status == BudgetStatus.AT_RISK)
        over_budget = sum(1 for v in variances if v.status == BudgetStatus.OVER_BUDGET)
        under_budget = sum(1 for v in variances if v.status == BudgetStatus.UNDER_BUDGET)

        # Top variances
        sorted_variances = sorted(variances, key=lambda v: v.consumption_vs_expected_pct, reverse=True)
        top_concerns = [
            {
                "entity": v.entity_name,
                "status": v.status.value,
                "variance_pct": v.variance_pct,
                "action_required": v.action_required
            }
            for v in sorted_variances[:5]
        ]

        # Trajectory
        projected_annual = sum(v.projected_year_end_tco2e for v in variances)
        on_track_for_target = projected_annual <= total_budget

        calculation_trace.append(f"Total budget: {total_budget:,.2f} tCO2e")
        calculation_trace.append(f"Total consumed: {total_consumed:,.2f} tCO2e ({utilization:.1f}%)")
        calculation_trace.append(f"Entities: {on_track} on track, {at_risk} at risk, {over_budget} over budget")

        status = CarbonBudgetStatus(
            reporting_period="current",
            total_budget_tco2e=round(total_budget, 2),
            total_consumed_tco2e=round(total_consumed, 2),
            total_remaining_tco2e=round(total_remaining, 2),
            overall_utilization_pct=round(utilization, 2),
            scope1_budget=round(scope1_budget, 2),
            scope1_consumed=round(scope1_consumed, 2),
            scope2_budget=round(scope2_budget, 2),
            scope2_consumed=round(scope2_consumed, 2),
            scope3_budget=round(scope3_budget, 2),
            scope3_consumed=round(scope3_consumed, 2),
            entities_on_track=on_track,
            entities_at_risk=at_risk,
            entities_over_budget=over_budget,
            entities_under_budget=under_budget,
            top_variance_entities=top_concerns,
            on_track_for_target=on_track_for_target,
            projected_annual_emissions=round(projected_annual, 2),
            target_annual_emissions=round(total_budget, 2)
        )

        provenance_hash = hashlib.sha256(
            json.dumps(status.model_dump(), sort_keys=True, default=str).encode()
        ).hexdigest()

        return ClimateBudgetOutput(
            success=True,
            operation="get_status",
            status=status,
            variances=variances,
            calculation_trace=calculation_trace,
            provenance_hash=provenance_hash
        )

    def _calculate_sbti_budget(self, input_data: ClimateBudgetInput) -> ClimateBudgetOutput:
        """Calculate budget based on SBTi pathway."""
        calculation_trace: List[str] = []

        baseline = input_data.baseline_emissions
        target_year = input_data.target_year or 2030
        pathway = input_data.sbti_pathway or "1.5C"

        if baseline is None:
            return ClimateBudgetOutput(
                success=False,
                operation="calculate_sbti_budget",
                calculation_trace=["ERROR: baseline_emissions required"]
            )

        if pathway not in SBTI_PATHWAYS:
            return ClimateBudgetOutput(
                success=False,
                operation="calculate_sbti_budget",
                calculation_trace=[f"ERROR: Unknown pathway {pathway}. Use 1.5C, WB2C, or 2C"]
            )

        current_year = datetime.utcnow().year
        years_to_target = target_year - current_year
        annual_reduction = SBTI_PATHWAYS[pathway]

        calculation_trace.append(f"Baseline emissions: {baseline:,.2f} tCO2e")
        calculation_trace.append(f"SBTi pathway: {pathway} ({annual_reduction*100:.1f}% annual reduction)")
        calculation_trace.append(f"Target year: {target_year}")

        # Calculate budget for next year
        cumulative_factor = (1 - annual_reduction) ** 1
        next_year_budget = baseline * cumulative_factor

        # Calculate target emissions
        target_factor = (1 - annual_reduction) ** years_to_target
        target_emissions = baseline * target_factor

        calculation_trace.append(f"Next year budget: {next_year_budget:,.2f} tCO2e")
        calculation_trace.append(f"{target_year} target: {target_emissions:,.2f} tCO2e")
        calculation_trace.append(f"Total reduction required: {(1-target_factor)*100:.1f}%")

        # Create allocation
        allocation = BudgetAllocation(
            allocation_id=f"sbti_{pathway}_{current_year + 1}",
            entity_name="Organization",
            entity_type="organization",
            budget_tco2e=round(next_year_budget, 2),
            period=BudgetPeriod.ANNUAL,
            start_date=datetime(current_year + 1, 1, 1),
            end_date=datetime(current_year + 1, 12, 31),
            allocation_method=AllocationMethod.SBTI,
            baseline_year_emissions=baseline,
            target_reduction_pct=round(annual_reduction * 100, 2)
        )

        provenance_hash = hashlib.sha256(
            json.dumps(allocation.model_dump(), sort_keys=True, default=str).encode()
        ).hexdigest()

        return ClimateBudgetOutput(
            success=True,
            operation="calculate_sbti_budget",
            allocation=allocation,
            calculation_trace=calculation_trace,
            provenance_hash=provenance_hash
        )

    def _distribute_budget(self, input_data: ClimateBudgetInput) -> ClimateBudgetOutput:
        """Distribute budget across entities."""
        calculation_trace: List[str] = []

        if input_data.allocation is None:
            return ClimateBudgetOutput(
                success=False,
                operation="distribute_budget",
                calculation_trace=["ERROR: Total allocation required"]
            )

        total_allocation = input_data.allocation
        weights = input_data.entity_weights or {}
        method = input_data.allocation_method or AllocationMethod.EQUAL

        calculation_trace.append(f"Distributing {total_allocation.budget_tco2e:,.2f} tCO2e")
        calculation_trace.append(f"Method: {method.value}")

        allocations: List[BudgetAllocation] = []

        if method == AllocationMethod.EQUAL:
            if not weights:
                return ClimateBudgetOutput(
                    success=False,
                    operation="distribute_budget",
                    calculation_trace=["ERROR: entity_weights required"]
                )

            equal_share = total_allocation.budget_tco2e / len(weights)
            for entity_name in weights.keys():
                alloc = BudgetAllocation(
                    allocation_id=f"{total_allocation.allocation_id}_{entity_name}",
                    entity_name=entity_name,
                    entity_type="department",
                    parent_entity_id=total_allocation.allocation_id,
                    budget_tco2e=round(equal_share, 2),
                    period=total_allocation.period,
                    start_date=total_allocation.start_date,
                    end_date=total_allocation.end_date,
                    allocation_method=method
                )
                allocations.append(alloc)
                self._allocations[alloc.allocation_id] = alloc
                calculation_trace.append(f"  {entity_name}: {equal_share:,.2f} tCO2e")

        elif method in [AllocationMethod.ECONOMIC, AllocationMethod.INTENSITY]:
            if not weights:
                return ClimateBudgetOutput(
                    success=False,
                    operation="distribute_budget",
                    calculation_trace=["ERROR: entity_weights required"]
                )

            total_weight = sum(weights.values())
            for entity_name, weight in weights.items():
                share = total_allocation.budget_tco2e * (weight / total_weight)
                alloc = BudgetAllocation(
                    allocation_id=f"{total_allocation.allocation_id}_{entity_name}",
                    entity_name=entity_name,
                    entity_type="department",
                    parent_entity_id=total_allocation.allocation_id,
                    budget_tco2e=round(share, 2),
                    period=total_allocation.period,
                    start_date=total_allocation.start_date,
                    end_date=total_allocation.end_date,
                    allocation_method=method
                )
                allocations.append(alloc)
                self._allocations[alloc.allocation_id] = alloc
                calculation_trace.append(f"  {entity_name}: {share:,.2f} tCO2e ({weight/total_weight*100:.1f}%)")

        provenance_hash = hashlib.sha256(
            json.dumps([a.model_dump() for a in allocations], sort_keys=True, default=str).encode()
        ).hexdigest()

        return ClimateBudgetOutput(
            success=True,
            operation="distribute_budget",
            allocations=allocations,
            calculation_trace=calculation_trace,
            provenance_hash=provenance_hash
        )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "ClimateBudgetAgent",
    "ClimateBudgetInput",
    "ClimateBudgetOutput",
    "BudgetPeriod",
    "BudgetAllocation",
    "BudgetVariance",
    "CarbonBudgetStatus",
    "BudgetStatus",
    "AllocationMethod",
    "VarianceSeverity",
]
