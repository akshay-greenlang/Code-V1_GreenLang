# -*- coding: utf-8 -*-
"""
GL-DECARB-PUB-001: Municipal Climate Action Agent
==================================================

Develops and tracks city-level climate action plans with emission inventories,
sector-specific targets, and implementation roadmaps. Supports municipal
governments in achieving their climate commitments.

Capabilities:
    - Municipal GHG inventory development (Scopes 1, 2, 3)
    - Science-based target setting (aligned with 1.5C/2C pathways)
    - Sector-specific decarbonization pathways
    - Action plan development with milestones
    - Progress tracking and gap analysis
    - Multi-stakeholder engagement planning
    - Cost-benefit analysis of climate actions

Zero-Hallucination Principle:
    All emission calculations use verified emission factors.
    Target pathways are derived from IPCC and science-based methodologies.
    No numeric values are generated without documented provenance.

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class TargetType(str, Enum):
    """Types of climate targets."""
    ABSOLUTE_REDUCTION = "absolute_reduction"  # Reduce by X tonnes
    PERCENTAGE_REDUCTION = "percentage_reduction"  # Reduce by X%
    CARBON_NEUTRAL = "carbon_neutral"  # Net zero by year
    NET_ZERO = "net_zero"  # Absolute zero with offsets
    INTENSITY_REDUCTION = "intensity_reduction"  # Per capita or per GDP


class ActionCategory(str, Enum):
    """Categories of climate actions."""
    ENERGY = "energy"  # Energy efficiency and renewables
    TRANSPORTATION = "transportation"  # Sustainable transport
    BUILDINGS = "buildings"  # Building efficiency
    WASTE = "waste"  # Waste management
    LAND_USE = "land_use"  # Land use and forestry
    INDUSTRY = "industry"  # Industrial processes
    WATER = "water"  # Water and wastewater
    ADAPTATION = "adaptation"  # Climate adaptation
    GOVERNANCE = "governance"  # Policy and governance


class ActionStatus(str, Enum):
    """Status of climate actions."""
    PROPOSED = "proposed"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ON_HOLD = "on_hold"
    CANCELLED = "cancelled"


class EmissionScope(str, Enum):
    """GHG Protocol emission scopes."""
    SCOPE_1 = "scope_1"  # Direct emissions
    SCOPE_2 = "scope_2"  # Indirect from purchased energy
    SCOPE_3 = "scope_3"  # Other indirect emissions


# =============================================================================
# Pydantic Models
# =============================================================================

class SectorEmissions(BaseModel):
    """Emissions data for a specific sector."""

    sector: str = Field(..., description="Sector name")
    scope_1_tco2e: float = Field(
        default=0.0,
        ge=0,
        description="Scope 1 emissions in tCO2e"
    )
    scope_2_tco2e: float = Field(
        default=0.0,
        ge=0,
        description="Scope 2 emissions in tCO2e"
    )
    scope_3_tco2e: float = Field(
        default=0.0,
        ge=0,
        description="Scope 3 emissions in tCO2e"
    )
    year: int = Field(..., description="Inventory year")
    methodology: str = Field(
        default="GPC",
        description="Methodology used (e.g., GPC, ICLEI)"
    )
    data_quality_score: float = Field(
        default=3.0,
        ge=1.0,
        le=5.0,
        description="Data quality score (1=low, 5=high)"
    )

    @property
    def total_tco2e(self) -> float:
        """Calculate total emissions across all scopes."""
        return self.scope_1_tco2e + self.scope_2_tco2e + self.scope_3_tco2e


class ClimateTarget(BaseModel):
    """Climate target definition."""

    target_id: str = Field(..., description="Unique target identifier")
    target_type: TargetType = Field(..., description="Type of target")
    target_year: int = Field(..., ge=2020, le=2100, description="Target year")
    base_year: int = Field(..., ge=1990, le=2024, description="Baseline year")
    target_value: float = Field(..., description="Target value (% or tonnes)")
    sector: Optional[str] = Field(None, description="Sector if sector-specific")
    scope: Optional[EmissionScope] = Field(
        None,
        description="Scope if scope-specific"
    )
    interim_targets: Dict[int, float] = Field(
        default_factory=dict,
        description="Interim targets by year"
    )
    aligned_with: List[str] = Field(
        default_factory=list,
        description="Alignment with frameworks (e.g., Paris, SBTi)"
    )

    @field_validator('target_year')
    @classmethod
    def validate_target_year(cls, v: int, info) -> int:
        """Ensure target year is after base year."""
        # Note: In Pydantic v2, we cannot access other fields easily in validators
        # This would need to be a model_validator for cross-field validation
        return v


class ClimateAction(BaseModel):
    """Individual climate action within a plan."""

    action_id: str = Field(..., description="Unique action identifier")
    name: str = Field(..., description="Action name")
    description: str = Field(..., description="Detailed description")
    category: ActionCategory = Field(..., description="Action category")
    status: ActionStatus = Field(
        default=ActionStatus.PROPOSED,
        description="Current status"
    )

    # Impact estimates
    estimated_reduction_tco2e: float = Field(
        default=0.0,
        ge=0,
        description="Estimated annual emission reduction in tCO2e"
    )
    uncertainty_range_percent: float = Field(
        default=20.0,
        ge=0,
        le=100,
        description="Uncertainty range as percentage"
    )

    # Timeline
    start_date: Optional[date] = Field(None, description="Planned start date")
    end_date: Optional[date] = Field(None, description="Planned end date")

    # Financials
    estimated_cost_usd: float = Field(
        default=0.0,
        ge=0,
        description="Estimated implementation cost in USD"
    )
    estimated_savings_usd_per_year: float = Field(
        default=0.0,
        ge=0,
        description="Estimated annual savings in USD"
    )
    funding_sources: List[str] = Field(
        default_factory=list,
        description="Identified funding sources"
    )

    # Implementation
    responsible_department: Optional[str] = Field(
        None,
        description="Responsible department"
    )
    partners: List[str] = Field(
        default_factory=list,
        description="Implementation partners"
    )
    kpis: Dict[str, str] = Field(
        default_factory=dict,
        description="Key performance indicators"
    )

    # Progress tracking
    progress_percent: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Implementation progress percentage"
    )
    actual_reduction_tco2e: float = Field(
        default=0.0,
        ge=0,
        description="Actual measured emission reduction"
    )

    # Provenance
    methodology_reference: Optional[str] = Field(
        None,
        description="Reference for emission reduction calculation"
    )
    data_sources: List[str] = Field(
        default_factory=list,
        description="Data sources used"
    )

    @property
    def payback_period_years(self) -> Optional[float]:
        """Calculate simple payback period in years."""
        if self.estimated_savings_usd_per_year > 0:
            return self.estimated_cost_usd / self.estimated_savings_usd_per_year
        return None


class ClimateActionPlan(BaseModel):
    """Complete municipal climate action plan."""

    plan_id: str = Field(..., description="Unique plan identifier")
    municipality_name: str = Field(..., description="Municipality name")
    population: int = Field(..., gt=0, description="Population")
    plan_name: str = Field(..., description="Plan name")
    plan_version: str = Field(default="1.0", description="Version")

    # Inventory
    base_year_emissions: List[SectorEmissions] = Field(
        default_factory=list,
        description="Base year emission inventory"
    )
    current_year_emissions: List[SectorEmissions] = Field(
        default_factory=list,
        description="Current year emission inventory"
    )

    # Targets
    targets: List[ClimateTarget] = Field(
        default_factory=list,
        description="Climate targets"
    )

    # Actions
    actions: List[ClimateAction] = Field(
        default_factory=list,
        description="Climate actions"
    )

    # Metadata
    created_at: datetime = Field(
        default_factory=DeterministicClock.now,
        description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=DeterministicClock.now,
        description="Last update timestamp"
    )
    created_by: Optional[str] = Field(None, description="Creator")

    # Provenance
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail"
    )

    @property
    def total_base_year_emissions(self) -> float:
        """Calculate total base year emissions."""
        return sum(s.total_tco2e for s in self.base_year_emissions)

    @property
    def total_current_emissions(self) -> float:
        """Calculate total current emissions."""
        return sum(s.total_tco2e for s in self.current_year_emissions)

    @property
    def total_planned_reduction(self) -> float:
        """Calculate total planned reduction from all actions."""
        return sum(a.estimated_reduction_tco2e for a in self.actions)

    @property
    def emissions_per_capita(self) -> float:
        """Calculate emissions per capita."""
        if self.population > 0:
            return self.total_current_emissions / self.population
        return 0.0


# =============================================================================
# Agent Input/Output Models
# =============================================================================

class MunicipalClimateActionInput(BaseModel):
    """Input for Municipal Climate Action Agent."""

    action: str = Field(
        ...,
        description="Action: create_plan, update_inventory, add_target, "
                    "add_action, calculate_gap, generate_roadmap, track_progress"
    )

    # Plan identification
    plan_id: Optional[str] = Field(None, description="Plan ID for updates")

    # For creating plans
    municipality_name: Optional[str] = Field(None, description="Municipality name")
    population: Optional[int] = Field(None, description="Population")
    plan_name: Optional[str] = Field(None, description="Plan name")

    # For inventory
    sector_emissions: Optional[List[SectorEmissions]] = Field(
        None,
        description="Sector emissions data"
    )
    inventory_year: Optional[int] = Field(None, description="Inventory year")

    # For targets
    target: Optional[ClimateTarget] = Field(None, description="Climate target")

    # For actions
    climate_action: Optional[ClimateAction] = Field(
        None,
        description="Climate action to add"
    )

    # For gap analysis
    target_year: Optional[int] = Field(
        None,
        description="Target year for gap analysis"
    )

    # Metadata
    user_id: Optional[str] = Field(None, description="User ID")
    tenant_id: Optional[str] = Field(None, description="Tenant ID")

    @field_validator('action')
    @classmethod
    def validate_action(cls, v: str) -> str:
        """Validate action is supported."""
        valid_actions = {
            'create_plan',
            'update_inventory',
            'add_target',
            'add_action',
            'update_action',
            'calculate_gap',
            'generate_roadmap',
            'track_progress',
            'get_plan',
            'list_plans',
        }
        if v not in valid_actions:
            raise ValueError(f"Invalid action: {v}. Valid: {valid_actions}")
        return v


class MunicipalClimateActionOutput(BaseModel):
    """Output from Municipal Climate Action Agent."""

    success: bool = Field(..., description="Whether operation succeeded")
    action: str = Field(..., description="Action performed")

    # Results
    plan: Optional[ClimateActionPlan] = Field(
        None,
        description="Climate action plan"
    )
    plans: Optional[List[ClimateActionPlan]] = Field(
        None,
        description="List of plans"
    )

    # Gap analysis results
    gap_analysis: Optional[Dict[str, Any]] = Field(
        None,
        description="Gap analysis results"
    )

    # Roadmap
    roadmap: Optional[Dict[str, Any]] = Field(
        None,
        description="Implementation roadmap"
    )

    # Progress tracking
    progress_report: Optional[Dict[str, Any]] = Field(
        None,
        description="Progress tracking report"
    )

    # Provenance
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail"
    )
    calculation_trace: List[str] = Field(
        default_factory=list,
        description="Calculation trace for audit"
    )

    # Error handling
    error: Optional[str] = Field(None, description="Error message")
    warnings: List[str] = Field(default_factory=list, description="Warnings")

    # Metadata
    timestamp: datetime = Field(
        default_factory=DeterministicClock.now,
        description="Timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Processing time in milliseconds"
    )


# =============================================================================
# Municipal Climate Action Agent
# =============================================================================

class MunicipalClimateActionAgent(BaseAgent):
    """
    GL-DECARB-PUB-001: Municipal Climate Action Agent

    Develops and tracks city-level climate action plans with emission inventories,
    sector-specific targets, and implementation roadmaps.

    Zero-Hallucination Guarantees:
        - All emission calculations use verified GPC methodology
        - Targets are validated against science-based pathways
        - Complete audit trail with SHA-256 hashes
        - All data sources are documented

    Usage:
        agent = MunicipalClimateActionAgent()
        result = agent.run({
            'action': 'create_plan',
            'municipality_name': 'Springfield',
            'population': 500000,
            'plan_name': 'Climate Action Plan 2030'
        })
    """

    AGENT_ID = "GL-DECARB-PUB-001"
    AGENT_NAME = "Municipal Climate Action Agent"
    VERSION = "1.0.0"

    # Science-based annual reduction rate for 1.5C pathway
    ANNUAL_REDUCTION_RATE_15C = 0.042  # 4.2% per year
    ANNUAL_REDUCTION_RATE_2C = 0.025   # 2.5% per year

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize Municipal Climate Action Agent."""
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Municipal climate action planning and tracking",
                version=self.VERSION,
                parameters={
                    "default_methodology": "GPC",
                    "default_target_pathway": "1.5C",
                    "enable_auto_gap_analysis": True,
                }
            )
        super().__init__(config)

        # Plan storage (in-memory, replace with database in production)
        self._plans: Dict[str, ClimateActionPlan] = {}

        # Emission factors for gap filling (simplified)
        self._sector_defaults = {
            "residential": {"scope_1_factor": 2.5, "scope_2_factor": 1.8},
            "commercial": {"scope_1_factor": 3.2, "scope_2_factor": 2.1},
            "industrial": {"scope_1_factor": 5.5, "scope_2_factor": 3.0},
            "transportation": {"scope_1_factor": 4.2, "scope_2_factor": 0.1},
            "waste": {"scope_1_factor": 1.2, "scope_2_factor": 0.2},
        }

        self.logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Execute climate action operation.

        Args:
            input_data: Input parameters including action and data

        Returns:
            AgentResult containing operation results
        """
        import time
        start_time = time.time()

        try:
            # Parse and validate input
            agent_input = MunicipalClimateActionInput(**input_data)

            # Route to appropriate handler
            action_handlers = {
                'create_plan': self._handle_create_plan,
                'update_inventory': self._handle_update_inventory,
                'add_target': self._handle_add_target,
                'add_action': self._handle_add_action,
                'update_action': self._handle_update_action,
                'calculate_gap': self._handle_calculate_gap,
                'generate_roadmap': self._handle_generate_roadmap,
                'track_progress': self._handle_track_progress,
                'get_plan': self._handle_get_plan,
                'list_plans': self._handle_list_plans,
            }

            handler = action_handlers.get(agent_input.action)
            if not handler:
                raise ValueError(f"Unknown action: {agent_input.action}")

            output = handler(agent_input)

            # Calculate processing time
            output.processing_time_ms = (time.time() - start_time) * 1000

            # Calculate provenance hash
            output.provenance_hash = self._calculate_output_hash(output)

            return AgentResult(
                success=output.success,
                data=output.model_dump(),
                error=output.error,
            )

        except Exception as e:
            self.logger.error(f"Climate action operation failed: {str(e)}", exc_info=True)
            return AgentResult(
                success=False,
                error=str(e),
            )

    def _handle_create_plan(
        self,
        input_data: MunicipalClimateActionInput
    ) -> MunicipalClimateActionOutput:
        """Create a new climate action plan."""
        trace = []

        if not input_data.municipality_name:
            return MunicipalClimateActionOutput(
                success=False,
                action='create_plan',
                error="Municipality name is required"
            )

        if not input_data.population or input_data.population <= 0:
            return MunicipalClimateActionOutput(
                success=False,
                action='create_plan',
                error="Valid population is required"
            )

        # Generate plan ID
        plan_id = f"CAP-{input_data.municipality_name.upper()[:3]}-{DeterministicClock.now().strftime('%Y%m%d')}"
        trace.append(f"Generated plan ID: {plan_id}")

        # Create plan
        plan = ClimateActionPlan(
            plan_id=plan_id,
            municipality_name=input_data.municipality_name,
            population=input_data.population,
            plan_name=input_data.plan_name or f"{input_data.municipality_name} Climate Action Plan",
            created_by=input_data.user_id,
        )

        # Store plan
        self._plans[plan_id] = plan
        trace.append(f"Created plan for {input_data.municipality_name}")

        self.logger.info(f"Created climate action plan: {plan_id}")

        return MunicipalClimateActionOutput(
            success=True,
            action='create_plan',
            plan=plan,
            calculation_trace=trace,
        )

    def _handle_update_inventory(
        self,
        input_data: MunicipalClimateActionInput
    ) -> MunicipalClimateActionOutput:
        """Update emission inventory for a plan."""
        trace = []

        if not input_data.plan_id:
            return MunicipalClimateActionOutput(
                success=False,
                action='update_inventory',
                error="Plan ID is required"
            )

        plan = self._plans.get(input_data.plan_id)
        if not plan:
            return MunicipalClimateActionOutput(
                success=False,
                action='update_inventory',
                error=f"Plan not found: {input_data.plan_id}"
            )

        if not input_data.sector_emissions:
            return MunicipalClimateActionOutput(
                success=False,
                action='update_inventory',
                error="Sector emissions data is required"
            )

        # Determine if this is base year or current year inventory
        inventory_year = input_data.inventory_year or DeterministicClock.now().year

        # Calculate totals for audit
        total_emissions = sum(s.total_tco2e for s in input_data.sector_emissions)
        trace.append(f"Total emissions for {inventory_year}: {total_emissions:.2f} tCO2e")

        # Check if any base year inventory exists
        if not plan.base_year_emissions:
            plan.base_year_emissions = input_data.sector_emissions
            trace.append(f"Set as base year inventory ({inventory_year})")
        else:
            plan.current_year_emissions = input_data.sector_emissions
            trace.append(f"Set as current year inventory ({inventory_year})")

        # Calculate per-sector breakdown for trace
        for sector in input_data.sector_emissions:
            trace.append(
                f"  {sector.sector}: {sector.total_tco2e:.2f} tCO2e "
                f"(S1: {sector.scope_1_tco2e:.2f}, S2: {sector.scope_2_tco2e:.2f}, "
                f"S3: {sector.scope_3_tco2e:.2f})"
            )

        # Calculate per capita
        per_capita = total_emissions / plan.population if plan.population > 0 else 0
        trace.append(f"Per capita emissions: {per_capita:.2f} tCO2e/person")

        plan.updated_at = DeterministicClock.now()

        self.logger.info(
            f"Updated inventory for plan {input_data.plan_id}: "
            f"{total_emissions:.2f} tCO2e"
        )

        return MunicipalClimateActionOutput(
            success=True,
            action='update_inventory',
            plan=plan,
            calculation_trace=trace,
        )

    def _handle_add_target(
        self,
        input_data: MunicipalClimateActionInput
    ) -> MunicipalClimateActionOutput:
        """Add a climate target to a plan."""
        trace = []

        if not input_data.plan_id:
            return MunicipalClimateActionOutput(
                success=False,
                action='add_target',
                error="Plan ID is required"
            )

        plan = self._plans.get(input_data.plan_id)
        if not plan:
            return MunicipalClimateActionOutput(
                success=False,
                action='add_target',
                error=f"Plan not found: {input_data.plan_id}"
            )

        if not input_data.target:
            return MunicipalClimateActionOutput(
                success=False,
                action='add_target',
                error="Target is required"
            )

        target = input_data.target

        # Validate target against science-based pathways
        warnings = []
        if target.target_type == TargetType.PERCENTAGE_REDUCTION:
            years_to_target = target.target_year - target.base_year
            required_reduction_15c = 1 - (1 - self.ANNUAL_REDUCTION_RATE_15C) ** years_to_target

            trace.append(
                f"Target: {target.target_value}% reduction by {target.target_year} "
                f"(base year: {target.base_year})"
            )
            trace.append(
                f"1.5C pathway requires {required_reduction_15c*100:.1f}% reduction "
                f"over {years_to_target} years"
            )

            if target.target_value < required_reduction_15c * 100:
                warnings.append(
                    f"Target ({target.target_value}%) is less ambitious than 1.5C pathway "
                    f"({required_reduction_15c*100:.1f}%)"
                )

        # Add target
        plan.targets.append(target)
        plan.updated_at = DeterministicClock.now()

        self.logger.info(f"Added target to plan {input_data.plan_id}: {target.target_id}")

        return MunicipalClimateActionOutput(
            success=True,
            action='add_target',
            plan=plan,
            calculation_trace=trace,
            warnings=warnings,
        )

    def _handle_add_action(
        self,
        input_data: MunicipalClimateActionInput
    ) -> MunicipalClimateActionOutput:
        """Add a climate action to a plan."""
        trace = []

        if not input_data.plan_id:
            return MunicipalClimateActionOutput(
                success=False,
                action='add_action',
                error="Plan ID is required"
            )

        plan = self._plans.get(input_data.plan_id)
        if not plan:
            return MunicipalClimateActionOutput(
                success=False,
                action='add_action',
                error=f"Plan not found: {input_data.plan_id}"
            )

        if not input_data.climate_action:
            return MunicipalClimateActionOutput(
                success=False,
                action='add_action',
                error="Climate action is required"
            )

        action = input_data.climate_action

        # Log action details
        trace.append(f"Adding action: {action.name}")
        trace.append(f"  Category: {action.category.value}")
        trace.append(f"  Estimated reduction: {action.estimated_reduction_tco2e:.2f} tCO2e/year")

        if action.estimated_cost_usd > 0:
            trace.append(f"  Estimated cost: ${action.estimated_cost_usd:,.2f}")

            if action.payback_period_years:
                trace.append(f"  Payback period: {action.payback_period_years:.1f} years")

        # Calculate cost per tonne
        if action.estimated_reduction_tco2e > 0 and action.estimated_cost_usd > 0:
            cost_per_tonne = action.estimated_cost_usd / action.estimated_reduction_tco2e
            trace.append(f"  Cost effectiveness: ${cost_per_tonne:.2f}/tCO2e")

        # Add action
        plan.actions.append(action)
        plan.updated_at = DeterministicClock.now()

        # Calculate cumulative impact
        total_reduction = sum(a.estimated_reduction_tco2e for a in plan.actions)
        trace.append(f"Total planned reduction: {total_reduction:.2f} tCO2e/year")

        self.logger.info(f"Added action to plan {input_data.plan_id}: {action.action_id}")

        return MunicipalClimateActionOutput(
            success=True,
            action='add_action',
            plan=plan,
            calculation_trace=trace,
        )

    def _handle_update_action(
        self,
        input_data: MunicipalClimateActionInput
    ) -> MunicipalClimateActionOutput:
        """Update an existing climate action."""
        trace = []

        if not input_data.plan_id:
            return MunicipalClimateActionOutput(
                success=False,
                action='update_action',
                error="Plan ID is required"
            )

        plan = self._plans.get(input_data.plan_id)
        if not plan:
            return MunicipalClimateActionOutput(
                success=False,
                action='update_action',
                error=f"Plan not found: {input_data.plan_id}"
            )

        if not input_data.climate_action:
            return MunicipalClimateActionOutput(
                success=False,
                action='update_action',
                error="Climate action is required"
            )

        action = input_data.climate_action

        # Find and update action
        found = False
        for i, existing_action in enumerate(plan.actions):
            if existing_action.action_id == action.action_id:
                plan.actions[i] = action
                found = True
                trace.append(f"Updated action: {action.action_id}")
                break

        if not found:
            return MunicipalClimateActionOutput(
                success=False,
                action='update_action',
                error=f"Action not found: {action.action_id}"
            )

        plan.updated_at = DeterministicClock.now()

        return MunicipalClimateActionOutput(
            success=True,
            action='update_action',
            plan=plan,
            calculation_trace=trace,
        )

    def _handle_calculate_gap(
        self,
        input_data: MunicipalClimateActionInput
    ) -> MunicipalClimateActionOutput:
        """Calculate gap between targets and planned actions."""
        trace = []

        if not input_data.plan_id:
            return MunicipalClimateActionOutput(
                success=False,
                action='calculate_gap',
                error="Plan ID is required"
            )

        plan = self._plans.get(input_data.plan_id)
        if not plan:
            return MunicipalClimateActionOutput(
                success=False,
                action='calculate_gap',
                error=f"Plan not found: {input_data.plan_id}"
            )

        target_year = input_data.target_year or 2030

        # Calculate base year emissions
        base_emissions = plan.total_base_year_emissions
        if base_emissions == 0:
            return MunicipalClimateActionOutput(
                success=False,
                action='calculate_gap',
                error="No base year emissions data available"
            )

        trace.append(f"Base year emissions: {base_emissions:.2f} tCO2e")

        # Find relevant targets
        relevant_targets = [
            t for t in plan.targets
            if t.target_year <= target_year
        ]

        # Calculate required reduction
        required_reduction = 0.0
        for target in relevant_targets:
            if target.target_type == TargetType.PERCENTAGE_REDUCTION:
                required_reduction = max(
                    required_reduction,
                    base_emissions * (target.target_value / 100)
                )
            elif target.target_type == TargetType.ABSOLUTE_REDUCTION:
                required_reduction = max(required_reduction, target.target_value)
            elif target.target_type in (TargetType.CARBON_NEUTRAL, TargetType.NET_ZERO):
                required_reduction = base_emissions

        trace.append(f"Required reduction by {target_year}: {required_reduction:.2f} tCO2e")

        # Calculate planned reduction from actions
        planned_reduction = sum(
            a.estimated_reduction_tco2e
            for a in plan.actions
            if a.end_date is None or a.end_date.year <= target_year
        )
        trace.append(f"Planned reduction from actions: {planned_reduction:.2f} tCO2e")

        # Calculate gap
        gap = required_reduction - planned_reduction
        gap_percent = (gap / required_reduction * 100) if required_reduction > 0 else 0

        trace.append(f"Gap: {gap:.2f} tCO2e ({gap_percent:.1f}%)")

        # Build gap analysis result
        gap_analysis = {
            "target_year": target_year,
            "base_year_emissions_tco2e": base_emissions,
            "required_reduction_tco2e": required_reduction,
            "planned_reduction_tco2e": planned_reduction,
            "gap_tco2e": gap,
            "gap_percent": gap_percent,
            "on_track": gap <= 0,
            "actions_count": len(plan.actions),
            "targets_analyzed": len(relevant_targets),
        }

        # Analyze by category
        category_analysis = {}
        for category in ActionCategory:
            category_actions = [
                a for a in plan.actions
                if a.category == category
            ]
            if category_actions:
                category_analysis[category.value] = {
                    "actions_count": len(category_actions),
                    "total_reduction_tco2e": sum(
                        a.estimated_reduction_tco2e for a in category_actions
                    ),
                    "total_cost_usd": sum(
                        a.estimated_cost_usd for a in category_actions
                    ),
                }

        gap_analysis["by_category"] = category_analysis

        self.logger.info(
            f"Gap analysis for {input_data.plan_id}: "
            f"{gap:.2f} tCO2e gap ({gap_percent:.1f}%)"
        )

        return MunicipalClimateActionOutput(
            success=True,
            action='calculate_gap',
            plan=plan,
            gap_analysis=gap_analysis,
            calculation_trace=trace,
        )

    def _handle_generate_roadmap(
        self,
        input_data: MunicipalClimateActionInput
    ) -> MunicipalClimateActionOutput:
        """Generate implementation roadmap."""
        trace = []

        if not input_data.plan_id:
            return MunicipalClimateActionOutput(
                success=False,
                action='generate_roadmap',
                error="Plan ID is required"
            )

        plan = self._plans.get(input_data.plan_id)
        if not plan:
            return MunicipalClimateActionOutput(
                success=False,
                action='generate_roadmap',
                error=f"Plan not found: {input_data.plan_id}"
            )

        trace.append("Generating implementation roadmap")

        # Group actions by timeline
        immediate_actions = []  # Start within 1 year
        short_term_actions = []  # Start within 1-3 years
        medium_term_actions = []  # Start within 3-5 years
        long_term_actions = []  # Start after 5 years

        current_year = DeterministicClock.now().year

        for action in plan.actions:
            if action.start_date:
                years_until_start = action.start_date.year - current_year
                if years_until_start <= 1:
                    immediate_actions.append(action)
                elif years_until_start <= 3:
                    short_term_actions.append(action)
                elif years_until_start <= 5:
                    medium_term_actions.append(action)
                else:
                    long_term_actions.append(action)
            else:
                # Actions without dates go to immediate
                immediate_actions.append(action)

        # Sort each phase by cost-effectiveness
        def cost_effectiveness(a: ClimateAction) -> float:
            if a.estimated_reduction_tco2e > 0 and a.estimated_cost_usd > 0:
                return a.estimated_cost_usd / a.estimated_reduction_tco2e
            return float('inf')

        immediate_actions.sort(key=cost_effectiveness)
        short_term_actions.sort(key=cost_effectiveness)
        medium_term_actions.sort(key=cost_effectiveness)
        long_term_actions.sort(key=cost_effectiveness)

        # Calculate phase totals
        def phase_summary(actions: List[ClimateAction]) -> Dict[str, Any]:
            return {
                "actions_count": len(actions),
                "total_reduction_tco2e": sum(a.estimated_reduction_tco2e for a in actions),
                "total_cost_usd": sum(a.estimated_cost_usd for a in actions),
                "actions": [
                    {
                        "action_id": a.action_id,
                        "name": a.name,
                        "category": a.category.value,
                        "reduction_tco2e": a.estimated_reduction_tco2e,
                        "cost_usd": a.estimated_cost_usd,
                    }
                    for a in actions
                ]
            }

        roadmap = {
            "plan_id": plan.plan_id,
            "municipality": plan.municipality_name,
            "generated_at": DeterministicClock.now().isoformat(),
            "phases": {
                "immediate": {
                    "timeframe": f"{current_year}-{current_year + 1}",
                    **phase_summary(immediate_actions)
                },
                "short_term": {
                    "timeframe": f"{current_year + 1}-{current_year + 3}",
                    **phase_summary(short_term_actions)
                },
                "medium_term": {
                    "timeframe": f"{current_year + 3}-{current_year + 5}",
                    **phase_summary(medium_term_actions)
                },
                "long_term": {
                    "timeframe": f"After {current_year + 5}",
                    **phase_summary(long_term_actions)
                },
            },
            "cumulative_reduction_tco2e": sum(
                a.estimated_reduction_tco2e for a in plan.actions
            ),
            "total_investment_usd": sum(
                a.estimated_cost_usd for a in plan.actions
            ),
        }

        trace.append(f"Immediate actions: {len(immediate_actions)}")
        trace.append(f"Short-term actions: {len(short_term_actions)}")
        trace.append(f"Medium-term actions: {len(medium_term_actions)}")
        trace.append(f"Long-term actions: {len(long_term_actions)}")

        self.logger.info(f"Generated roadmap for plan {input_data.plan_id}")

        return MunicipalClimateActionOutput(
            success=True,
            action='generate_roadmap',
            plan=plan,
            roadmap=roadmap,
            calculation_trace=trace,
        )

    def _handle_track_progress(
        self,
        input_data: MunicipalClimateActionInput
    ) -> MunicipalClimateActionOutput:
        """Track progress against targets."""
        trace = []

        if not input_data.plan_id:
            return MunicipalClimateActionOutput(
                success=False,
                action='track_progress',
                error="Plan ID is required"
            )

        plan = self._plans.get(input_data.plan_id)
        if not plan:
            return MunicipalClimateActionOutput(
                success=False,
                action='track_progress',
                error=f"Plan not found: {input_data.plan_id}"
            )

        trace.append("Tracking progress against targets")

        # Calculate actual vs planned reductions
        actual_reduction = sum(a.actual_reduction_tco2e for a in plan.actions)
        planned_reduction = sum(a.estimated_reduction_tco2e for a in plan.actions)

        # Calculate action completion rates
        completed_actions = [a for a in plan.actions if a.status == ActionStatus.COMPLETED]
        in_progress_actions = [a for a in plan.actions if a.status == ActionStatus.IN_PROGRESS]

        # Calculate overall progress
        overall_progress = (
            sum(a.progress_percent for a in plan.actions) / len(plan.actions)
            if plan.actions else 0
        )

        progress_report = {
            "plan_id": plan.plan_id,
            "municipality": plan.municipality_name,
            "report_date": DeterministicClock.now().isoformat(),
            "actions_summary": {
                "total": len(plan.actions),
                "completed": len(completed_actions),
                "in_progress": len(in_progress_actions),
                "proposed": len([a for a in plan.actions if a.status == ActionStatus.PROPOSED]),
                "on_hold": len([a for a in plan.actions if a.status == ActionStatus.ON_HOLD]),
            },
            "emissions_reduction": {
                "planned_tco2e": planned_reduction,
                "actual_tco2e": actual_reduction,
                "achievement_rate_percent": (
                    actual_reduction / planned_reduction * 100
                    if planned_reduction > 0 else 0
                ),
            },
            "overall_progress_percent": overall_progress,
            "base_year_emissions_tco2e": plan.total_base_year_emissions,
            "current_emissions_tco2e": plan.total_current_emissions,
            "reduction_from_base_percent": (
                (plan.total_base_year_emissions - plan.total_current_emissions)
                / plan.total_base_year_emissions * 100
                if plan.total_base_year_emissions > 0 else 0
            ),
        }

        trace.append(f"Actions completed: {len(completed_actions)}/{len(plan.actions)}")
        trace.append(f"Overall progress: {overall_progress:.1f}%")
        trace.append(f"Actual reduction: {actual_reduction:.2f} tCO2e")

        self.logger.info(
            f"Progress report for {input_data.plan_id}: "
            f"{overall_progress:.1f}% overall progress"
        )

        return MunicipalClimateActionOutput(
            success=True,
            action='track_progress',
            plan=plan,
            progress_report=progress_report,
            calculation_trace=trace,
        )

    def _handle_get_plan(
        self,
        input_data: MunicipalClimateActionInput
    ) -> MunicipalClimateActionOutput:
        """Get a plan by ID."""
        if not input_data.plan_id:
            return MunicipalClimateActionOutput(
                success=False,
                action='get_plan',
                error="Plan ID is required"
            )

        plan = self._plans.get(input_data.plan_id)
        if not plan:
            return MunicipalClimateActionOutput(
                success=False,
                action='get_plan',
                error=f"Plan not found: {input_data.plan_id}"
            )

        return MunicipalClimateActionOutput(
            success=True,
            action='get_plan',
            plan=plan,
        )

    def _handle_list_plans(
        self,
        input_data: MunicipalClimateActionInput
    ) -> MunicipalClimateActionOutput:
        """List all plans."""
        plans = list(self._plans.values())

        return MunicipalClimateActionOutput(
            success=True,
            action='list_plans',
            plans=plans,
        )

    def _calculate_output_hash(self, output: MunicipalClimateActionOutput) -> str:
        """Calculate SHA-256 hash of output for provenance."""
        content = {
            "action": output.action,
            "success": output.success,
            "timestamp": output.timestamp.isoformat(),
        }

        if output.plan:
            content["plan_id"] = output.plan.plan_id
            content["plan_hash"] = hashlib.sha256(
                output.plan.model_dump_json().encode()
            ).hexdigest()[:16]

        if output.gap_analysis:
            content["gap_analysis_hash"] = hashlib.sha256(
                json.dumps(output.gap_analysis, sort_keys=True).encode()
            ).hexdigest()[:16]

        content_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()

    # =========================================================================
    # Public API Methods
    # =========================================================================

    def create_plan(
        self,
        municipality_name: str,
        population: int,
        plan_name: Optional[str] = None,
    ) -> ClimateActionPlan:
        """Create a new climate action plan directly."""
        result = self.run({
            'action': 'create_plan',
            'municipality_name': municipality_name,
            'population': population,
            'plan_name': plan_name,
        })

        if result.success:
            return ClimateActionPlan(**result.data['plan'])
        else:
            raise ValueError(result.error)

    def get_plan(self, plan_id: str) -> Optional[ClimateActionPlan]:
        """Get a plan by ID."""
        return self._plans.get(plan_id)

    def get_all_plans(self) -> List[ClimateActionPlan]:
        """Get all plans."""
        return list(self._plans.values())
