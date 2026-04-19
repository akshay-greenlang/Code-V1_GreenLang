# -*- coding: utf-8 -*-
"""
ActionPlanEngine - PACK-034 ISO 50001 EnMS Engine 7
====================================================

ISO 50001 Clause 6.2 compliant engine for creating, managing, and tracking
energy objectives, targets, and action plans.  Implements SMART objective
validation, financial analysis (simple payback, NPV, IRR, BCR), multi-criteria
action prioritisation, progress tracking with milestone monitoring, overdue
detection, Gantt data generation, and portfolio-level aggregation.

Calculation Methodology:
    SMART Objective Scoring:
        Each criterion (S, M, A, R, T) scores 0 or 20 points.
        smart_score = sum(20 * criterion_met for each of S, M, A, R, T)

    Financial Metrics:
        Simple Payback  = estimated_cost / estimated_savings_cost
        NPV = -cost + sum( savings_t / (1 + r)^t  for t in 1..n )
        IRR = r  such that NPV(r) = 0  (bisection, 100 iterations)
        BCR = PV(savings) / cost

    Savings Estimation:
        energy_savings_kwh  = baseline_kwh * improvement_pct
        cost_savings        = energy_savings_kwh * energy_price_per_kwh
        co2e_reduction_kg   = energy_savings_kwh * grid_emission_factor

    Progress Tracking:
        overall_pct = completed_items / total_items * 100
        on_track: items where status != overdue AND today <= due_date
        delayed:  items where status == overdue OR (today > due_date AND not completed)

    Multi-Criteria Prioritisation:
        score = w_payback * norm_payback + w_savings * norm_savings
              + w_cost_effectiveness * norm_cer + w_urgency * norm_urgency
        Normalised via min-max to [0, 1]; lower payback = better.

Regulatory References:
    - ISO 50001:2018 Clause 6.2 - Objectives, energy targets and planning
    - ISO 50001:2018 Clause 6.3 - Energy review
    - ISO 50001:2018 Clause 9.1 - Monitoring, measurement, analysis
    - IPMVP (International Performance Measurement and Verification Protocol)
      Options A-D for M&V of energy savings
    - EN 16247-1:2022 Energy audits - General requirements
    - ISO 50006:2023 Energy baselines and EnPIs

Zero-Hallucination:
    - All formulas are standard engineering economics (no LLM in calc path)
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result
    - IPMVP verification method descriptions from official protocol

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-034 ISO 50001 EnMS
Engine:  7 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100)."""
    return _safe_divide(part * Decimal("100"), whole)

def _round_val(value: Decimal, places: int = 6) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

def _today() -> date:
    """Return current UTC date."""
    return datetime.now(timezone.utc).date()

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ObjectiveType(str, Enum):
    """Type of energy objective per ISO 50001 Clause 6.2.

    REDUCTION: Absolute energy consumption reduction.
    EFFICIENCY_IMPROVEMENT: Energy performance improvement (EnPI).
    RENEWABLE_INTEGRATION: Increase renewable energy share.
    AWARENESS: Staff energy awareness and training.
    PROCUREMENT: Energy-efficient procurement practices.
    MONITORING_ENHANCEMENT: Improved monitoring and measurement.
    """
    REDUCTION = "reduction"
    EFFICIENCY_IMPROVEMENT = "efficiency_improvement"
    RENEWABLE_INTEGRATION = "renewable_integration"
    AWARENESS = "awareness"
    PROCUREMENT = "procurement"
    MONITORING_ENHANCEMENT = "monitoring_enhancement"

class TargetScope(str, Enum):
    """Scope boundary for an energy target.

    SCOPE1: Direct emissions (on-site combustion, fleet).
    SCOPE2: Indirect emissions (purchased electricity, steam).
    SCOPE3: Value chain emissions.
    ALL_SCOPES: All scopes combined.
    SPECIFIC_SEU: Specific significant energy use.
    FACILITY_WIDE: Entire facility.
    """
    SCOPE1 = "scope1"
    SCOPE2 = "scope2"
    SCOPE3 = "scope3"
    ALL_SCOPES = "all_scopes"
    SPECIFIC_SEU = "specific_seu"
    FACILITY_WIDE = "facility_wide"

class ActionStatus(str, Enum):
    """Lifecycle status of an action item or plan.

    PLANNED: Defined but not yet approved.
    APPROVED: Approved and awaiting start.
    IN_PROGRESS: Currently being implemented.
    COMPLETED: Successfully completed.
    ON_HOLD: Temporarily paused.
    CANCELLED: Permanently cancelled.
    OVERDUE: Past due date without completion.
    """
    PLANNED = "planned"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ON_HOLD = "on_hold"
    CANCELLED = "cancelled"
    OVERDUE = "overdue"

class ActionPriority(str, Enum):
    """Priority level for action items.

    CRITICAL: Must be addressed immediately.
    HIGH: Should be addressed within current period.
    MEDIUM: Schedule within next planning cycle.
    LOW: Address when resources are available.
    """
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ResourceType(str, Enum):
    """Type of resource required for an action plan.

    CAPITAL: Capital expenditure (equipment, infrastructure).
    OPERATIONAL: Operational expenditure (services, consumables).
    HUMAN: Staff time and labour.
    TECHNICAL: Technical expertise and consulting.
    EXTERNAL: External contractor or vendor services.
    """
    CAPITAL = "capital"
    OPERATIONAL = "operational"
    HUMAN = "human"
    TECHNICAL = "technical"
    EXTERNAL = "external"

class VerificationMethod(str, Enum):
    """IPMVP measurement and verification options.

    METERED_SAVINGS: Option B - Retrofit isolation, all parameter measurement.
    CALCULATED_SAVINGS: Option A - Retrofit isolation, key parameter measurement.
    STIPULATED_SAVINGS: Option A variant - Stipulated (engineering estimate).
    CALIBRATED_SIMULATION: Option D - Calibrated whole-building simulation.
    """
    METERED_SAVINGS = "metered_savings"
    CALCULATED_SAVINGS = "calculated_savings"
    STIPULATED_SAVINGS = "stipulated_savings"
    CALIBRATED_SIMULATION = "calibrated_simulation"

# ---------------------------------------------------------------------------
# Constants / Reference Data
# ---------------------------------------------------------------------------

DEFAULT_DISCOUNT_RATE: Decimal = Decimal("0.08")
DEFAULT_ANALYSIS_YEARS: int = 10
DEFAULT_ENERGY_PRICE_PER_KWH: Decimal = Decimal("0.12")
DEFAULT_GRID_EMISSION_FACTOR: Decimal = Decimal("0.4")  # kgCO2e per kWh
MAX_IRR_ITERATIONS: int = 100
IRR_TOLERANCE: Decimal = Decimal("0.0001")

# Typical savings ranges by common energy efficiency measure category.
# Values are (low_pct, high_pct) of baseline consumption.
TYPICAL_SAVINGS_BY_MEASURE: Dict[str, Dict[str, Any]] = {
    "led_lighting_retrofit": {
        "description": "Replace fluorescent/HID with LED lighting",
        "typical_savings_pct_low": Decimal("30"),
        "typical_savings_pct_high": Decimal("60"),
        "typical_payback_years": Decimal("2"),
        "category": "lighting",
    },
    "hvac_optimization": {
        "description": "HVAC scheduling, setpoint optimization, and controls",
        "typical_savings_pct_low": Decimal("10"),
        "typical_savings_pct_high": Decimal("30"),
        "typical_payback_years": Decimal("3"),
        "category": "hvac",
    },
    "vfd_motors": {
        "description": "Variable frequency drives on pumps and fans",
        "typical_savings_pct_low": Decimal("20"),
        "typical_savings_pct_high": Decimal("50"),
        "typical_payback_years": Decimal("2.5"),
        "category": "motors",
    },
    "compressed_air_optimization": {
        "description": "Leak repair, pressure reduction, controls upgrade",
        "typical_savings_pct_low": Decimal("15"),
        "typical_savings_pct_high": Decimal("35"),
        "typical_payback_years": Decimal("1.5"),
        "category": "compressed_air",
    },
    "building_envelope": {
        "description": "Insulation, glazing, air sealing improvements",
        "typical_savings_pct_low": Decimal("10"),
        "typical_savings_pct_high": Decimal("25"),
        "typical_payback_years": Decimal("7"),
        "category": "envelope",
    },
    "heat_recovery": {
        "description": "Waste heat recovery from exhaust, processes, or equipment",
        "typical_savings_pct_low": Decimal("15"),
        "typical_savings_pct_high": Decimal("40"),
        "typical_payback_years": Decimal("4"),
        "category": "thermal",
    },
    "boiler_upgrade": {
        "description": "High-efficiency boiler or condensing boiler replacement",
        "typical_savings_pct_low": Decimal("10"),
        "typical_savings_pct_high": Decimal("25"),
        "typical_payback_years": Decimal("5"),
        "category": "thermal",
    },
    "solar_pv": {
        "description": "Rooftop or ground-mount solar PV installation",
        "typical_savings_pct_low": Decimal("10"),
        "typical_savings_pct_high": Decimal("40"),
        "typical_payback_years": Decimal("6"),
        "category": "renewable",
    },
    "bms_upgrade": {
        "description": "Building management system installation or upgrade",
        "typical_savings_pct_low": Decimal("10"),
        "typical_savings_pct_high": Decimal("20"),
        "typical_payback_years": Decimal("4"),
        "category": "controls",
    },
    "power_factor_correction": {
        "description": "Capacitor bank installation for power factor improvement",
        "typical_savings_pct_low": Decimal("2"),
        "typical_savings_pct_high": Decimal("8"),
        "typical_payback_years": Decimal("2"),
        "category": "electrical",
    },
    "steam_system_optimization": {
        "description": "Steam trap repair, insulation, condensate return",
        "typical_savings_pct_low": Decimal("10"),
        "typical_savings_pct_high": Decimal("30"),
        "typical_payback_years": Decimal("1.5"),
        "category": "thermal",
    },
    "process_optimization": {
        "description": "Production scheduling, batch optimization, lean energy",
        "typical_savings_pct_low": Decimal("5"),
        "typical_savings_pct_high": Decimal("15"),
        "typical_payback_years": Decimal("1"),
        "category": "process",
    },
}

# ISO 50001 Clause 6.2 required elements for action plan completeness.
ACTION_PLAN_CHECKLIST: Dict[str, str] = {
    "what_will_be_done": "Description of the actions to be taken",
    "resources_required": "Financial, human, and technical resources needed",
    "responsible_person": "Person accountable for implementation",
    "completion_timeframe": "Target completion date or timeframe",
    "method_of_verification": "How results will be evaluated (IPMVP option)",
    "method_of_verifying_improvement": "How energy performance improvement is verified",
    "expected_results": "Quantified expected energy savings or improvement",
    "integration_with_enms": "How the action plan integrates with the EnMS",
}

# IPMVP Options A-D descriptions per the International Performance
# Measurement and Verification Protocol (EVO 10000-1:2022).
VERIFICATION_METHOD_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    VerificationMethod.CALCULATED_SAVINGS.value: {
        "name": "IPMVP Option A - Retrofit Isolation: Key Parameter Measurement",
        "description": (
            "Savings are determined by field measurement of the key "
            "performance parameter(s) which define the energy use of the "
            "affected system. Parameters not selected for measurement are "
            "estimated based on historical data, manufacturer specifications, "
            "or engineering judgement."
        ),
        "best_for": (
            "Single retrofit measures where key parameter can be measured "
            "and other parameters are predictable (e.g. lighting retrofit "
            "where operating hours are stipulated)."
        ),
        "accuracy": "Moderate - depends on quality of stipulated values",
    },
    VerificationMethod.METERED_SAVINGS.value: {
        "name": "IPMVP Option B - Retrofit Isolation: All Parameter Measurement",
        "description": (
            "Savings are determined by field measurement of all energy use "
            "parameters of the affected system. Short-term or continuous "
            "post-retrofit metering is used to quantify actual performance."
        ),
        "best_for": (
            "Individual systems where all parameters can be measured "
            "(e.g. VFD on a pump where both power and flow are metered)."
        ),
        "accuracy": "High - direct measurement of all relevant parameters",
    },
    VerificationMethod.STIPULATED_SAVINGS.value: {
        "name": "IPMVP Option A Variant - Engineering Estimate (Stipulated)",
        "description": (
            "Savings are estimated using engineering calculations based on "
            "equipment specifications, operational data, and accepted "
            "engineering methods.  No post-installation measurement is "
            "performed; savings are stipulated."
        ),
        "best_for": (
            "Low-cost or low-risk measures where the cost of measurement "
            "would exceed the value of improved accuracy (e.g. pipe "
            "insulation, weather-stripping)."
        ),
        "accuracy": "Low to moderate - no post-measurement verification",
    },
    VerificationMethod.CALIBRATED_SIMULATION.value: {
        "name": "IPMVP Option D - Calibrated Simulation",
        "description": (
            "Savings are determined through simulation of the facility's "
            "energy use.  The simulation model must be calibrated against "
            "actual utility data (monthly or hourly) so that it predicts "
            "patterns consistent with actual measured data.  Savings are "
            "the difference between baseline and post-retrofit simulations."
        ),
        "best_for": (
            "Whole-building or multi-measure projects where interactive "
            "effects make isolation impractical (e.g. deep retrofit of "
            "envelope + HVAC + lighting simultaneously)."
        ),
        "accuracy": "Moderate to high - depends on calibration quality",
    },
}

# Default criteria weights for action plan prioritisation.
DEFAULT_PRIORITY_WEIGHTS: Dict[str, Decimal] = {
    "payback": Decimal("0.30"),
    "savings": Decimal("0.25"),
    "cost_effectiveness": Decimal("0.25"),
    "urgency": Decimal("0.20"),
}

# Priority urgency scores by ActionPriority enum.
PRIORITY_URGENCY_SCORES: Dict[str, Decimal] = {
    ActionPriority.CRITICAL.value: Decimal("1.0"),
    ActionPriority.HIGH.value: Decimal("0.75"),
    ActionPriority.MEDIUM.value: Decimal("0.50"),
    ActionPriority.LOW.value: Decimal("0.25"),
}

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class SMARTObjective(BaseModel):
    """SMART criteria validation result for an energy objective.

    Each criterion (Specific, Measurable, Achievable, Relevant, Time-bound)
    is evaluated as a boolean.  The smart_score ranges from 0 to 100.

    Attributes:
        objective_id: Linked objective identifier.
        objective_text: The objective text that was validated.
        is_specific: Contains clear, unambiguous target.
        is_measurable: Has quantifiable metric and unit.
        is_achievable: Target is within realistic range.
        is_relevant: Aligned with energy policy and SEUs.
        is_time_bound: Has explicit deadline or timeframe.
        smart_score: Composite score 0-100.
        validation_notes: Explanatory notes per criterion.
    """
    objective_id: str = Field(
        default_factory=_new_uuid, description="Linked objective ID"
    )
    objective_text: str = Field(
        default="", max_length=2000, description="Objective text validated"
    )
    is_specific: bool = Field(
        default=False, description="Contains clear, unambiguous target"
    )
    is_measurable: bool = Field(
        default=False, description="Has quantifiable metric and unit"
    )
    is_achievable: bool = Field(
        default=False, description="Target within realistic range"
    )
    is_relevant: bool = Field(
        default=False, description="Aligned with energy policy"
    )
    is_time_bound: bool = Field(
        default=False, description="Has explicit deadline"
    )
    smart_score: Decimal = Field(
        default=Decimal("0"), ge=0, le=100,
        description="Composite SMART score 0-100"
    )
    validation_notes: List[str] = Field(
        default_factory=list, description="Per-criterion notes"
    )

class EnergyObjective(BaseModel):
    """An energy objective per ISO 50001 Clause 6.2.

    Attributes:
        objective_id: Unique objective identifier.
        enms_id: Parent EnMS identifier.
        objective_text: Full text of the objective.
        objective_type: Classification of the objective.
        target_scope: Scope boundary for this objective.
        baseline_value: Baseline energy value (kWh, GJ, etc.).
        target_value: Target energy value to achieve.
        target_unit: Unit of measurement (kWh, GJ, kgCO2e, etc.).
        target_date: Deadline for achieving the target.
        responsible_person: Accountable person or role.
        smart_validation: SMART criteria validation result.
        status: Current lifecycle status.
    """
    objective_id: str = Field(
        default_factory=_new_uuid, description="Unique objective ID"
    )
    enms_id: str = Field(
        default="", description="Parent EnMS identifier"
    )
    objective_text: str = Field(
        default="", max_length=2000, description="Objective text"
    )
    objective_type: ObjectiveType = Field(
        default=ObjectiveType.REDUCTION, description="Objective classification"
    )
    target_scope: TargetScope = Field(
        default=TargetScope.FACILITY_WIDE, description="Scope boundary"
    )
    baseline_value: Decimal = Field(
        default=Decimal("0"), ge=0, description="Baseline energy value"
    )
    target_value: Decimal = Field(
        default=Decimal("0"), ge=0, description="Target energy value"
    )
    target_unit: str = Field(
        default="kWh", max_length=50, description="Unit of measurement"
    )
    target_date: Optional[date] = Field(
        default=None, description="Target completion date"
    )
    responsible_person: str = Field(
        default="", max_length=200, description="Accountable person"
    )
    smart_validation: Optional[SMARTObjective] = Field(
        default=None, description="SMART validation result"
    )
    status: ActionStatus = Field(
        default=ActionStatus.PLANNED, description="Lifecycle status"
    )

    @field_validator("objective_type", mode="before")
    @classmethod
    def coerce_objective_type(cls, v: Any) -> Any:
        """Accept string values for ObjectiveType."""
        if isinstance(v, str):
            valid = {t.value for t in ObjectiveType}
            if v not in valid:
                raise ValueError(
                    f"Unknown objective type '{v}'. Must be one of: {sorted(valid)}"
                )
        return v

    @field_validator("target_scope", mode="before")
    @classmethod
    def coerce_target_scope(cls, v: Any) -> Any:
        """Accept string values for TargetScope."""
        if isinstance(v, str):
            valid = {t.value for t in TargetScope}
            if v not in valid:
                raise ValueError(
                    f"Unknown target scope '{v}'. Must be one of: {sorted(valid)}"
                )
        return v

    @field_validator("status", mode="before")
    @classmethod
    def coerce_status(cls, v: Any) -> Any:
        """Accept string values for ActionStatus."""
        if isinstance(v, str):
            valid = {t.value for t in ActionStatus}
            if v not in valid:
                raise ValueError(
                    f"Unknown status '{v}'. Must be one of: {sorted(valid)}"
                )
        return v

class EnergyTarget(BaseModel):
    """An energy target linked to an objective per ISO 50001 Clause 6.2.

    Attributes:
        target_id: Unique target identifier.
        objective_id: Parent objective identifier.
        description: Target description.
        target_value: Quantified target value.
        target_unit: Unit of measurement.
        baseline_value: Baseline value for comparison.
        achievement_pct: Current achievement percentage.
        interim_milestones: List of milestone definitions.
        verification_method: IPMVP verification option.
    """
    target_id: str = Field(
        default_factory=_new_uuid, description="Unique target ID"
    )
    objective_id: str = Field(
        default="", description="Parent objective ID"
    )
    description: str = Field(
        default="", max_length=2000, description="Target description"
    )
    target_value: Decimal = Field(
        default=Decimal("0"), ge=0, description="Quantified target value"
    )
    target_unit: str = Field(
        default="kWh", max_length=50, description="Unit of measurement"
    )
    baseline_value: Decimal = Field(
        default=Decimal("0"), ge=0, description="Baseline value"
    )
    achievement_pct: Decimal = Field(
        default=Decimal("0"), ge=0, le=Decimal("200"),
        description="Achievement percentage"
    )
    interim_milestones: List[Dict[str, Any]] = Field(
        default_factory=list, description="Milestone definitions"
    )
    verification_method: VerificationMethod = Field(
        default=VerificationMethod.CALCULATED_SAVINGS,
        description="IPMVP verification option"
    )

    @field_validator("verification_method", mode="before")
    @classmethod
    def coerce_verification_method(cls, v: Any) -> Any:
        """Accept string values for VerificationMethod."""
        if isinstance(v, str):
            valid = {t.value for t in VerificationMethod}
            if v not in valid:
                raise ValueError(
                    f"Unknown verification method '{v}'. Must be one of: {sorted(valid)}"
                )
        return v

class ResourceRequirement(BaseModel):
    """A resource needed for an action plan.

    Attributes:
        resource_type: Category of resource.
        description: What the resource is used for.
        estimated_cost: Estimated cost of this resource.
        currency: ISO 4217 currency code.
        source: Where the resource comes from.
        approved: Whether the resource has been approved.
    """
    resource_type: ResourceType = Field(
        default=ResourceType.CAPITAL, description="Resource category"
    )
    description: str = Field(
        default="", max_length=1000, description="Resource description"
    )
    estimated_cost: Decimal = Field(
        default=Decimal("0"), ge=0, description="Estimated cost"
    )
    currency: str = Field(
        default="EUR", max_length=3, description="ISO 4217 currency code"
    )
    source: str = Field(
        default="", max_length=500, description="Resource source"
    )
    approved: bool = Field(
        default=False, description="Approved flag"
    )

    @field_validator("resource_type", mode="before")
    @classmethod
    def coerce_resource_type(cls, v: Any) -> Any:
        """Accept string values for ResourceType."""
        if isinstance(v, str):
            valid = {t.value for t in ResourceType}
            if v not in valid:
                raise ValueError(
                    f"Unknown resource type '{v}'. Must be one of: {sorted(valid)}"
                )
        return v

class ActionItem(BaseModel):
    """An individual action item within an action plan.

    Attributes:
        item_id: Unique action item identifier.
        plan_id: Parent action plan identifier.
        description: What needs to be done.
        assigned_to: Person or role responsible.
        department: Organisational department.
        start_date: Planned start date.
        due_date: Planned due date.
        completed_date: Actual completion date (if completed).
        status: Current lifecycle status.
        priority: Priority level.
        dependencies: IDs of prerequisite action items.
        notes: Additional notes or context.
    """
    item_id: str = Field(
        default_factory=_new_uuid, description="Unique item ID"
    )
    plan_id: str = Field(
        default="", description="Parent action plan ID"
    )
    description: str = Field(
        default="", max_length=2000, description="Action description"
    )
    assigned_to: str = Field(
        default="", max_length=200, description="Assigned person"
    )
    department: str = Field(
        default="", max_length=200, description="Department"
    )
    start_date: Optional[date] = Field(
        default=None, description="Planned start date"
    )
    due_date: Optional[date] = Field(
        default=None, description="Planned due date"
    )
    completed_date: Optional[date] = Field(
        default=None, description="Actual completion date"
    )
    status: ActionStatus = Field(
        default=ActionStatus.PLANNED, description="Lifecycle status"
    )
    priority: ActionPriority = Field(
        default=ActionPriority.MEDIUM, description="Priority level"
    )
    dependencies: List[str] = Field(
        default_factory=list, description="Prerequisite item IDs"
    )
    notes: str = Field(
        default="", max_length=5000, description="Notes"
    )

    @field_validator("status", mode="before")
    @classmethod
    def coerce_item_status(cls, v: Any) -> Any:
        """Accept string values for ActionStatus."""
        if isinstance(v, str):
            valid = {t.value for t in ActionStatus}
            if v not in valid:
                raise ValueError(
                    f"Unknown status '{v}'. Must be one of: {sorted(valid)}"
                )
        return v

    @field_validator("priority", mode="before")
    @classmethod
    def coerce_item_priority(cls, v: Any) -> Any:
        """Accept string values for ActionPriority."""
        if isinstance(v, str):
            valid = {t.value for t in ActionPriority}
            if v not in valid:
                raise ValueError(
                    f"Unknown priority '{v}'. Must be one of: {sorted(valid)}"
                )
        return v

class ActionPlan(BaseModel):
    """An action plan linked to an energy target per ISO 50001 Clause 6.2.

    Attributes:
        plan_id: Unique plan identifier.
        target_id: Linked energy target identifier.
        plan_name: Human-readable plan name.
        description: Detailed plan description.
        responsible_person: Person accountable for the plan.
        department: Organisational department.
        resources: Required resources.
        estimated_cost: Total estimated implementation cost.
        estimated_savings_kwh: Expected annual energy savings (kWh).
        estimated_savings_cost: Expected annual cost savings.
        simple_payback_years: Simple payback period.
        npv: Net present value.
        irr_pct: Internal rate of return (percentage).
        start_date: Planned start date.
        end_date: Planned end date.
        items: Action items within this plan.
        verification_method: IPMVP verification option.
        status: Current lifecycle status.
    """
    plan_id: str = Field(
        default_factory=_new_uuid, description="Unique plan ID"
    )
    target_id: str = Field(
        default="", description="Linked energy target ID"
    )
    plan_name: str = Field(
        default="", max_length=500, description="Plan name"
    )
    description: str = Field(
        default="", max_length=5000, description="Plan description"
    )
    responsible_person: str = Field(
        default="", max_length=200, description="Accountable person"
    )
    department: str = Field(
        default="", max_length=200, description="Department"
    )
    resources: List[ResourceRequirement] = Field(
        default_factory=list, description="Required resources"
    )
    estimated_cost: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total estimated cost"
    )
    estimated_savings_kwh: Decimal = Field(
        default=Decimal("0"), ge=0, description="Annual savings (kWh)"
    )
    estimated_savings_cost: Decimal = Field(
        default=Decimal("0"), ge=0, description="Annual cost savings"
    )
    simple_payback_years: Decimal = Field(
        default=Decimal("0"), ge=0, description="Simple payback (years)"
    )
    npv: Optional[Decimal] = Field(
        default=None, description="Net present value"
    )
    irr_pct: Optional[Decimal] = Field(
        default=None, description="Internal rate of return (%)"
    )
    start_date: Optional[date] = Field(
        default=None, description="Planned start date"
    )
    end_date: Optional[date] = Field(
        default=None, description="Planned end date"
    )
    items: List[ActionItem] = Field(
        default_factory=list, description="Action items"
    )
    verification_method: VerificationMethod = Field(
        default=VerificationMethod.CALCULATED_SAVINGS,
        description="IPMVP verification option"
    )
    status: ActionStatus = Field(
        default=ActionStatus.PLANNED, description="Lifecycle status"
    )

    @field_validator("verification_method", mode="before")
    @classmethod
    def coerce_plan_verification(cls, v: Any) -> Any:
        """Accept string values for VerificationMethod."""
        if isinstance(v, str):
            valid = {t.value for t in VerificationMethod}
            if v not in valid:
                raise ValueError(
                    f"Unknown verification method '{v}'. Must be one of: {sorted(valid)}"
                )
        return v

    @field_validator("status", mode="before")
    @classmethod
    def coerce_plan_status(cls, v: Any) -> Any:
        """Accept string values for ActionStatus."""
        if isinstance(v, str):
            valid = {t.value for t in ActionStatus}
            if v not in valid:
                raise ValueError(
                    f"Unknown status '{v}'. Must be one of: {sorted(valid)}"
                )
        return v

class ActionPlanPortfolio(BaseModel):
    """Aggregated portfolio of objectives, targets, and action plans.

    Attributes:
        portfolio_id: Unique portfolio identifier.
        enms_id: Parent EnMS identifier.
        objectives: All energy objectives.
        targets: All energy targets.
        plans: All action plans.
        total_investment: Sum of all plan estimated costs.
        total_estimated_savings_kwh: Sum of annual energy savings.
        total_estimated_savings_cost: Sum of annual cost savings.
        portfolio_payback_years: Portfolio-level simple payback.
        overall_progress_pct: Weighted progress across all plans.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 provenance hash.
        calculation_time_ms: Engine processing time in milliseconds.
    """
    portfolio_id: str = Field(
        default_factory=_new_uuid, description="Portfolio ID"
    )
    enms_id: str = Field(
        default="", description="Parent EnMS ID"
    )
    objectives: List[EnergyObjective] = Field(
        default_factory=list, description="Energy objectives"
    )
    targets: List[EnergyTarget] = Field(
        default_factory=list, description="Energy targets"
    )
    plans: List[ActionPlan] = Field(
        default_factory=list, description="Action plans"
    )
    total_investment: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total investment"
    )
    total_estimated_savings_kwh: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total savings (kWh)"
    )
    total_estimated_savings_cost: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total cost savings"
    )
    portfolio_payback_years: Decimal = Field(
        default=Decimal("0"), ge=0, description="Portfolio payback (years)"
    )
    overall_progress_pct: Decimal = Field(
        default=Decimal("0"), ge=0, le=Decimal("100"),
        description="Overall progress percentage"
    )
    calculated_at: datetime = Field(
        default_factory=utcnow, description="Calculation timestamp"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )
    calculation_time_ms: int = Field(
        default=0, ge=0, description="Processing time (ms)"
    )

# ---------------------------------------------------------------------------
# Model Rebuild (required for `from __future__ import annotations`)
# ---------------------------------------------------------------------------
SMARTObjective.model_rebuild()
EnergyObjective.model_rebuild()
EnergyTarget.model_rebuild()
ResourceRequirement.model_rebuild()
ActionItem.model_rebuild()
ActionPlan.model_rebuild()
ActionPlanPortfolio.model_rebuild()

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ActionPlanEngine:
    """ISO 50001 Clause 6.2 action plan engine.

    Creates and manages energy objectives, targets, and action plans.
    Provides SMART validation, financial analysis (payback, NPV, IRR, BCR),
    multi-criteria prioritisation, progress tracking, overdue detection,
    and portfolio aggregation.

    All arithmetic uses ``Decimal`` for deterministic, audit-grade precision.
    Every result carries a SHA-256 provenance hash.

    Usage::

        engine = ActionPlanEngine()
        obj = engine.create_objective(
            enms_id="enms-001",
            text="Reduce electricity consumption by 15% by 2027-12-31",
            obj_type=ObjectiveType.REDUCTION,
            target_value=Decimal("15"),
        )
        smart = engine.validate_smart(obj)
        print(f"SMART score: {smart.smart_score}")

    Regulatory References:
        - ISO 50001:2018 Clause 6.2
        - IPMVP EVO 10000-1:2022
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise ActionPlanEngine.

        Args:
            config: Optional overrides.  Supported keys:
                - discount_rate (Decimal): NPV/IRR discount rate
                - analysis_years (int): financial analysis horizon
                - energy_price_per_kwh (Decimal): energy cost per kWh
                - grid_emission_factor (Decimal): kgCO2e per kWh
        """
        self.config = config or {}
        self._discount_rate = _decimal(
            self.config.get("discount_rate", DEFAULT_DISCOUNT_RATE)
        )
        self._analysis_years = int(
            self.config.get("analysis_years", DEFAULT_ANALYSIS_YEARS)
        )
        self._energy_price = _decimal(
            self.config.get("energy_price_per_kwh", DEFAULT_ENERGY_PRICE_PER_KWH)
        )
        self._grid_ef = _decimal(
            self.config.get("grid_emission_factor", DEFAULT_GRID_EMISSION_FACTOR)
        )
        logger.info(
            "ActionPlanEngine v%s initialised (discount=%.2f, years=%d, "
            "price=%.4f, ef=%.4f)",
            self.engine_version,
            float(self._discount_rate),
            self._analysis_years,
            float(self._energy_price),
            float(self._grid_ef),
        )

    # ------------------------------------------------------------------ #
    # Objective Management                                                 #
    # ------------------------------------------------------------------ #

    def create_objective(
        self,
        enms_id: str,
        text: str,
        obj_type: ObjectiveType,
        target_value: Decimal,
        target_scope: TargetScope = TargetScope.FACILITY_WIDE,
        baseline_value: Decimal = Decimal("0"),
        target_unit: str = "kWh",
        target_date: Optional[date] = None,
        responsible_person: str = "",
    ) -> EnergyObjective:
        """Create a new energy objective per ISO 50001 Clause 6.2.

        Args:
            enms_id: Parent EnMS identifier.
            text: Full text of the objective.
            obj_type: Objective classification.
            target_value: Quantified target value.
            target_scope: Scope boundary.
            baseline_value: Baseline energy value for comparison.
            target_unit: Unit of measurement.
            target_date: Deadline for the objective.
            responsible_person: Accountable person or role.

        Returns:
            EnergyObjective with SMART validation attached.
        """
        t0 = time.perf_counter()
        logger.info(
            "Creating objective: enms=%s, type=%s, target=%s %s",
            enms_id, obj_type.value, str(target_value), target_unit,
        )

        objective = EnergyObjective(
            enms_id=enms_id,
            objective_text=text,
            objective_type=obj_type,
            target_scope=target_scope,
            baseline_value=_decimal(baseline_value),
            target_value=_decimal(target_value),
            target_unit=target_unit,
            target_date=target_date,
            responsible_person=responsible_person,
            status=ActionStatus.PLANNED,
        )

        # Auto-validate SMART criteria
        smart = self.validate_smart(objective)
        objective.smart_validation = smart

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Objective created: id=%s, smart_score=%s (%.1f ms)",
            objective.objective_id, str(smart.smart_score), elapsed,
        )
        return objective

    def validate_smart(self, objective: EnergyObjective) -> SMARTObjective:
        """Validate an objective against SMART criteria.

        Checks:
            Specific:   objective_text is non-empty and contains a numeric target
            Measurable: target_value > 0 and target_unit is set
            Achievable: target_value is within plausible range
            Relevant:   objective_type is set and aligned
            Time-bound: target_date is set and in the future

        Each met criterion contributes 20 points to the smart_score (max 100).

        Args:
            objective: The energy objective to validate.

        Returns:
            SMARTObjective with per-criterion results and composite score.
        """
        t0 = time.perf_counter()
        notes: List[str] = []
        score = Decimal("0")

        # --- Specific ---
        is_specific = self._check_specific(objective)
        if is_specific:
            score += Decimal("20")
            notes.append("SPECIFIC: Objective contains clear target description.")
        else:
            notes.append(
                "SPECIFIC: Objective text is empty or lacks numeric target."
            )

        # --- Measurable ---
        is_measurable = self._check_measurable(objective)
        if is_measurable:
            score += Decimal("20")
            notes.append("MEASURABLE: Quantifiable target and unit defined.")
        else:
            notes.append(
                "MEASURABLE: Missing target value > 0 or target unit."
            )

        # --- Achievable ---
        is_achievable = self._check_achievable(objective)
        if is_achievable:
            score += Decimal("20")
            notes.append("ACHIEVABLE: Target is within plausible range.")
        else:
            notes.append(
                "ACHIEVABLE: Target appears unrealistic (> 100% reduction "
                "or negative improvement)."
            )

        # --- Relevant ---
        is_relevant = self._check_relevant(objective)
        if is_relevant:
            score += Decimal("20")
            notes.append("RELEVANT: Objective type is aligned with energy policy.")
        else:
            notes.append(
                "RELEVANT: Objective type or scope is not clearly defined."
            )

        # --- Time-bound ---
        is_time_bound = self._check_time_bound(objective)
        if is_time_bound:
            score += Decimal("20")
            notes.append("TIME-BOUND: Target date is set and in the future.")
        else:
            notes.append(
                "TIME-BOUND: No target date set or target date is in the past."
            )

        smart = SMARTObjective(
            objective_id=objective.objective_id,
            objective_text=objective.objective_text,
            is_specific=is_specific,
            is_measurable=is_measurable,
            is_achievable=is_achievable,
            is_relevant=is_relevant,
            is_time_bound=is_time_bound,
            smart_score=score,
            validation_notes=notes,
        )

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.debug(
            "SMART validation: id=%s, score=%s (%.1f ms)",
            objective.objective_id, str(score), elapsed,
        )
        return smart

    def _check_specific(self, objective: EnergyObjective) -> bool:
        """Check if objective is specific (clear, unambiguous target).

        Args:
            objective: Energy objective to check.

        Returns:
            True if the objective text is non-empty and contains digits.
        """
        text = objective.objective_text.strip()
        if not text:
            return False
        # Must contain at least one digit to indicate a numeric target
        return any(c.isdigit() for c in text)

    def _check_measurable(self, objective: EnergyObjective) -> bool:
        """Check if objective is measurable (quantifiable metric).

        Args:
            objective: Energy objective to check.

        Returns:
            True if target_value > 0 and target_unit is set.
        """
        return (
            objective.target_value > Decimal("0")
            and len(objective.target_unit.strip()) > 0
        )

    def _check_achievable(self, objective: EnergyObjective) -> bool:
        """Check if objective is achievable (realistic range).

        For reduction objectives, check target is not > 100% of baseline.
        For other types, ensure target_value is positive and reasonable.

        Args:
            objective: Energy objective to check.

        Returns:
            True if the target is within a plausible range.
        """
        if objective.target_value <= Decimal("0"):
            return False

        if objective.objective_type == ObjectiveType.REDUCTION:
            # If target is percentage-like and baseline is set
            if objective.baseline_value > Decimal("0"):
                reduction_pct = _safe_pct(
                    objective.target_value, objective.baseline_value
                )
                # More than 100% reduction is unrealistic
                if reduction_pct > Decimal("100"):
                    return False
            # If target_value represents percentage directly
            if objective.target_unit in ("%", "percent", "pct"):
                if objective.target_value > Decimal("100"):
                    return False

        return True

    def _check_relevant(self, objective: EnergyObjective) -> bool:
        """Check if objective is relevant (aligned with energy policy).

        Args:
            objective: Energy objective to check.

        Returns:
            True if objective_type is valid and scope is set.
        """
        return (
            objective.objective_type is not None
            and objective.target_scope is not None
            and len(objective.objective_text.strip()) > 0
        )

    def _check_time_bound(self, objective: EnergyObjective) -> bool:
        """Check if objective is time-bound (has future deadline).

        Args:
            objective: Energy objective to check.

        Returns:
            True if target_date is set and is in the future.
        """
        if objective.target_date is None:
            return False
        return objective.target_date > _today()

    # ------------------------------------------------------------------ #
    # Target Management                                                    #
    # ------------------------------------------------------------------ #

    def create_target(
        self,
        objective_id: str,
        description: str,
        target_value: Decimal,
        target_unit: str = "kWh",
        baseline_value: Decimal = Decimal("0"),
        interim_milestones: Optional[List[Dict[str, Any]]] = None,
        verification_method: VerificationMethod = VerificationMethod.CALCULATED_SAVINGS,
    ) -> EnergyTarget:
        """Create an energy target linked to an objective.

        Args:
            objective_id: Parent objective identifier.
            description: Target description.
            target_value: Quantified target value.
            target_unit: Unit of measurement.
            baseline_value: Baseline value for comparison.
            interim_milestones: Optional milestone definitions.
            verification_method: IPMVP verification option.

        Returns:
            EnergyTarget with initial achievement at 0%.
        """
        t0 = time.perf_counter()
        logger.info(
            "Creating target: objective=%s, value=%s %s",
            objective_id, str(target_value), target_unit,
        )

        target = EnergyTarget(
            objective_id=objective_id,
            description=description,
            target_value=_decimal(target_value),
            target_unit=target_unit,
            baseline_value=_decimal(baseline_value),
            achievement_pct=Decimal("0"),
            interim_milestones=interim_milestones or [],
            verification_method=verification_method,
        )

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Target created: id=%s (%.1f ms)", target.target_id, elapsed,
        )
        return target

    # ------------------------------------------------------------------ #
    # Action Plan Management                                               #
    # ------------------------------------------------------------------ #

    def create_action_plan(
        self,
        target_id: str,
        plan_data: Dict[str, Any],
    ) -> ActionPlan:
        """Create an action plan from a data dictionary.

        Automatically calculates estimated savings, payback, and creates
        action items from the plan_data['items'] list if provided.

        Args:
            target_id: Linked energy target identifier.
            plan_data: Dictionary containing plan fields.  Expected keys:
                - plan_name (str)
                - description (str)
                - responsible_person (str)
                - department (str)
                - estimated_cost (Decimal)
                - estimated_savings_kwh (Decimal)
                - estimated_savings_cost (Decimal, optional - computed if absent)
                - start_date (date or str)
                - end_date (date or str)
                - verification_method (str)
                - resources (list of dict)
                - items (list of dict)

        Returns:
            ActionPlan with financial metrics computed.
        """
        t0 = time.perf_counter()
        logger.info(
            "Creating action plan for target=%s, name=%s",
            target_id, plan_data.get("plan_name", ""),
        )

        plan_id = _new_uuid()

        # Parse resources
        resources = self._parse_resources(plan_data.get("resources", []))

        # Parse action items
        raw_items = plan_data.get("items", [])
        items = self._parse_action_items(plan_id, raw_items)

        # Compute estimated cost from resources if not provided
        estimated_cost = _decimal(plan_data.get("estimated_cost", Decimal("0")))
        if estimated_cost == Decimal("0") and resources:
            estimated_cost = sum(r.estimated_cost for r in resources)

        # Estimated savings
        savings_kwh = _decimal(plan_data.get("estimated_savings_kwh", Decimal("0")))
        savings_cost = _decimal(plan_data.get("estimated_savings_cost", Decimal("0")))
        if savings_cost == Decimal("0") and savings_kwh > Decimal("0"):
            savings_cost = savings_kwh * self._energy_price

        # Simple payback
        payback = _safe_divide(
            estimated_cost, savings_cost, Decimal("99")
        )

        # Parse dates
        start_date = self._parse_date(plan_data.get("start_date"))
        end_date = self._parse_date(plan_data.get("end_date"))

        # Verification method
        vm_raw = plan_data.get(
            "verification_method", VerificationMethod.CALCULATED_SAVINGS.value
        )
        if isinstance(vm_raw, str):
            try:
                vm = VerificationMethod(vm_raw)
            except ValueError:
                vm = VerificationMethod.CALCULATED_SAVINGS
        else:
            vm = vm_raw

        plan = ActionPlan(
            plan_id=plan_id,
            target_id=target_id,
            plan_name=plan_data.get("plan_name", ""),
            description=plan_data.get("description", ""),
            responsible_person=plan_data.get("responsible_person", ""),
            department=plan_data.get("department", ""),
            resources=resources,
            estimated_cost=_round_val(estimated_cost, 2),
            estimated_savings_kwh=_round_val(savings_kwh, 2),
            estimated_savings_cost=_round_val(savings_cost, 2),
            simple_payback_years=_round_val(payback, 2),
            start_date=start_date,
            end_date=end_date,
            items=items,
            verification_method=vm,
            status=ActionStatus.PLANNED,
        )

        # Compute NPV and IRR
        fin_metrics = self.calculate_financial_metrics(plan)
        plan.npv = _round_val(_decimal(fin_metrics.get("npv", Decimal("0"))), 2)
        plan.irr_pct = _round_val(
            _decimal(fin_metrics.get("irr_pct", Decimal("0"))), 2
        )

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Action plan created: id=%s, cost=%s, savings=%s kWh, "
            "payback=%s yr (%.1f ms)",
            plan.plan_id,
            str(plan.estimated_cost),
            str(plan.estimated_savings_kwh),
            str(plan.simple_payback_years),
            elapsed,
        )
        return plan

    def _parse_resources(
        self,
        raw_resources: List[Dict[str, Any]],
    ) -> List[ResourceRequirement]:
        """Parse resource requirement dictionaries into models.

        Args:
            raw_resources: List of resource data dicts.

        Returns:
            List of ResourceRequirement models.
        """
        resources: List[ResourceRequirement] = []
        for rd in raw_resources:
            try:
                res = ResourceRequirement(
                    resource_type=rd.get("resource_type", ResourceType.CAPITAL.value),
                    description=rd.get("description", ""),
                    estimated_cost=_decimal(rd.get("estimated_cost", Decimal("0"))),
                    currency=rd.get("currency", "EUR"),
                    source=rd.get("source", ""),
                    approved=rd.get("approved", False),
                )
                resources.append(res)
            except (ValueError, TypeError) as exc:
                logger.warning("Skipping invalid resource: %s", str(exc))
        return resources

    def _parse_action_items(
        self,
        plan_id: str,
        raw_items: List[Dict[str, Any]],
    ) -> List[ActionItem]:
        """Parse action item dictionaries into models.

        Args:
            plan_id: Parent plan identifier.
            raw_items: List of action item data dicts.

        Returns:
            List of ActionItem models.
        """
        items: List[ActionItem] = []
        for rd in raw_items:
            try:
                item = ActionItem(
                    plan_id=plan_id,
                    description=rd.get("description", ""),
                    assigned_to=rd.get("assigned_to", ""),
                    department=rd.get("department", ""),
                    start_date=self._parse_date(rd.get("start_date")),
                    due_date=self._parse_date(rd.get("due_date")),
                    completed_date=self._parse_date(rd.get("completed_date")),
                    status=rd.get("status", ActionStatus.PLANNED.value),
                    priority=rd.get("priority", ActionPriority.MEDIUM.value),
                    dependencies=rd.get("dependencies", []),
                    notes=rd.get("notes", ""),
                )
                items.append(item)
            except (ValueError, TypeError) as exc:
                logger.warning("Skipping invalid action item: %s", str(exc))
        return items

    def _parse_date(self, value: Any) -> Optional[date]:
        """Parse a date from string or date object.

        Args:
            value: Date string (YYYY-MM-DD) or date object.

        Returns:
            Parsed date or None.
        """
        if value is None:
            return None
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                return date.fromisoformat(value)
            except (ValueError, TypeError):
                logger.warning("Unable to parse date: %s", value)
                return None
        return None

    # ------------------------------------------------------------------ #
    # Savings Estimation                                                   #
    # ------------------------------------------------------------------ #

    def calculate_savings_estimate(
        self,
        plan: ActionPlan,
        energy_price_per_kwh: Optional[Decimal] = None,
        grid_emission_factor: Optional[Decimal] = None,
    ) -> Dict[str, Any]:
        """Calculate savings estimates for an action plan.

        Args:
            plan: The action plan to analyse.
            energy_price_per_kwh: Override energy price (default from config).
            grid_emission_factor: Override grid emission factor.

        Returns:
            Dict with keys:
                - energy_savings_kwh: Annual electricity savings.
                - cost_savings: Annual monetary savings.
                - co2e_reduction_kg: Annual CO2e reduction in kg.
                - co2e_reduction_tonnes: Annual CO2e reduction in tonnes.
                - provenance_hash: SHA-256 hash.
        """
        t0 = time.perf_counter()
        price = energy_price_per_kwh or self._energy_price
        ef = grid_emission_factor or self._grid_ef

        savings_kwh = plan.estimated_savings_kwh
        cost_savings = savings_kwh * price
        co2e_kg = savings_kwh * ef
        co2e_tonnes = _safe_divide(co2e_kg, Decimal("1000"))

        result = {
            "plan_id": plan.plan_id,
            "energy_savings_kwh": _round_val(savings_kwh, 2),
            "cost_savings": _round_val(cost_savings, 2),
            "co2e_reduction_kg": _round_val(co2e_kg, 2),
            "co2e_reduction_tonnes": _round_val(co2e_tonnes, 4),
            "energy_price_per_kwh": _round_val(price, 4),
            "grid_emission_factor": _round_val(ef, 4),
        }
        result["provenance_hash"] = _compute_hash(result)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Savings estimate: plan=%s, kwh=%s, cost=%s, co2e=%s t "
            "(%.1f ms)",
            plan.plan_id,
            str(result["energy_savings_kwh"]),
            str(result["cost_savings"]),
            str(result["co2e_reduction_tonnes"]),
            elapsed,
        )
        return result

    # ------------------------------------------------------------------ #
    # Financial Metrics                                                    #
    # ------------------------------------------------------------------ #

    def calculate_financial_metrics(
        self,
        plan: ActionPlan,
        discount_rate: Optional[Decimal] = None,
        analysis_years: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Calculate financial metrics for an action plan.

        Computes simple payback, NPV, IRR (bisection), and BCR.

        Args:
            plan: The action plan to analyse.
            discount_rate: Override discount rate (default from config).
            analysis_years: Override analysis period (default from config).

        Returns:
            Dict with keys:
                - simple_payback_years
                - npv
                - irr_pct
                - bcr (benefit-cost ratio)
                - total_discounted_savings
                - provenance_hash
        """
        t0 = time.perf_counter()
        r = discount_rate or self._discount_rate
        n = analysis_years or self._analysis_years

        cost = plan.estimated_cost
        annual_savings = plan.estimated_savings_cost

        # Simple payback
        simple_payback = _safe_divide(cost, annual_savings, Decimal("99"))

        # NPV = -cost + sum( savings / (1+r)^t )
        npv = self._compute_npv(cost, annual_savings, r, n)

        # Total discounted savings (PV of savings stream)
        pv_savings = self._compute_pv_savings(annual_savings, r, n)

        # BCR = PV(savings) / cost
        bcr = _safe_divide(pv_savings, cost, Decimal("0"))

        # IRR via bisection
        irr_pct = self._compute_irr(cost, annual_savings, n)

        result = {
            "plan_id": plan.plan_id,
            "simple_payback_years": _round_val(simple_payback, 2),
            "npv": _round_val(npv, 2),
            "irr_pct": _round_val(irr_pct, 2),
            "bcr": _round_val(bcr, 4),
            "total_discounted_savings": _round_val(pv_savings, 2),
            "discount_rate": _round_val(r, 4),
            "analysis_years": n,
        }
        result["provenance_hash"] = _compute_hash(result)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Financial metrics: plan=%s, payback=%s yr, NPV=%s, "
            "IRR=%s%%, BCR=%s (%.1f ms)",
            plan.plan_id,
            str(result["simple_payback_years"]),
            str(result["npv"]),
            str(result["irr_pct"]),
            str(result["bcr"]),
            elapsed,
        )
        return result

    def _compute_npv(
        self,
        cost: Decimal,
        annual_savings: Decimal,
        discount_rate: Decimal,
        years: int,
    ) -> Decimal:
        """Compute net present value.

        NPV = -cost + sum( annual_savings / (1+r)^t  for t=1..years )

        Args:
            cost: Upfront investment.
            annual_savings: Annual savings (constant).
            discount_rate: Discount rate.
            years: Analysis period.

        Returns:
            NPV value.
        """
        pv = Decimal("0")
        for t in range(1, years + 1):
            discount_factor = (Decimal("1") + discount_rate) ** _decimal(t)
            pv += _safe_divide(annual_savings, discount_factor)
        return pv - cost

    def _compute_pv_savings(
        self,
        annual_savings: Decimal,
        discount_rate: Decimal,
        years: int,
    ) -> Decimal:
        """Compute present value of a constant savings stream.

        PV = sum( annual_savings / (1+r)^t  for t=1..years )

        Args:
            annual_savings: Annual savings (constant).
            discount_rate: Discount rate.
            years: Analysis period.

        Returns:
            Present value of savings.
        """
        pv = Decimal("0")
        for t in range(1, years + 1):
            discount_factor = (Decimal("1") + discount_rate) ** _decimal(t)
            pv += _safe_divide(annual_savings, discount_factor)
        return pv

    def _compute_irr(
        self,
        cost: Decimal,
        annual_savings: Decimal,
        years: int,
    ) -> Decimal:
        """Compute IRR via bisection method.

        Finds rate r where NPV(r) = 0.

        Args:
            cost: Upfront investment.
            annual_savings: Annual savings (constant).
            years: Analysis period.

        Returns:
            IRR as a percentage (e.g. 25.00 for 25%).
        """
        if cost <= Decimal("0") or annual_savings <= Decimal("0"):
            return Decimal("0")

        lo = Decimal("-0.50")
        hi = Decimal("5.00")

        for _ in range(MAX_IRR_ITERATIONS):
            mid = (lo + hi) / Decimal("2")
            npv_mid = -cost
            for t in range(1, years + 1):
                denom = (Decimal("1") + mid) ** _decimal(t)
                if denom != Decimal("0"):
                    npv_mid += _safe_divide(annual_savings, denom)

            if abs(npv_mid) < Decimal("1"):
                return mid * Decimal("100")
            elif npv_mid > Decimal("0"):
                lo = mid
            else:
                hi = mid

            if abs(hi - lo) < IRR_TOLERANCE:
                break

        return ((lo + hi) / Decimal("2")) * Decimal("100")

    # ------------------------------------------------------------------ #
    # Action Prioritisation                                                #
    # ------------------------------------------------------------------ #

    def prioritize_actions(
        self,
        plans: List[ActionPlan],
        criteria_weights: Optional[Dict[str, Decimal]] = None,
    ) -> List[Dict[str, Any]]:
        """Prioritise action plans using multi-criteria scoring.

        Criteria:
            payback: Lower payback = higher score (normalised, inverted).
            savings: Higher savings = higher score (normalised).
            cost_effectiveness: savings / cost ratio (normalised).
            urgency: Based on plan priority of highest-priority item.

        Args:
            plans: List of action plans to rank.
            criteria_weights: Optional weight overrides (keys: payback,
                savings, cost_effectiveness, urgency).

        Returns:
            List of dicts with rank, plan_id, plan_name, weighted_score,
            and individual criterion scores, sorted by weighted_score desc.
        """
        t0 = time.perf_counter()
        if not plans:
            return []

        weights = criteria_weights or DEFAULT_PRIORITY_WEIGHTS.copy()
        logger.info(
            "Prioritising %d action plans with weights=%s",
            len(plans), {k: str(v) for k, v in weights.items()},
        )

        # Extract raw values
        raw_paybacks: Dict[str, Decimal] = {}
        raw_savings: Dict[str, Decimal] = {}
        raw_cer: Dict[str, Decimal] = {}
        raw_urgency: Dict[str, Decimal] = {}

        for p in plans:
            raw_paybacks[p.plan_id] = p.simple_payback_years
            raw_savings[p.plan_id] = p.estimated_savings_cost
            raw_cer[p.plan_id] = _safe_divide(
                p.estimated_savings_cost, p.estimated_cost, Decimal("0")
            )
            # Urgency from highest-priority item
            raw_urgency[p.plan_id] = self._plan_urgency_score(p)

        # Normalise each criterion (min-max)
        norm_paybacks = self._normalise_min_max(raw_paybacks, minimize=True)
        norm_savings = self._normalise_min_max(raw_savings, minimize=False)
        norm_cer = self._normalise_min_max(raw_cer, minimize=False)
        norm_urgency = self._normalise_min_max(raw_urgency, minimize=False)

        # Compute weighted scores
        w_payback = _decimal(weights.get("payback", Decimal("0.30")))
        w_savings = _decimal(weights.get("savings", Decimal("0.25")))
        w_cer = _decimal(weights.get("cost_effectiveness", Decimal("0.25")))
        w_urgency = _decimal(weights.get("urgency", Decimal("0.20")))

        scored: List[Dict[str, Any]] = []
        plan_lookup = {p.plan_id: p for p in plans}

        for pid in raw_paybacks:
            p = plan_lookup[pid]
            ws = (
                norm_paybacks.get(pid, Decimal("0")) * w_payback
                + norm_savings.get(pid, Decimal("0")) * w_savings
                + norm_cer.get(pid, Decimal("0")) * w_cer
                + norm_urgency.get(pid, Decimal("0")) * w_urgency
            )
            scored.append({
                "plan_id": pid,
                "plan_name": p.plan_name,
                "weighted_score": _round_val(ws, 6),
                "payback_score": _round_val(
                    norm_paybacks.get(pid, Decimal("0")), 4
                ),
                "savings_score": _round_val(
                    norm_savings.get(pid, Decimal("0")), 4
                ),
                "cost_effectiveness_score": _round_val(
                    norm_cer.get(pid, Decimal("0")), 4
                ),
                "urgency_score": _round_val(
                    norm_urgency.get(pid, Decimal("0")), 4
                ),
                "estimated_cost": str(p.estimated_cost),
                "estimated_savings_cost": str(p.estimated_savings_cost),
                "simple_payback_years": str(p.simple_payback_years),
            })

        # Sort descending by weighted_score
        scored.sort(key=lambda x: x["weighted_score"], reverse=True)

        # Assign ranks
        for rank, entry in enumerate(scored, start=1):
            entry["rank"] = rank

        # Provenance
        provenance = _compute_hash(scored)
        for entry in scored:
            entry["provenance_hash"] = provenance

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Prioritisation complete: %d plans ranked, hash=%s (%.1f ms)",
            len(scored), provenance[:16], elapsed,
        )
        return scored

    def _plan_urgency_score(self, plan: ActionPlan) -> Decimal:
        """Derive urgency score from highest-priority item in plan.

        Args:
            plan: Action plan to evaluate.

        Returns:
            Urgency score 0.0-1.0.
        """
        if not plan.items:
            return Decimal("0.50")

        max_urgency = Decimal("0")
        for item in plan.items:
            u = _decimal(
                PRIORITY_URGENCY_SCORES.get(item.priority.value, Decimal("0.50"))
            )
            if u > max_urgency:
                max_urgency = u
        return max_urgency

    def _normalise_min_max(
        self,
        values: Dict[str, Decimal],
        minimize: bool = False,
    ) -> Dict[str, Decimal]:
        """Min-max normalise a set of values to [0, 1].

        Args:
            values: Raw values by ID.
            minimize: If True, lower raw value = higher normalised score.

        Returns:
            Normalised values by ID.
        """
        if not values:
            return {}

        vals = list(values.values())
        v_min = min(vals)
        v_max = max(vals)
        spread = v_max - v_min

        result: Dict[str, Decimal] = {}
        for k, raw in values.items():
            if spread == Decimal("0"):
                result[k] = Decimal("0.5")
            elif minimize:
                result[k] = _safe_divide(v_max - raw, spread)
            else:
                result[k] = _safe_divide(raw - v_min, spread)
        return result

    # ------------------------------------------------------------------ #
    # Progress Tracking                                                    #
    # ------------------------------------------------------------------ #

    def track_progress(
        self,
        portfolio: ActionPlanPortfolio,
    ) -> Dict[str, Any]:
        """Track overall progress across a portfolio.

        Computes completion percentage, counts on-track / delayed / completed
        items, and tracks milestone status per plan.

        Args:
            portfolio: The action plan portfolio to track.

        Returns:
            Dict with:
                - overall_completion_pct: Weighted completion across plans.
                - total_items: Total action items.
                - completed_items: Completed action items.
                - in_progress_items: In-progress items.
                - on_track_items: Items on track (not overdue).
                - delayed_items: Overdue or past-due items.
                - on_hold_items: Items on hold.
                - cancelled_items: Cancelled items.
                - plan_progress: Per-plan progress detail.
                - milestone_status: Milestone tracking per target.
                - provenance_hash: SHA-256 hash.
        """
        t0 = time.perf_counter()
        today = _today()

        total_items = 0
        completed_items = 0
        in_progress_items = 0
        on_track_items = 0
        delayed_items = 0
        on_hold_items = 0
        cancelled_items = 0

        plan_progress: List[Dict[str, Any]] = []

        for plan in portfolio.plans:
            plan_total = len(plan.items)
            plan_completed = 0
            plan_delayed = 0
            plan_on_track = 0

            for item in plan.items:
                total_items += 1

                if item.status == ActionStatus.COMPLETED:
                    completed_items += 1
                    plan_completed += 1
                elif item.status == ActionStatus.IN_PROGRESS:
                    in_progress_items += 1
                    if item.due_date is not None and today > item.due_date:
                        delayed_items += 1
                        plan_delayed += 1
                    else:
                        on_track_items += 1
                        plan_on_track += 1
                elif item.status == ActionStatus.OVERDUE:
                    delayed_items += 1
                    plan_delayed += 1
                elif item.status == ActionStatus.ON_HOLD:
                    on_hold_items += 1
                elif item.status == ActionStatus.CANCELLED:
                    cancelled_items += 1
                elif item.status in (ActionStatus.PLANNED, ActionStatus.APPROVED):
                    if item.due_date is not None and today > item.due_date:
                        delayed_items += 1
                        plan_delayed += 1
                    else:
                        on_track_items += 1
                        plan_on_track += 1

            plan_pct = _safe_pct(
                _decimal(plan_completed), _decimal(plan_total)
            ) if plan_total > 0 else Decimal("0")

            plan_progress.append({
                "plan_id": plan.plan_id,
                "plan_name": plan.plan_name,
                "total_items": plan_total,
                "completed": plan_completed,
                "delayed": plan_delayed,
                "on_track": plan_on_track,
                "completion_pct": _round_val(plan_pct, 2),
                "status": plan.status.value,
            })

        # Overall completion
        overall_pct = _safe_pct(
            _decimal(completed_items), _decimal(total_items)
        ) if total_items > 0 else Decimal("0")

        # Milestone tracking
        milestone_status = self._track_milestones(portfolio.targets)

        result = {
            "portfolio_id": portfolio.portfolio_id,
            "overall_completion_pct": _round_val(overall_pct, 2),
            "total_items": total_items,
            "completed_items": completed_items,
            "in_progress_items": in_progress_items,
            "on_track_items": on_track_items,
            "delayed_items": delayed_items,
            "on_hold_items": on_hold_items,
            "cancelled_items": cancelled_items,
            "plan_progress": plan_progress,
            "milestone_status": milestone_status,
            "tracked_at": utcnow().isoformat(),
        }
        result["provenance_hash"] = _compute_hash(result)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Progress tracked: portfolio=%s, overall=%s%%, "
            "completed=%d/%d, delayed=%d (%.1f ms)",
            portfolio.portfolio_id,
            str(result["overall_completion_pct"]),
            completed_items, total_items, delayed_items, elapsed,
        )
        return result

    def _track_milestones(
        self,
        targets: List[EnergyTarget],
    ) -> List[Dict[str, Any]]:
        """Track milestone status for all targets.

        Args:
            targets: List of energy targets with milestones.

        Returns:
            List of milestone status dicts per target.
        """
        today = _today()
        results: List[Dict[str, Any]] = []

        for target in targets:
            if not target.interim_milestones:
                continue

            milestone_data: List[Dict[str, Any]] = []
            for ms in target.interim_milestones:
                ms_date = self._parse_date(ms.get("date"))
                ms_value = _decimal(ms.get("value", Decimal("0")))
                ms_description = ms.get("description", "")

                is_past = ms_date is not None and ms_date <= today
                is_achieved = (
                    target.achievement_pct >= ms_value
                    if ms_value > Decimal("0")
                    else False
                )

                status = "upcoming"
                if is_past and is_achieved:
                    status = "achieved"
                elif is_past and not is_achieved:
                    status = "missed"

                milestone_data.append({
                    "description": ms_description,
                    "date": ms_date.isoformat() if ms_date else None,
                    "target_value": str(ms_value),
                    "status": status,
                })

            results.append({
                "target_id": target.target_id,
                "milestones": milestone_data,
            })

        return results

    # ------------------------------------------------------------------ #
    # Overdue Detection                                                    #
    # ------------------------------------------------------------------ #

    def check_overdue_items(
        self,
        portfolio: ActionPlanPortfolio,
    ) -> List[Dict[str, Any]]:
        """Identify all overdue action items in a portfolio.

        An item is overdue if:
            - status is OVERDUE, or
            - due_date < today AND status is not COMPLETED/CANCELLED

        Args:
            portfolio: The action plan portfolio to check.

        Returns:
            List of dicts with plan_id, item_id, description, due_date,
            days_overdue, assigned_to, priority.
        """
        t0 = time.perf_counter()
        today = _today()
        overdue: List[Dict[str, Any]] = []

        for plan in portfolio.plans:
            for item in plan.items:
                is_overdue = False

                if item.status == ActionStatus.OVERDUE:
                    is_overdue = True
                elif (
                    item.due_date is not None
                    and today > item.due_date
                    and item.status not in (
                        ActionStatus.COMPLETED,
                        ActionStatus.CANCELLED,
                    )
                ):
                    is_overdue = True

                if is_overdue:
                    days_overdue = 0
                    if item.due_date is not None:
                        days_overdue = (today - item.due_date).days

                    overdue.append({
                        "plan_id": plan.plan_id,
                        "plan_name": plan.plan_name,
                        "item_id": item.item_id,
                        "description": item.description,
                        "assigned_to": item.assigned_to,
                        "department": item.department,
                        "due_date": (
                            item.due_date.isoformat() if item.due_date else None
                        ),
                        "days_overdue": days_overdue,
                        "priority": item.priority.value,
                        "status": item.status.value,
                    })

        # Sort by days overdue (most overdue first), then by priority
        priority_order = {
            ActionPriority.CRITICAL.value: 0,
            ActionPriority.HIGH.value: 1,
            ActionPriority.MEDIUM.value: 2,
            ActionPriority.LOW.value: 3,
        }
        overdue.sort(
            key=lambda x: (
                -x["days_overdue"],
                priority_order.get(x["priority"], 9),
            )
        )

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Overdue check: portfolio=%s, %d overdue items found (%.1f ms)",
            portfolio.portfolio_id, len(overdue), elapsed,
        )
        return overdue

    # ------------------------------------------------------------------ #
    # Gantt Data Generation                                                #
    # ------------------------------------------------------------------ #

    def generate_gantt_data(
        self,
        plans: List[ActionPlan],
    ) -> List[Dict[str, Any]]:
        """Generate timeline data suitable for Gantt chart visualisation.

        Produces one entry per action plan with start/end dates, and
        nested entries per action item with dependencies.

        Args:
            plans: List of action plans to visualise.

        Returns:
            List of dicts with:
                - id: plan or item ID
                - name: display name
                - start: ISO date string
                - end: ISO date string
                - progress: 0-100
                - dependencies: list of prerequisite IDs
                - type: 'plan' or 'item'
                - status: lifecycle status
                - children: nested item entries (for plans)
        """
        t0 = time.perf_counter()
        gantt_data: List[Dict[str, Any]] = []

        for plan in plans:
            # Plan-level entry
            plan_start = plan.start_date
            plan_end = plan.end_date

            # Compute progress from items
            total_items = len(plan.items)
            completed = sum(
                1 for i in plan.items
                if i.status == ActionStatus.COMPLETED
            )
            progress = float(_safe_pct(
                _decimal(completed), _decimal(total_items)
            )) if total_items > 0 else 0.0

            # Child item entries
            children: List[Dict[str, Any]] = []
            for item in plan.items:
                item_progress = 100.0 if item.status == ActionStatus.COMPLETED else (
                    50.0 if item.status == ActionStatus.IN_PROGRESS else 0.0
                )

                children.append({
                    "id": item.item_id,
                    "name": item.description[:100],
                    "start": (
                        item.start_date.isoformat() if item.start_date else None
                    ),
                    "end": (
                        item.due_date.isoformat() if item.due_date else None
                    ),
                    "progress": item_progress,
                    "dependencies": item.dependencies,
                    "type": "item",
                    "status": item.status.value,
                    "assigned_to": item.assigned_to,
                    "priority": item.priority.value,
                })

            gantt_data.append({
                "id": plan.plan_id,
                "name": plan.plan_name,
                "start": plan_start.isoformat() if plan_start else None,
                "end": plan_end.isoformat() if plan_end else None,
                "progress": round(progress, 1),
                "dependencies": [],
                "type": "plan",
                "status": plan.status.value,
                "responsible_person": plan.responsible_person,
                "children": children,
            })

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Gantt data generated: %d plans, %d total items (%.1f ms)",
            len(plans),
            sum(len(p.items) for p in plans),
            elapsed,
        )
        return gantt_data

    # ------------------------------------------------------------------ #
    # Portfolio Management                                                 #
    # ------------------------------------------------------------------ #

    def create_portfolio(
        self,
        enms_id: str,
        objectives: List[EnergyObjective],
        targets: List[EnergyTarget],
        plans: List[ActionPlan],
    ) -> ActionPlanPortfolio:
        """Create an aggregated portfolio of objectives, targets, and plans.

        Computes portfolio-level totals for investment, savings, and payback.

        Args:
            enms_id: Parent EnMS identifier.
            objectives: All energy objectives.
            targets: All energy targets.
            plans: All action plans.

        Returns:
            ActionPlanPortfolio with aggregated metrics and provenance hash.
        """
        t0 = time.perf_counter()
        logger.info(
            "Creating portfolio: enms=%s, objectives=%d, targets=%d, plans=%d",
            enms_id, len(objectives), len(targets), len(plans),
        )

        # Aggregate financial totals
        total_investment = sum(
            (p.estimated_cost for p in plans), Decimal("0")
        )
        total_savings_kwh = sum(
            (p.estimated_savings_kwh for p in plans), Decimal("0")
        )
        total_savings_cost = sum(
            (p.estimated_savings_cost for p in plans), Decimal("0")
        )

        # Portfolio payback
        portfolio_payback = _safe_divide(
            total_investment, total_savings_cost, Decimal("99")
        )

        # Overall progress from action items
        total_items = sum(len(p.items) for p in plans)
        completed_items = sum(
            sum(1 for i in p.items if i.status == ActionStatus.COMPLETED)
            for p in plans
        )
        overall_progress = _safe_pct(
            _decimal(completed_items), _decimal(total_items)
        ) if total_items > 0 else Decimal("0")

        elapsed_ms = int((time.perf_counter() - t0) * 1000.0)

        portfolio = ActionPlanPortfolio(
            enms_id=enms_id,
            objectives=objectives,
            targets=targets,
            plans=plans,
            total_investment=_round_val(total_investment, 2),
            total_estimated_savings_kwh=_round_val(total_savings_kwh, 2),
            total_estimated_savings_cost=_round_val(total_savings_cost, 2),
            portfolio_payback_years=_round_val(portfolio_payback, 2),
            overall_progress_pct=_round_val(overall_progress, 2),
            calculation_time_ms=elapsed_ms,
        )
        portfolio.provenance_hash = _compute_hash(portfolio)

        logger.info(
            "Portfolio created: id=%s, investment=%s, savings=%s kWh, "
            "payback=%s yr, progress=%s%%, hash=%s (%d ms)",
            portfolio.portfolio_id,
            str(portfolio.total_investment),
            str(portfolio.total_estimated_savings_kwh),
            str(portfolio.portfolio_payback_years),
            str(portfolio.overall_progress_pct),
            portfolio.provenance_hash[:16],
            elapsed_ms,
        )
        return portfolio

    # ------------------------------------------------------------------ #
    # Plan Completeness Validation                                         #
    # ------------------------------------------------------------------ #

    def validate_plan_completeness(
        self,
        plan: ActionPlan,
    ) -> Dict[str, Any]:
        """Validate an action plan against ISO 50001 Clause 6.2 requirements.

        Checks all required elements per ACTION_PLAN_CHECKLIST, verifies
        resource allocation, and validates timeline feasibility.

        Args:
            plan: The action plan to validate.

        Returns:
            Dict with:
                - plan_id: Plan identifier.
                - is_complete: True if all required elements are present.
                - checklist_results: Per-element pass/fail.
                - missing_elements: List of missing required elements.
                - resource_validation: Resource allocation status.
                - timeline_validation: Timeline feasibility status.
                - warnings: Non-blocking issues.
                - completeness_score: 0-100 score.
                - provenance_hash: SHA-256 hash.
        """
        t0 = time.perf_counter()
        logger.info(
            "Validating plan completeness: plan=%s", plan.plan_id,
        )

        checklist_results: Dict[str, bool] = {}
        missing: List[str] = []
        warnings: List[str] = []

        # --- what_will_be_done ---
        has_description = len(plan.description.strip()) > 0
        checklist_results["what_will_be_done"] = has_description
        if not has_description:
            missing.append("what_will_be_done")

        # --- resources_required ---
        has_resources = len(plan.resources) > 0
        checklist_results["resources_required"] = has_resources
        if not has_resources:
            missing.append("resources_required")

        # --- responsible_person ---
        has_responsible = len(plan.responsible_person.strip()) > 0
        checklist_results["responsible_person"] = has_responsible
        if not has_responsible:
            missing.append("responsible_person")

        # --- completion_timeframe ---
        has_timeframe = plan.end_date is not None
        checklist_results["completion_timeframe"] = has_timeframe
        if not has_timeframe:
            missing.append("completion_timeframe")

        # --- method_of_verification ---
        has_verification = plan.verification_method is not None
        checklist_results["method_of_verification"] = has_verification
        if not has_verification:
            missing.append("method_of_verification")

        # --- method_of_verifying_improvement ---
        # Considered present if savings are quantified
        has_improvement_verification = (
            plan.estimated_savings_kwh > Decimal("0")
            or plan.estimated_savings_cost > Decimal("0")
        )
        checklist_results["method_of_verifying_improvement"] = (
            has_improvement_verification
        )
        if not has_improvement_verification:
            missing.append("method_of_verifying_improvement")

        # --- expected_results ---
        has_expected_results = plan.estimated_savings_kwh > Decimal("0")
        checklist_results["expected_results"] = has_expected_results
        if not has_expected_results:
            missing.append("expected_results")

        # --- integration_with_enms ---
        # Considered present if target_id is linked
        has_integration = len(plan.target_id.strip()) > 0
        checklist_results["integration_with_enms"] = has_integration
        if not has_integration:
            missing.append("integration_with_enms")

        # Resource validation
        resource_validation = self._validate_resources(plan)

        # Timeline validation
        timeline_validation = self._validate_timeline(plan)

        # Add timeline warnings
        if not timeline_validation.get("is_valid", True):
            for issue in timeline_validation.get("issues", []):
                warnings.append(f"Timeline: {issue}")

        # Add resource warnings
        if not resource_validation.get("all_approved", True):
            warnings.append(
                "Resources: Not all resources have been approved."
            )

        # Completeness score
        total_checks = len(ACTION_PLAN_CHECKLIST)
        passed_checks = sum(1 for v in checklist_results.values() if v)
        completeness_score = _safe_pct(
            _decimal(passed_checks), _decimal(total_checks)
        )

        is_complete = len(missing) == 0

        result = {
            "plan_id": plan.plan_id,
            "plan_name": plan.plan_name,
            "is_complete": is_complete,
            "checklist_results": checklist_results,
            "missing_elements": missing,
            "resource_validation": resource_validation,
            "timeline_validation": timeline_validation,
            "warnings": warnings,
            "completeness_score": _round_val(completeness_score, 2),
        }
        result["provenance_hash"] = _compute_hash(result)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Plan validation: plan=%s, complete=%s, score=%s%%, "
            "missing=%d elements (%.1f ms)",
            plan.plan_id,
            str(is_complete),
            str(result["completeness_score"]),
            len(missing),
            elapsed,
        )
        return result

    def _validate_resources(
        self,
        plan: ActionPlan,
    ) -> Dict[str, Any]:
        """Validate resource allocation for a plan.

        Args:
            plan: Action plan to validate.

        Returns:
            Dict with resource validation details.
        """
        total_resource_cost = sum(
            (r.estimated_cost for r in plan.resources), Decimal("0")
        )
        approved_cost = sum(
            (r.estimated_cost for r in plan.resources if r.approved),
            Decimal("0"),
        )
        all_approved = all(r.approved for r in plan.resources) if plan.resources else False

        # Check if resource total matches estimated cost
        cost_match = abs(total_resource_cost - plan.estimated_cost) < Decimal("1")

        return {
            "total_resources": len(plan.resources),
            "total_resource_cost": _round_val(total_resource_cost, 2),
            "approved_cost": _round_val(approved_cost, 2),
            "all_approved": all_approved,
            "cost_matches_estimate": cost_match,
            "resource_types": list(set(
                r.resource_type.value for r in plan.resources
            )),
        }

    def _validate_timeline(
        self,
        plan: ActionPlan,
    ) -> Dict[str, Any]:
        """Validate timeline feasibility for a plan.

        Args:
            plan: Action plan to validate.

        Returns:
            Dict with timeline validation details.
        """
        issues: List[str] = []
        today = _today()

        # Check plan dates
        if plan.start_date is not None and plan.end_date is not None:
            if plan.start_date > plan.end_date:
                issues.append("Plan start date is after end date.")

            duration_days = (plan.end_date - plan.start_date).days
            if duration_days < 1:
                issues.append("Plan duration is less than 1 day.")

        if plan.end_date is not None and plan.end_date < today:
            issues.append("Plan end date is in the past.")

        # Check item dates
        for item in plan.items:
            if item.start_date and item.due_date:
                if item.start_date > item.due_date:
                    issues.append(
                        f"Item '{item.description[:50]}' has start after due date."
                    )

            if item.due_date and plan.end_date:
                if item.due_date > plan.end_date:
                    issues.append(
                        f"Item '{item.description[:50]}' due date exceeds plan end."
                    )

            if item.start_date and plan.start_date:
                if item.start_date < plan.start_date:
                    issues.append(
                        f"Item '{item.description[:50]}' starts before plan start."
                    )

        # Check dependency ordering
        item_lookup: Dict[str, ActionItem] = {
            i.item_id: i for i in plan.items
        }
        for item in plan.items:
            for dep_id in item.dependencies:
                dep = item_lookup.get(dep_id)
                if dep is not None:
                    if (
                        dep.due_date is not None
                        and item.start_date is not None
                        and dep.due_date > item.start_date
                    ):
                        issues.append(
                            f"Item '{item.description[:50]}' starts before "
                            f"dependency '{dep.description[:50]}' is due."
                        )

        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "total_items": len(plan.items),
            "items_with_dates": sum(
                1 for i in plan.items
                if i.start_date is not None and i.due_date is not None
            ),
        }

    # ------------------------------------------------------------------ #
    # Reference Data Accessors                                             #
    # ------------------------------------------------------------------ #

    def get_typical_savings(
        self,
        measure_key: str,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve typical savings data for a common measure type.

        Args:
            measure_key: Key from TYPICAL_SAVINGS_BY_MEASURE.

        Returns:
            Savings data dict or None if not found.
        """
        data = TYPICAL_SAVINGS_BY_MEASURE.get(measure_key)
        if data is None:
            logger.warning("Unknown measure key: %s", measure_key)
            return None
        return dict(data)

    def list_typical_measures(self) -> List[Dict[str, Any]]:
        """List all typical energy efficiency measures with savings ranges.

        Returns:
            List of dicts with measure_key, description, savings range.
        """
        results: List[Dict[str, Any]] = []
        for key, data in TYPICAL_SAVINGS_BY_MEASURE.items():
            results.append({
                "measure_key": key,
                "description": data["description"],
                "typical_savings_pct_low": str(data["typical_savings_pct_low"]),
                "typical_savings_pct_high": str(data["typical_savings_pct_high"]),
                "typical_payback_years": str(data["typical_payback_years"]),
                "category": data["category"],
            })
        return results

    def get_verification_method_info(
        self,
        method: VerificationMethod,
    ) -> Dict[str, str]:
        """Retrieve IPMVP verification method description.

        Args:
            method: Verification method enum value.

        Returns:
            Dict with name, description, best_for, accuracy.
        """
        info = VERIFICATION_METHOD_DESCRIPTIONS.get(method.value)
        if info is None:
            return {
                "name": method.value,
                "description": "No description available.",
                "best_for": "",
                "accuracy": "",
            }
        return dict(info)

    def get_action_plan_checklist(self) -> Dict[str, str]:
        """Retrieve the ISO 50001 Clause 6.2 action plan checklist.

        Returns:
            Dict mapping required element to its description.
        """
        return dict(ACTION_PLAN_CHECKLIST)

    # ------------------------------------------------------------------ #
    # Batch Operations                                                     #
    # ------------------------------------------------------------------ #

    def batch_calculate_financial_metrics(
        self,
        plans: List[ActionPlan],
        discount_rate: Optional[Decimal] = None,
        analysis_years: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Calculate financial metrics for multiple plans.

        Args:
            plans: List of action plans.
            discount_rate: Override discount rate.
            analysis_years: Override analysis period.

        Returns:
            List of financial metrics dicts, one per plan.
        """
        t0 = time.perf_counter()
        results: List[Dict[str, Any]] = []

        for plan in plans:
            metrics = self.calculate_financial_metrics(
                plan, discount_rate, analysis_years
            )
            results.append(metrics)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Batch financial metrics: %d plans processed (%.1f ms)",
            len(plans), elapsed,
        )
        return results

    def batch_validate_plans(
        self,
        plans: List[ActionPlan],
    ) -> List[Dict[str, Any]]:
        """Validate completeness for multiple plans.

        Args:
            plans: List of action plans to validate.

        Returns:
            List of validation result dicts, one per plan.
        """
        t0 = time.perf_counter()
        results: List[Dict[str, Any]] = []

        for plan in plans:
            validation = self.validate_plan_completeness(plan)
            results.append(validation)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Batch plan validation: %d plans processed (%.1f ms)",
            len(plans), elapsed,
        )
        return results

    # ------------------------------------------------------------------ #
    # Summary & Reporting                                                  #
    # ------------------------------------------------------------------ #

    def generate_portfolio_summary(
        self,
        portfolio: ActionPlanPortfolio,
    ) -> Dict[str, Any]:
        """Generate a comprehensive portfolio summary for reporting.

        Combines financial, progress, and compliance data.

        Args:
            portfolio: The portfolio to summarise.

        Returns:
            Dict with financial_summary, progress_summary,
            objective_summary, risk_summary, provenance_hash.
        """
        t0 = time.perf_counter()

        # Financial summary
        financial = {
            "total_investment": str(portfolio.total_investment),
            "total_estimated_savings_kwh": str(
                portfolio.total_estimated_savings_kwh
            ),
            "total_estimated_savings_cost": str(
                portfolio.total_estimated_savings_cost
            ),
            "portfolio_payback_years": str(portfolio.portfolio_payback_years),
            "plan_count": len(portfolio.plans),
        }

        # NPV/IRR per plan
        plan_financials: List[Dict[str, Any]] = []
        for plan in portfolio.plans:
            plan_financials.append({
                "plan_id": plan.plan_id,
                "plan_name": plan.plan_name,
                "estimated_cost": str(plan.estimated_cost),
                "estimated_savings_kwh": str(plan.estimated_savings_kwh),
                "estimated_savings_cost": str(plan.estimated_savings_cost),
                "simple_payback_years": str(plan.simple_payback_years),
                "npv": str(plan.npv) if plan.npv is not None else "N/A",
                "irr_pct": (
                    str(plan.irr_pct) if plan.irr_pct is not None else "N/A"
                ),
                "status": plan.status.value,
            })
        financial["plan_details"] = plan_financials

        # Progress summary
        progress = self.track_progress(portfolio)

        # Objective summary
        objective_summary: List[Dict[str, Any]] = []
        for obj in portfolio.objectives:
            smart_score = Decimal("0")
            if obj.smart_validation is not None:
                smart_score = obj.smart_validation.smart_score

            objective_summary.append({
                "objective_id": obj.objective_id,
                "objective_text": obj.objective_text[:200],
                "type": obj.objective_type.value,
                "scope": obj.target_scope.value,
                "baseline": str(obj.baseline_value),
                "target": str(obj.target_value),
                "unit": obj.target_unit,
                "smart_score": str(smart_score),
                "status": obj.status.value,
            })

        # Risk summary (overdue items)
        overdue = self.check_overdue_items(portfolio)
        risk_summary = {
            "overdue_count": len(overdue),
            "critical_overdue": sum(
                1 for o in overdue
                if o["priority"] == ActionPriority.CRITICAL.value
            ),
            "high_overdue": sum(
                1 for o in overdue
                if o["priority"] == ActionPriority.HIGH.value
            ),
            "max_days_overdue": (
                max(o["days_overdue"] for o in overdue) if overdue else 0
            ),
            "overdue_items": overdue[:10],  # Top 10 most overdue
        }

        result = {
            "portfolio_id": portfolio.portfolio_id,
            "enms_id": portfolio.enms_id,
            "financial_summary": financial,
            "progress_summary": {
                "overall_completion_pct": progress["overall_completion_pct"],
                "total_items": progress["total_items"],
                "completed_items": progress["completed_items"],
                "delayed_items": progress["delayed_items"],
                "on_track_items": progress["on_track_items"],
            },
            "objective_summary": objective_summary,
            "risk_summary": risk_summary,
            "generated_at": utcnow().isoformat(),
        }
        result["provenance_hash"] = _compute_hash(result)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Portfolio summary generated: portfolio=%s, hash=%s (%.1f ms)",
            portfolio.portfolio_id,
            result["provenance_hash"][:16],
            elapsed,
        )
        return result

    # ------------------------------------------------------------------ #
    # Status Transition                                                    #
    # ------------------------------------------------------------------ #

    def update_item_status(
        self,
        item: ActionItem,
        new_status: ActionStatus,
        completed_date: Optional[date] = None,
        notes: str = "",
    ) -> ActionItem:
        """Update the status of an action item with validation.

        Validates allowed status transitions:
            PLANNED     -> APPROVED, CANCELLED
            APPROVED    -> IN_PROGRESS, ON_HOLD, CANCELLED
            IN_PROGRESS -> COMPLETED, ON_HOLD, OVERDUE, CANCELLED
            ON_HOLD     -> IN_PROGRESS, CANCELLED
            OVERDUE     -> IN_PROGRESS, COMPLETED, CANCELLED

        Args:
            item: The action item to update.
            new_status: Target status.
            completed_date: Date of completion (auto-set if COMPLETED).
            notes: Additional notes for the transition.

        Returns:
            Updated ActionItem.

        Raises:
            ValueError: If the status transition is not allowed.
        """
        allowed_transitions: Dict[ActionStatus, List[ActionStatus]] = {
            ActionStatus.PLANNED: [
                ActionStatus.APPROVED, ActionStatus.CANCELLED,
            ],
            ActionStatus.APPROVED: [
                ActionStatus.IN_PROGRESS, ActionStatus.ON_HOLD,
                ActionStatus.CANCELLED,
            ],
            ActionStatus.IN_PROGRESS: [
                ActionStatus.COMPLETED, ActionStatus.ON_HOLD,
                ActionStatus.OVERDUE, ActionStatus.CANCELLED,
            ],
            ActionStatus.ON_HOLD: [
                ActionStatus.IN_PROGRESS, ActionStatus.CANCELLED,
            ],
            ActionStatus.OVERDUE: [
                ActionStatus.IN_PROGRESS, ActionStatus.COMPLETED,
                ActionStatus.CANCELLED,
            ],
            ActionStatus.COMPLETED: [],
            ActionStatus.CANCELLED: [],
        }

        current = item.status
        allowed = allowed_transitions.get(current, [])

        if new_status not in allowed:
            raise ValueError(
                f"Cannot transition from {current.value} to {new_status.value}. "
                f"Allowed transitions: {[s.value for s in allowed]}"
            )

        item.status = new_status

        if new_status == ActionStatus.COMPLETED:
            item.completed_date = completed_date or _today()

        if notes:
            existing = item.notes
            separator = "\n---\n" if existing else ""
            item.notes = f"{existing}{separator}[{_today().isoformat()}] {notes}"

        logger.info(
            "Item status updated: item=%s, %s -> %s",
            item.item_id, current.value, new_status.value,
        )
        return item

    def update_plan_status(
        self,
        plan: ActionPlan,
    ) -> ActionPlan:
        """Auto-update plan status based on item statuses.

        Logic:
            - All items COMPLETED -> plan COMPLETED
            - Any item IN_PROGRESS -> plan IN_PROGRESS
            - Any item OVERDUE -> plan IN_PROGRESS (with warning)
            - All items ON_HOLD/CANCELLED -> plan ON_HOLD
            - Otherwise -> plan remains current status

        Args:
            plan: The action plan to update.

        Returns:
            Updated ActionPlan.
        """
        if not plan.items:
            return plan

        statuses = [item.status for item in plan.items]

        all_completed = all(
            s == ActionStatus.COMPLETED for s in statuses
        )
        any_in_progress = any(
            s == ActionStatus.IN_PROGRESS for s in statuses
        )
        any_overdue = any(
            s == ActionStatus.OVERDUE for s in statuses
        )
        all_stopped = all(
            s in (ActionStatus.ON_HOLD, ActionStatus.CANCELLED)
            for s in statuses
        )

        if all_completed:
            plan.status = ActionStatus.COMPLETED
            logger.info(
                "Plan auto-completed: plan=%s (all items completed)",
                plan.plan_id,
            )
        elif any_overdue or any_in_progress:
            plan.status = ActionStatus.IN_PROGRESS
            if any_overdue:
                logger.warning(
                    "Plan has overdue items: plan=%s", plan.plan_id,
                )
        elif all_stopped:
            plan.status = ActionStatus.ON_HOLD
            logger.info(
                "Plan on hold: plan=%s (all items stopped)", plan.plan_id,
            )

        return plan

    # ------------------------------------------------------------------ #
    # Target Achievement Tracking                                          #
    # ------------------------------------------------------------------ #

    def update_target_achievement(
        self,
        target: EnergyTarget,
        actual_value: Decimal,
    ) -> EnergyTarget:
        """Update target achievement based on actual measured value.

        Calculates achievement_pct as:
            If baseline > target (reduction):
                pct = (baseline - actual) / (baseline - target) * 100
            Else (improvement):
                pct = (actual - baseline) / (target - baseline) * 100

        Args:
            target: The energy target to update.
            actual_value: Actual measured value.

        Returns:
            Updated EnergyTarget with recalculated achievement_pct.
        """
        baseline = target.baseline_value
        target_val = target.target_value

        if baseline > target_val:
            # Reduction target: lower actual = better
            improvement = baseline - actual_value
            target_improvement = baseline - target_val
            achievement = _safe_pct(improvement, target_improvement)
        elif target_val > baseline:
            # Improvement target: higher actual = better
            improvement = actual_value - baseline
            target_improvement = target_val - baseline
            achievement = _safe_pct(improvement, target_improvement)
        else:
            # Target equals baseline
            achievement = Decimal("100") if actual_value == target_val else Decimal("0")

        # Clamp to 0-200% (allow over-achievement)
        achievement = max(Decimal("0"), min(achievement, Decimal("200")))

        target.achievement_pct = _round_val(achievement, 2)

        logger.info(
            "Target achievement updated: target=%s, actual=%s, "
            "achievement=%s%%",
            target.target_id, str(actual_value),
            str(target.achievement_pct),
        )
        return target
