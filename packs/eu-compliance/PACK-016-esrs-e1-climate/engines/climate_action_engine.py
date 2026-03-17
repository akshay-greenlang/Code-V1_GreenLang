# -*- coding: utf-8 -*-
"""
ClimateActionEngine - PACK-016 ESRS E1 Climate Engine 5
=========================================================

Manages climate policies (E1-2) and actions/resources (E1-3) per ESRS E1.

Under the European Sustainability Reporting Standards (ESRS), ESRS E1-2
requires the undertaking to disclose the policies it has in place to manage
its material impacts, risks, and opportunities related to climate change
mitigation and adaptation.  ESRS E1-3 requires disclosure of the key
actions taken and planned, and the resources mobilised, to achieve the
undertaking's climate-related targets.

ESRS E1-2 (Policies) Framework:
    - Para 22: The undertaking shall describe its policies adopted to
      manage its material impacts, risks, and opportunities related to
      climate change mitigation and adaptation.
    - Para 23: The description shall include: (a) the scope of each
      policy; (b) the body responsible for implementing the policy;
      (c) third-party standards or initiatives referenced.
    - Para 24: The policies shall cover own operations and the upstream
      and downstream value chain where relevant.

ESRS E1-3 (Actions and Resources) Framework:
    - Para 26: The undertaking shall disclose its climate change
      mitigation and adaptation actions and the resources allocated
      to their implementation.
    - Para 27: For each action: expected outcome (GHG reduction),
      time horizon, current status, resources required (CapEx/OpEx).
    - Para 28: Actions shall be linked to specific targets (E1-4)
      and policies (E1-2).

Regulatory References:
    - EU Delegated Regulation 2023/2772 (ESRS Set 1)
    - ESRS E1 Climate Change, Para 22-28
    - ESRS E1 Application Requirements AR E1-13 through AR E1-23
    - EU Taxonomy Regulation 2020/852 (substantial contribution criteria)
    - EFRAG Implementation Guidance IG 1

Zero-Hallucination:
    - Resource allocation uses deterministic summation
    - Completeness scoring uses deterministic ratio calculation
    - Taxonomy alignment is a lookup-based classification
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-016 ESRS E1 Climate Change
Status:  Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, Pydantic model, or other).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal:
    """Convert value to Decimal safely.

    Args:
        value: Numeric value (int, float, str, or Decimal).

    Returns:
        Decimal representation.
    """
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _safe_divide(
    numerator: float, denominator: float, default: float = 0.0
) -> float:
    """Safely divide two numbers, returning *default* on zero denominator."""
    if denominator == 0.0:
        return default
    return numerator / denominator


def _round2(value: float) -> float:
    """Round to 2 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    ))


def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(
        Decimal("0.001"), rounding=ROUND_HALF_UP
    ))


def _round4(value: float) -> float:
    """Round to 4 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(
        Decimal("0.0001"), rounding=ROUND_HALF_UP
    ))


def _round_val(value: Decimal, places: int = 3) -> Decimal:
    """Round a Decimal value to the specified number of decimal places.

    Uses ROUND_HALF_UP for regulatory consistency.

    Args:
        value: Decimal value to round.
        places: Number of decimal places (default 3).

    Returns:
        Rounded Decimal value.
    """
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PolicyType(str, Enum):
    """Type of climate-related policy per ESRS E1-2.

    Policies may address different aspects of the undertaking's
    climate change response, from direct emission reduction through
    mitigation to managing physical climate risks through adaptation.
    """
    MITIGATION = "mitigation"
    ADAPTATION = "adaptation"
    ENERGY_EFFICIENCY = "energy_efficiency"
    RENEWABLE_PROCUREMENT = "renewable_procurement"
    SUPPLY_CHAIN = "supply_chain"
    PRODUCT_DESIGN = "product_design"


class PolicyScope(str, Enum):
    """Scope of a climate policy per ESRS E1-2 Para 23(a).

    Indicates the organisational coverage of the policy, from the
    entire group to individual sites or the external supply chain.
    """
    GROUP_WIDE = "group_wide"
    BUSINESS_UNIT = "business_unit"
    SITE_LEVEL = "site_level"
    SUPPLY_CHAIN = "supply_chain"


class ActionCategory(str, Enum):
    """Category of climate action per ESRS E1-3 and AR E1-18.

    Represents the type of decarbonisation lever or adaptation measure
    being pursued.  Aligns with commonly recognised categories of
    climate mitigation and adaptation actions.
    """
    ENERGY_EFFICIENCY = "energy_efficiency"
    FUEL_SWITCHING = "fuel_switching"
    ELECTRIFICATION = "electrification"
    RENEWABLE_ENERGY = "renewable_energy"
    PROCESS_CHANGE = "process_change"
    CARBON_CAPTURE = "carbon_capture"
    NATURE_BASED = "nature_based"
    SUPPLY_CHAIN = "supply_chain"
    PRODUCT_REDESIGN = "product_redesign"
    OFFSET = "offset"


class ActionStatus(str, Enum):
    """Status of a climate action per ESRS E1-3 Para 27.

    Tracks the implementation lifecycle of each climate action from
    planning through completion or cancellation.
    """
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ON_HOLD = "on_hold"


class ResourceType(str, Enum):
    """Type of resource allocated to climate actions per ESRS E1-3 Para 27.

    Resources may be capital expenditure, operating expenditure,
    human resources, or technology investments.
    """
    CAPEX = "capex"
    OPEX = "opex"
    HUMAN = "human"
    TECHNOLOGY = "technology"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# Required ESRS E1-2 data points for policies disclosure.
# Each key is a data point identifier; the value is a human-readable label.
E1_2_DATAPOINTS: Dict[str, str] = {
    "e1_2_dp01": "Whether the undertaking has adopted policies to manage climate change mitigation and adaptation",
    "e1_2_dp02": "Description of the scope of each policy (activities, geographies, value chain stages)",
    "e1_2_dp03": "Body or individual responsible for implementing the policy",
    "e1_2_dp04": "Third-party standards or initiatives that the policy is based on",
    "e1_2_dp05": "Description of how the policy addresses climate change mitigation",
    "e1_2_dp06": "Description of how the policy addresses climate change adaptation",
    "e1_2_dp07": "Whether the policy covers the undertaking's own operations",
    "e1_2_dp08": "Whether the policy covers the upstream value chain",
    "e1_2_dp09": "Whether the policy covers the downstream value chain",
    "e1_2_dp10": "Date of adoption and most recent review of each policy",
    "e1_2_dp11": "How the policy contributes to achieving climate targets (E1-4)",
    "e1_2_dp12": "Whether stakeholders were consulted in developing the policy",
}

# Required ESRS E1-3 data points for actions and resources disclosure.
E1_3_DATAPOINTS: Dict[str, str] = {
    "e1_3_dp01": "Key climate change mitigation actions taken and planned",
    "e1_3_dp02": "Key climate change adaptation actions taken and planned",
    "e1_3_dp03": "Expected GHG emission reduction for each mitigation action (tCO2e)",
    "e1_3_dp04": "Time horizon for achieving the expected outcome of each action",
    "e1_3_dp05": "Current status of each action (planned, in progress, completed)",
    "e1_3_dp06": "Amount of current and future CapEx allocated to each action",
    "e1_3_dp07": "Amount of current and future OpEx allocated to each action",
    "e1_3_dp08": "Total CapEx for all climate change mitigation and adaptation actions",
    "e1_3_dp09": "Total OpEx for all climate change mitigation and adaptation actions",
    "e1_3_dp10": "How the actions contribute to achieving climate targets (E1-4)",
    "e1_3_dp11": "Whether the actions are consistent with the transition plan (E1-1)",
    "e1_3_dp12": "Whether any actions are Taxonomy-aligned (EU Taxonomy Regulation)",
    "e1_3_dp13": "Responsible entity or individual for each action",
    "e1_3_dp14": "Actions grouped by category (energy efficiency, renewable energy, etc.)",
}


# EU Taxonomy substantial contribution mapping for climate actions.
# Maps action categories to EU Taxonomy climate objectives.
ACTION_TAXONOMY_ALIGNMENT: Dict[str, Dict[str, Any]] = {
    "energy_efficiency": {
        "objective": "climate_change_mitigation",
        "description": "Improvements in energy efficiency contribute to climate "
                       "change mitigation per EU Taxonomy Art. 10(1)(a)",
        "substantial_contribution_criteria": "30% energy savings threshold",
        "nace_sectors": ["C", "D", "F", "L"],
    },
    "fuel_switching": {
        "objective": "climate_change_mitigation",
        "description": "Switching from high-carbon to low-carbon fuels contributes "
                       "to climate change mitigation per EU Taxonomy Art. 10(1)(b)",
        "substantial_contribution_criteria": "Life-cycle GHG reduction threshold",
        "nace_sectors": ["D", "H"],
    },
    "electrification": {
        "objective": "climate_change_mitigation",
        "description": "Electrification of processes reduces direct emissions when "
                       "powered by low-carbon electricity per EU Taxonomy Art. 10(1)(c)",
        "substantial_contribution_criteria": "Below emission intensity benchmark",
        "nace_sectors": ["C", "H"],
    },
    "renewable_energy": {
        "objective": "climate_change_mitigation",
        "description": "Renewable energy generation and procurement contributes "
                       "to climate change mitigation per EU Taxonomy Art. 10(1)(a)",
        "substantial_contribution_criteria": "Life-cycle emissions below threshold",
        "nace_sectors": ["D35.11"],
    },
    "process_change": {
        "objective": "climate_change_mitigation",
        "description": "Process changes that reduce direct emissions contribute to "
                       "climate change mitigation per EU Taxonomy Art. 10(1)(d)",
        "substantial_contribution_criteria": "Sector-specific emission benchmarks",
        "nace_sectors": ["C"],
    },
    "carbon_capture": {
        "objective": "climate_change_mitigation",
        "description": "Carbon capture and storage contributes to climate change "
                       "mitigation per EU Taxonomy Art. 10(1)(e)",
        "substantial_contribution_criteria": "Permanent storage of captured CO2",
        "nace_sectors": ["C", "D", "E"],
    },
    "nature_based": {
        "objective": "climate_change_mitigation",
        "description": "Nature-based solutions (afforestation, reforestation) "
                       "contribute to climate change mitigation per Art. 10(1)(f)",
        "substantial_contribution_criteria": "Net carbon sequestration demonstrated",
        "nace_sectors": ["A"],
    },
    "supply_chain": {
        "objective": "climate_change_mitigation",
        "description": "Supply chain decarbonisation through supplier engagement, "
                       "sustainable procurement, and logistics optimisation",
        "substantial_contribution_criteria": "Scope 3 reduction demonstrated",
        "nace_sectors": ["G", "H"],
    },
    "product_redesign": {
        "objective": "climate_change_mitigation",
        "description": "Product redesign for lower carbon footprint through "
                       "materials substitution, circular design, and extended life",
        "substantial_contribution_criteria": "Life-cycle carbon reduction",
        "nace_sectors": ["C"],
    },
    "offset": {
        "objective": "beyond_value_chain_mitigation",
        "description": "Carbon offsets are NOT Taxonomy-eligible and should not "
                       "be used in lieu of emission reductions per SBTi guidance",
        "substantial_contribution_criteria": "N/A - offsets not Taxonomy-eligible",
        "nace_sectors": [],
    },
}


# Policy type descriptions for reporting.
POLICY_TYPE_DESCRIPTIONS: Dict[str, str] = {
    "mitigation": "Climate change mitigation policy addressing GHG emission reductions",
    "adaptation": "Climate change adaptation policy addressing physical risk management",
    "energy_efficiency": "Energy efficiency policy targeting reduced energy consumption per unit output",
    "renewable_procurement": "Renewable energy procurement policy for increasing clean energy share",
    "supply_chain": "Supply chain climate policy for value chain emission reduction",
    "product_design": "Product design policy for lower-carbon products and services",
}


# Action category descriptions for reporting.
ACTION_CATEGORY_DESCRIPTIONS: Dict[str, str] = {
    "energy_efficiency": "Energy efficiency improvements (building insulation, process optimisation, lighting, HVAC)",
    "fuel_switching": "Fuel switching from high-carbon to low-carbon fuels (coal to gas, gas to hydrogen)",
    "electrification": "Electrification of processes and transport (heat pumps, EVs, electric furnaces)",
    "renewable_energy": "Renewable energy generation or procurement (solar, wind, PPAs, RECs)",
    "process_change": "Industrial process changes to reduce direct emissions (catalyst, feedstock, technique)",
    "carbon_capture": "Carbon capture, utilisation, and storage (CCS, CCU, DACCS, BECCS)",
    "nature_based": "Nature-based solutions (afforestation, reforestation, soil carbon, wetlands restoration)",
    "supply_chain": "Supply chain decarbonisation (supplier engagement, sustainable procurement, logistics)",
    "product_redesign": "Product redesign for lower life-cycle emissions (materials, circular design, durability)",
    "offset": "Carbon offset purchases (voluntary market credits, compliance market allowances)",
}


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class ClimatePolicy(BaseModel):
    """A climate-related policy per ESRS E1-2.

    Represents a policy the undertaking has adopted to manage its
    material impacts, risks, and opportunities related to climate
    change mitigation and adaptation.
    """
    policy_id: str = Field(
        default_factory=_new_uuid,
        description="Unique policy identifier",
    )
    name: str = Field(
        ...,
        description="Name of the climate policy",
        min_length=1,
        max_length=500,
    )
    policy_type: PolicyType = Field(
        ...,
        description="Type of climate policy",
    )
    scope: PolicyScope = Field(
        default=PolicyScope.GROUP_WIDE,
        description="Organisational scope of the policy",
    )
    description: str = Field(
        default="",
        description="Detailed description of the policy",
        max_length=5000,
    )
    adoption_date: Optional[date] = Field(
        default=None,
        description="Date the policy was adopted",
    )
    review_cycle: str = Field(
        default="annual",
        description="Review cycle for the policy (annual, biennial, etc.)",
        max_length=100,
    )
    responsible_body: str = Field(
        default="",
        description="Body or individual responsible for implementing the policy",
        max_length=500,
    )
    covers_value_chain: bool = Field(
        default=False,
        description="Whether the policy covers the value chain (upstream/downstream)",
    )
    covers_own_operations: bool = Field(
        default=True,
        description="Whether the policy covers the undertaking's own operations",
    )
    covers_upstream: bool = Field(
        default=False,
        description="Whether the policy covers upstream value chain",
    )
    covers_downstream: bool = Field(
        default=False,
        description="Whether the policy covers downstream value chain",
    )
    third_party_standards: List[str] = Field(
        default_factory=list,
        description="Third-party standards or initiatives referenced",
    )
    linked_target_ids: List[str] = Field(
        default_factory=list,
        description="IDs of E1-4 targets this policy contributes to achieving",
    )
    stakeholders_consulted: bool = Field(
        default=False,
        description="Whether stakeholders were consulted in policy development",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash",
    )

    @field_validator("name")
    @classmethod
    def validate_name_not_empty(cls, v: str) -> str:
        """Validate that policy name is not empty or whitespace."""
        if not v.strip():
            raise ValueError("Policy name cannot be empty")
        return v.strip()


class ClimateAction(BaseModel):
    """A climate action per ESRS E1-3.

    Represents a key action taken or planned, and the resources
    allocated to its implementation, for climate change mitigation
    or adaptation.
    """
    action_id: str = Field(
        default_factory=_new_uuid,
        description="Unique action identifier",
    )
    name: str = Field(
        ...,
        description="Name of the climate action",
        min_length=1,
        max_length=500,
    )
    category: ActionCategory = Field(
        ...,
        description="Category of the climate action",
    )
    description: str = Field(
        default="",
        description="Detailed description of the action",
        max_length=5000,
    )
    expected_reduction_tco2e: Decimal = Field(
        default=Decimal("0.000"),
        description="Expected GHG emission reduction in tCO2e",
        ge=0,
    )
    capex_amount: Decimal = Field(
        default=Decimal("0.00"),
        description="Capital expenditure allocated (in reporting currency)",
        ge=0,
    )
    opex_amount: Decimal = Field(
        default=Decimal("0.00"),
        description="Operating expenditure allocated (in reporting currency)",
        ge=0,
    )
    currency: str = Field(
        default="EUR",
        description="Currency code for financial amounts",
        max_length=3,
    )
    start_date: Optional[date] = Field(
        default=None,
        description="Start date of the action",
    )
    end_date: Optional[date] = Field(
        default=None,
        description="Expected or actual end date of the action",
    )
    status: ActionStatus = Field(
        default=ActionStatus.PLANNED,
        description="Current status of the action",
    )
    responsible_entity: str = Field(
        default="",
        description="Entity or individual responsible for implementing the action",
        max_length=500,
    )
    taxonomy_aligned: bool = Field(
        default=False,
        description="Whether the action is EU Taxonomy-aligned",
    )
    linked_policy_ids: List[str] = Field(
        default_factory=list,
        description="IDs of E1-2 policies this action implements",
    )
    linked_target_ids: List[str] = Field(
        default_factory=list,
        description="IDs of E1-4 targets this action contributes to",
    )
    consistent_with_transition_plan: bool = Field(
        default=False,
        description="Whether the action is consistent with E1-1 transition plan",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash",
    )

    @field_validator("name")
    @classmethod
    def validate_name_not_empty(cls, v: str) -> str:
        """Validate that action name is not empty or whitespace."""
        if not v.strip():
            raise ValueError("Action name cannot be empty")
        return v.strip()


class ResourceAllocation(BaseModel):
    """Resource allocation for a climate action per ESRS E1-3 Para 27.

    Tracks the type, amount, and period of resources mobilised for
    each climate action.
    """
    allocation_id: str = Field(
        default_factory=_new_uuid,
        description="Unique allocation identifier",
    )
    resource_type: ResourceType = Field(
        ...,
        description="Type of resource allocated",
    )
    amount: Decimal = Field(
        ...,
        description="Amount of resource allocated (monetary or FTE)",
        ge=0,
    )
    currency: str = Field(
        default="EUR",
        description="Currency code (for CAPEX/OPEX resources)",
        max_length=3,
    )
    period: str = Field(
        default="",
        description="Period for the allocation (e.g., '2025', '2025-2030')",
        max_length=100,
    )
    action_id: str = Field(
        ...,
        description="ID of the climate action this resource is allocated to",
        min_length=1,
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash",
    )


class ClimateActionResult(BaseModel):
    """Result of climate action plan compilation per ESRS E1-2 and E1-3.

    Contains the complete inventory of policies and actions with
    aggregated resource summaries, status breakdowns, and
    completeness assessments.
    """
    result_id: str = Field(
        default_factory=_new_uuid,
        description="Unique result identifier",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION,
        description="Engine version used for this compilation",
    )
    calculated_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp of compilation (UTC)",
    )
    policies: List[ClimatePolicy] = Field(
        default_factory=list,
        description="List of climate policies (E1-2)",
    )
    actions: List[ClimateAction] = Field(
        default_factory=list,
        description="List of climate actions (E1-3)",
    )
    total_policies: int = Field(
        default=0,
        description="Total number of policies registered",
    )
    total_actions: int = Field(
        default=0,
        description="Total number of actions registered",
    )
    total_capex: Decimal = Field(
        default=Decimal("0.00"),
        description="Total CapEx across all actions",
    )
    total_opex: Decimal = Field(
        default=Decimal("0.00"),
        description="Total OpEx across all actions",
    )
    total_expected_reduction_tco2e: Decimal = Field(
        default=Decimal("0.000"),
        description="Total expected GHG reduction across all actions (tCO2e)",
    )
    actions_by_status: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of actions grouped by status",
    )
    actions_by_category: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of actions grouped by category",
    )
    policies_by_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of policies grouped by type",
    )
    policies_by_scope: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of policies grouped by scope",
    )
    resource_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Resource allocation summary by type",
    )
    taxonomy_aligned_count: int = Field(
        default=0,
        description="Number of actions that are EU Taxonomy-aligned",
    )
    taxonomy_aligned_capex: Decimal = Field(
        default=Decimal("0.00"),
        description="Total CapEx of Taxonomy-aligned actions",
    )
    taxonomy_aligned_pct: float = Field(
        default=0.0,
        description="Percentage of actions that are Taxonomy-aligned",
    )
    completeness_score_e1_2: float = Field(
        default=0.0,
        description="Completeness score for E1-2 data points (0-100)",
    )
    completeness_score_e1_3: float = Field(
        default=0.0,
        description="Completeness score for E1-3 data points (0-100)",
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Processing time in milliseconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of the entire result",
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ClimateActionEngine:
    """Climate action and policy engine per ESRS E1-2 and E1-3.

    Provides deterministic, zero-hallucination compilation of:
    - Climate policy registration and tracking (E1-2)
    - Climate action registration and tracking (E1-3)
    - Resource allocation calculation (CapEx, OpEx, human, technology)
    - EU Taxonomy alignment assessment for climate actions
    - Completeness validation against E1-2 and E1-3 data points
    - Data point extraction for XBRL tagging

    All calculations are bit-perfect reproducible.  No LLM is used
    in any calculation path.

    Usage::

        engine = ClimateActionEngine()
        policy = ClimatePolicy(
            name="Group Climate Mitigation Policy",
            policy_type=PolicyType.MITIGATION,
            scope=PolicyScope.GROUP_WIDE,
        )
        registered_policy = engine.register_policy(policy)

        action = ClimateAction(
            name="Solar PV installation at HQ",
            category=ActionCategory.RENEWABLE_ENERGY,
            expected_reduction_tco2e=Decimal("500.0"),
            capex_amount=Decimal("1200000.00"),
            status=ActionStatus.IN_PROGRESS,
        )
        registered_action = engine.register_action(action)

        result = engine.build_action_plan(
            policies=[registered_policy],
            actions=[registered_action],
            resources=[],
        )
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self) -> None:
        """Initialise ClimateActionEngine."""
        self._policies: List[ClimatePolicy] = []
        self._actions: List[ClimateAction] = []
        self._resources: List[ResourceAllocation] = []
        logger.info(
            "ClimateActionEngine v%s initialised", self.engine_version
        )

    # ------------------------------------------------------------------ #
    # Policy Registration                                                  #
    # ------------------------------------------------------------------ #

    def register_policy(self, policy: ClimatePolicy) -> ClimatePolicy:
        """Register a climate policy per ESRS E1-2.

        Assigns a provenance hash and adds the policy to the internal
        registry.  If the policy already has an ID, it is preserved;
        otherwise a new UUID is generated.

        Args:
            policy: ClimatePolicy to register.

        Returns:
            Registered ClimatePolicy with provenance hash.

        Raises:
            ValueError: If policy name is empty.
        """
        t0 = time.perf_counter()

        if not policy.name.strip():
            raise ValueError("Policy name cannot be empty")

        # Ensure unique ID
        if not policy.policy_id:
            policy.policy_id = _new_uuid()

        # Compute provenance hash
        policy.provenance_hash = _compute_hash(policy)

        # Add to internal registry
        self._policies.append(policy)

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)
        logger.info(
            "Registered policy '%s' (id=%s) in %.3f ms",
            policy.name,
            policy.policy_id,
            elapsed_ms,
        )
        return policy

    # ------------------------------------------------------------------ #
    # Action Registration                                                  #
    # ------------------------------------------------------------------ #

    def register_action(self, action: ClimateAction) -> ClimateAction:
        """Register a climate action per ESRS E1-3.

        Assigns a provenance hash and adds the action to the internal
        registry.

        Args:
            action: ClimateAction to register.

        Returns:
            Registered ClimateAction with provenance hash.

        Raises:
            ValueError: If action name is empty.
        """
        t0 = time.perf_counter()

        if not action.name.strip():
            raise ValueError("Action name cannot be empty")

        if not action.action_id:
            action.action_id = _new_uuid()

        action.provenance_hash = _compute_hash(action)

        self._actions.append(action)

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)
        logger.info(
            "Registered action '%s' (id=%s, category=%s) in %.3f ms",
            action.name,
            action.action_id,
            action.category.value,
            elapsed_ms,
        )
        return action

    # ------------------------------------------------------------------ #
    # Resource Registration                                                #
    # ------------------------------------------------------------------ #

    def register_resource(
        self, resource: ResourceAllocation
    ) -> ResourceAllocation:
        """Register a resource allocation for a climate action.

        Args:
            resource: ResourceAllocation to register.

        Returns:
            Registered ResourceAllocation with provenance hash.
        """
        if not resource.allocation_id:
            resource.allocation_id = _new_uuid()

        resource.provenance_hash = _compute_hash(resource)
        self._resources.append(resource)

        logger.info(
            "Registered resource allocation: type=%s, amount=%s, action_id=%s",
            resource.resource_type.value,
            resource.amount,
            resource.action_id,
        )
        return resource

    # ------------------------------------------------------------------ #
    # Action Plan Builder                                                  #
    # ------------------------------------------------------------------ #

    def build_action_plan(
        self,
        policies: Optional[List[ClimatePolicy]] = None,
        actions: Optional[List[ClimateAction]] = None,
        resources: Optional[List[ResourceAllocation]] = None,
    ) -> ClimateActionResult:
        """Build the complete climate action plan per E1-2 and E1-3.

        Aggregates all registered policies, actions, and resources into
        a single result with summary statistics, completeness scoring,
        and provenance tracking.

        Args:
            policies: List of policies (uses internal registry if None).
            actions: List of actions (uses internal registry if None).
            resources: List of resources (uses internal registry if None).

        Returns:
            ClimateActionResult with complete aggregation.
        """
        t0 = time.perf_counter()

        if policies is None:
            policies = list(self._policies)
        if actions is None:
            actions = list(self._actions)
        if resources is None:
            resources = list(self._resources)

        # Aggregate actions
        total_capex = Decimal("0.00")
        total_opex = Decimal("0.00")
        total_reduction = Decimal("0.000")
        actions_by_status: Dict[str, int] = {}
        actions_by_category: Dict[str, int] = {}
        taxonomy_aligned_count = 0
        taxonomy_aligned_capex = Decimal("0.00")

        for action in actions:
            total_capex += action.capex_amount
            total_opex += action.opex_amount
            total_reduction += action.expected_reduction_tco2e

            status_key = action.status.value
            actions_by_status[status_key] = (
                actions_by_status.get(status_key, 0) + 1
            )

            cat_key = action.category.value
            actions_by_category[cat_key] = (
                actions_by_category.get(cat_key, 0) + 1
            )

            if action.taxonomy_aligned:
                taxonomy_aligned_count += 1
                taxonomy_aligned_capex += action.capex_amount

        # Aggregate policies
        policies_by_type: Dict[str, int] = {}
        policies_by_scope: Dict[str, int] = {}

        for policy in policies:
            type_key = policy.policy_type.value
            policies_by_type[type_key] = (
                policies_by_type.get(type_key, 0) + 1
            )

            scope_key = policy.scope.value
            policies_by_scope[scope_key] = (
                policies_by_scope.get(scope_key, 0) + 1
            )

        # Resource summary
        resource_summary = self.calculate_resource_allocation(
            actions, resources
        )

        # Taxonomy alignment percentage
        taxonomy_pct = _round2(
            _safe_divide(
                float(taxonomy_aligned_count),
                float(len(actions)),
                0.0,
            ) * 100.0
        )

        # Completeness scoring
        e1_2_completeness = self._calculate_e1_2_completeness(policies)
        e1_3_completeness = self._calculate_e1_3_completeness(
            actions, resources
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = ClimateActionResult(
            policies=policies,
            actions=actions,
            total_policies=len(policies),
            total_actions=len(actions),
            total_capex=_round_val(total_capex, 2),
            total_opex=_round_val(total_opex, 2),
            total_expected_reduction_tco2e=_round_val(total_reduction, 3),
            actions_by_status=actions_by_status,
            actions_by_category=actions_by_category,
            policies_by_type=policies_by_type,
            policies_by_scope=policies_by_scope,
            resource_summary=resource_summary,
            taxonomy_aligned_count=taxonomy_aligned_count,
            taxonomy_aligned_capex=_round_val(taxonomy_aligned_capex, 2),
            taxonomy_aligned_pct=taxonomy_pct,
            completeness_score_e1_2=e1_2_completeness,
            completeness_score_e1_3=e1_3_completeness,
            processing_time_ms=elapsed_ms,
        )

        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Built action plan: %d policies, %d actions, total CapEx=%s, "
            "total OpEx=%s, total reduction=%s tCO2e in %.3f ms",
            len(policies),
            len(actions),
            total_capex,
            total_opex,
            total_reduction,
            elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------ #
    # Resource Allocation Calculation                                      #
    # ------------------------------------------------------------------ #

    def calculate_resource_allocation(
        self,
        actions: List[ClimateAction],
        resources: Optional[List[ResourceAllocation]] = None,
    ) -> Dict[str, Any]:
        """Calculate resource allocation summary across all actions.

        Aggregates CapEx and OpEx from actions and any additional
        resource allocations, producing a summary by resource type.

        Args:
            actions: List of climate actions with embedded CapEx/OpEx.
            resources: Optional additional resource allocations.

        Returns:
            Dict with resource summary by type.
        """
        summary: Dict[str, Any] = {
            "capex_from_actions": Decimal("0.00"),
            "opex_from_actions": Decimal("0.00"),
            "capex_from_allocations": Decimal("0.00"),
            "opex_from_allocations": Decimal("0.00"),
            "human_resources_allocated": Decimal("0.00"),
            "technology_investments": Decimal("0.00"),
            "total_capex": Decimal("0.00"),
            "total_opex": Decimal("0.00"),
            "total_resources": Decimal("0.00"),
            "by_action": {},
        }

        # Sum from actions
        for action in actions:
            summary["capex_from_actions"] += action.capex_amount
            summary["opex_from_actions"] += action.opex_amount

            summary["by_action"][action.action_id] = {
                "name": action.name,
                "capex": str(action.capex_amount),
                "opex": str(action.opex_amount),
                "additional_resources": [],
            }

        # Sum from resource allocations
        if resources:
            for res in resources:
                if res.resource_type == ResourceType.CAPEX:
                    summary["capex_from_allocations"] += res.amount
                elif res.resource_type == ResourceType.OPEX:
                    summary["opex_from_allocations"] += res.amount
                elif res.resource_type == ResourceType.HUMAN:
                    summary["human_resources_allocated"] += res.amount
                elif res.resource_type == ResourceType.TECHNOLOGY:
                    summary["technology_investments"] += res.amount

                if res.action_id in summary["by_action"]:
                    summary["by_action"][res.action_id][
                        "additional_resources"
                    ].append({
                        "type": res.resource_type.value,
                        "amount": str(res.amount),
                        "period": res.period,
                    })

        # Totals
        summary["total_capex"] = (
            summary["capex_from_actions"]
            + summary["capex_from_allocations"]
        )
        summary["total_opex"] = (
            summary["opex_from_actions"]
            + summary["opex_from_allocations"]
        )
        summary["total_resources"] = (
            summary["total_capex"]
            + summary["total_opex"]
            + summary["human_resources_allocated"]
            + summary["technology_investments"]
        )

        # Convert Decimal to str for serialisation
        for key in [
            "capex_from_actions",
            "opex_from_actions",
            "capex_from_allocations",
            "opex_from_allocations",
            "human_resources_allocated",
            "technology_investments",
            "total_capex",
            "total_opex",
            "total_resources",
        ]:
            summary[key] = str(summary[key])

        return summary

    # ------------------------------------------------------------------ #
    # EU Taxonomy Alignment Assessment                                     #
    # ------------------------------------------------------------------ #

    def assess_taxonomy_alignment(
        self, actions: List[ClimateAction]
    ) -> Dict[str, Any]:
        """Assess EU Taxonomy alignment of climate actions.

        Evaluates each action's category against the EU Taxonomy
        substantial contribution criteria for climate change mitigation
        and adaptation.

        Args:
            actions: List of climate actions to assess.

        Returns:
            Dict with alignment assessment per action and summary.
        """
        t0 = time.perf_counter()

        assessment: Dict[str, Any] = {
            "total_actions": len(actions),
            "taxonomy_eligible_count": 0,
            "taxonomy_aligned_count": 0,
            "non_eligible_count": 0,
            "total_capex_eligible": Decimal("0.00"),
            "total_capex_aligned": Decimal("0.00"),
            "eligible_pct": 0.0,
            "aligned_pct": 0.0,
            "actions": [],
        }

        for action in actions:
            category_key = action.category.value
            taxonomy_info = ACTION_TAXONOMY_ALIGNMENT.get(category_key, {})

            is_eligible = bool(
                taxonomy_info
                and taxonomy_info.get("objective") != "beyond_value_chain_mitigation"
            )

            action_assessment = {
                "action_id": action.action_id,
                "name": action.name,
                "category": category_key,
                "taxonomy_eligible": is_eligible,
                "taxonomy_aligned": action.taxonomy_aligned and is_eligible,
                "objective": taxonomy_info.get("objective", "N/A"),
                "substantial_contribution_criteria": taxonomy_info.get(
                    "substantial_contribution_criteria", "N/A"
                ),
                "capex": str(action.capex_amount),
            }

            if is_eligible:
                assessment["taxonomy_eligible_count"] += 1
                assessment["total_capex_eligible"] += action.capex_amount

                if action.taxonomy_aligned:
                    assessment["taxonomy_aligned_count"] += 1
                    assessment["total_capex_aligned"] += action.capex_amount
            else:
                assessment["non_eligible_count"] += 1

            assessment["actions"].append(action_assessment)

        # Percentages
        assessment["eligible_pct"] = _round2(
            _safe_divide(
                float(assessment["taxonomy_eligible_count"]),
                float(len(actions)),
                0.0,
            ) * 100.0
        )
        assessment["aligned_pct"] = _round2(
            _safe_divide(
                float(assessment["taxonomy_aligned_count"]),
                float(len(actions)),
                0.0,
            ) * 100.0
        )

        # Convert Decimal to str for serialisation
        assessment["total_capex_eligible"] = str(
            assessment["total_capex_eligible"]
        )
        assessment["total_capex_aligned"] = str(
            assessment["total_capex_aligned"]
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)
        logger.info(
            "Taxonomy alignment assessment: %d eligible, %d aligned "
            "of %d actions in %.3f ms",
            assessment["taxonomy_eligible_count"],
            assessment["taxonomy_aligned_count"],
            len(actions),
            elapsed_ms,
        )
        return assessment

    # ------------------------------------------------------------------ #
    # E1-2 Completeness Validation                                         #
    # ------------------------------------------------------------------ #

    def validate_completeness_e1_2(
        self, result: ClimateActionResult
    ) -> Dict[str, Any]:
        """Validate completeness of E1-2 data points.

        Checks each required E1-2 data point against the result to
        determine coverage.

        Args:
            result: ClimateActionResult to validate.

        Returns:
            Dict with data point coverage status and completeness score.
        """
        datapoints_status: Dict[str, Dict[str, Any]] = {}
        covered = 0

        policies = result.policies
        has_policies = len(policies) > 0

        checks = {
            "e1_2_dp01": has_policies,
            "e1_2_dp02": any(p.scope is not None for p in policies),
            "e1_2_dp03": any(
                bool(p.responsible_body) for p in policies
            ),
            "e1_2_dp04": any(
                len(p.third_party_standards) > 0 for p in policies
            ),
            "e1_2_dp05": any(
                p.policy_type == PolicyType.MITIGATION for p in policies
            ),
            "e1_2_dp06": any(
                p.policy_type == PolicyType.ADAPTATION for p in policies
            ),
            "e1_2_dp07": any(p.covers_own_operations for p in policies),
            "e1_2_dp08": any(p.covers_upstream for p in policies),
            "e1_2_dp09": any(p.covers_downstream for p in policies),
            "e1_2_dp10": any(
                p.adoption_date is not None for p in policies
            ),
            "e1_2_dp11": any(
                len(p.linked_target_ids) > 0 for p in policies
            ),
            "e1_2_dp12": any(p.stakeholders_consulted for p in policies),
        }

        for dp_id, dp_label in E1_2_DATAPOINTS.items():
            is_covered = checks.get(dp_id, False)
            if is_covered:
                covered += 1
            datapoints_status[dp_id] = {
                "label": dp_label,
                "covered": is_covered,
                "status": "COMPLETE" if is_covered else "MISSING",
            }

        total = len(E1_2_DATAPOINTS)
        score = _round2(_safe_divide(float(covered), float(total), 0.0) * 100.0)

        return {
            "disclosure_requirement": "E1-2",
            "title": "Policies related to climate change mitigation and adaptation",
            "total_datapoints": total,
            "covered_datapoints": covered,
            "missing_datapoints": total - covered,
            "completeness_score": score,
            "datapoints": datapoints_status,
            "provenance_hash": _compute_hash(datapoints_status),
        }

    # ------------------------------------------------------------------ #
    # E1-3 Completeness Validation                                         #
    # ------------------------------------------------------------------ #

    def validate_completeness_e1_3(
        self, result: ClimateActionResult
    ) -> Dict[str, Any]:
        """Validate completeness of E1-3 data points.

        Checks each required E1-3 data point against the result to
        determine coverage.

        Args:
            result: ClimateActionResult to validate.

        Returns:
            Dict with data point coverage status and completeness score.
        """
        datapoints_status: Dict[str, Dict[str, Any]] = {}
        covered = 0

        actions = result.actions
        has_actions = len(actions) > 0

        has_mitigation = any(
            a.category != ActionCategory.OFFSET for a in actions
        )
        has_adaptation = any(
            a.category == ActionCategory.NATURE_BASED for a in actions
        )
        has_reduction = any(
            a.expected_reduction_tco2e > 0 for a in actions
        )
        has_dates = any(
            a.end_date is not None for a in actions
        )
        has_status = any(
            a.status is not None for a in actions
        )
        has_capex = any(a.capex_amount > 0 for a in actions)
        has_opex = any(a.opex_amount > 0 for a in actions)
        has_targets = any(
            len(a.linked_target_ids) > 0 for a in actions
        )
        has_transition = any(
            a.consistent_with_transition_plan for a in actions
        )
        has_taxonomy = any(a.taxonomy_aligned for a in actions)
        has_responsible = any(
            bool(a.responsible_entity) for a in actions
        )

        checks = {
            "e1_3_dp01": has_mitigation and has_actions,
            "e1_3_dp02": has_adaptation,
            "e1_3_dp03": has_reduction,
            "e1_3_dp04": has_dates,
            "e1_3_dp05": has_status,
            "e1_3_dp06": has_capex,
            "e1_3_dp07": has_opex,
            "e1_3_dp08": result.total_capex > 0,
            "e1_3_dp09": result.total_opex > 0,
            "e1_3_dp10": has_targets,
            "e1_3_dp11": has_transition,
            "e1_3_dp12": has_taxonomy,
            "e1_3_dp13": has_responsible,
            "e1_3_dp14": len(result.actions_by_category) > 0,
        }

        for dp_id, dp_label in E1_3_DATAPOINTS.items():
            is_covered = checks.get(dp_id, False)
            if is_covered:
                covered += 1
            datapoints_status[dp_id] = {
                "label": dp_label,
                "covered": is_covered,
                "status": "COMPLETE" if is_covered else "MISSING",
            }

        total = len(E1_3_DATAPOINTS)
        score = _round2(_safe_divide(float(covered), float(total), 0.0) * 100.0)

        return {
            "disclosure_requirement": "E1-3",
            "title": "Actions and resources related to climate change",
            "total_datapoints": total,
            "covered_datapoints": covered,
            "missing_datapoints": total - covered,
            "completeness_score": score,
            "datapoints": datapoints_status,
            "provenance_hash": _compute_hash(datapoints_status),
        }

    # ------------------------------------------------------------------ #
    # E1-2 Data Point Extraction                                           #
    # ------------------------------------------------------------------ #

    def get_e1_2_datapoints(
        self, result: ClimateActionResult
    ) -> Dict[str, Any]:
        """Extract structured E1-2 data points for XBRL tagging.

        Returns a dict of all E1-2 data points with their values,
        suitable for XBRL tagging and digital submission.

        Args:
            result: ClimateActionResult to extract from.

        Returns:
            Dict mapping data point IDs to values.
        """
        policies = result.policies
        has_policies = len(policies) > 0

        datapoints: Dict[str, Any] = {
            "e1_2_dp01": {
                "value": has_policies,
                "label": E1_2_DATAPOINTS["e1_2_dp01"],
                "xbrl_element": "esrs:ClimatePoliciesAdopted",
            },
            "e1_2_dp02": {
                "value": [
                    {
                        "policy_name": p.name,
                        "scope": p.scope.value,
                        "description": p.description,
                    }
                    for p in policies
                ],
                "label": E1_2_DATAPOINTS["e1_2_dp02"],
                "xbrl_element": "esrs:ClimatePolicyScopeDescription",
            },
            "e1_2_dp03": {
                "value": [
                    {
                        "policy_name": p.name,
                        "responsible_body": p.responsible_body,
                    }
                    for p in policies
                    if p.responsible_body
                ],
                "label": E1_2_DATAPOINTS["e1_2_dp03"],
                "xbrl_element": "esrs:ClimatePolicyResponsibleBody",
            },
            "e1_2_dp04": {
                "value": [
                    {
                        "policy_name": p.name,
                        "standards": p.third_party_standards,
                    }
                    for p in policies
                    if p.third_party_standards
                ],
                "label": E1_2_DATAPOINTS["e1_2_dp04"],
                "xbrl_element": "esrs:ClimatePolicyThirdPartyStandards",
            },
            "e1_2_dp05": {
                "value": [
                    p.name for p in policies
                    if p.policy_type == PolicyType.MITIGATION
                ],
                "label": E1_2_DATAPOINTS["e1_2_dp05"],
                "xbrl_element": "esrs:ClimateMitigationPolicyDescription",
            },
            "e1_2_dp06": {
                "value": [
                    p.name for p in policies
                    if p.policy_type == PolicyType.ADAPTATION
                ],
                "label": E1_2_DATAPOINTS["e1_2_dp06"],
                "xbrl_element": "esrs:ClimateAdaptationPolicyDescription",
            },
            "e1_2_dp07": {
                "value": any(p.covers_own_operations for p in policies),
                "label": E1_2_DATAPOINTS["e1_2_dp07"],
                "xbrl_element": "esrs:ClimatePolicyCoversOwnOperations",
            },
            "e1_2_dp08": {
                "value": any(p.covers_upstream for p in policies),
                "label": E1_2_DATAPOINTS["e1_2_dp08"],
                "xbrl_element": "esrs:ClimatePolicyCoversUpstream",
            },
            "e1_2_dp09": {
                "value": any(p.covers_downstream for p in policies),
                "label": E1_2_DATAPOINTS["e1_2_dp09"],
                "xbrl_element": "esrs:ClimatePolicyCoversDownstream",
            },
            "e1_2_dp10": {
                "value": [
                    {
                        "policy_name": p.name,
                        "adoption_date": str(p.adoption_date)
                        if p.adoption_date
                        else None,
                        "review_cycle": p.review_cycle,
                    }
                    for p in policies
                ],
                "label": E1_2_DATAPOINTS["e1_2_dp10"],
                "xbrl_element": "esrs:ClimatePolicyAdoptionDate",
            },
            "e1_2_dp11": {
                "value": [
                    {
                        "policy_name": p.name,
                        "linked_targets": p.linked_target_ids,
                    }
                    for p in policies
                    if p.linked_target_ids
                ],
                "label": E1_2_DATAPOINTS["e1_2_dp11"],
                "xbrl_element": "esrs:ClimatePolicyTargetContribution",
            },
            "e1_2_dp12": {
                "value": any(p.stakeholders_consulted for p in policies),
                "label": E1_2_DATAPOINTS["e1_2_dp12"],
                "xbrl_element": "esrs:ClimatePolicyStakeholderConsultation",
            },
        }

        datapoints["provenance_hash"] = _compute_hash(datapoints)
        return datapoints

    # ------------------------------------------------------------------ #
    # E1-3 Data Point Extraction                                           #
    # ------------------------------------------------------------------ #

    def get_e1_3_datapoints(
        self, result: ClimateActionResult
    ) -> Dict[str, Any]:
        """Extract structured E1-3 data points for XBRL tagging.

        Returns a dict of all E1-3 data points with their values,
        suitable for XBRL tagging and digital submission.

        Args:
            result: ClimateActionResult to extract from.

        Returns:
            Dict mapping data point IDs to values.
        """
        actions = result.actions

        mitigation_actions = [
            a for a in actions if a.category != ActionCategory.OFFSET
        ]
        adaptation_actions = [
            a for a in actions
            if a.category == ActionCategory.NATURE_BASED
        ]

        datapoints: Dict[str, Any] = {
            "e1_3_dp01": {
                "value": [
                    {
                        "name": a.name,
                        "category": a.category.value,
                        "description": a.description,
                        "status": a.status.value,
                    }
                    for a in mitigation_actions
                ],
                "label": E1_3_DATAPOINTS["e1_3_dp01"],
                "xbrl_element": "esrs:ClimateMitigationActions",
            },
            "e1_3_dp02": {
                "value": [
                    {
                        "name": a.name,
                        "description": a.description,
                        "status": a.status.value,
                    }
                    for a in adaptation_actions
                ],
                "label": E1_3_DATAPOINTS["e1_3_dp02"],
                "xbrl_element": "esrs:ClimateAdaptationActions",
            },
            "e1_3_dp03": {
                "value": [
                    {
                        "name": a.name,
                        "expected_reduction_tco2e": str(
                            a.expected_reduction_tco2e
                        ),
                    }
                    for a in actions
                    if a.expected_reduction_tco2e > 0
                ],
                "label": E1_3_DATAPOINTS["e1_3_dp03"],
                "xbrl_element": "esrs:ClimateActionExpectedReduction",
            },
            "e1_3_dp04": {
                "value": [
                    {
                        "name": a.name,
                        "start_date": str(a.start_date) if a.start_date else None,
                        "end_date": str(a.end_date) if a.end_date else None,
                    }
                    for a in actions
                ],
                "label": E1_3_DATAPOINTS["e1_3_dp04"],
                "xbrl_element": "esrs:ClimateActionTimeHorizon",
            },
            "e1_3_dp05": {
                "value": result.actions_by_status,
                "label": E1_3_DATAPOINTS["e1_3_dp05"],
                "xbrl_element": "esrs:ClimateActionStatus",
            },
            "e1_3_dp06": {
                "value": [
                    {
                        "name": a.name,
                        "capex": str(a.capex_amount),
                        "currency": a.currency,
                    }
                    for a in actions
                    if a.capex_amount > 0
                ],
                "label": E1_3_DATAPOINTS["e1_3_dp06"],
                "xbrl_element": "esrs:ClimateActionCapEx",
            },
            "e1_3_dp07": {
                "value": [
                    {
                        "name": a.name,
                        "opex": str(a.opex_amount),
                        "currency": a.currency,
                    }
                    for a in actions
                    if a.opex_amount > 0
                ],
                "label": E1_3_DATAPOINTS["e1_3_dp07"],
                "xbrl_element": "esrs:ClimateActionOpEx",
            },
            "e1_3_dp08": {
                "value": str(result.total_capex),
                "label": E1_3_DATAPOINTS["e1_3_dp08"],
                "xbrl_element": "esrs:ClimateActionTotalCapEx",
            },
            "e1_3_dp09": {
                "value": str(result.total_opex),
                "label": E1_3_DATAPOINTS["e1_3_dp09"],
                "xbrl_element": "esrs:ClimateActionTotalOpEx",
            },
            "e1_3_dp10": {
                "value": [
                    {
                        "name": a.name,
                        "linked_targets": a.linked_target_ids,
                    }
                    for a in actions
                    if a.linked_target_ids
                ],
                "label": E1_3_DATAPOINTS["e1_3_dp10"],
                "xbrl_element": "esrs:ClimateActionTargetContribution",
            },
            "e1_3_dp11": {
                "value": any(
                    a.consistent_with_transition_plan for a in actions
                ),
                "label": E1_3_DATAPOINTS["e1_3_dp11"],
                "xbrl_element": "esrs:ClimateActionTransitionPlanConsistency",
            },
            "e1_3_dp12": {
                "value": result.taxonomy_aligned_count > 0,
                "label": E1_3_DATAPOINTS["e1_3_dp12"],
                "xbrl_element": "esrs:ClimateActionTaxonomyAligned",
            },
            "e1_3_dp13": {
                "value": [
                    {
                        "name": a.name,
                        "responsible_entity": a.responsible_entity,
                    }
                    for a in actions
                    if a.responsible_entity
                ],
                "label": E1_3_DATAPOINTS["e1_3_dp13"],
                "xbrl_element": "esrs:ClimateActionResponsibleEntity",
            },
            "e1_3_dp14": {
                "value": result.actions_by_category,
                "label": E1_3_DATAPOINTS["e1_3_dp14"],
                "xbrl_element": "esrs:ClimateActionByCategory",
            },
        }

        datapoints["provenance_hash"] = _compute_hash(datapoints)
        return datapoints

    # ------------------------------------------------------------------ #
    # Summary and Reporting Utilities                                      #
    # ------------------------------------------------------------------ #

    def get_action_summary(
        self, action: ClimateAction
    ) -> Dict[str, Any]:
        """Return a structured summary of a single climate action.

        Useful for action-level reporting and audit documentation.

        Args:
            action: ClimateAction to summarise.

        Returns:
            Dict with action details and taxonomy alignment info.
        """
        taxonomy_info = ACTION_TAXONOMY_ALIGNMENT.get(
            action.category.value, {}
        )

        return {
            "action_id": action.action_id,
            "name": action.name,
            "category": action.category.value,
            "category_description": ACTION_CATEGORY_DESCRIPTIONS.get(
                action.category.value, ""
            ),
            "status": action.status.value,
            "expected_reduction_tco2e": str(action.expected_reduction_tco2e),
            "capex": str(action.capex_amount),
            "opex": str(action.opex_amount),
            "currency": action.currency,
            "start_date": str(action.start_date) if action.start_date else None,
            "end_date": str(action.end_date) if action.end_date else None,
            "responsible_entity": action.responsible_entity,
            "taxonomy_aligned": action.taxonomy_aligned,
            "taxonomy_objective": taxonomy_info.get("objective", "N/A"),
            "taxonomy_criteria": taxonomy_info.get(
                "substantial_contribution_criteria", "N/A"
            ),
            "linked_policies": action.linked_policy_ids,
            "linked_targets": action.linked_target_ids,
            "consistent_with_transition_plan": action.consistent_with_transition_plan,
            "provenance_hash": action.provenance_hash,
        }

    def get_policy_summary(
        self, policy: ClimatePolicy
    ) -> Dict[str, Any]:
        """Return a structured summary of a single climate policy.

        Args:
            policy: ClimatePolicy to summarise.

        Returns:
            Dict with policy details.
        """
        return {
            "policy_id": policy.policy_id,
            "name": policy.name,
            "type": policy.policy_type.value,
            "type_description": POLICY_TYPE_DESCRIPTIONS.get(
                policy.policy_type.value, ""
            ),
            "scope": policy.scope.value,
            "description": policy.description,
            "adoption_date": str(policy.adoption_date)
            if policy.adoption_date
            else None,
            "review_cycle": policy.review_cycle,
            "responsible_body": policy.responsible_body,
            "covers_own_operations": policy.covers_own_operations,
            "covers_upstream": policy.covers_upstream,
            "covers_downstream": policy.covers_downstream,
            "third_party_standards": policy.third_party_standards,
            "linked_targets": policy.linked_target_ids,
            "stakeholders_consulted": policy.stakeholders_consulted,
            "provenance_hash": policy.provenance_hash,
        }

    def get_reduction_by_category(
        self, actions: List[ClimateAction]
    ) -> Dict[str, str]:
        """Calculate total expected GHG reduction by action category.

        Args:
            actions: List of climate actions.

        Returns:
            Dict mapping category to total expected reduction (tCO2e).
        """
        by_category: Dict[str, Decimal] = {}

        for action in actions:
            cat_key = action.category.value
            if cat_key not in by_category:
                by_category[cat_key] = Decimal("0.000")
            by_category[cat_key] += action.expected_reduction_tco2e

        return {
            k: str(_round_val(v, 3)) for k, v in by_category.items()
        }

    def get_capex_by_category(
        self, actions: List[ClimateAction]
    ) -> Dict[str, str]:
        """Calculate total CapEx by action category.

        Args:
            actions: List of climate actions.

        Returns:
            Dict mapping category to total CapEx.
        """
        by_category: Dict[str, Decimal] = {}

        for action in actions:
            cat_key = action.category.value
            if cat_key not in by_category:
                by_category[cat_key] = Decimal("0.00")
            by_category[cat_key] += action.capex_amount

        return {
            k: str(_round_val(v, 2)) for k, v in by_category.items()
        }

    def get_actions_by_status(
        self, actions: List[ClimateAction], status: ActionStatus
    ) -> List[ClimateAction]:
        """Filter actions by status.

        Args:
            actions: List of climate actions.
            status: ActionStatus to filter by.

        Returns:
            Filtered list of actions matching the status.
        """
        return [a for a in actions if a.status == status]

    def get_actions_by_category(
        self, actions: List[ClimateAction], category: ActionCategory
    ) -> List[ClimateAction]:
        """Filter actions by category.

        Args:
            actions: List of climate actions.
            category: ActionCategory to filter by.

        Returns:
            Filtered list of actions matching the category.
        """
        return [a for a in actions if a.category == category]

    def clear_registry(self) -> None:
        """Clear all registered policies, actions, and resources."""
        self._policies.clear()
        self._actions.clear()
        self._resources.clear()
        logger.info("ClimateActionEngine registry cleared")

    # ------------------------------------------------------------------ #
    # Private: Completeness Calculation                                    #
    # ------------------------------------------------------------------ #

    def _calculate_e1_2_completeness(
        self, policies: List[ClimatePolicy]
    ) -> float:
        """Calculate E1-2 completeness score from policies list.

        Uses the same logic as validate_completeness_e1_2 but returns
        only the score.

        Args:
            policies: List of climate policies.

        Returns:
            Completeness score (0-100).
        """
        if not policies:
            return 0.0

        covered = 0
        total = len(E1_2_DATAPOINTS)

        checks = [
            len(policies) > 0,
            any(p.scope is not None for p in policies),
            any(bool(p.responsible_body) for p in policies),
            any(len(p.third_party_standards) > 0 for p in policies),
            any(p.policy_type == PolicyType.MITIGATION for p in policies),
            any(p.policy_type == PolicyType.ADAPTATION for p in policies),
            any(p.covers_own_operations for p in policies),
            any(p.covers_upstream for p in policies),
            any(p.covers_downstream for p in policies),
            any(p.adoption_date is not None for p in policies),
            any(len(p.linked_target_ids) > 0 for p in policies),
            any(p.stakeholders_consulted for p in policies),
        ]

        covered = sum(1 for c in checks if c)
        return _round2(_safe_divide(float(covered), float(total), 0.0) * 100.0)

    def _calculate_e1_3_completeness(
        self,
        actions: List[ClimateAction],
        resources: Optional[List[ResourceAllocation]] = None,
    ) -> float:
        """Calculate E1-3 completeness score from actions list.

        Uses the same logic as validate_completeness_e1_3 but returns
        only the score.

        Args:
            actions: List of climate actions.
            resources: Optional list of resource allocations.

        Returns:
            Completeness score (0-100).
        """
        if not actions:
            return 0.0

        total = len(E1_3_DATAPOINTS)

        checks = [
            any(a.category != ActionCategory.OFFSET for a in actions),
            any(a.category == ActionCategory.NATURE_BASED for a in actions),
            any(a.expected_reduction_tco2e > 0 for a in actions),
            any(a.end_date is not None for a in actions),
            any(a.status is not None for a in actions),
            any(a.capex_amount > 0 for a in actions),
            any(a.opex_amount > 0 for a in actions),
            sum(a.capex_amount for a in actions) > 0,
            sum(a.opex_amount for a in actions) > 0,
            any(len(a.linked_target_ids) > 0 for a in actions),
            any(a.consistent_with_transition_plan for a in actions),
            any(a.taxonomy_aligned for a in actions),
            any(bool(a.responsible_entity) for a in actions),
            True,  # actions_by_category is always populated if actions exist
        ]

        covered = sum(1 for c in checks if c)
        return _round2(_safe_divide(float(covered), float(total), 0.0) * 100.0)
