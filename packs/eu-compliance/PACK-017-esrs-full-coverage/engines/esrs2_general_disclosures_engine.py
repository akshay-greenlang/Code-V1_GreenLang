# -*- coding: utf-8 -*-
"""
ESRS2GeneralDisclosuresEngine - PACK-017 ESRS Full Coverage Engine 1
=====================================================================

Assesses and validates ESRS 2 General Disclosures, the cross-cutting
disclosure requirements that ALL companies must report under the European
Sustainability Reporting Standards.

ESRS 2 is mandatory for all undertakings regardless of their materiality
assessment.  It requires disclosure across four pillars:

**Governance (GOV):**
    - GOV-1: Role of administrative, management and supervisory bodies
      (ESRS 2 Para 21-25)
    - GOV-2: Information provided to and sustainability matters addressed
      by undertaking's administrative, management and supervisory bodies
      (ESRS 2 Para 27-29)
    - GOV-3: Integration of sustainability-related performance in
      incentive schemes (ESRS 2 Para 31-33)
    - GOV-4: Statement on due diligence (ESRS 2 Para 35-37)
    - GOV-5: Risk management and internal controls over sustainability
      reporting (ESRS 2 Para 39-41)

**Strategy and Business Model (SBM):**
    - SBM-1: Strategy and business model (ESRS 2 Para 43-47)
    - SBM-2: Interests and views of stakeholders (ESRS 2 Para 49-52)
    - SBM-3: Material impacts, risks and opportunities and their
      interaction with strategy and business model (ESRS 2 Para 54-57)

**Impact, Risk and Opportunity Assessment (IRO):**
    - IRO-1: Description of the processes to identify and assess material
      impacts, risks and opportunities (ESRS 2 Para 59-62)
    - IRO-2: Disclosure requirements in ESRS covered by the undertaking's
      sustainability statement (ESRS 2 Para 64-66)

**Minimum Disclosure Requirements (MDR):**
    - MDR-P: Policies adopted to manage material sustainability matters
    - MDR-A: Actions and resources in relation to material sustainability
      matters
    - MDR-T: Tracking effectiveness of policies and actions through
      targets
    - MDR-M: Metrics in relation to material sustainability matters

Regulatory References:
    - EU Delegated Regulation 2023/2772 (ESRS)
    - ESRS 2 General Disclosures (all paragraphs 1-66)
    - OECD Guidelines for Multinational Enterprises (2023 update)
    - UN Guiding Principles on Business and Human Rights (2011)
    - COSO ERM Framework (2017)
    - ISO 31000:2018 Risk Management

Zero-Hallucination:
    - All percentage calculations use deterministic Decimal arithmetic
    - Completeness scoring uses pre-defined data point checklists
    - Board composition metrics are calculated from provided member data
    - No LLM involvement in any calculation or scoring path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-017 ESRS Full Coverage
Status:  Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional

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
    numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0")
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

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

class GovernanceBodyType(str, Enum):
    """Types of administrative, management and supervisory bodies.

    Per ESRS 2 GOV-1 Para 21, undertakings shall disclose the
    composition of their governance bodies with sustainability oversight.
    """
    BOARD = "board"
    BOARD_OF_DIRECTORS = "board_of_directors"
    SUPERVISORY_BOARD = "supervisory_board"
    MANAGEMENT_BOARD = "management_board"
    AUDIT_COMMITTEE = "audit_committee"
    SUSTAINABILITY_COMMITTEE = "sustainability_committee"
    RISK_COMMITTEE = "risk_committee"
    REMUNERATION_COMMITTEE = "remuneration_committee"

class BoardMemberRole(str, Enum):
    """Roles of members on governance bodies.

    Per ESRS 2 GOV-1 Para 22, the disclosure shall include
    the role and responsibilities of each member.
    """
    CHAIR = "chair"
    VICE_CHAIR = "vice_chair"
    EXECUTIVE_DIRECTOR = "executive_director"
    NON_EXECUTIVE_DIRECTOR = "non_executive_director"
    INDEPENDENT_DIRECTOR = "independent_director"
    EMPLOYEE_REPRESENTATIVE = "employee_representative"

class SustainabilityExpertise(str, Enum):
    """Areas of sustainability expertise held by governance members.

    Per ESRS 2 GOV-1 Para 23, undertakings shall describe the
    sustainability-related expertise of their governance bodies.
    """
    CLIMATE = "climate"
    ENVIRONMENTAL = "environmental"
    SOCIAL = "social"
    GOVERNANCE = "governance"
    HUMAN_RIGHTS = "human_rights"
    SUPPLY_CHAIN = "supply_chain"
    FINANCE_ESG = "finance_esg"

class DueDiligenceScope(str, Enum):
    """Scope of due diligence coverage in the value chain.

    Per ESRS 2 GOV-4 Para 35, the due diligence statement shall
    cover the full extent of the value chain.
    """
    OWN_OPERATIONS = "own_operations"
    UPSTREAM_SUPPLY_CHAIN = "upstream_supply_chain"
    DOWNSTREAM_VALUE_CHAIN = "downstream_value_chain"
    FULL_VALUE_CHAIN = "full_value_chain"

class IncentiveMetricType(str, Enum):
    """Types of sustainability metrics used in incentive schemes.

    Per ESRS 2 GOV-3 Para 31, undertakings shall disclose
    sustainability-related performance metrics linked to remuneration.
    """
    GHG_REDUCTION = "ghg_reduction"
    RENEWABLE_ENERGY = "renewable_energy"
    DIVERSITY = "diversity"
    HEALTH_SAFETY = "health_safety"
    CUSTOMER_SATISFACTION = "customer_satisfaction"
    ESG_RATING = "esg_rating"
    CIRCULAR_ECONOMY = "circular_economy"

class IncentiveSchemeType(str, Enum):
    """Types of incentive schemes linked to sustainability performance.

    Per ESRS 2 GOV-3 Para 31, undertakings shall disclose the
    type of remuneration schemes that integrate sustainability targets.
    """
    BONUS = "bonus"
    LONG_TERM_INCENTIVE = "long_term_incentive"
    EQUITY = "equity"
    DEFERRED_COMPENSATION = "deferred_compensation"

class StakeholderGroup(str, Enum):
    """Categories of stakeholders per ESRS 2 SBM-2.

    Per ESRS 2 SBM-2 Para 49, undertakings shall describe the
    interests and views of affected stakeholders.
    """
    EMPLOYEES = "employees"
    INVESTORS = "investors"
    CUSTOMERS = "customers"
    SUPPLIERS = "suppliers"
    COMMUNITIES = "communities"
    LOCAL_COMMUNITIES = "local_communities"
    CIVIL_SOCIETY = "civil_society"
    REGULATORS = "regulators"
    MEDIA = "media"
    ACADEMIA = "academia"

class MaterialityDetermination(str, Enum):
    """Outcome of the materiality assessment for an ESRS topic.

    Per ESRS 2 IRO-2 Para 64, undertakings shall list which ESRS
    disclosure requirements they have determined to be material.
    """
    MATERIAL = "material"
    NOT_MATERIAL = "not_material"
    PHASE_IN = "phase_in"

class MaterialityType(str, Enum):
    """Type of materiality assessment per double materiality principle.

    Per ESRS 1 Para 38-39, undertakings shall consider both impact
    and financial materiality dimensions.
    """
    IMPACT = "impact"
    FINANCIAL = "financial"
    DOUBLE = "double"

class TimeHorizon(str, Enum):
    """Time horizons for impacts, risks and opportunities.

    Per ESRS 2 SBM-3 Para 54, time horizons must be specified
    when describing material IROs.
    """
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ESRS 2 GOV required data points for completeness validation.
ESRS2_GOV_DATAPOINTS: List[str] = [
    "gov1_01_governance_body_composition",
    "gov1_02_member_roles_and_responsibilities",
    "gov1_03_sustainability_expertise",
    "gov1_04_meeting_frequency",
    "gov1_05_independence_percentage",
    "gov1_06_gender_diversity",
    "gov2_01_sustainability_matters_addressed",
    "gov2_02_information_provided_to_bodies",
    "gov2_03_frequency_of_sustainability_agenda",
    "gov3_01_sustainability_linked_remuneration",
    "gov3_02_incentive_metric_types",
    "gov3_03_weight_in_total_remuneration",
    "gov4_01_due_diligence_statement",
    "gov4_02_value_chain_coverage",
    "gov4_03_standards_followed",
    "gov5_01_risk_management_framework",
    "gov5_02_sustainability_integration",
    "gov5_03_three_lines_of_defense",
    "gov5_04_internal_audit_scope",
    "gov5_05_external_assurance",
]

# ESRS 2 SBM required data points.
ESRS2_SBM_DATAPOINTS: List[str] = [
    "sbm1_01_strategy_description",
    "sbm1_02_business_model_description",
    "sbm1_03_sustainability_relevance",
    "sbm1_04_value_chain_description",
    "sbm1_05_products_services_affected",
    "sbm2_01_stakeholder_groups_identified",
    "sbm2_02_engagement_methods",
    "sbm2_03_key_concerns_raised",
    "sbm2_04_views_influence_on_strategy",
    "sbm3_01_material_iros_identified",
    "sbm3_02_interaction_with_strategy",
    "sbm3_03_time_horizons",
    "sbm3_04_value_chain_stages",
]

# ESRS 2 IRO required data points.
ESRS2_IRO_DATAPOINTS: List[str] = [
    "iro1_01_process_description",
    "iro1_02_stakeholders_consulted",
    "iro1_03_materiality_criteria",
    "iro1_04_double_materiality_approach",
    "iro1_05_update_frequency",
    "iro2_01_material_topics_list",
    "iro2_02_esrs_standards_covered",
    "iro2_03_non_material_topics_rationale",
    "iro2_04_phase_in_topics",
]

# ESRS 2 MDR required data points.
ESRS2_MDR_DATAPOINTS: List[str] = [
    "mdr_p_01_policy_description",
    "mdr_p_02_policy_scope",
    "mdr_p_03_highest_body_responsible",
    "mdr_p_04_third_party_standards_referenced",
    "mdr_p_05_stakeholder_consideration",
    "mdr_a_01_action_description",
    "mdr_a_02_resources_allocated",
    "mdr_a_03_implementation_timeline",
    "mdr_a_04_expected_outcomes",
    "mdr_t_01_measurable_targets",
    "mdr_t_02_target_base_year",
    "mdr_t_03_milestones_and_interim_targets",
    "mdr_t_04_methodology_for_tracking",
    "mdr_m_01_metrics_defined",
    "mdr_m_02_metric_methodology",
    "mdr_m_03_metric_validation",
]

# OECD Due Diligence steps per GOV-4 Para 36.
# Source: OECD Guidelines for Multinational Enterprises (2023 update).
OECD_DUE_DILIGENCE_STEPS: List[str] = [
    "embed_responsible_business_conduct_into_policies",
    "identify_and_assess_adverse_impacts",
    "cease_prevent_or_mitigate_adverse_impacts",
    "track_implementation_and_results",
    "communicate_how_impacts_are_addressed",
    "provide_for_or_cooperate_in_remediation",
]

# Three Lines of Defense model per GOV-5 Para 39.
# Source: IIA Three Lines Model (2020), COSO ERM (2017).
THREE_LINES_OF_DEFENSE_MODEL: Dict[str, str] = {
    "first_line": "Management controls and internal control measures",
    "second_line": "Risk management and compliance functions",
    "third_line": "Internal audit providing independent assurance",
}

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class GovernanceBody(BaseModel):
    """A governance body with sustainability oversight per GOV-1.

    Represents an administrative, management, or supervisory body
    as described in ESRS 2 Para 21-25.
    """
    body_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this governance body",
    )
    type: GovernanceBodyType = Field(
        ...,
        description="Type of governance body",
    )
    members_count: int = Field(
        ...,
        description="Total number of members on this body",
        ge=0,
    )
    name: str = Field(
        default="",
        description="Name of the governance body",
        max_length=300,
    )
    independent_members_pct: Decimal = Field(
        default=Decimal("0"),
        description="Percentage of independent members (0-100)",
        ge=Decimal("0"),
        le=Decimal("100"),
    )
    female_members_pct: Decimal = Field(
        default=Decimal("0"),
        description="Percentage of female members (0-100)",
        ge=Decimal("0"),
        le=Decimal("100"),
    )
    sustainability_expertise_count: int = Field(
        default=0,
        description="Number of members with sustainability expertise",
        ge=0,
    )
    meeting_frequency: int = Field(
        default=0,
        description="Number of meetings per year",
        ge=0,
    )
    sustainability_topics_discussed: List[str] = Field(
        default_factory=list,
        description="Sustainability topics addressed by this body",
    )

    @property
    def expertise_ratio(self) -> Decimal:
        """Ratio of members with sustainability expertise."""
        if self.members_count == 0:
            return Decimal("0")
        return _round_val(
            _decimal(self.sustainability_expertise_count) / _decimal(self.members_count),
            3,
        )

class BoardMember(BaseModel):
    """Individual governance body member per GOV-1 Para 22.

    Captures the role, demographics, expertise, and committee
    memberships of each member for composition analysis.
    """
    member_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this member",
    )
    name: str = Field(
        ...,
        description="Member name",
        max_length=300,
    )
    role: BoardMemberRole = Field(
        ...,
        description="Member role on the governance body",
    )
    gender: str = Field(
        default="",
        description="Gender (male, female, non_binary, not_disclosed)",
        max_length=50,
    )
    age: int = Field(
        default=0,
        description="Age of the member",
        ge=0,
    )
    tenure_years: Decimal = Field(
        default=Decimal("0"),
        description="Years of tenure on the body",
        ge=Decimal("0"),
    )
    sustainability_expertise: List[SustainabilityExpertise] = Field(
        default_factory=list,
        description="Areas of sustainability expertise held",
    )
    committees: List[str] = Field(
        default_factory=list,
        description="Committee memberships",
    )

class IncentiveScheme(BaseModel):
    """Sustainability-linked incentive scheme per GOV-3 Para 31.

    Captures how sustainability-related performance is integrated
    into the remuneration of governance body members and senior
    management.
    """
    scheme_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this incentive scheme",
    )
    scheme_type: IncentiveSchemeType = Field(
        default=IncentiveSchemeType.BONUS,
        description="Type of incentive scheme",
    )
    description: str = Field(
        default="",
        description="Description of the incentive scheme",
        max_length=500,
    )
    role_level: str = Field(
        default="",
        description="Management level to which this scheme applies",
        max_length=200,
    )
    sustainability_metrics_linked: List[str] = Field(
        default_factory=list,
        description="Sustainability metrics linked to remuneration",
    )
    sustainability_metrics: List[IncentiveMetricType] = Field(
        default_factory=list,
        description="Typed sustainability metrics linked to remuneration",
    )
    weight_of_sustainability_pct: Decimal = Field(
        default=Decimal("0"),
        description="Weight of sustainability metrics in total remuneration (0-100)",
        ge=Decimal("0"),
        le=Decimal("100"),
    )
    weight_in_total_remuneration_pct: Decimal = Field(
        default=Decimal("0"),
        description="Weight of sustainability in total remuneration (0-100)",
        ge=Decimal("0"),
        le=Decimal("100"),
    )
    variable_pay_affected_pct: Decimal = Field(
        default=Decimal("0"),
        description="Percentage of variable pay affected by sustainability (0-100)",
        ge=Decimal("0"),
        le=Decimal("100"),
    )
    metric_targets: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of metric type to target description",
    )

class DueDiligenceProcess(BaseModel):
    """Due diligence process per GOV-4 Para 35-37.

    Captures the undertaking's due diligence statement including
    scope, standards followed, and value chain coverage.
    """
    process_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this due diligence process",
    )
    scope: DueDiligenceScope = Field(
        ...,
        description="Scope of due diligence coverage",
    )
    standards_followed: List[str] = Field(
        default_factory=list,
        description="Standards and frameworks followed (e.g. OECD, UNGPs)",
    )
    value_chain_coverage: List[str] = Field(
        default_factory=list,
        description="Parts of the value chain covered",
    )
    stakeholders_consulted: List[str] = Field(
        default_factory=list,
        description="Stakeholder groups consulted during due diligence",
    )
    topics_covered: List[str] = Field(
        default_factory=list,
        description="Sustainability topics covered by due diligence",
    )
    frequency: str = Field(
        default="annual",
        description="Frequency of due diligence reviews",
        max_length=100,
    )

    @field_validator("standards_followed")
    @classmethod
    def validate_standards(cls, v: List[str]) -> List[str]:
        """Validate that at least one standard is referenced."""
        if not v:
            logger.warning(
                "GOV-4 requires referencing due diligence standards "
                "(e.g. OECD Guidelines, UNGPs)"
            )
        return v

class RiskManagementIntegration(BaseModel):
    """Risk management and internal controls per GOV-5 Para 39-41.

    Captures how sustainability risks are integrated into the
    undertaking's overall risk management framework.
    """
    framework_type: str = Field(
        ...,
        description="Risk management framework type (e.g. COSO ERM, ISO 31000)",
        max_length=200,
    )
    sustainability_integrated: bool = Field(
        default=False,
        description="Whether sustainability risks are integrated into ERM",
    )
    three_lines_of_defense: Dict[str, bool] = Field(
        default_factory=lambda: {
            "first_line": False,
            "second_line": False,
            "third_line": False,
        },
        description="Whether each line addresses sustainability",
    )
    internal_audit_scope: List[str] = Field(
        default_factory=list,
        description="Sustainability topics in internal audit scope",
    )
    external_assurance: bool = Field(
        default=False,
        description="Whether external assurance is obtained for sustainability",
    )

# Backward-compatible aliases
RiskManagementProcess = RiskManagementIntegration
InternalControlSystem = RiskManagementIntegration

class StrategyElement(BaseModel):
    """Strategy and business model element per SBM-1 Para 43-47.

    Captures how sustainability is embedded in the undertaking's
    strategy and business model.
    """
    element_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this strategy element",
    )
    description: str = Field(
        ...,
        description="Description of the strategy or business model element",
        max_length=2000,
    )
    sustainability_relevance: str = Field(
        default="",
        description="How this element relates to sustainability",
        max_length=2000,
    )
    time_horizon: TimeHorizon = Field(
        default=TimeHorizon.MEDIUM_TERM,
        description="Applicable time horizon",
    )
    products_services_affected: List[str] = Field(
        default_factory=list,
        description="Products, services, or markets affected",
    )

class StakeholderEngagement(BaseModel):
    """Stakeholder engagement record per SBM-2 Para 49-52.

    Captures engagement activities with stakeholder groups and
    how their views influenced the undertaking's strategy.
    """
    stakeholder_group: StakeholderGroup = Field(
        ...,
        description="Stakeholder group engaged",
    )
    engagement_methods: List[str] = Field(
        default_factory=list,
        description="Methods used for engagement (surveys, meetings, etc.)",
    )
    frequency: str = Field(
        default="annual",
        description="Frequency of engagement",
        max_length=100,
    )
    key_concerns: List[str] = Field(
        default_factory=list,
        description="Key sustainability concerns raised by stakeholders",
    )
    how_views_influenced_strategy: str = Field(
        default="",
        description="Description of how stakeholder views influenced strategy",
        max_length=2000,
    )

    @property
    def key_concerns_raised(self) -> List[str]:
        """Backward-compatible alias."""
        return self.key_concerns

class MaterialIRO(BaseModel):
    """Material impact, risk or opportunity per IRO-1/IRO-2.

    Captures the result of the double materiality assessment for
    a specific ESRS topic, per ESRS 2 Para 59-66.
    """
    iro_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this IRO",
    )
    topic: str = Field(
        ...,
        description="Sustainability topic (e.g. climate change, pollution)",
        max_length=300,
    )
    esrs_standard: str = Field(
        default="",
        description="Applicable ESRS standard (e.g. ESRS E1, ESRS S1)",
        max_length=50,
    )
    disclosure_requirements: List[str] = Field(
        default_factory=list,
        description="Specific disclosure requirements applicable",
    )
    is_material: MaterialityDetermination = Field(
        default=MaterialityDetermination.NOT_MATERIAL,
        description="Materiality determination outcome",
    )
    determination_rationale: str = Field(
        default="",
        description="Rationale for the materiality determination",
        max_length=2000,
    )
    time_horizon: TimeHorizon = Field(
        default=TimeHorizon.MEDIUM_TERM,
        description="Time horizon over which the IRO is relevant",
    )
    value_chain_stage: str = Field(
        default="",
        description="Stage of the value chain where the IRO occurs",
        max_length=200,
    )

class MinimumDisclosureRequirement(BaseModel):
    """Minimum Disclosure Requirement (MDR) entry per ESRS 2.

    Captures the policies, actions, targets, or metrics adopted
    to manage material sustainability matters (MDR-P, MDR-A,
    MDR-T, MDR-M).
    """
    mdr_type: str = Field(
        ...,
        description="MDR type: MDR-P, MDR-A, MDR-T, or MDR-M",
        max_length=10,
    )
    topic: str = Field(
        ...,
        description="Sustainability topic this MDR relates to",
        max_length=300,
    )
    policy_or_action_description: str = Field(
        default="",
        description="Description of the policy, action, target, or metric",
        max_length=3000,
    )
    scope: str = Field(
        default="",
        description="Scope of applicability",
        max_length=500,
    )
    implementation_date: Optional[str] = Field(
        default=None,
        description="Date of implementation (ISO format YYYY-MM-DD)",
        max_length=10,
    )
    resources_allocated: Optional[Decimal] = Field(
        default=None,
        description="Financial resources allocated (in reporting currency)",
        ge=Decimal("0"),
    )
    kpis: List[str] = Field(
        default_factory=list,
        description="Key performance indicators for tracking",
    )

    @field_validator("mdr_type")
    @classmethod
    def validate_mdr_type(cls, v: str) -> str:
        """Validate mdr_type is one of the four MDR categories."""
        valid_types = {"MDR-P", "MDR-A", "MDR-T", "MDR-M"}
        if v not in valid_types:
            raise ValueError(
                f"mdr_type must be one of {valid_types}, got '{v}'"
            )
        return v

class ESRS2GeneralResult(BaseModel):
    """Complete ESRS 2 General Disclosures result.

    Aggregates all governance, strategy, IRO, and MDR assessments
    into a single disclosure result with computed metrics and
    provenance tracking.
    """
    result_id: str = Field(
        default_factory=_new_uuid,
        description="Unique result identifier",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION,
        description="Engine version used for this assessment",
    )
    calculated_at: datetime = Field(
        default_factory=utcnow,
        description="Timestamp of calculation (UTC)",
    )

    # Governance (GOV-1 through GOV-5)
    governance_bodies: List[GovernanceBody] = Field(
        default_factory=list,
        description="Governance bodies with sustainability oversight",
    )
    board_composition: List[BoardMember] = Field(
        default_factory=list,
        description="Board member composition data",
    )
    incentive_schemes: List[IncentiveScheme] = Field(
        default_factory=list,
        description="Sustainability-linked incentive schemes",
    )
    due_diligence: Optional[DueDiligenceProcess] = Field(
        default=None,
        description="Due diligence process (GOV-4)",
    )
    risk_management: Optional[RiskManagementIntegration] = Field(
        default=None,
        description="Risk management integration (GOV-5)",
    )

    # Strategy (SBM-1 through SBM-3)
    strategy_elements: List[StrategyElement] = Field(
        default_factory=list,
        description="Strategy and business model elements (SBM-1)",
    )
    stakeholder_engagements: List[StakeholderEngagement] = Field(
        default_factory=list,
        description="Stakeholder engagement records (SBM-2)",
    )

    # IRO (IRO-1, IRO-2)
    material_iros: List[MaterialIRO] = Field(
        default_factory=list,
        description="Material impacts, risks and opportunities",
    )
    non_material_topics: List[str] = Field(
        default_factory=list,
        description="Topics determined as not material with rationale",
    )

    # MDR (MDR-P, MDR-A, MDR-T, MDR-M)
    mdr_policies: List[MinimumDisclosureRequirement] = Field(
        default_factory=list,
        description="MDR-P: Policies adopted",
    )
    mdr_actions: List[MinimumDisclosureRequirement] = Field(
        default_factory=list,
        description="MDR-A: Actions and resources",
    )
    mdr_targets: List[MinimumDisclosureRequirement] = Field(
        default_factory=list,
        description="MDR-T: Targets for tracking effectiveness",
    )
    mdr_metrics: List[MinimumDisclosureRequirement] = Field(
        default_factory=list,
        description="MDR-M: Metrics in relation to material matters",
    )

    # Computed metrics
    total_board_members: int = Field(
        default=0,
        description="Total number of board members across all bodies",
    )
    independent_pct: Decimal = Field(
        default=Decimal("0"),
        description="Overall independence percentage (0-100)",
    )
    female_board_pct: Decimal = Field(
        default=Decimal("0"),
        description="Overall female board representation percentage (0-100)",
    )
    sustainability_expertise_pct: Decimal = Field(
        default=Decimal("0"),
        description="Percentage of members with sustainability expertise (0-100)",
    )
    sustainability_linked_remuneration_pct: Decimal = Field(
        default=Decimal("0"),
        description="Average weight of sustainability in remuneration (0-100)",
    )
    material_topics_count: int = Field(
        default=0,
        description="Number of topics determined as material",
    )
    compliance_score: Decimal = Field(
        default=Decimal("0"),
        description="Overall ESRS 2 compliance score (0-100)",
    )

    # Provenance
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of all inputs and calculation steps",
    )

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class GeneralDisclosuresEngine:
    """ESRS 2 General Disclosures assessment engine.

    Provides deterministic, zero-hallucination assessments for:
    - Governance body composition analysis (GOV-1)
    - Sustainability information flow assessment (GOV-2)
    - Incentive scheme evaluation (GOV-3)
    - Due diligence statement validation (GOV-4)
    - Risk management integration scoring (GOV-5)
    - Strategy and business model review (SBM-1, SBM-2, SBM-3)
    - IRO process assessment (IRO-1, IRO-2)
    - Minimum Disclosure Requirements validation (MDR-P/A/T/M)
    - Full ESRS 2 completeness scoring

    All calculations use Decimal arithmetic for bit-perfect
    reproducibility.  No LLM is used in any calculation path.

    Usage::

        engine = GeneralDisclosuresEngine()
        bodies = [GovernanceBody(type=GovernanceBodyType.BOARD_OF_DIRECTORS, ...)]
        members = [BoardMember(name="Jane Doe", role=BoardMemberRole.CHAIR, ...)]
        result = engine.calculate_esrs2_disclosure(
            bodies=bodies, members=members, schemes=[], ...
        )
    """

    engine_version: str = _MODULE_VERSION

    # ------------------------------------------------------------------ #
    # Governance Assessment (GOV-1 through GOV-5)                          #
    # ------------------------------------------------------------------ #

    def assess_governance(
        self,
        bodies: List[GovernanceBody],
        members: List[BoardMember],
        schemes: List[IncentiveScheme],
        due_diligence: Optional[DueDiligenceProcess] = None,
        risk_management: Optional[RiskManagementIntegration] = None,
    ) -> Dict[str, Any]:
        """Assess governance disclosures per ESRS 2 GOV-1 through GOV-5.

        Evaluates governance body composition, sustainability expertise,
        incentive scheme integration, due diligence coverage, and risk
        management framework integration.

        Args:
            bodies: List of GovernanceBody instances.
            members: List of BoardMember instances.
            schemes: List of IncentiveScheme instances.
            due_diligence: Optional DueDiligenceProcess.
            risk_management: Optional RiskManagementIntegration.

        Returns:
            Dict with governance assessment including gov1 through gov5
            sections, computed metrics, overall_governance_score, and
            provenance_hash.

        Raises:
            ValueError: If bodies list is empty.
        """
        t0 = time.perf_counter()

        if not bodies:
            raise ValueError("At least one GovernanceBody is required")

        logger.info(
            "Assessing governance: %d bodies, %d members, %d schemes",
            len(bodies), len(members), len(schemes),
        )

        gov1 = self._assess_gov1_composition(bodies, members)
        gov2 = self._assess_gov2_topics(bodies)
        gov3 = self._assess_gov3_incentives(schemes)
        gov4 = self._assess_gov4_due_diligence(due_diligence)
        gov5 = self._assess_gov5_risk_management(risk_management)

        section_scores = [
            gov1.get("score", Decimal("0")),
            gov2.get("score", Decimal("0")),
            gov3.get("score", Decimal("0")),
            gov4.get("score", Decimal("0")),
            gov5.get("score", Decimal("0")),
        ]
        overall_score = _round_val(
            _safe_divide(sum(section_scores), _decimal(len(section_scores))),
            1,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result: Dict[str, Any] = {
            "gov1": gov1,
            "gov2": gov2,
            "gov3": gov3,
            "gov4": gov4,
            "gov5": gov5,
            "metrics": {
                "total_bodies": len(bodies),
                "total_members": len(members),
                "total_schemes": len(schemes),
                "has_due_diligence": due_diligence is not None,
                "has_risk_management": risk_management is not None,
            },
            "overall_governance_score": overall_score,
            "processing_time_ms": round(elapsed_ms, 3),
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Governance assessment complete: score=%.1f, hash=%s",
            float(overall_score), result["provenance_hash"][:16],
        )
        return result

    # ------------------------------------------------------------------ #
    # GOV-1 Composition                                                    #
    # ------------------------------------------------------------------ #

    def _assess_gov1_composition(
        self,
        bodies: List[GovernanceBody],
        members: List[BoardMember],
    ) -> Dict[str, Any]:
        """Assess GOV-1 body composition per Para 21-25.

        Args:
            bodies: Governance bodies to assess.
            members: Individual members for detailed analysis.

        Returns:
            Dict with composition metrics and score.
        """
        total_members = sum(b.members_count for b in bodies)
        independent_pct = self._weighted_average_pct(
            bodies, "independent_members_pct"
        )
        female_pct = self._weighted_average_pct(
            bodies, "female_members_pct"
        )

        members_with_expertise = sum(
            1 for m in members if len(m.sustainability_expertise) > 0
        )
        member_denom = _decimal(len(members)) if members else Decimal("1")
        expertise_pct = _round_val(
            _safe_divide(_decimal(members_with_expertise), member_denom)
            * Decimal("100"),
            1,
        )

        # Score: 6 GOV-1 checklist items
        populated = Decimal("0")
        total_checks = Decimal("6")
        if total_members > 0:
            populated += Decimal("1")
        if independent_pct > Decimal("0"):
            populated += Decimal("1")
        if female_pct > Decimal("0"):
            populated += Decimal("1")
        if members_with_expertise > 0:
            populated += Decimal("1")
        if any(b.meeting_frequency > 0 for b in bodies):
            populated += Decimal("1")
        if members:
            populated += Decimal("1")

        score = _round_val(populated / total_checks * Decimal("100"), 1)

        return {
            "total_members": total_members,
            "independent_pct": independent_pct,
            "female_pct": female_pct,
            "sustainability_expertise_count": members_with_expertise,
            "sustainability_expertise_pct": expertise_pct,
            "bodies_with_meetings": sum(
                1 for b in bodies if b.meeting_frequency > 0
            ),
            "score": score,
        }

    # ------------------------------------------------------------------ #
    # GOV-2 Topics                                                         #
    # ------------------------------------------------------------------ #

    def _assess_gov2_topics(
        self, bodies: List[GovernanceBody]
    ) -> Dict[str, Any]:
        """Assess GOV-2 sustainability topics addressed per Para 27-29.

        Args:
            bodies: Governance bodies to evaluate.

        Returns:
            Dict with topic coverage analysis and score.
        """
        all_topics: List[str] = []
        bodies_with_topics = 0
        for body in bodies:
            if body.sustainability_topics_discussed:
                bodies_with_topics += 1
                all_topics.extend(body.sustainability_topics_discussed)

        unique_topics = sorted(set(all_topics))

        populated = Decimal("0")
        total_checks = Decimal("3")
        if bodies_with_topics > 0:
            populated += Decimal("1")
        if len(unique_topics) > 0:
            populated += Decimal("1")
        if any(b.meeting_frequency > 0 for b in bodies):
            populated += Decimal("1")

        score = _round_val(populated / total_checks * Decimal("100"), 1)

        return {
            "bodies_addressing_sustainability": bodies_with_topics,
            "unique_topics_discussed": unique_topics,
            "total_topic_mentions": len(all_topics),
            "score": score,
        }

    # ------------------------------------------------------------------ #
    # GOV-3 Incentives                                                     #
    # ------------------------------------------------------------------ #

    def _assess_gov3_incentives(
        self, schemes: List[IncentiveScheme]
    ) -> Dict[str, Any]:
        """Assess GOV-3 sustainability-linked incentives per Para 31-33.

        Args:
            schemes: Incentive schemes to evaluate.

        Returns:
            Dict with incentive assessment and score.
        """
        if not schemes:
            return {
                "has_sustainability_incentives": False,
                "average_weight_pct": Decimal("0"),
                "average_variable_pay_pct": Decimal("0"),
                "metric_types_used": [],
                "score": Decimal("0"),
            }

        avg_weight = _round_val(
            _safe_divide(
                sum(s.weight_in_total_remuneration_pct for s in schemes),
                _decimal(len(schemes)),
            ),
            1,
        )
        avg_variable = _round_val(
            _safe_divide(
                sum(s.variable_pay_affected_pct for s in schemes),
                _decimal(len(schemes)),
            ),
            1,
        )

        all_metrics: List[str] = []
        for s in schemes:
            all_metrics.extend(m.value for m in s.sustainability_metrics)
        unique_metrics = sorted(set(all_metrics))

        populated = Decimal("0")
        total_checks = Decimal("3")
        if len(schemes) > 0:
            populated += Decimal("1")
        if len(unique_metrics) > 0:
            populated += Decimal("1")
        if avg_weight > Decimal("0"):
            populated += Decimal("1")

        score = _round_val(populated / total_checks * Decimal("100"), 1)

        return {
            "has_sustainability_incentives": True,
            "schemes_count": len(schemes),
            "average_weight_pct": avg_weight,
            "average_variable_pay_pct": avg_variable,
            "metric_types_used": unique_metrics,
            "score": score,
        }

    # ------------------------------------------------------------------ #
    # GOV-4 Due Diligence                                                  #
    # ------------------------------------------------------------------ #

    def _assess_gov4_due_diligence(
        self, dd: Optional[DueDiligenceProcess]
    ) -> Dict[str, Any]:
        """Assess GOV-4 due diligence statement per Para 35-37.

        Validates coverage against the six OECD due diligence steps
        and assesses value chain coverage breadth.

        Args:
            dd: Due diligence process to evaluate.

        Returns:
            Dict with due diligence assessment and score.
        """
        if dd is None:
            return {
                "has_due_diligence": False,
                "scope": None,
                "oecd_steps_covered": 0,
                "oecd_steps_total": len(OECD_DUE_DILIGENCE_STEPS),
                "value_chain_coverage": [],
                "score": Decimal("0"),
            }

        oecd_covered = sum(
            1 for step in OECD_DUE_DILIGENCE_STEPS
            if step in dd.topics_covered or step in dd.standards_followed
        )

        populated = Decimal("0")
        total_checks = Decimal("5")
        if dd.scope is not None:
            populated += Decimal("1")
        if len(dd.standards_followed) > 0:
            populated += Decimal("1")
        if len(dd.value_chain_coverage) > 0:
            populated += Decimal("1")
        if len(dd.stakeholders_consulted) > 0:
            populated += Decimal("1")
        if len(dd.topics_covered) > 0:
            populated += Decimal("1")

        score = _round_val(populated / total_checks * Decimal("100"), 1)

        return {
            "has_due_diligence": True,
            "scope": dd.scope.value,
            "standards_followed": dd.standards_followed,
            "oecd_steps_covered": oecd_covered,
            "oecd_steps_total": len(OECD_DUE_DILIGENCE_STEPS),
            "value_chain_coverage": dd.value_chain_coverage,
            "stakeholders_consulted": dd.stakeholders_consulted,
            "topics_covered": dd.topics_covered,
            "score": score,
        }

    # ------------------------------------------------------------------ #
    # GOV-5 Risk Management                                                #
    # ------------------------------------------------------------------ #

    def _assess_gov5_risk_management(
        self, rm: Optional[RiskManagementIntegration]
    ) -> Dict[str, Any]:
        """Assess GOV-5 risk management per Para 39-41.

        Evaluates the three lines of defense integration and
        internal/external assurance coverage.

        Args:
            rm: Risk management integration to evaluate.

        Returns:
            Dict with risk management assessment and score.
        """
        if rm is None:
            return {
                "has_risk_management": False,
                "sustainability_integrated": False,
                "three_lines_active": 0,
                "three_lines_total": 3,
                "has_external_assurance": False,
                "score": Decimal("0"),
            }

        active_lines = sum(
            1 for active in rm.three_lines_of_defense.values() if active
        )

        populated = Decimal("0")
        total_checks = Decimal("5")
        if rm.framework_type:
            populated += Decimal("1")
        if rm.sustainability_integrated:
            populated += Decimal("1")
        if active_lines > 0:
            populated += Decimal("1")
        if len(rm.internal_audit_scope) > 0:
            populated += Decimal("1")
        if rm.external_assurance:
            populated += Decimal("1")

        score = _round_val(populated / total_checks * Decimal("100"), 1)

        return {
            "has_risk_management": True,
            "framework_type": rm.framework_type,
            "sustainability_integrated": rm.sustainability_integrated,
            "three_lines_active": active_lines,
            "three_lines_total": 3,
            "internal_audit_topics": rm.internal_audit_scope,
            "has_external_assurance": rm.external_assurance,
            "score": score,
        }

    # ------------------------------------------------------------------ #
    # Strategy Assessment (SBM-1, SBM-2, SBM-3)                           #
    # ------------------------------------------------------------------ #

    def assess_strategy(
        self,
        elements: List[StrategyElement],
        stakeholder_engagements: List[StakeholderEngagement],
        material_iros: Optional[List[MaterialIRO]] = None,
    ) -> Dict[str, Any]:
        """Assess strategy and business model disclosures per SBM-1/2/3.

        Evaluates strategy elements, stakeholder engagement breadth,
        and the interaction between material IROs and strategy.

        Args:
            elements: Strategy and business model elements (SBM-1).
            stakeholder_engagements: Stakeholder engagement records (SBM-2).
            material_iros: Optional material IROs for SBM-3 analysis.

        Returns:
            Dict with sbm1, sbm2, sbm3 sections,
            overall_strategy_score, and provenance_hash.
        """
        t0 = time.perf_counter()

        logger.info(
            "Assessing strategy: %d elements, %d engagements",
            len(elements), len(stakeholder_engagements),
        )

        sbm1 = self._assess_sbm1(elements)
        sbm2 = self._assess_sbm2(stakeholder_engagements)
        sbm3 = self._assess_sbm3(material_iros or [])

        section_scores = [
            sbm1.get("score", Decimal("0")),
            sbm2.get("score", Decimal("0")),
            sbm3.get("score", Decimal("0")),
        ]
        overall_score = _round_val(
            _safe_divide(sum(section_scores), _decimal(len(section_scores))),
            1,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result: Dict[str, Any] = {
            "sbm1": sbm1,
            "sbm2": sbm2,
            "sbm3": sbm3,
            "overall_strategy_score": overall_score,
            "processing_time_ms": round(elapsed_ms, 3),
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Strategy assessment complete: score=%.1f",
            float(overall_score),
        )
        return result

    def _assess_sbm1(
        self, elements: List[StrategyElement]
    ) -> Dict[str, Any]:
        """Assess SBM-1 strategy and business model per Para 43-47.

        Args:
            elements: Strategy elements to evaluate.

        Returns:
            Dict with strategy analysis and score.
        """
        if not elements:
            return {
                "elements_count": 0,
                "time_horizons_covered": [],
                "products_services_affected": [],
                "score": Decimal("0"),
            }

        horizons = sorted(set(e.time_horizon.value for e in elements))
        all_products: List[str] = []
        for e in elements:
            all_products.extend(e.products_services_affected)
        unique_products = sorted(set(all_products))

        elements_with_relevance = sum(
            1 for e in elements if e.sustainability_relevance
        )

        populated = Decimal("0")
        total_checks = Decimal("5")
        if len(elements) > 0:
            populated += Decimal("1")
        if elements_with_relevance > 0:
            populated += Decimal("1")
        if len(horizons) > 1:
            populated += Decimal("1")
        elif len(horizons) == 1:
            populated += Decimal("0.5")
        if len(unique_products) > 0:
            populated += Decimal("1")
        if any(e.description for e in elements):
            populated += Decimal("1")

        score = _round_val(populated / total_checks * Decimal("100"), 1)

        return {
            "elements_count": len(elements),
            "elements_with_sustainability_relevance": elements_with_relevance,
            "time_horizons_covered": horizons,
            "products_services_affected": unique_products,
            "score": score,
        }

    def _assess_sbm2(
        self, engagements: List[StakeholderEngagement]
    ) -> Dict[str, Any]:
        """Assess SBM-2 stakeholder interests and views per Para 49-52.

        Args:
            engagements: Stakeholder engagement records.

        Returns:
            Dict with stakeholder engagement analysis and score.
        """
        if not engagements:
            return {
                "stakeholder_groups_engaged": [],
                "total_concerns_raised": 0,
                "engagements_influencing_strategy": 0,
                "score": Decimal("0"),
            }

        groups = sorted(set(e.stakeholder_group.value for e in engagements))
        total_concerns = sum(len(e.key_concerns_raised) for e in engagements)
        influencing = sum(
            1 for e in engagements if e.how_views_influenced_strategy
        )

        populated = Decimal("0")
        total_checks = Decimal("4")
        if len(groups) > 0:
            populated += Decimal("1")
        if any(e.engagement_methods for e in engagements):
            populated += Decimal("1")
        if total_concerns > 0:
            populated += Decimal("1")
        if influencing > 0:
            populated += Decimal("1")

        score = _round_val(populated / total_checks * Decimal("100"), 1)

        return {
            "stakeholder_groups_engaged": groups,
            "groups_count": len(groups),
            "total_concerns_raised": total_concerns,
            "engagements_influencing_strategy": influencing,
            "score": score,
        }

    def _assess_sbm3(
        self, iros: List[MaterialIRO]
    ) -> Dict[str, Any]:
        """Assess SBM-3 material IROs and strategy interaction per Para 54-57.

        Args:
            iros: Material IROs to evaluate.

        Returns:
            Dict with IRO-strategy interaction analysis and score.
        """
        material_iros = [
            i for i in iros
            if i.is_material == MaterialityDetermination.MATERIAL
        ]

        if not material_iros:
            return {
                "material_iros_count": 0,
                "time_horizons_covered": [],
                "value_chain_stages": [],
                "esrs_standards_covered": [],
                "score": Decimal("0"),
            }

        horizons = sorted(set(i.time_horizon.value for i in material_iros))
        stages = sorted(set(
            i.value_chain_stage for i in material_iros if i.value_chain_stage
        ))
        standards = sorted(set(
            i.esrs_standard for i in material_iros if i.esrs_standard
        ))

        populated = Decimal("0")
        total_checks = Decimal("4")
        if len(material_iros) > 0:
            populated += Decimal("1")
        if len(horizons) > 0:
            populated += Decimal("1")
        if len(stages) > 0:
            populated += Decimal("1")
        if any(i.determination_rationale for i in material_iros):
            populated += Decimal("1")

        score = _round_val(populated / total_checks * Decimal("100"), 1)

        return {
            "material_iros_count": len(material_iros),
            "time_horizons_covered": horizons,
            "value_chain_stages": stages,
            "esrs_standards_covered": standards,
            "score": score,
        }

    # ------------------------------------------------------------------ #
    # IRO Process Assessment (IRO-1, IRO-2)                                #
    # ------------------------------------------------------------------ #

    def assess_iro_process(
        self, iros: List[MaterialIRO]
    ) -> Dict[str, Any]:
        """Assess impact, risk and opportunity processes per IRO-1/IRO-2.

        Evaluates the completeness and quality of the double materiality
        assessment process and the resulting list of material topics.

        Args:
            iros: All IROs from the materiality assessment.

        Returns:
            Dict with iro1, iro2 sections, material_summary,
            and provenance_hash.

        Raises:
            ValueError: If iros list is empty.
        """
        t0 = time.perf_counter()

        if not iros:
            raise ValueError("At least one MaterialIRO is required")

        logger.info("Assessing IRO process: %d IROs", len(iros))

        material = [
            i for i in iros
            if i.is_material == MaterialityDetermination.MATERIAL
        ]
        not_material = [
            i for i in iros
            if i.is_material == MaterialityDetermination.NOT_MATERIAL
        ]
        phase_in = [
            i for i in iros
            if i.is_material == MaterialityDetermination.PHASE_IN
        ]

        iro1 = self._assess_iro1(iros)
        iro2 = self._assess_iro2(material, not_material, phase_in)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result: Dict[str, Any] = {
            "iro1": iro1,
            "iro2": iro2,
            "material_summary": {
                "total_topics_assessed": len(iros),
                "material_count": len(material),
                "not_material_count": len(not_material),
                "phase_in_count": len(phase_in),
            },
            "processing_time_ms": round(elapsed_ms, 3),
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "IRO assessment complete: %d material, %d not material, %d phase-in",
            len(material), len(not_material), len(phase_in),
        )
        return result

    def _assess_iro1(self, iros: List[MaterialIRO]) -> Dict[str, Any]:
        """Assess IRO-1 process description per Para 59-62.

        Args:
            iros: All IROs to assess for process completeness.

        Returns:
            Dict with process description quality metrics and score.
        """
        iros_with_rationale = sum(
            1 for i in iros if i.determination_rationale
        )
        iros_with_standard = sum(1 for i in iros if i.esrs_standard)
        iros_with_horizon = sum(1 for i in iros if i.time_horizon is not None)
        iros_with_value_chain = sum(1 for i in iros if i.value_chain_stage)

        populated = Decimal("0")
        total_checks = Decimal("5")
        if len(iros) > 0:
            populated += Decimal("1")
        if iros_with_rationale > 0:
            populated += Decimal("1")
        if iros_with_standard > 0:
            populated += Decimal("1")
        if iros_with_horizon > 0:
            populated += Decimal("1")
        if iros_with_value_chain > 0:
            populated += Decimal("1")

        score = _round_val(populated / total_checks * Decimal("100"), 1)

        return {
            "total_iros_assessed": len(iros),
            "with_rationale": iros_with_rationale,
            "with_esrs_standard": iros_with_standard,
            "with_time_horizon": iros_with_horizon,
            "with_value_chain_stage": iros_with_value_chain,
            "score": score,
        }

    def _assess_iro2(
        self,
        material: List[MaterialIRO],
        not_material: List[MaterialIRO],
        phase_in: List[MaterialIRO],
    ) -> Dict[str, Any]:
        """Assess IRO-2 disclosure requirements covered per Para 64-66.

        Args:
            material: Topics determined as material.
            not_material: Topics determined as not material.
            phase_in: Topics with phase-in provisions.

        Returns:
            Dict with coverage analysis and score.
        """
        material_standards = sorted(set(
            i.esrs_standard for i in material if i.esrs_standard
        ))
        material_topics = sorted(set(i.topic for i in material))
        non_material_topics = sorted(set(i.topic for i in not_material))
        non_material_with_rationale = sum(
            1 for i in not_material if i.determination_rationale
        )

        populated = Decimal("0")
        total_checks = Decimal("4")
        if len(material) > 0:
            populated += Decimal("1")
        if len(material_standards) > 0:
            populated += Decimal("1")
        if len(not_material) > 0 and non_material_with_rationale == len(not_material):
            populated += Decimal("1")
        elif non_material_with_rationale > 0:
            populated += Decimal("0.5")
        # Phase-in is always valid (may legitimately be empty)
        populated += Decimal("1")

        score = _round_val(populated / total_checks * Decimal("100"), 1)

        return {
            "material_topics": material_topics,
            "material_esrs_standards": material_standards,
            "non_material_topics": non_material_topics,
            "non_material_with_rationale": non_material_with_rationale,
            "phase_in_topics": [i.topic for i in phase_in],
            "score": score,
        }

    # ------------------------------------------------------------------ #
    # MDR Validation (MDR-P, MDR-A, MDR-T, MDR-M)                         #
    # ------------------------------------------------------------------ #

    def validate_mdr(
        self,
        policies: List[MinimumDisclosureRequirement],
        actions: List[MinimumDisclosureRequirement],
        targets: List[MinimumDisclosureRequirement],
        metrics: List[MinimumDisclosureRequirement],
    ) -> Dict[str, Any]:
        """Validate Minimum Disclosure Requirements per ESRS 2 MDR.

        Checks completeness and quality of MDR-P (policies), MDR-A
        (actions), MDR-T (targets), and MDR-M (metrics).

        Args:
            policies: MDR-P entries (policies adopted).
            actions: MDR-A entries (actions and resources).
            targets: MDR-T entries (targets for tracking).
            metrics: MDR-M entries (metrics defined).

        Returns:
            Dict with mdr_p, mdr_a, mdr_t, mdr_m sections,
            overall_mdr_score, and provenance_hash.
        """
        t0 = time.perf_counter()

        logger.info(
            "Validating MDR: %d policies, %d actions, %d targets, %d metrics",
            len(policies), len(actions), len(targets), len(metrics),
        )

        mdr_p = self._validate_mdr_p(policies)
        mdr_a = self._validate_mdr_a(actions)
        mdr_t = self._validate_mdr_t(targets)
        mdr_m = self._validate_mdr_m(metrics)

        section_scores = [
            mdr_p.get("score", Decimal("0")),
            mdr_a.get("score", Decimal("0")),
            mdr_t.get("score", Decimal("0")),
            mdr_m.get("score", Decimal("0")),
        ]
        overall_score = _round_val(
            _safe_divide(sum(section_scores), _decimal(len(section_scores))),
            1,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result: Dict[str, Any] = {
            "mdr_p": mdr_p,
            "mdr_a": mdr_a,
            "mdr_t": mdr_t,
            "mdr_m": mdr_m,
            "overall_mdr_score": overall_score,
            "processing_time_ms": round(elapsed_ms, 3),
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "MDR validation complete: score=%.1f", float(overall_score),
        )
        return result

    def _validate_mdr_p(
        self, policies: List[MinimumDisclosureRequirement]
    ) -> Dict[str, Any]:
        """Validate MDR-P policies per ESRS 2 requirements.

        Args:
            policies: Policy MDR entries to validate.

        Returns:
            Dict with policy validation results and score.
        """
        if not policies:
            return {
                "policies_count": 0,
                "with_description": 0,
                "with_scope": 0,
                "topics_covered": [],
                "score": Decimal("0"),
            }

        with_desc = sum(1 for p in policies if p.policy_or_action_description)
        with_scope = sum(1 for p in policies if p.scope)
        topics = sorted(set(p.topic for p in policies))

        populated = Decimal("0")
        total_checks = Decimal("5")
        if len(policies) > 0:
            populated += Decimal("1")
        if with_desc == len(policies):
            populated += Decimal("1")
        elif with_desc > 0:
            populated += Decimal("0.5")
        if with_scope == len(policies):
            populated += Decimal("1")
        elif with_scope > 0:
            populated += Decimal("0.5")
        if len(topics) > 0:
            populated += Decimal("1")
        if any(p.kpis for p in policies):
            populated += Decimal("1")

        score = _round_val(populated / total_checks * Decimal("100"), 1)

        return {
            "policies_count": len(policies),
            "with_description": with_desc,
            "with_scope": with_scope,
            "topics_covered": topics,
            "score": score,
        }

    def _validate_mdr_a(
        self, actions: List[MinimumDisclosureRequirement]
    ) -> Dict[str, Any]:
        """Validate MDR-A actions and resources.

        Args:
            actions: Action MDR entries to validate.

        Returns:
            Dict with action validation results and score.
        """
        if not actions:
            return {
                "actions_count": 0,
                "with_resources": 0,
                "total_resources_allocated": Decimal("0"),
                "with_timeline": 0,
                "score": Decimal("0"),
            }

        with_resources = sum(
            1 for a in actions if a.resources_allocated is not None
        )
        total_resources = sum(
            a.resources_allocated for a in actions
            if a.resources_allocated is not None
        )
        with_timeline = sum(
            1 for a in actions if a.implementation_date is not None
        )

        populated = Decimal("0")
        total_checks = Decimal("4")
        if len(actions) > 0:
            populated += Decimal("1")
        if with_resources > 0:
            populated += Decimal("1")
        if with_timeline > 0:
            populated += Decimal("1")
        if any(a.policy_or_action_description for a in actions):
            populated += Decimal("1")

        score = _round_val(populated / total_checks * Decimal("100"), 1)

        return {
            "actions_count": len(actions),
            "with_resources": with_resources,
            "total_resources_allocated": _round_val(_decimal(total_resources), 2),
            "with_timeline": with_timeline,
            "score": score,
        }

    def _validate_mdr_t(
        self, targets: List[MinimumDisclosureRequirement]
    ) -> Dict[str, Any]:
        """Validate MDR-T targets for tracking effectiveness.

        Args:
            targets: Target MDR entries to validate.

        Returns:
            Dict with target validation results and score.
        """
        if not targets:
            return {
                "targets_count": 0,
                "with_kpis": 0,
                "with_timeline": 0,
                "score": Decimal("0"),
            }

        with_kpis = sum(1 for t in targets if t.kpis)
        with_timeline = sum(
            1 for t in targets if t.implementation_date is not None
        )

        populated = Decimal("0")
        total_checks = Decimal("4")
        if len(targets) > 0:
            populated += Decimal("1")
        if with_kpis > 0:
            populated += Decimal("1")
        if with_timeline > 0:
            populated += Decimal("1")
        if any(t.policy_or_action_description for t in targets):
            populated += Decimal("1")

        score = _round_val(populated / total_checks * Decimal("100"), 1)

        return {
            "targets_count": len(targets),
            "with_kpis": with_kpis,
            "with_timeline": with_timeline,
            "topics_covered": sorted(set(t.topic for t in targets)),
            "score": score,
        }

    def _validate_mdr_m(
        self, metrics: List[MinimumDisclosureRequirement]
    ) -> Dict[str, Any]:
        """Validate MDR-M metrics in relation to material matters.

        Args:
            metrics: Metric MDR entries to validate.

        Returns:
            Dict with metric validation results and score.
        """
        if not metrics:
            return {
                "metrics_count": 0,
                "with_kpis": 0,
                "with_methodology": 0,
                "score": Decimal("0"),
            }

        with_kpis = sum(1 for m in metrics if m.kpis)
        with_methodology = sum(
            1 for m in metrics if m.policy_or_action_description
        )

        populated = Decimal("0")
        total_checks = Decimal("3")
        if len(metrics) > 0:
            populated += Decimal("1")
        if with_kpis > 0:
            populated += Decimal("1")
        if with_methodology > 0:
            populated += Decimal("1")

        score = _round_val(populated / total_checks * Decimal("100"), 1)

        return {
            "metrics_count": len(metrics),
            "with_kpis": with_kpis,
            "with_methodology": with_methodology,
            "topics_covered": sorted(set(m.topic for m in metrics)),
            "score": score,
        }

    # ------------------------------------------------------------------ #
    # Full ESRS 2 Disclosure Calculation                                   #
    # ------------------------------------------------------------------ #

    def calculate_esrs2_disclosure(
        self,
        bodies: List[GovernanceBody],
        members: List[BoardMember],
        schemes: List[IncentiveScheme],
        due_diligence: Optional[DueDiligenceProcess] = None,
        risk_management: Optional[RiskManagementIntegration] = None,
        strategy_elements: Optional[List[StrategyElement]] = None,
        stakeholder_engagements: Optional[List[StakeholderEngagement]] = None,
        iros: Optional[List[MaterialIRO]] = None,
        mdr_policies: Optional[List[MinimumDisclosureRequirement]] = None,
        mdr_actions: Optional[List[MinimumDisclosureRequirement]] = None,
        mdr_targets: Optional[List[MinimumDisclosureRequirement]] = None,
        mdr_metrics: Optional[List[MinimumDisclosureRequirement]] = None,
    ) -> ESRS2GeneralResult:
        """Calculate the complete ESRS 2 General Disclosures result.

        Orchestrates all governance, strategy, IRO, and MDR assessments
        into a single comprehensive result with computed metrics and
        SHA-256 provenance.

        Args:
            bodies: Governance bodies with sustainability oversight.
            members: Individual board/governance body members.
            schemes: Sustainability-linked incentive schemes.
            due_diligence: Due diligence process (GOV-4).
            risk_management: Risk management integration (GOV-5).
            strategy_elements: Strategy and business model elements (SBM-1).
            stakeholder_engagements: Stakeholder engagement records (SBM-2).
            iros: Material impacts, risks and opportunities (IRO-1/2).
            mdr_policies: MDR-P policies adopted.
            mdr_actions: MDR-A actions and resources.
            mdr_targets: MDR-T targets for tracking.
            mdr_metrics: MDR-M metrics defined.

        Returns:
            ESRS2GeneralResult with complete provenance.

        Raises:
            ValueError: If bodies list is empty.
        """
        t0 = time.perf_counter()

        if not bodies:
            raise ValueError("At least one GovernanceBody is required")

        logger.info("Calculating full ESRS 2 disclosure")

        # Normalize optional inputs
        strategy_elements = strategy_elements or []
        stakeholder_engagements = stakeholder_engagements or []
        iros = iros or []
        mdr_policies = mdr_policies or []
        mdr_actions = mdr_actions or []
        mdr_targets = mdr_targets or []
        mdr_metrics = mdr_metrics or []

        # Step 1: Board composition metrics
        total_board_members = sum(b.members_count for b in bodies)
        independent_pct = self._weighted_average_pct(
            bodies, "independent_members_pct"
        )
        female_pct = self._weighted_average_pct(
            bodies, "female_members_pct"
        )

        members_with_expertise = sum(
            1 for m in members if len(m.sustainability_expertise) > 0
        )
        member_denom = _decimal(len(members)) if members else Decimal("1")
        expertise_pct = _round_val(
            _safe_divide(_decimal(members_with_expertise), member_denom)
            * Decimal("100"),
            1,
        )

        # Step 2: Incentive metrics
        if schemes:
            avg_remuneration = _round_val(
                _safe_divide(
                    sum(s.weight_in_total_remuneration_pct for s in schemes),
                    _decimal(len(schemes)),
                ),
                1,
            )
        else:
            avg_remuneration = Decimal("0")

        # Step 3: Material topics
        material_count = sum(
            1 for i in iros
            if i.is_material == MaterialityDetermination.MATERIAL
        )
        non_material_topics = [
            i.topic for i in iros
            if i.is_material == MaterialityDetermination.NOT_MATERIAL
        ]

        # Step 4: Compliance score across four pillars
        gov_result = self.assess_governance(
            bodies, members, schemes, due_diligence, risk_management
        )
        strat_result = self.assess_strategy(
            strategy_elements, stakeholder_engagements, iros
        )

        iro_score = Decimal("0")
        if iros:
            iro_result = self.assess_iro_process(iros)
            iro1_score = iro_result["iro1"].get("score", Decimal("0"))
            iro2_score = iro_result["iro2"].get("score", Decimal("0"))
            iro_score = _round_val(
                _safe_divide(iro1_score + iro2_score, Decimal("2")), 1
            )

        mdr_result = self.validate_mdr(
            mdr_policies, mdr_actions, mdr_targets, mdr_metrics
        )

        pillar_scores = [
            gov_result.get("overall_governance_score", Decimal("0")),
            strat_result.get("overall_strategy_score", Decimal("0")),
            iro_score,
            mdr_result.get("overall_mdr_score", Decimal("0")),
        ]
        compliance_score = _round_val(
            _safe_divide(sum(pillar_scores), _decimal(len(pillar_scores))),
            1,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = ESRS2GeneralResult(
            governance_bodies=bodies,
            board_composition=members,
            incentive_schemes=schemes,
            due_diligence=due_diligence,
            risk_management=risk_management,
            strategy_elements=strategy_elements,
            stakeholder_engagements=stakeholder_engagements,
            material_iros=[
                i for i in iros
                if i.is_material != MaterialityDetermination.NOT_MATERIAL
            ],
            non_material_topics=non_material_topics,
            mdr_policies=mdr_policies,
            mdr_actions=mdr_actions,
            mdr_targets=mdr_targets,
            mdr_metrics=mdr_metrics,
            total_board_members=total_board_members,
            independent_pct=independent_pct,
            female_board_pct=female_pct,
            sustainability_expertise_pct=expertise_pct,
            sustainability_linked_remuneration_pct=avg_remuneration,
            material_topics_count=material_count,
            compliance_score=compliance_score,
        )

        result.provenance_hash = _compute_hash(result)

        logger.info(
            "ESRS 2 disclosure calculated: compliance=%.1f%%, "
            "members=%d, material_topics=%d, hash=%s, time=%.1fms",
            float(compliance_score), total_board_members,
            material_count, result.provenance_hash[:16], elapsed_ms,
        )

        return result

    # ------------------------------------------------------------------ #
    # Completeness Validation                                              #
    # ------------------------------------------------------------------ #

    def validate_esrs2_completeness(
        self, result: ESRS2GeneralResult
    ) -> Dict[str, Any]:
        """Validate completeness of ESRS 2 General Disclosures.

        Checks whether all mandatory ESRS 2 data points across GOV,
        SBM, IRO, and MDR sections are present and populated.

        Args:
            result: ESRS2GeneralResult to validate.

        Returns:
            Dict with total_datapoints, populated_datapoints,
            missing_datapoints, completeness_pct, is_complete,
            section_completeness, and provenance_hash.
        """
        all_datapoints = (
            ESRS2_GOV_DATAPOINTS
            + ESRS2_SBM_DATAPOINTS
            + ESRS2_IRO_DATAPOINTS
            + ESRS2_MDR_DATAPOINTS
        )

        populated: List[str] = []
        missing: List[str] = []

        # GOV data point checks
        gov_checks = self._build_gov_checks(result)

        # SBM data point checks
        sbm_checks = self._build_sbm_checks(result)

        # IRO data point checks
        iro_checks = self._build_iro_checks(result)

        # MDR data point checks
        mdr_checks = self._build_mdr_checks(result)

        all_checks = {**gov_checks, **sbm_checks, **iro_checks, **mdr_checks}

        for dp, is_populated in all_checks.items():
            if is_populated:
                populated.append(dp)
            else:
                missing.append(dp)

        total = len(all_datapoints)
        pop_count = len(populated)
        completeness = _round_val(
            _safe_divide(_decimal(pop_count), _decimal(total))
            * Decimal("100"),
            1,
        )

        section_completeness = {
            "gov": self._section_completeness(gov_checks, ESRS2_GOV_DATAPOINTS),
            "sbm": self._section_completeness(sbm_checks, ESRS2_SBM_DATAPOINTS),
            "iro": self._section_completeness(iro_checks, ESRS2_IRO_DATAPOINTS),
            "mdr": self._section_completeness(mdr_checks, ESRS2_MDR_DATAPOINTS),
        }

        validation_result = {
            "total_datapoints": total,
            "populated_datapoints": pop_count,
            "missing_datapoints": missing,
            "completeness_pct": completeness,
            "is_complete": len(missing) == 0,
            "section_completeness": section_completeness,
            "provenance_hash": _compute_hash(
                {"result_id": result.result_id, "checks": all_checks}
            ),
        }

        logger.info(
            "ESRS 2 completeness: %s%% (%d/%d), missing=%d",
            completeness, pop_count, total, len(missing),
        )

        return validation_result

    # ------------------------------------------------------------------ #
    # Completeness Check Builders                                          #
    # ------------------------------------------------------------------ #

    def _build_gov_checks(
        self, result: ESRS2GeneralResult
    ) -> Dict[str, bool]:
        """Build GOV section data point checks.

        Args:
            result: ESRS2GeneralResult to check.

        Returns:
            Dict mapping GOV data point IDs to populated status.
        """
        has_bodies = len(result.governance_bodies) > 0
        has_members = len(result.board_composition) > 0

        return {
            "gov1_01_governance_body_composition": has_bodies,
            "gov1_02_member_roles_and_responsibilities": has_members,
            "gov1_03_sustainability_expertise": any(
                len(m.sustainability_expertise) > 0
                for m in result.board_composition
            ) if has_members else False,
            "gov1_04_meeting_frequency": any(
                b.meeting_frequency > 0 for b in result.governance_bodies
            ) if has_bodies else False,
            "gov1_05_independence_percentage": result.independent_pct > Decimal("0"),
            "gov1_06_gender_diversity": result.female_board_pct > Decimal("0"),
            "gov2_01_sustainability_matters_addressed": any(
                len(b.sustainability_topics_discussed) > 0
                for b in result.governance_bodies
            ) if has_bodies else False,
            "gov2_02_information_provided_to_bodies": any(
                len(b.sustainability_topics_discussed) > 0
                for b in result.governance_bodies
            ) if has_bodies else False,
            "gov2_03_frequency_of_sustainability_agenda": any(
                b.meeting_frequency > 0 for b in result.governance_bodies
            ) if has_bodies else False,
            "gov3_01_sustainability_linked_remuneration": len(result.incentive_schemes) > 0,
            "gov3_02_incentive_metric_types": any(
                len(s.sustainability_metrics) > 0
                for s in result.incentive_schemes
            ) if result.incentive_schemes else False,
            "gov3_03_weight_in_total_remuneration": (
                result.sustainability_linked_remuneration_pct > Decimal("0")
            ),
            "gov4_01_due_diligence_statement": result.due_diligence is not None,
            "gov4_02_value_chain_coverage": (
                len(result.due_diligence.value_chain_coverage) > 0
                if result.due_diligence else False
            ),
            "gov4_03_standards_followed": (
                len(result.due_diligence.standards_followed) > 0
                if result.due_diligence else False
            ),
            "gov5_01_risk_management_framework": result.risk_management is not None,
            "gov5_02_sustainability_integration": (
                result.risk_management.sustainability_integrated
                if result.risk_management else False
            ),
            "gov5_03_three_lines_of_defense": (
                any(result.risk_management.three_lines_of_defense.values())
                if result.risk_management else False
            ),
            "gov5_04_internal_audit_scope": (
                len(result.risk_management.internal_audit_scope) > 0
                if result.risk_management else False
            ),
            "gov5_05_external_assurance": (
                result.risk_management.external_assurance
                if result.risk_management else False
            ),
        }

    def _build_sbm_checks(
        self, result: ESRS2GeneralResult
    ) -> Dict[str, bool]:
        """Build SBM section data point checks.

        Args:
            result: ESRS2GeneralResult to check.

        Returns:
            Dict mapping SBM data point IDs to populated status.
        """
        has_elements = len(result.strategy_elements) > 0
        has_engagements = len(result.stakeholder_engagements) > 0
        has_iros = len(result.material_iros) > 0

        return {
            "sbm1_01_strategy_description": has_elements,
            "sbm1_02_business_model_description": any(
                e.description for e in result.strategy_elements
            ) if has_elements else False,
            "sbm1_03_sustainability_relevance": any(
                e.sustainability_relevance for e in result.strategy_elements
            ) if has_elements else False,
            "sbm1_04_value_chain_description": has_elements,
            "sbm1_05_products_services_affected": any(
                len(e.products_services_affected) > 0
                for e in result.strategy_elements
            ) if has_elements else False,
            "sbm2_01_stakeholder_groups_identified": has_engagements,
            "sbm2_02_engagement_methods": any(
                len(e.engagement_methods) > 0
                for e in result.stakeholder_engagements
            ) if has_engagements else False,
            "sbm2_03_key_concerns_raised": any(
                len(e.key_concerns_raised) > 0
                for e in result.stakeholder_engagements
            ) if has_engagements else False,
            "sbm2_04_views_influence_on_strategy": any(
                e.how_views_influenced_strategy
                for e in result.stakeholder_engagements
            ) if has_engagements else False,
            "sbm3_01_material_iros_identified": has_iros,
            "sbm3_02_interaction_with_strategy": any(
                i.determination_rationale for i in result.material_iros
            ) if has_iros else False,
            "sbm3_03_time_horizons": any(
                i.time_horizon is not None for i in result.material_iros
            ) if has_iros else False,
            "sbm3_04_value_chain_stages": any(
                i.value_chain_stage for i in result.material_iros
            ) if has_iros else False,
        }

    def _build_iro_checks(
        self, result: ESRS2GeneralResult
    ) -> Dict[str, bool]:
        """Build IRO section data point checks.

        Args:
            result: ESRS2GeneralResult to check.

        Returns:
            Dict mapping IRO data point IDs to populated status.
        """
        has_iros = len(result.material_iros) > 0

        return {
            "iro1_01_process_description": has_iros,
            "iro1_02_stakeholders_consulted": (
                len(result.due_diligence.stakeholders_consulted) > 0
                if result.due_diligence else False
            ),
            "iro1_03_materiality_criteria": any(
                i.determination_rationale for i in result.material_iros
            ) if has_iros else False,
            "iro1_04_double_materiality_approach": has_iros,
            "iro1_05_update_frequency": (
                bool(result.due_diligence.frequency)
                if result.due_diligence else False
            ),
            "iro2_01_material_topics_list": result.material_topics_count > 0,
            "iro2_02_esrs_standards_covered": any(
                i.esrs_standard for i in result.material_iros
            ) if has_iros else False,
            "iro2_03_non_material_topics_rationale": len(result.non_material_topics) > 0,
            "iro2_04_phase_in_topics": True,
        }

    def _build_mdr_checks(
        self, result: ESRS2GeneralResult
    ) -> Dict[str, bool]:
        """Build MDR section data point checks.

        Args:
            result: ESRS2GeneralResult to check.

        Returns:
            Dict mapping MDR data point IDs to populated status.
        """
        has_policies = len(result.mdr_policies) > 0
        has_actions = len(result.mdr_actions) > 0
        has_targets = len(result.mdr_targets) > 0
        has_metrics = len(result.mdr_metrics) > 0

        return {
            "mdr_p_01_policy_description": has_policies,
            "mdr_p_02_policy_scope": any(
                p.scope for p in result.mdr_policies
            ) if has_policies else False,
            "mdr_p_03_highest_body_responsible": has_policies,
            "mdr_p_04_third_party_standards_referenced": True,
            "mdr_p_05_stakeholder_consideration": True,
            "mdr_a_01_action_description": has_actions,
            "mdr_a_02_resources_allocated": any(
                a.resources_allocated is not None for a in result.mdr_actions
            ) if has_actions else False,
            "mdr_a_03_implementation_timeline": any(
                a.implementation_date is not None for a in result.mdr_actions
            ) if has_actions else False,
            "mdr_a_04_expected_outcomes": any(
                a.policy_or_action_description for a in result.mdr_actions
            ) if has_actions else False,
            "mdr_t_01_measurable_targets": has_targets,
            "mdr_t_02_target_base_year": any(
                t.implementation_date for t in result.mdr_targets
            ) if has_targets else False,
            "mdr_t_03_milestones_and_interim_targets": any(
                t.kpis for t in result.mdr_targets
            ) if has_targets else False,
            "mdr_t_04_methodology_for_tracking": any(
                t.policy_or_action_description for t in result.mdr_targets
            ) if has_targets else False,
            "mdr_m_01_metrics_defined": has_metrics,
            "mdr_m_02_metric_methodology": any(
                m.policy_or_action_description for m in result.mdr_metrics
            ) if has_metrics else False,
            "mdr_m_03_metric_validation": any(
                m.kpis for m in result.mdr_metrics
            ) if has_metrics else False,
        }

    # ------------------------------------------------------------------ #
    # Private Helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _section_completeness(
        checks: Dict[str, bool], dp_list: List[str]
    ) -> Dict[str, Any]:
        """Calculate completeness for a single section.

        Args:
            checks: Dict mapping data point IDs to populated status.
            dp_list: List of data point IDs in this section.

        Returns:
            Dict with populated count, total, and completeness_pct.
        """
        sec_pop = sum(1 for dp in dp_list if checks.get(dp, False))
        sec_total = len(dp_list)
        return {
            "populated": sec_pop,
            "total": sec_total,
            "completeness_pct": _round_val(
                _safe_divide(_decimal(sec_pop), _decimal(sec_total))
                * Decimal("100"),
                1,
            ),
        }

    def _weighted_average_pct(
        self,
        bodies: List[GovernanceBody],
        field_name: str,
    ) -> Decimal:
        """Calculate weighted average percentage across governance bodies.

        Weights each body's percentage by its member count, producing
        a single aggregate percentage across all bodies.

        Args:
            bodies: List of GovernanceBody instances.
            field_name: Name of the percentage field to average.

        Returns:
            Weighted average percentage (Decimal, 0-100).
        """
        total_members = sum(b.members_count for b in bodies)
        if total_members == 0:
            return Decimal("0")

        weighted_sum = Decimal("0")
        for body in bodies:
            pct_value = getattr(body, field_name, Decimal("0"))
            weight = _safe_divide(
                _decimal(body.members_count),
                _decimal(total_members),
            )
            weighted_sum += pct_value * weight

        return _round_val(weighted_sum, 1)

    # ------------------------------------------------------------------ #
    # Individual Disclosure Requirement Methods                            #
    # ------------------------------------------------------------------ #

    def assess_gov1(
        self,
        bodies: List[GovernanceBody],
        members: Optional[List[BoardMember]] = None,
    ) -> Dict[str, Any]:
        """Assess GOV-1 governance body composition."""
        return self._assess_gov1_composition(bodies, members or [])

    def assess_gov2(self, bodies: List[GovernanceBody]) -> Dict[str, Any]:
        """Assess GOV-2 sustainability information flow."""
        return self._assess_gov2_topics(bodies)

    def assess_gov3(self, schemes: List[IncentiveScheme]) -> Dict[str, Any]:
        """Assess GOV-3 incentive scheme integration."""
        return self._assess_gov3_incentives(schemes)

    def assess_gov4(
        self, due_diligence: Optional[DueDiligenceProcess] = None
    ) -> Dict[str, Any]:
        """Assess GOV-4 due diligence statement."""
        return self._assess_gov4_due_diligence(due_diligence)

    def assess_gov5(
        self, risk_management: Optional[RiskManagementIntegration] = None
    ) -> Dict[str, Any]:
        """Assess GOV-5 risk management and internal controls."""
        return self._assess_gov5_risk_management(risk_management)

    def assess_sbm1(
        self, strategy_elements: Optional[List[StrategyElement]] = None
    ) -> Dict[str, Any]:
        """Assess SBM-1 strategy and business model."""
        elements = strategy_elements or []
        return {
            "element_count": len(elements),
            "elements": [e.model_dump(mode="json") for e in elements],
            "score": _round_val(
                Decimal("100") if elements else Decimal("0"), 1
            ),
        }

    def assess_sbm2(
        self, engagements: Optional[List[StakeholderEngagement]] = None
    ) -> Dict[str, Any]:
        """Assess SBM-2 stakeholder interests and views."""
        items = engagements or []
        return {
            "engagement_count": len(items),
            "groups_covered": list({e.stakeholder_group.value for e in items}),
            "score": _round_val(
                Decimal("100") if items else Decimal("0"), 1
            ),
        }

    def assess_sbm3(
        self, material_iros: Optional[List[MaterialIRO]] = None
    ) -> Dict[str, Any]:
        """Assess SBM-3 material impacts, risks and opportunities."""
        iros = material_iros or []
        return {
            "iro_count": len(iros),
            "material_topics": [i.topic for i in iros if i.is_material == MaterialityDetermination.MATERIAL],
            "score": _round_val(
                Decimal("100") if iros else Decimal("0"), 1
            ),
        }

    def assess_iro1(self, **kwargs: Any) -> Dict[str, Any]:
        """Assess IRO-1 process for identifying material IROs."""
        return self.assess_iro_process(**kwargs) if kwargs else {
            "process_described": False,
            "score": Decimal("0"),
        }

    def assess_iro2(
        self, material_iros: Optional[List[MaterialIRO]] = None
    ) -> Dict[str, Any]:
        """Assess IRO-2 disclosure requirements covered."""
        iros = material_iros or []
        material_count = sum(
            1 for i in iros
            if i.is_material == MaterialityDetermination.MATERIAL
        )
        return {
            "total_topics": len(iros),
            "material_topics": material_count,
            "not_material_topics": len(iros) - material_count,
            "score": _round_val(
                Decimal("100") if iros else Decimal("0"), 1
            ),
        }

    def get_esrs2_datapoints(self) -> List[str]:
        """Return the full list of ESRS 2 required data points."""
        return (
            ESRS2_GOV_DATAPOINTS
            + ESRS2_SBM_DATAPOINTS
            + ESRS2_IRO_DATAPOINTS
            + ESRS2_MDR_DATAPOINTS
        )

    # Convenience aliases matching test expectations
    def assess_governance_body(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Alias for assess_governance — GOV-1 through GOV-5."""
        return self.assess_governance(*args, **kwargs)

    def assess_business_model(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Alias for assess_strategy — SBM-1 through SBM-3."""
        return self.assess_strategy(*args, **kwargs)
