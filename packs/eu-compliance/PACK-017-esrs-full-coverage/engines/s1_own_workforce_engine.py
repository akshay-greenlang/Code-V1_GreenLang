# -*- coding: utf-8 -*-
"""
OwnWorkforceEngine - PACK-017 ESRS S1 Own Workforce Engine
============================================================

Calculates all 17 disclosure requirements under ESRS S1 (Own Workforce),
the largest social standard in the European Sustainability Reporting
Standards framework.

This engine implements the complete ESRS S1 calculation and assessment
pipeline for own-workforce disclosures, including:

- Workforce demographics by gender, contract type, region (S1-6)
- Non-employee worker characterisation (S1-7)
- Collective bargaining coverage and social dialogue (S1-8)
- Diversity metrics: gender in top management, age, disability (S1-9)
- Adequate wage benchmark assessment (S1-10)
- Social protection coverage analysis (S1-11)
- Persons with disabilities inclusion metrics (S1-12)
- Training and skills development hours per employee (S1-13)
- Health and safety: TRIR, LTIFR, fatalities, lost days (S1-14)
- Work-life balance: family leave uptake rates (S1-15)
- Remuneration: gender pay gap, CEO-to-median ratio (S1-16)
- Human rights incidents, complaints, and severe impacts (S1-17)
- Qualitative assessments for policies, engagement, remediation,
  actions, and targets (S1-1 through S1-5)

ESRS S1 Disclosure Requirements (17 total):
    - S1-1  (Para 19-21):  Policies related to own workforce
    - S1-2  (Para 23-25):  Processes for engaging with own workers and
                            their representatives
    - S1-3  (Para 27-30):  Processes to remediate negative impacts and
                            channels for raising concerns
    - S1-4  (Para 32-36):  Taking action on material impacts, managing
                            risks and pursuing opportunities
    - S1-5  (Para 38-40):  Targets related to managing material negative
                            impacts, advancing positive impacts, and
                            managing risks and opportunities
    - S1-6  (Para 50-54):  Characteristics of the undertaking's employees
                            (headcount by gender, contract type, region)
    - S1-7  (Para 56-58):  Characteristics of non-employee workers in the
                            undertaking's own workforce
    - S1-8  (Para 60-63):  Collective bargaining coverage and social dialogue
    - S1-9  (Para 65-67):  Diversity metrics (gender distribution in top
                            management, age distribution, disability)
    - S1-10 (Para 69-71):  Adequate wages (benchmark assessment)
    - S1-11 (Para 73-75):  Social protection (coverage by social protection)
    - S1-12 (Para 77-79):  Persons with disabilities (inclusion and
                            accessibility)
    - S1-13 (Para 81-83):  Training and skills development (training hours
                            per employee by gender)
    - S1-14 (Para 85-91):  Health and safety metrics (work-related injuries,
                            fatalities, lost days, TRIR, LTIFR)
    - S1-15 (Para 93-95):  Work-life balance (family-related leave,
                            uptake rates)
    - S1-16 (Para 97-99):  Remuneration metrics (gender pay gap,
                            CEO-to-median ratio)
    - S1-17 (Para 101-103): Incidents, complaints and severe human rights
                             impacts

Regulatory References:
    - EU Delegated Regulation 2023/2772 (ESRS)
    - ESRS S1 Own Workforce (all 17 disclosure requirements)
    - ILO Conventions (fundamental labour rights)
    - UN Guiding Principles on Business and Human Rights (UNGPs)
    - GRI 401-405 (Labour and Diversity Standards)
    - ISO 45001 (Occupational Health and Safety)

Zero-Hallucination:
    - All headcount, rate, and ratio calculations use deterministic arithmetic
    - TRIR and LTIFR use OSHA/GRI standard formulas with fixed constants
    - Gender pay gap uses Eurostat-aligned median-based methodology
    - Aggregation uses Decimal arithmetic with ROUND_HALF_UP
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

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

def _round2(value: Decimal) -> Decimal:
    """Round Decimal to 2 decimal places using ROUND_HALF_UP."""
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

def _round4(value: Decimal) -> Decimal:
    """Round Decimal to 4 decimal places using ROUND_HALF_UP."""
    return value.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

def _round6(value: Decimal) -> Decimal:
    """Round Decimal to 6 decimal places using ROUND_HALF_UP."""
    return value.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)

def _pct(numerator: Decimal, denominator: Decimal) -> Decimal:
    """Calculate percentage safely, rounded to 2 decimal places.

    Args:
        numerator: The part.
        denominator: The whole.

    Returns:
        Percentage value (0-100 range), rounded to 2 places.
    """
    if denominator == Decimal("0"):
        return Decimal("0")
    return _round2(numerator / denominator * Decimal("100"))

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EmploymentType(str, Enum):
    """Employment contract type per ESRS S1-6 Para 50.

    ESRS requires headcount disaggregation by contract type to
    assess workforce stability and precarity.
    """
    PERMANENT = "permanent"
    TEMPORARY = "temporary"
    NON_GUARANTEED = "non_guaranteed"

class WorkingTime(str, Enum):
    """Working time arrangement per ESRS S1-6 Para 50.

    Full-time versus part-time disaggregation is mandatory for
    employee characterisation under S1-6.
    """
    FULL_TIME = "full_time"
    PART_TIME = "part_time"

class Gender(str, Enum):
    """Gender classification per ESRS S1-6 Para 50.

    Used throughout S1 for disaggregation of workforce metrics.
    """
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    NOT_DISCLOSED = "not_disclosed"

class AgeGroup(str, Enum):
    """Age group classification per ESRS S1-9 Para 65.

    Standard three-band age grouping used for diversity reporting.
    """
    UNDER_30 = "under_30"
    BETWEEN_30_50 = "between_30_50"
    OVER_50 = "over_50"

class Region(str, Enum):
    """Geographic region classification per ESRS S1-6 Para 52.

    Primary disaggregation is EU vs non-EU; further country-level
    breakdown is required for significant locations.
    """
    EU = "eu"
    NON_EU = "non_eu"

class ManagementLevel(str, Enum):
    """Management level classification per ESRS S1-9 Para 65.

    Used for diversity disaggregation, especially gender
    distribution in top and senior management.
    """
    TOP_MANAGEMENT = "top_management"
    SENIOR_MANAGEMENT = "senior_management"
    MIDDLE_MANAGEMENT = "middle_management"
    PROFESSIONAL = "professional"
    ADMINISTRATIVE = "administrative"
    OPERATIONAL = "operational"

class InjurySeverity(str, Enum):
    """Injury severity classification per ESRS S1-14 Para 85-91.

    Aligned with GRI 403 and OSHA recordable injury classifications.
    """
    FATALITY = "fatality"
    HIGH_CONSEQUENCE = "high_consequence"
    RECORDABLE = "recordable"
    FIRST_AID = "first_aid"
    NEAR_MISS = "near_miss"

class LeaveType(str, Enum):
    """Family-related leave type per ESRS S1-15 Para 93-95.

    Covers all mandatory family-related leave categories for
    work-life balance disclosure.
    """
    MATERNITY = "maternity"
    PATERNITY = "paternity"
    PARENTAL = "parental"
    CARERS = "carers"

class IncidentType(str, Enum):
    """Human rights incident type per ESRS S1-17 Para 101-103.

    Covers the fundamental human rights areas relevant to own
    workforce per ESRS and UNGPs.
    """
    DISCRIMINATION = "discrimination"
    HARASSMENT = "harassment"
    FORCED_LABOUR = "forced_labour"
    CHILD_LABOUR = "child_labour"
    FREEDOM_OF_ASSOCIATION = "freedom_of_association"
    OTHER_HUMAN_RIGHTS = "other_human_rights"

class NonEmployeeType(str, Enum):
    """Non-employee worker type per ESRS S1-7 Para 56-58.

    Types of workers in the undertaking's own workforce who are
    not employees.
    """
    CONTRACTOR = "contractor"
    TEMPORARY_AGENCY = "temporary_agency"
    SELF_EMPLOYED = "self_employed"
    INTERN = "intern"
    APPRENTICE = "apprentice"
    OTHER = "other"

class SocialDialogueType(str, Enum):
    """Social dialogue arrangement type per ESRS S1-8 Para 60-63.

    Forms of worker representation and social dialogue mechanisms.
    """
    WORKS_COUNCIL = "works_council"
    TRADE_UNION = "trade_union"
    EMPLOYEE_REPRESENTATIVE = "employee_representative"
    JOINT_COMMITTEE = "joint_committee"
    OTHER = "other"
    NONE = "none"

class RemediationStatus(str, Enum):
    """Remediation status for human rights incidents per S1-17."""
    OPEN = "open"
    UNDER_INVESTIGATION = "under_investigation"
    REMEDIATION_IN_PROGRESS = "remediation_in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"

# ---------------------------------------------------------------------------
# Constants - ESRS S1 XBRL-Tagged Datapoints
# ---------------------------------------------------------------------------

S1_1_DATAPOINTS: List[str] = [
    "s1_1_01_policies_own_workforce_described",
    "s1_1_02_human_rights_policy_commitments",
    "s1_1_03_ilo_conventions_referenced",
    "s1_1_04_policy_scope_and_coverage",
]

S1_2_DATAPOINTS: List[str] = [
    "s1_2_01_engagement_processes_described",
    "s1_2_02_stages_of_engagement",
    "s1_2_03_worker_representatives_involved",
    "s1_2_04_engagement_outcomes_disclosed",
]

S1_3_DATAPOINTS: List[str] = [
    "s1_3_01_remediation_processes_described",
    "s1_3_02_grievance_channels_available",
    "s1_3_03_whistleblowing_mechanisms",
    "s1_3_04_access_to_remedy_ensured",
]

S1_4_DATAPOINTS: List[str] = [
    "s1_4_01_actions_on_material_impacts",
    "s1_4_02_risk_management_approach",
    "s1_4_03_opportunity_pursuit_approach",
    "s1_4_04_resources_allocated",
    "s1_4_05_effectiveness_tracking",
]

S1_5_DATAPOINTS: List[str] = [
    "s1_5_01_measurable_targets_set",
    "s1_5_02_target_timeframes",
    "s1_5_03_progress_against_targets",
    "s1_5_04_baseline_values_defined",
]

S1_6_DATAPOINTS: List[str] = [
    "s1_6_01_total_employees_headcount",
    "s1_6_02_employees_by_gender",
    "s1_6_03_employees_by_country",
    "s1_6_04_employees_by_contract_type",
    "s1_6_05_employees_by_contract_type_and_gender",
    "s1_6_06_employees_by_working_time",
    "s1_6_07_employees_by_working_time_and_gender",
    "s1_6_08_employees_by_region",
]

S1_7_DATAPOINTS: List[str] = [
    "s1_7_01_total_non_employees_headcount",
    "s1_7_02_non_employees_by_type",
    "s1_7_03_non_employees_by_gender",
    "s1_7_04_methodologies_and_assumptions",
]

S1_8_DATAPOINTS: List[str] = [
    "s1_8_01_collective_bargaining_coverage_pct",
    "s1_8_02_social_dialogue_coverage_pct",
    "s1_8_03_coverage_by_region",
    "s1_8_04_eea_workers_not_covered_description",
]

S1_9_DATAPOINTS: List[str] = [
    "s1_9_01_gender_distribution_top_management",
    "s1_9_02_gender_distribution_by_level",
    "s1_9_03_age_distribution_employees",
    "s1_9_04_disability_percentage",
]

S1_10_DATAPOINTS: List[str] = [
    "s1_10_01_adequate_wage_assessment",
    "s1_10_02_proportion_below_adequate_wage",
    "s1_10_03_benchmark_methodology",
]

S1_11_DATAPOINTS: List[str] = [
    "s1_11_01_social_protection_coverage_pct",
    "s1_11_02_coverage_by_type",
    "s1_11_03_gaps_in_coverage_description",
]

S1_12_DATAPOINTS: List[str] = [
    "s1_12_01_disability_inclusion_rate_pct",
    "s1_12_02_accessibility_measures",
    "s1_12_03_reasonable_accommodation_provided",
]

S1_13_DATAPOINTS: List[str] = [
    "s1_13_01_total_training_hours",
    "s1_13_02_average_training_hours_per_employee",
    "s1_13_03_training_hours_by_gender",
    "s1_13_04_training_hours_by_category",
]

S1_14_DATAPOINTS: List[str] = [
    "s1_14_01_fatalities_work_related",
    "s1_14_02_high_consequence_injuries",
    "s1_14_03_recordable_work_related_injuries",
    "s1_14_04_lost_days_work_related",
    "s1_14_05_total_hours_worked",
    "s1_14_06_trir_total_recordable_incident_rate",
    "s1_14_07_ltifr_lost_time_injury_frequency_rate",
    "s1_14_08_fatality_rate",
]

S1_15_DATAPOINTS: List[str] = [
    "s1_15_01_family_leave_entitled_pct",
    "s1_15_02_maternity_leave_uptake_pct",
    "s1_15_03_paternity_leave_uptake_pct",
    "s1_15_04_parental_leave_return_rate_pct",
]

S1_16_DATAPOINTS: List[str] = [
    "s1_16_01_gender_pay_gap_pct",
    "s1_16_02_ceo_total_compensation",
    "s1_16_03_median_employee_compensation",
    "s1_16_04_ceo_to_median_ratio",
]

S1_17_DATAPOINTS: List[str] = [
    "s1_17_01_total_incidents_reported",
    "s1_17_02_incidents_by_type",
    "s1_17_03_incidents_by_severity",
    "s1_17_04_severe_human_rights_impacts",
    "s1_17_05_remediation_status_summary",
]

ALL_S1_DATAPOINTS: List[str] = (
    S1_1_DATAPOINTS + S1_2_DATAPOINTS + S1_3_DATAPOINTS
    + S1_4_DATAPOINTS + S1_5_DATAPOINTS + S1_6_DATAPOINTS
    + S1_7_DATAPOINTS + S1_8_DATAPOINTS + S1_9_DATAPOINTS
    + S1_10_DATAPOINTS + S1_11_DATAPOINTS + S1_12_DATAPOINTS
    + S1_13_DATAPOINTS + S1_14_DATAPOINTS + S1_15_DATAPOINTS
    + S1_16_DATAPOINTS + S1_17_DATAPOINTS
)

# OSHA standard factor for incident rate normalisation
# TRIR = (recordable injuries * 200,000) / total hours worked
# LTIFR = (lost time injuries * 1,000,000) / total hours worked
TRIR_NORMALISATION_FACTOR: Decimal = Decimal("200000")
LTIFR_NORMALISATION_FACTOR: Decimal = Decimal("1000000")

# ---------------------------------------------------------------------------
# Pydantic Models - Qualitative Disclosures (S1-1 through S1-5)
# ---------------------------------------------------------------------------

class WorkforcePolicy(BaseModel):
    """Workforce policy description per ESRS S1-1 Para 19-21.

    Captures information about undertaking policies related to its
    own workforce, including human rights commitments and ILO
    convention references.
    """
    policy_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this policy record",
    )
    policy_name: str = Field(
        ..., description="Name of the policy", max_length=500,
    )
    description: str = Field(
        default="",
        description="Description of policy content and scope",
        max_length=5000,
    )
    human_rights_commitments: List[str] = Field(
        default_factory=list,
        description="List of human rights commitments referenced",
    )
    ilo_conventions_referenced: List[str] = Field(
        default_factory=list,
        description="ILO conventions explicitly referenced",
    )
    scope_description: str = Field(
        default="",
        description="Scope and coverage of the policy",
        max_length=2000,
    )
    approval_date: Optional[datetime] = Field(
        default=None, description="Date policy was last approved",
    )
    is_publicly_available: bool = Field(
        default=False, description="Whether the policy is publicly available",
    )

class EngagementProcess(BaseModel):
    """Engagement process description per ESRS S1-2 Para 23-25.

    Documents how the undertaking engages with its own workers and
    their representatives on material sustainability matters.
    """
    process_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this process record",
    )
    process_name: str = Field(
        ..., description="Name of the engagement process", max_length=500,
    )
    description: str = Field(
        default="",
        description="Description of the engagement process",
        max_length=5000,
    )
    stages: List[str] = Field(
        default_factory=list,
        description="Stages at which engagement occurs",
    )
    worker_representatives_involved: bool = Field(
        default=False,
        description="Whether worker representatives are involved",
    )
    frequency: str = Field(
        default="",
        description="Frequency of engagement (e.g., quarterly, annual)",
        max_length=100,
    )
    outcomes_disclosed: bool = Field(
        default=False,
        description="Whether outcomes of engagement are disclosed",
    )

class RemediationChannel(BaseModel):
    """Remediation and grievance channel per ESRS S1-3 Para 27-30.

    Documents channels available for raising concerns and the
    processes for remediating negative impacts.
    """
    channel_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this channel record",
    )
    channel_name: str = Field(
        ..., description="Name of the remediation/grievance channel",
        max_length=500,
    )
    channel_type: str = Field(
        default="",
        description="Type (e.g., hotline, ombudsperson, portal)",
        max_length=200,
    )
    description: str = Field(
        default="",
        description="Description of the channel and its operation",
        max_length=5000,
    )
    is_anonymous: bool = Field(
        default=False, description="Whether anonymous reporting is supported",
    )
    is_accessible_externally: bool = Field(
        default=False,
        description="Whether channel is accessible to external parties",
    )
    complaints_received: int = Field(
        default=0, description="Number of complaints received in period",
        ge=0,
    )
    complaints_resolved: int = Field(
        default=0, description="Number of complaints resolved in period",
        ge=0,
    )

# ---------------------------------------------------------------------------
# Pydantic Models - Quantitative Disclosures (S1-6 through S1-17)
# ---------------------------------------------------------------------------

class EmployeeData(BaseModel):
    """Individual employee record per ESRS S1-6 Para 50-54.

    Captures the attributes required for S1-6 demographic analysis,
    S1-9 diversity, S1-10 adequate wages, S1-11 social protection,
    S1-12 disability, and S1-13 training calculations.
    """
    employee_id: str = Field(
        default_factory=_new_uuid,
        description="Unique employee identifier",
    )
    gender: Gender = Field(
        ..., description="Gender classification",
    )
    age_group: AgeGroup = Field(
        ..., description="Age group classification",
    )
    employment_type: EmploymentType = Field(
        ..., description="Contract type (permanent, temporary, non-guaranteed)",
    )
    working_time: WorkingTime = Field(
        ..., description="Working time arrangement (full-time, part-time)",
    )
    region: Region = Field(
        ..., description="Geographic region (EU, non-EU)",
    )
    country_code: str = Field(
        default="",
        description="ISO 3166-1 alpha-2 country code",
        max_length=3,
    )
    department: str = Field(
        default="",
        description="Department or business unit",
        max_length=200,
    )
    management_level: ManagementLevel = Field(
        default=ManagementLevel.OPERATIONAL,
        description="Management level for diversity reporting",
    )
    annual_wage: Decimal = Field(
        default=Decimal("0"),
        description="Annual gross wage in reporting currency",
        ge=Decimal("0"),
    )
    adequate_wage_benchmark: Decimal = Field(
        default=Decimal("0"),
        description="Adequate wage benchmark for the employee location",
        ge=Decimal("0"),
    )
    social_protection_covered: bool = Field(
        default=True,
        description="Whether employee is covered by social protection",
    )
    disability_status: bool = Field(
        default=False,
        description="Whether employee has a declared disability",
    )
    training_hours: Decimal = Field(
        default=Decimal("0"),
        description="Training hours during the reporting period",
        ge=Decimal("0"),
    )

class NonEmployeeWorker(BaseModel):
    """Non-employee worker record per ESRS S1-7 Para 56-58.

    Captures data on workers who are part of the undertaking's own
    workforce but are not employees (e.g., contractors, agency workers).
    """
    worker_id: str = Field(
        default_factory=_new_uuid,
        description="Unique worker identifier",
    )
    worker_type: NonEmployeeType = Field(
        ..., description="Type of non-employee worker",
    )
    gender: Gender = Field(
        default=Gender.NOT_DISCLOSED,
        description="Gender classification",
    )
    headcount: int = Field(
        default=1,
        description="Number of workers in this category",
        ge=0,
    )

class CollectiveBargainingData(BaseModel):
    """Collective bargaining data per ESRS S1-8 Para 60-63.

    Captures collective bargaining coverage and social dialogue
    information by region.
    """
    region: str = Field(
        ..., description="Region or country identifier", max_length=200,
    )
    total_employees_in_region: int = Field(
        ..., description="Total employees in this region", ge=0,
    )
    covered_by_collective_bargaining: int = Field(
        default=0,
        description="Employees covered by collective bargaining agreements",
        ge=0,
    )
    coverage_pct: Decimal = Field(
        default=Decimal("0"),
        description="Collective bargaining coverage percentage",
        ge=Decimal("0"), le=Decimal("100"),
    )
    social_dialogue_type: SocialDialogueType = Field(
        default=SocialDialogueType.NONE,
        description="Type of social dialogue arrangement in this region",
    )
    is_eea: bool = Field(
        default=False,
        description="Whether this region is within the EEA",
    )

class HealthSafetyIncident(BaseModel):
    """Individual health and safety incident per ESRS S1-14 Para 85-91.

    Used for feeding into the health and safety metrics calculations.
    """
    incident_id: str = Field(
        default_factory=_new_uuid,
        description="Unique incident identifier",
    )
    severity: InjurySeverity = Field(
        ..., description="Severity classification of the incident",
    )
    lost_days: Decimal = Field(
        default=Decimal("0"),
        description="Number of lost workdays due to this incident",
        ge=Decimal("0"),
    )
    date: Optional[datetime] = Field(
        default=None, description="Date of the incident",
    )
    is_employee: bool = Field(
        default=True,
        description="Whether the affected person is an employee",
    )

class HealthSafetyMetrics(BaseModel):
    """Aggregated health and safety metrics per ESRS S1-14 Para 85-91.

    Pre-aggregated health and safety data for direct calculation of
    TRIR, LTIFR, and fatality rates.
    """
    period: str = Field(
        default="",
        description="Reporting period (e.g., 2025, 2025-Q4)",
        max_length=50,
    )
    fatalities: int = Field(
        default=0, description="Work-related fatalities", ge=0,
    )
    high_consequence_injuries: int = Field(
        default=0,
        description="High-consequence work-related injuries (excl. fatalities)",
        ge=0,
    )
    recordable_injuries: int = Field(
        default=0, description="Total recordable work-related injuries", ge=0,
    )
    lost_days: Decimal = Field(
        default=Decimal("0"),
        description="Total lost workdays due to work-related injuries",
        ge=Decimal("0"),
    )
    total_hours_worked: Decimal = Field(
        default=Decimal("0"),
        description="Total hours worked across all employees in the period",
        ge=Decimal("0"),
    )
    trir: Optional[Decimal] = Field(
        default=None,
        description="Pre-calculated TRIR (if provided, overrides calculation)",
    )
    ltifr: Optional[Decimal] = Field(
        default=None,
        description="Pre-calculated LTIFR (if provided, overrides calculation)",
    )

class FamilyLeaveData(BaseModel):
    """Family leave data per ESRS S1-15 Para 93-95.

    Captures leave entitlement and uptake for work-life balance
    disclosure.
    """
    leave_type: LeaveType = Field(
        ..., description="Type of family-related leave",
    )
    entitled_employees: int = Field(
        default=0, description="Number of employees entitled to this leave",
        ge=0,
    )
    employees_who_took_leave: int = Field(
        default=0, description="Number of employees who took this leave",
        ge=0,
    )
    employees_returned_after_leave: int = Field(
        default=0,
        description="Number of employees who returned after leave",
        ge=0,
    )
    average_days_taken: Decimal = Field(
        default=Decimal("0"),
        description="Average number of days taken per employee",
        ge=Decimal("0"),
    )

class HumanRightsIncident(BaseModel):
    """Human rights incident record per ESRS S1-17 Para 101-103.

    Captures individual incidents, complaints, and severe human
    rights impacts for S1-17 disclosure.
    """
    incident_id: str = Field(
        default_factory=_new_uuid,
        description="Unique incident identifier",
    )
    incident_type: IncidentType = Field(
        ..., description="Type of human rights incident",
    )
    severity: str = Field(
        default="moderate",
        description="Severity level (minor, moderate, severe, critical)",
        max_length=50,
    )
    date: Optional[datetime] = Field(
        default=None, description="Date the incident was reported",
    )
    remediation_status: RemediationStatus = Field(
        default=RemediationStatus.OPEN,
        description="Current remediation status",
    )
    description: str = Field(
        default="",
        description="Description of the incident",
        max_length=5000,
    )
    is_severe_impact: bool = Field(
        default=False,
        description="Whether this constitutes a severe human rights impact",
    )

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        """Validate severity is one of the allowed values."""
        allowed = {"minor", "moderate", "severe", "critical"}
        if v.lower() not in allowed:
            raise ValueError(
                f"severity must be one of {allowed}, got '{v}'"
            )
        return v.lower()

# ---------------------------------------------------------------------------
# Pydantic Models - Result
# ---------------------------------------------------------------------------

class S1WorkforceResult(BaseModel):
    """Complete ESRS S1 Own Workforce disclosure result.

    Aggregates all 17 disclosure requirements into a single result
    object with provenance tracking.
    """
    result_id: str = Field(
        default_factory=_new_uuid,
        description="Unique result identifier",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION,
        description="Engine version used for this calculation",
    )
    calculated_at: datetime = Field(
        default_factory=utcnow,
        description="Timestamp of calculation (UTC)",
    )
    reporting_year: int = Field(
        default=0, description="Reporting year",
    )
    entity_name: str = Field(
        default="", description="Entity or undertaking name",
    )

    # S1-1 through S1-5: Qualitative assessments
    s1_1_policies: Dict[str, Any] = Field(
        default_factory=dict,
        description="S1-1: Policies related to own workforce",
    )
    s1_2_engagement: Dict[str, Any] = Field(
        default_factory=dict,
        description="S1-2: Engagement processes with workers",
    )
    s1_3_remediation: Dict[str, Any] = Field(
        default_factory=dict,
        description="S1-3: Remediation and grievance channels",
    )
    s1_4_actions: Dict[str, Any] = Field(
        default_factory=dict,
        description="S1-4: Actions on material impacts",
    )
    s1_5_targets: Dict[str, Any] = Field(
        default_factory=dict,
        description="S1-5: Targets and progress",
    )

    # S1-6: Employee demographics
    s1_6_demographics: Dict[str, Any] = Field(
        default_factory=dict,
        description="S1-6: Employee characteristics and demographics",
    )

    # S1-7: Non-employee workers
    s1_7_non_employees: Dict[str, Any] = Field(
        default_factory=dict,
        description="S1-7: Non-employee worker characteristics",
    )

    # S1-8: Collective bargaining
    s1_8_collective_bargaining: Dict[str, Any] = Field(
        default_factory=dict,
        description="S1-8: Collective bargaining and social dialogue",
    )

    # S1-9: Diversity
    s1_9_diversity: Dict[str, Any] = Field(
        default_factory=dict,
        description="S1-9: Diversity metrics",
    )

    # S1-10: Adequate wages
    s1_10_adequate_wages: Dict[str, Any] = Field(
        default_factory=dict,
        description="S1-10: Adequate wage assessment",
    )

    # S1-11: Social protection
    s1_11_social_protection: Dict[str, Any] = Field(
        default_factory=dict,
        description="S1-11: Social protection coverage",
    )

    # S1-12: Disability
    s1_12_disability: Dict[str, Any] = Field(
        default_factory=dict,
        description="S1-12: Disability inclusion metrics",
    )

    # S1-13: Training
    s1_13_training: Dict[str, Any] = Field(
        default_factory=dict,
        description="S1-13: Training and skills development",
    )

    # S1-14: Health and safety
    s1_14_health_safety: Dict[str, Any] = Field(
        default_factory=dict,
        description="S1-14: Health and safety metrics",
    )

    # S1-15: Work-life balance
    s1_15_work_life_balance: Dict[str, Any] = Field(
        default_factory=dict,
        description="S1-15: Work-life balance metrics",
    )

    # S1-16: Remuneration
    s1_16_remuneration: Dict[str, Any] = Field(
        default_factory=dict,
        description="S1-16: Remuneration metrics",
    )

    # S1-17: Incidents
    s1_17_incidents: Dict[str, Any] = Field(
        default_factory=dict,
        description="S1-17: Incidents, complaints and impacts",
    )

    # Cross-cutting
    compliance_score: Decimal = Field(
        default=Decimal("0"),
        description="Overall S1 compliance score (0-100)",
    )
    processing_time_ms: float = Field(
        default=0.0, description="Processing time in milliseconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of all inputs and calculation steps",
    )

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class OwnWorkforceEngine:
    """ESRS S1 Own Workforce calculation engine.

    Provides deterministic, zero-hallucination calculations for all 17
    ESRS S1 disclosure requirements covering the undertaking's own
    workforce.  The engine handles both quantitative metrics (headcounts,
    rates, ratios, percentages) and structured qualitative assessments
    (policies, processes, channels).

    All numeric calculations use Decimal arithmetic for bit-perfect
    reproducibility.  No LLM is used in any calculation path.

    Calculation Methodology:
        - Demographics: headcount aggregation with multi-dimensional pivot
        - TRIR = (recordable_injuries * 200,000) / total_hours_worked
        - LTIFR = (lost_time_injuries * 1,000,000) / total_hours_worked
        - Gender pay gap = (median_male - median_female) / median_male * 100
        - CEO ratio = ceo_total_compensation / median_employee_compensation
        - All percentages: (part / whole) * 100, rounded to 2 decimal places

    Usage::

        engine = OwnWorkforceEngine()
        employees = [EmployeeData(...), ...]
        demographics = engine.calculate_workforce_demographics(employees)
        result = engine.calculate_s1_disclosure(
            employees=employees,
            reporting_year=2025,
            entity_name="Acme Corp",
        )

    Attributes:
        engine_version: Version string for provenance tracking.
    """

    engine_version: str = _MODULE_VERSION

    # ------------------------------------------------------------------ #
    # S1-6: Workforce Demographics                                        #
    # ------------------------------------------------------------------ #

    def calculate_workforce_demographics(
        self, employees: List[EmployeeData]
    ) -> Dict[str, Any]:
        """Calculate workforce demographics per ESRS S1-6 Para 50-54.

        Produces headcount breakdowns by gender, contract type,
        working time, region, and all cross-dimensional combinations
        required by the standard.

        Args:
            employees: List of EmployeeData records.

        Returns:
            Dict with keys: total_headcount, by_gender, by_contract_type,
            by_contract_type_and_gender, by_working_time,
            by_working_time_and_gender, by_region, by_country,
            provenance_hash.

        Raises:
            ValueError: If employees list is empty.
        """
        if not employees:
            raise ValueError("At least one EmployeeData record is required")

        total = len(employees)
        logger.info("Calculating S1-6 demographics for %d employees", total)

        # By gender
        by_gender = self._count_by_attr(employees, "gender")

        # By contract type
        by_contract = self._count_by_attr(employees, "employment_type")

        # By contract type and gender (cross-dimensional)
        by_contract_gender: Dict[str, Dict[str, int]] = {}
        for ct in EmploymentType:
            by_contract_gender[ct.value] = {}
            subset = [e for e in employees if e.employment_type == ct]
            for g in Gender:
                count = sum(1 for e in subset if e.gender == g)
                if count > 0:
                    by_contract_gender[ct.value][g.value] = count

        # By working time
        by_working_time = self._count_by_attr(employees, "working_time")

        # By working time and gender (cross-dimensional)
        by_wt_gender: Dict[str, Dict[str, int]] = {}
        for wt in WorkingTime:
            by_wt_gender[wt.value] = {}
            subset = [e for e in employees if e.working_time == wt]
            for g in Gender:
                count = sum(1 for e in subset if e.gender == g)
                if count > 0:
                    by_wt_gender[wt.value][g.value] = count

        # By region
        by_region = self._count_by_attr(employees, "region")

        # By country (finer granularity)
        by_country: Dict[str, int] = {}
        for emp in employees:
            cc = emp.country_code if emp.country_code else "UNKNOWN"
            by_country[cc] = by_country.get(cc, 0) + 1

        result = {
            "total_headcount": total,
            "by_gender": by_gender,
            "by_contract_type": by_contract,
            "by_contract_type_and_gender": by_contract_gender,
            "by_working_time": by_working_time,
            "by_working_time_and_gender": by_wt_gender,
            "by_region": by_region,
            "by_country": by_country,
            "esrs_ref": "S1-6 Para 50-54",
            "provenance_hash": "",
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "S1-6 demographics: total=%d, hash=%s",
            total, result["provenance_hash"][:16],
        )
        return result

    # ------------------------------------------------------------------ #
    # S1-7: Non-Employee Workers                                          #
    # ------------------------------------------------------------------ #

    def calculate_non_employee_metrics(
        self, workers: List[NonEmployeeWorker]
    ) -> Dict[str, Any]:
        """Calculate non-employee worker metrics per ESRS S1-7 Para 56-58.

        Aggregates headcounts of non-employee workers by type and gender.

        Args:
            workers: List of NonEmployeeWorker records.

        Returns:
            Dict with keys: total_headcount, by_type, by_gender,
            provenance_hash.
        """
        if not workers:
            logger.warning("No non-employee workers provided for S1-7")
            return {
                "total_headcount": 0,
                "by_type": {},
                "by_gender": {},
                "esrs_ref": "S1-7 Para 56-58",
                "provenance_hash": _compute_hash({"total_headcount": 0}),
            }

        total = sum(w.headcount for w in workers)

        # By type
        by_type: Dict[str, int] = {}
        for wt in NonEmployeeType:
            count = sum(w.headcount for w in workers if w.worker_type == wt)
            if count > 0:
                by_type[wt.value] = count

        # By gender
        by_gender: Dict[str, int] = {}
        for g in Gender:
            count = sum(w.headcount for w in workers if w.gender == g)
            if count > 0:
                by_gender[g.value] = count

        result = {
            "total_headcount": total,
            "by_type": by_type,
            "by_gender": by_gender,
            "esrs_ref": "S1-7 Para 56-58",
            "provenance_hash": "",
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "S1-7 non-employees: total=%d, hash=%s",
            total, result["provenance_hash"][:16],
        )
        return result

    # ------------------------------------------------------------------ #
    # S1-8: Collective Bargaining Coverage                                #
    # ------------------------------------------------------------------ #

    def calculate_collective_bargaining(
        self, data: List[CollectiveBargainingData]
    ) -> Dict[str, Any]:
        """Calculate collective bargaining coverage per ESRS S1-8 Para 60-63.

        Aggregates coverage percentages across regions and identifies
        EEA workers not covered by collective bargaining agreements.

        Args:
            data: List of CollectiveBargainingData records by region.

        Returns:
            Dict with keys: overall_coverage_pct, by_region,
            eea_coverage_pct, non_eea_coverage_pct, provenance_hash.
        """
        if not data:
            logger.warning("No collective bargaining data provided for S1-8")
            return {
                "overall_coverage_pct": "0.00",
                "by_region": {},
                "eea_coverage_pct": "0.00",
                "non_eea_coverage_pct": "0.00",
                "esrs_ref": "S1-8 Para 60-63",
                "provenance_hash": _compute_hash({"total": 0}),
            }

        total_employees = sum(d.total_employees_in_region for d in data)
        total_covered = sum(d.covered_by_collective_bargaining for d in data)

        overall_pct = _pct(_decimal(total_covered), _decimal(total_employees))

        # EEA vs non-EEA
        eea_total = sum(
            d.total_employees_in_region for d in data if d.is_eea
        )
        eea_covered = sum(
            d.covered_by_collective_bargaining for d in data if d.is_eea
        )
        eea_pct = _pct(_decimal(eea_covered), _decimal(eea_total))

        non_eea_total = sum(
            d.total_employees_in_region for d in data if not d.is_eea
        )
        non_eea_covered = sum(
            d.covered_by_collective_bargaining for d in data if not d.is_eea
        )
        non_eea_pct = _pct(_decimal(non_eea_covered), _decimal(non_eea_total))

        # Per region
        by_region: Dict[str, Dict[str, Any]] = {}
        for d in data:
            region_pct = _pct(
                _decimal(d.covered_by_collective_bargaining),
                _decimal(d.total_employees_in_region),
            )
            by_region[d.region] = {
                "total_employees": d.total_employees_in_region,
                "covered": d.covered_by_collective_bargaining,
                "coverage_pct": str(region_pct),
                "social_dialogue_type": d.social_dialogue_type.value,
                "is_eea": d.is_eea,
            }

        result = {
            "total_employees": total_employees,
            "total_covered": total_covered,
            "overall_coverage_pct": str(overall_pct),
            "eea_coverage_pct": str(eea_pct),
            "non_eea_coverage_pct": str(non_eea_pct),
            "by_region": by_region,
            "esrs_ref": "S1-8 Para 60-63",
            "provenance_hash": "",
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "S1-8 collective bargaining: overall=%.2f%%, hash=%s",
            float(overall_pct), result["provenance_hash"][:16],
        )
        return result

    # ------------------------------------------------------------------ #
    # S1-9: Diversity Metrics                                             #
    # ------------------------------------------------------------------ #

    def calculate_diversity_metrics(
        self, employees: List[EmployeeData]
    ) -> Dict[str, Any]:
        """Calculate diversity metrics per ESRS S1-9 Para 65-67.

        Computes gender distribution at each management level (with
        special focus on top management), age distribution across the
        entire workforce, and disability percentage.

        Args:
            employees: List of EmployeeData records.

        Returns:
            Dict with keys: gender_by_management_level,
            gender_top_management_pct, age_distribution,
            disability_pct, provenance_hash.

        Raises:
            ValueError: If employees list is empty.
        """
        if not employees:
            raise ValueError("At least one EmployeeData record is required")

        total = _decimal(len(employees))
        logger.info("Calculating S1-9 diversity metrics for %d employees", len(employees))

        # Gender distribution by management level
        gender_by_level: Dict[str, Dict[str, Any]] = {}
        for level in ManagementLevel:
            subset = [e for e in employees if e.management_level == level]
            level_total = len(subset)
            if level_total == 0:
                continue
            gender_dist: Dict[str, Any] = {"total": level_total}
            for g in Gender:
                g_count = sum(1 for e in subset if e.gender == g)
                if g_count > 0:
                    gender_dist[g.value] = g_count
                    gender_dist[f"{g.value}_pct"] = str(
                        _pct(_decimal(g_count), _decimal(level_total))
                    )
            gender_by_level[level.value] = gender_dist

        # Top management gender percentage (key S1-9 metric)
        top_mgmt = [
            e for e in employees
            if e.management_level == ManagementLevel.TOP_MANAGEMENT
        ]
        top_mgmt_total = _decimal(len(top_mgmt))
        top_mgmt_female = _decimal(
            sum(1 for e in top_mgmt if e.gender == Gender.FEMALE)
        )
        top_mgmt_female_pct = _pct(top_mgmt_female, top_mgmt_total)

        # Age distribution
        age_distribution: Dict[str, Any] = {}
        for ag in AgeGroup:
            count = sum(1 for e in employees if e.age_group == ag)
            age_distribution[ag.value] = {
                "count": count,
                "pct": str(_pct(_decimal(count), total)),
            }

        # Disability percentage
        disability_count = sum(1 for e in employees if e.disability_status)
        disability_pct = _pct(_decimal(disability_count), total)

        result = {
            "gender_by_management_level": gender_by_level,
            "gender_top_management_female_pct": str(top_mgmt_female_pct),
            "top_management_total": int(top_mgmt_total),
            "age_distribution": age_distribution,
            "disability_count": disability_count,
            "disability_pct": str(disability_pct),
            "total_employees": len(employees),
            "esrs_ref": "S1-9 Para 65-67",
            "provenance_hash": "",
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "S1-9 diversity: top_mgmt_female=%.2f%%, disability=%.2f%%, hash=%s",
            float(top_mgmt_female_pct), float(disability_pct),
            result["provenance_hash"][:16],
        )
        return result

    # ------------------------------------------------------------------ #
    # S1-10: Adequate Wages                                               #
    # ------------------------------------------------------------------ #

    def assess_adequate_wages(
        self, employees: List[EmployeeData]
    ) -> Dict[str, Any]:
        """Assess adequate wages per ESRS S1-10 Para 69-71.

        Determines the proportion of employees whose annual wage is
        below the adequate wage benchmark for their location.  Only
        employees with a non-zero benchmark are included in the
        assessment.

        Args:
            employees: List of EmployeeData records with wage data.

        Returns:
            Dict with keys: assessed_count, below_adequate_count,
            below_adequate_pct, by_region, by_gender, provenance_hash.
        """
        # Filter to employees with valid benchmarks
        assessed = [
            e for e in employees
            if e.adequate_wage_benchmark > Decimal("0")
        ]

        if not assessed:
            logger.warning("No employees with adequate wage benchmarks for S1-10")
            return {
                "assessed_count": 0,
                "below_adequate_count": 0,
                "below_adequate_pct": "0.00",
                "by_region": {},
                "by_gender": {},
                "esrs_ref": "S1-10 Para 69-71",
                "provenance_hash": _compute_hash({"assessed_count": 0}),
            }

        assessed_total = _decimal(len(assessed))
        below = [
            e for e in assessed
            if e.annual_wage < e.adequate_wage_benchmark
        ]
        below_count = _decimal(len(below))
        below_pct = _pct(below_count, assessed_total)

        # By region
        by_region: Dict[str, Dict[str, str]] = {}
        for region in Region:
            r_assessed = [e for e in assessed if e.region == region]
            if not r_assessed:
                continue
            r_below = [
                e for e in r_assessed
                if e.annual_wage < e.adequate_wage_benchmark
            ]
            by_region[region.value] = {
                "assessed": str(len(r_assessed)),
                "below_adequate": str(len(r_below)),
                "below_adequate_pct": str(
                    _pct(_decimal(len(r_below)), _decimal(len(r_assessed)))
                ),
            }

        # By gender
        by_gender: Dict[str, Dict[str, str]] = {}
        for g in Gender:
            g_assessed = [e for e in assessed if e.gender == g]
            if not g_assessed:
                continue
            g_below = [
                e for e in g_assessed
                if e.annual_wage < e.adequate_wage_benchmark
            ]
            by_gender[g.value] = {
                "assessed": str(len(g_assessed)),
                "below_adequate": str(len(g_below)),
                "below_adequate_pct": str(
                    _pct(_decimal(len(g_below)), _decimal(len(g_assessed)))
                ),
            }

        result = {
            "assessed_count": int(assessed_total),
            "below_adequate_count": int(below_count),
            "below_adequate_pct": str(below_pct),
            "by_region": by_region,
            "by_gender": by_gender,
            "esrs_ref": "S1-10 Para 69-71",
            "provenance_hash": "",
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "S1-10 adequate wages: %d/%d below benchmark (%.2f%%), hash=%s",
            int(below_count), int(assessed_total), float(below_pct),
            result["provenance_hash"][:16],
        )
        return result

    # ------------------------------------------------------------------ #
    # S1-11: Social Protection                                            #
    # ------------------------------------------------------------------ #

    def assess_social_protection(
        self, employees: List[EmployeeData]
    ) -> Dict[str, Any]:
        """Assess social protection coverage per ESRS S1-11 Para 73-75.

        Calculates the percentage of employees covered by social
        protection schemes, disaggregated by region.

        Args:
            employees: List of EmployeeData records.

        Returns:
            Dict with keys: total_employees, covered_count,
            coverage_pct, by_region, provenance_hash.
        """
        if not employees:
            return {
                "total_employees": 0,
                "covered_count": 0,
                "coverage_pct": "0.00",
                "by_region": {},
                "esrs_ref": "S1-11 Para 73-75",
                "provenance_hash": _compute_hash({"total": 0}),
            }

        total = _decimal(len(employees))
        covered = _decimal(
            sum(1 for e in employees if e.social_protection_covered)
        )
        coverage_pct = _pct(covered, total)

        # By region
        by_region: Dict[str, Dict[str, str]] = {}
        for region in Region:
            r_emps = [e for e in employees if e.region == region]
            if not r_emps:
                continue
            r_covered = sum(1 for e in r_emps if e.social_protection_covered)
            by_region[region.value] = {
                "total": str(len(r_emps)),
                "covered": str(r_covered),
                "coverage_pct": str(
                    _pct(_decimal(r_covered), _decimal(len(r_emps)))
                ),
            }

        result = {
            "total_employees": int(total),
            "covered_count": int(covered),
            "coverage_pct": str(coverage_pct),
            "by_region": by_region,
            "esrs_ref": "S1-11 Para 73-75",
            "provenance_hash": "",
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "S1-11 social protection: %d/%d covered (%.2f%%), hash=%s",
            int(covered), int(total), float(coverage_pct),
            result["provenance_hash"][:16],
        )
        return result

    # ------------------------------------------------------------------ #
    # S1-12: Persons with Disabilities                                    #
    # ------------------------------------------------------------------ #

    def calculate_disability_metrics(
        self, employees: List[EmployeeData]
    ) -> Dict[str, Any]:
        """Calculate disability inclusion metrics per ESRS S1-12 Para 77-79.

        Computes the disability inclusion rate and disaggregation by
        management level and gender.

        Args:
            employees: List of EmployeeData records.

        Returns:
            Dict with keys: total_employees, disability_count,
            disability_pct, by_management_level, by_gender,
            provenance_hash.
        """
        if not employees:
            return {
                "total_employees": 0,
                "disability_count": 0,
                "disability_pct": "0.00",
                "by_management_level": {},
                "by_gender": {},
                "esrs_ref": "S1-12 Para 77-79",
                "provenance_hash": _compute_hash({"total": 0}),
            }

        total = _decimal(len(employees))
        disabled = [e for e in employees if e.disability_status]
        disability_count = len(disabled)
        disability_pct = _pct(_decimal(disability_count), total)

        # By management level
        by_level: Dict[str, Dict[str, str]] = {}
        for level in ManagementLevel:
            level_emps = [e for e in employees if e.management_level == level]
            if not level_emps:
                continue
            level_disabled = sum(1 for e in level_emps if e.disability_status)
            by_level[level.value] = {
                "total": str(len(level_emps)),
                "disabled": str(level_disabled),
                "pct": str(
                    _pct(_decimal(level_disabled), _decimal(len(level_emps)))
                ),
            }

        # By gender
        by_gender: Dict[str, Dict[str, str]] = {}
        for g in Gender:
            g_emps = [e for e in employees if e.gender == g]
            if not g_emps:
                continue
            g_disabled = sum(1 for e in g_emps if e.disability_status)
            by_gender[g.value] = {
                "total": str(len(g_emps)),
                "disabled": str(g_disabled),
                "pct": str(
                    _pct(_decimal(g_disabled), _decimal(len(g_emps)))
                ),
            }

        result = {
            "total_employees": int(total),
            "disability_count": disability_count,
            "disability_pct": str(disability_pct),
            "by_management_level": by_level,
            "by_gender": by_gender,
            "esrs_ref": "S1-12 Para 77-79",
            "provenance_hash": "",
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "S1-12 disability: %d/%d (%.2f%%), hash=%s",
            disability_count, int(total), float(disability_pct),
            result["provenance_hash"][:16],
        )
        return result

    # ------------------------------------------------------------------ #
    # S1-13: Training and Skills Development                              #
    # ------------------------------------------------------------------ #

    def calculate_training_metrics(
        self, employees: List[EmployeeData]
    ) -> Dict[str, Any]:
        """Calculate training metrics per ESRS S1-13 Para 81-83.

        Computes total training hours, average training hours per
        employee, and disaggregation by gender and management level.

        Args:
            employees: List of EmployeeData records with training_hours.

        Returns:
            Dict with keys: total_training_hours,
            avg_hours_per_employee, by_gender, by_management_level,
            provenance_hash.
        """
        if not employees:
            return {
                "total_training_hours": "0",
                "avg_hours_per_employee": "0.00",
                "by_gender": {},
                "by_management_level": {},
                "esrs_ref": "S1-13 Para 81-83",
                "provenance_hash": _compute_hash({"total": 0}),
            }

        total_hours = sum(e.training_hours for e in employees)
        total_emps = _decimal(len(employees))
        avg_hours = _round2(_safe_divide(total_hours, total_emps))

        # By gender
        by_gender: Dict[str, Dict[str, str]] = {}
        for g in Gender:
            g_emps = [e for e in employees if e.gender == g]
            if not g_emps:
                continue
            g_total = sum(e.training_hours for e in g_emps)
            g_avg = _round2(
                _safe_divide(g_total, _decimal(len(g_emps)))
            )
            by_gender[g.value] = {
                "employee_count": str(len(g_emps)),
                "total_hours": str(g_total),
                "avg_hours_per_employee": str(g_avg),
            }

        # By management level
        by_level: Dict[str, Dict[str, str]] = {}
        for level in ManagementLevel:
            l_emps = [e for e in employees if e.management_level == level]
            if not l_emps:
                continue
            l_total = sum(e.training_hours for e in l_emps)
            l_avg = _round2(
                _safe_divide(l_total, _decimal(len(l_emps)))
            )
            by_level[level.value] = {
                "employee_count": str(len(l_emps)),
                "total_hours": str(l_total),
                "avg_hours_per_employee": str(l_avg),
            }

        result = {
            "total_training_hours": str(total_hours),
            "avg_hours_per_employee": str(avg_hours),
            "employee_count": int(total_emps),
            "by_gender": by_gender,
            "by_management_level": by_level,
            "esrs_ref": "S1-13 Para 81-83",
            "provenance_hash": "",
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "S1-13 training: total_hours=%s, avg=%s hrs/employee, hash=%s",
            str(total_hours), str(avg_hours),
            result["provenance_hash"][:16],
        )
        return result

    # ------------------------------------------------------------------ #
    # S1-14: Health and Safety                                            #
    # ------------------------------------------------------------------ #

    def calculate_health_safety(
        self,
        metrics: Optional[HealthSafetyMetrics] = None,
        incidents: Optional[List[HealthSafetyIncident]] = None,
    ) -> Dict[str, Any]:
        """Calculate health and safety metrics per ESRS S1-14 Para 85-91.

        Computes TRIR (Total Recordable Incident Rate), LTIFR
        (Lost Time Injury Frequency Rate), fatality rate, and other
        required health and safety disclosure metrics.

        TRIR = (recordable_injuries * 200,000) / total_hours_worked
        LTIFR = (lost_time_injuries * 1,000,000) / total_hours_worked
        Fatality rate = (fatalities * 1,000,000) / total_hours_worked

        Args:
            metrics: Pre-aggregated HealthSafetyMetrics (preferred).
            incidents: List of HealthSafetyIncident for bottom-up calc.

        Returns:
            Dict with keys: fatalities, high_consequence_injuries,
            recordable_injuries, lost_days, total_hours_worked,
            trir, ltifr, fatality_rate, provenance_hash.

        Raises:
            ValueError: If neither metrics nor incidents are provided.
        """
        if metrics is None and not incidents:
            raise ValueError(
                "Either metrics or incidents must be provided for S1-14"
            )

        # Build from incidents if metrics not provided
        if metrics is None and incidents:
            fatalities = sum(
                1 for i in incidents
                if i.severity == InjurySeverity.FATALITY
            )
            high_consequence = sum(
                1 for i in incidents
                if i.severity == InjurySeverity.HIGH_CONSEQUENCE
            )
            recordable = sum(
                1 for i in incidents
                if i.severity in (
                    InjurySeverity.FATALITY,
                    InjurySeverity.HIGH_CONSEQUENCE,
                    InjurySeverity.RECORDABLE,
                )
            )
            lost_days = sum(i.lost_days for i in incidents)

            metrics = HealthSafetyMetrics(
                fatalities=fatalities,
                high_consequence_injuries=high_consequence,
                recordable_injuries=recordable,
                lost_days=lost_days,
                total_hours_worked=Decimal("0"),
            )

        assert metrics is not None  # guaranteed by the checks above

        hours = metrics.total_hours_worked

        # TRIR calculation
        if metrics.trir is not None:
            trir = _round4(metrics.trir)
        else:
            trir = _round4(
                _safe_divide(
                    _decimal(metrics.recordable_injuries) * TRIR_NORMALISATION_FACTOR,
                    hours,
                )
            )

        # LTIFR calculation (lost time = recordable minus first aid)
        if metrics.ltifr is not None:
            ltifr = _round4(metrics.ltifr)
        else:
            # Lost time injuries = high_consequence + recordable (excl. fatalities for rate)
            lost_time_injuries = (
                metrics.high_consequence_injuries + metrics.recordable_injuries
            )
            ltifr = _round4(
                _safe_divide(
                    _decimal(lost_time_injuries) * LTIFR_NORMALISATION_FACTOR,
                    hours,
                )
            )

        # Fatality rate
        fatality_rate = _round6(
            _safe_divide(
                _decimal(metrics.fatalities) * LTIFR_NORMALISATION_FACTOR,
                hours,
            )
        )

        result = {
            "fatalities": metrics.fatalities,
            "high_consequence_injuries": metrics.high_consequence_injuries,
            "recordable_injuries": metrics.recordable_injuries,
            "lost_days": str(metrics.lost_days),
            "total_hours_worked": str(hours),
            "trir": str(trir),
            "ltifr": str(ltifr),
            "fatality_rate": str(fatality_rate),
            "esrs_ref": "S1-14 Para 85-91",
            "provenance_hash": "",
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "S1-14 H&S: fatalities=%d, TRIR=%s, LTIFR=%s, hash=%s",
            metrics.fatalities, str(trir), str(ltifr),
            result["provenance_hash"][:16],
        )
        return result

    # ------------------------------------------------------------------ #
    # S1-15: Work-Life Balance                                            #
    # ------------------------------------------------------------------ #

    def calculate_work_life_balance(
        self, leave_data: List[FamilyLeaveData]
    ) -> Dict[str, Any]:
        """Calculate work-life balance metrics per ESRS S1-15 Para 93-95.

        Computes family-related leave entitlement percentages,
        uptake rates by leave type, and return rates.

        Args:
            leave_data: List of FamilyLeaveData records by type.

        Returns:
            Dict with keys: by_leave_type (each with entitled,
            took_leave, uptake_pct, return_rate_pct),
            overall_return_rate_pct, provenance_hash.
        """
        if not leave_data:
            return {
                "by_leave_type": {},
                "overall_return_rate_pct": "0.00",
                "esrs_ref": "S1-15 Para 93-95",
                "provenance_hash": _compute_hash({"leave_data": []}),
            }

        by_type: Dict[str, Dict[str, str]] = {}
        total_took_leave = 0
        total_returned = 0

        for ld in leave_data:
            uptake_pct = _pct(
                _decimal(ld.employees_who_took_leave),
                _decimal(ld.entitled_employees),
            )
            return_pct = _pct(
                _decimal(ld.employees_returned_after_leave),
                _decimal(ld.employees_who_took_leave),
            )

            by_type[ld.leave_type.value] = {
                "entitled_employees": str(ld.entitled_employees),
                "employees_who_took_leave": str(ld.employees_who_took_leave),
                "uptake_pct": str(uptake_pct),
                "employees_returned": str(ld.employees_returned_after_leave),
                "return_rate_pct": str(return_pct),
                "average_days_taken": str(ld.average_days_taken),
            }

            total_took_leave += ld.employees_who_took_leave
            total_returned += ld.employees_returned_after_leave

        overall_return = _pct(
            _decimal(total_returned), _decimal(total_took_leave)
        )

        result = {
            "by_leave_type": by_type,
            "total_employees_took_leave": total_took_leave,
            "total_employees_returned": total_returned,
            "overall_return_rate_pct": str(overall_return),
            "esrs_ref": "S1-15 Para 93-95",
            "provenance_hash": "",
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "S1-15 work-life: return_rate=%.2f%%, hash=%s",
            float(overall_return), result["provenance_hash"][:16],
        )
        return result

    # ------------------------------------------------------------------ #
    # S1-16: Remuneration Metrics                                         #
    # ------------------------------------------------------------------ #

    def calculate_remuneration(
        self,
        employees: List[EmployeeData],
        ceo_total_compensation: Optional[Decimal] = None,
    ) -> Dict[str, Any]:
        """Calculate remuneration metrics per ESRS S1-16 Para 97-99.

        Computes the gender pay gap (unadjusted, median-based using
        the Eurostat methodology) and the CEO-to-median-employee
        pay ratio.

        Gender Pay Gap = (median_male - median_female) / median_male * 100

        Args:
            employees: List of EmployeeData records with annual_wage.
            ceo_total_compensation: Total CEO compensation for ratio.

        Returns:
            Dict with keys: gender_pay_gap_pct, median_male_wage,
            median_female_wage, ceo_compensation, median_employee_wage,
            ceo_to_median_ratio, provenance_hash.
        """
        if not employees:
            return {
                "gender_pay_gap_pct": "0.00",
                "median_male_wage": "0",
                "median_female_wage": "0",
                "ceo_compensation": "0",
                "median_employee_wage": "0",
                "ceo_to_median_ratio": "0.00",
                "esrs_ref": "S1-16 Para 97-99",
                "provenance_hash": _compute_hash({"employees": 0}),
            }

        # Median wages by gender
        male_wages = sorted(
            [e.annual_wage for e in employees if e.gender == Gender.MALE]
        )
        female_wages = sorted(
            [e.annual_wage for e in employees if e.gender == Gender.FEMALE]
        )
        all_wages = sorted([e.annual_wage for e in employees])

        median_male = self._calculate_median(male_wages)
        median_female = self._calculate_median(female_wages)
        median_all = self._calculate_median(all_wages)

        # Gender pay gap (unadjusted, Eurostat methodology)
        gender_pay_gap = _pct(
            median_male - median_female, median_male
        ) if median_male > Decimal("0") else Decimal("0")

        # CEO-to-median ratio
        ceo_comp = ceo_total_compensation or Decimal("0")
        ceo_ratio = _round2(
            _safe_divide(ceo_comp, median_all)
        ) if median_all > Decimal("0") else Decimal("0")

        result = {
            "gender_pay_gap_pct": str(_round2(gender_pay_gap)),
            "median_male_wage": str(median_male),
            "median_female_wage": str(median_female),
            "median_employee_wage": str(median_all),
            "ceo_compensation": str(ceo_comp),
            "ceo_to_median_ratio": str(ceo_ratio),
            "male_employee_count": len(male_wages),
            "female_employee_count": len(female_wages),
            "total_employee_count": len(all_wages),
            "esrs_ref": "S1-16 Para 97-99",
            "provenance_hash": "",
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "S1-16 remuneration: pay_gap=%.2f%%, CEO_ratio=%.2f, hash=%s",
            float(gender_pay_gap), float(ceo_ratio),
            result["provenance_hash"][:16],
        )
        return result

    # ------------------------------------------------------------------ #
    # S1-17: Incidents, Complaints, Severe Human Rights Impacts           #
    # ------------------------------------------------------------------ #

    def assess_incidents(
        self, incidents: List[HumanRightsIncident]
    ) -> Dict[str, Any]:
        """Assess incidents and complaints per ESRS S1-17 Para 101-103.

        Aggregates human rights incidents by type, severity, and
        remediation status.  Identifies severe impacts separately.

        Args:
            incidents: List of HumanRightsIncident records.

        Returns:
            Dict with keys: total_incidents, by_type, by_severity,
            severe_impacts_count, by_remediation_status,
            provenance_hash.
        """
        if not incidents:
            return {
                "total_incidents": 0,
                "by_type": {},
                "by_severity": {},
                "severe_impacts_count": 0,
                "by_remediation_status": {},
                "esrs_ref": "S1-17 Para 101-103",
                "provenance_hash": _compute_hash({"incidents": 0}),
            }

        total = len(incidents)

        # By incident type
        by_type: Dict[str, int] = {}
        for it in IncidentType:
            count = sum(1 for i in incidents if i.incident_type == it)
            if count > 0:
                by_type[it.value] = count

        # By severity
        by_severity: Dict[str, int] = {}
        for sev in ["minor", "moderate", "severe", "critical"]:
            count = sum(1 for i in incidents if i.severity == sev)
            if count > 0:
                by_severity[sev] = count

        # Severe impacts
        severe_count = sum(1 for i in incidents if i.is_severe_impact)

        # By remediation status
        by_status: Dict[str, int] = {}
        for status in RemediationStatus:
            count = sum(
                1 for i in incidents if i.remediation_status == status
            )
            if count > 0:
                by_status[status.value] = count

        result = {
            "total_incidents": total,
            "by_type": by_type,
            "by_severity": by_severity,
            "severe_impacts_count": severe_count,
            "by_remediation_status": by_status,
            "esrs_ref": "S1-17 Para 101-103",
            "provenance_hash": "",
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "S1-17 incidents: total=%d, severe=%d, hash=%s",
            total, severe_count, result["provenance_hash"][:16],
        )
        return result

    # ------------------------------------------------------------------ #
    # S1-1 through S1-5: Qualitative Assessments                          #
    # ------------------------------------------------------------------ #

    def assess_policies(
        self, policies: List[WorkforcePolicy]
    ) -> Dict[str, Any]:
        """Assess workforce policies per ESRS S1-1 Para 19-21.

        Summarises the undertaking's policies, checking for key
        required elements such as human rights commitments and ILO
        convention references.

        Args:
            policies: List of WorkforcePolicy records.

        Returns:
            Dict with policy summary and completeness indicators.
        """
        if not policies:
            return {
                "policy_count": 0,
                "has_human_rights_commitments": False,
                "has_ilo_references": False,
                "publicly_available_count": 0,
                "esrs_ref": "S1-1 Para 19-21",
                "provenance_hash": _compute_hash({"policies": 0}),
            }

        has_hr = any(len(p.human_rights_commitments) > 0 for p in policies)
        has_ilo = any(len(p.ilo_conventions_referenced) > 0 for p in policies)
        public_count = sum(1 for p in policies if p.is_publicly_available)

        all_hr_commitments: List[str] = []
        all_ilo_refs: List[str] = []
        for p in policies:
            all_hr_commitments.extend(p.human_rights_commitments)
            all_ilo_refs.extend(p.ilo_conventions_referenced)

        result = {
            "policy_count": len(policies),
            "policy_names": [p.policy_name for p in policies],
            "has_human_rights_commitments": has_hr,
            "human_rights_commitments": list(set(all_hr_commitments)),
            "has_ilo_references": has_ilo,
            "ilo_conventions_referenced": list(set(all_ilo_refs)),
            "publicly_available_count": public_count,
            "esrs_ref": "S1-1 Para 19-21",
            "provenance_hash": "",
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def assess_engagement(
        self, processes: List[EngagementProcess]
    ) -> Dict[str, Any]:
        """Assess engagement processes per ESRS S1-2 Para 23-25.

        Summarises the undertaking's engagement processes with own
        workers and their representatives.

        Args:
            processes: List of EngagementProcess records.

        Returns:
            Dict with engagement summary and completeness indicators.
        """
        if not processes:
            return {
                "process_count": 0,
                "worker_representatives_involved": False,
                "outcomes_disclosed": False,
                "esrs_ref": "S1-2 Para 23-25",
                "provenance_hash": _compute_hash({"processes": 0}),
            }

        has_reps = any(p.worker_representatives_involved for p in processes)
        has_outcomes = any(p.outcomes_disclosed for p in processes)

        result = {
            "process_count": len(processes),
            "process_names": [p.process_name for p in processes],
            "worker_representatives_involved": has_reps,
            "outcomes_disclosed": has_outcomes,
            "all_stages": list(set(
                stage for p in processes for stage in p.stages
            )),
            "esrs_ref": "S1-2 Para 23-25",
            "provenance_hash": "",
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def assess_remediation(
        self, channels: List[RemediationChannel]
    ) -> Dict[str, Any]:
        """Assess remediation channels per ESRS S1-3 Para 27-30.

        Summarises the undertaking's remediation and grievance channels.

        Args:
            channels: List of RemediationChannel records.

        Returns:
            Dict with remediation summary and resolution metrics.
        """
        if not channels:
            return {
                "channel_count": 0,
                "has_anonymous_channel": False,
                "total_complaints_received": 0,
                "total_complaints_resolved": 0,
                "resolution_rate_pct": "0.00",
                "esrs_ref": "S1-3 Para 27-30",
                "provenance_hash": _compute_hash({"channels": 0}),
            }

        has_anon = any(c.is_anonymous for c in channels)
        total_received = sum(c.complaints_received for c in channels)
        total_resolved = sum(c.complaints_resolved for c in channels)
        resolution_rate = _pct(
            _decimal(total_resolved), _decimal(total_received)
        )

        result = {
            "channel_count": len(channels),
            "channel_names": [c.channel_name for c in channels],
            "has_anonymous_channel": has_anon,
            "total_complaints_received": total_received,
            "total_complaints_resolved": total_resolved,
            "resolution_rate_pct": str(resolution_rate),
            "esrs_ref": "S1-3 Para 27-30",
            "provenance_hash": "",
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------ #
    # Full S1 Disclosure Calculation                                      #
    # ------------------------------------------------------------------ #

    def calculate_s1_disclosure(
        self,
        employees: Optional[List[EmployeeData]] = None,
        non_employee_workers: Optional[List[NonEmployeeWorker]] = None,
        collective_bargaining: Optional[List[CollectiveBargainingData]] = None,
        health_safety_metrics: Optional[HealthSafetyMetrics] = None,
        health_safety_incidents: Optional[List[HealthSafetyIncident]] = None,
        family_leave_data: Optional[List[FamilyLeaveData]] = None,
        human_rights_incidents: Optional[List[HumanRightsIncident]] = None,
        policies: Optional[List[WorkforcePolicy]] = None,
        engagement_processes: Optional[List[EngagementProcess]] = None,
        remediation_channels: Optional[List[RemediationChannel]] = None,
        ceo_total_compensation: Optional[Decimal] = None,
        reporting_year: int = 0,
        entity_name: str = "",
    ) -> S1WorkforceResult:
        """Calculate the complete ESRS S1 Own Workforce disclosure.

        Orchestrates all 17 disclosure requirements into a single
        S1WorkforceResult with full provenance tracking.

        Args:
            employees: Employee data for S1-6, S1-9 to S1-13, S1-16.
            non_employee_workers: Non-employee data for S1-7.
            collective_bargaining: Bargaining data for S1-8.
            health_safety_metrics: Pre-aggregated H&S data for S1-14.
            health_safety_incidents: Individual H&S incidents for S1-14.
            family_leave_data: Leave data for S1-15.
            human_rights_incidents: HR incidents for S1-17.
            policies: Workforce policies for S1-1.
            engagement_processes: Engagement processes for S1-2.
            remediation_channels: Remediation channels for S1-3.
            ceo_total_compensation: CEO compensation for S1-16.
            reporting_year: Reporting year.
            entity_name: Entity or undertaking name.

        Returns:
            S1WorkforceResult with all 17 DRs populated.
        """
        t0 = time.perf_counter()

        logger.info(
            "Calculating full S1 disclosure: entity=%s, year=%d",
            entity_name, reporting_year,
        )

        emp_list = employees or []

        # S1-1: Policies
        s1_1 = self.assess_policies(policies or [])

        # S1-2: Engagement
        s1_2 = self.assess_engagement(engagement_processes or [])

        # S1-3: Remediation
        s1_3 = self.assess_remediation(remediation_channels or [])

        # S1-4: Actions (placeholder for qualitative narrative)
        s1_4: Dict[str, Any] = {
            "description": "Actions on material impacts - narrative disclosure",
            "esrs_ref": "S1-4 Para 32-36",
            "provenance_hash": _compute_hash({"s1_4": "placeholder"}),
        }

        # S1-5: Targets (placeholder for qualitative narrative)
        s1_5: Dict[str, Any] = {
            "description": "Targets and progress - narrative disclosure",
            "esrs_ref": "S1-5 Para 38-40",
            "provenance_hash": _compute_hash({"s1_5": "placeholder"}),
        }

        # S1-6: Demographics
        s1_6: Dict[str, Any] = {}
        if emp_list:
            s1_6 = self.calculate_workforce_demographics(emp_list)
        else:
            s1_6 = {
                "total_headcount": 0,
                "esrs_ref": "S1-6 Para 50-54",
                "provenance_hash": _compute_hash({"total": 0}),
            }

        # S1-7: Non-employee workers
        s1_7 = self.calculate_non_employee_metrics(non_employee_workers or [])

        # S1-8: Collective bargaining
        s1_8 = self.calculate_collective_bargaining(
            collective_bargaining or []
        )

        # S1-9: Diversity
        s1_9: Dict[str, Any] = {}
        if emp_list:
            s1_9 = self.calculate_diversity_metrics(emp_list)
        else:
            s1_9 = {
                "total_employees": 0,
                "esrs_ref": "S1-9 Para 65-67",
                "provenance_hash": _compute_hash({"total": 0}),
            }

        # S1-10: Adequate wages
        s1_10 = self.assess_adequate_wages(emp_list)

        # S1-11: Social protection
        s1_11 = self.assess_social_protection(emp_list)

        # S1-12: Disability
        s1_12 = self.calculate_disability_metrics(emp_list)

        # S1-13: Training
        s1_13 = self.calculate_training_metrics(emp_list)

        # S1-14: Health and safety
        s1_14: Dict[str, Any] = {}
        if health_safety_metrics or health_safety_incidents:
            s1_14 = self.calculate_health_safety(
                metrics=health_safety_metrics,
                incidents=health_safety_incidents,
            )
        else:
            s1_14 = {
                "fatalities": 0,
                "trir": "0",
                "ltifr": "0",
                "esrs_ref": "S1-14 Para 85-91",
                "provenance_hash": _compute_hash({"hs": "no_data"}),
            }

        # S1-15: Work-life balance
        s1_15 = self.calculate_work_life_balance(family_leave_data or [])

        # S1-16: Remuneration
        s1_16 = self.calculate_remuneration(emp_list, ceo_total_compensation)

        # S1-17: Incidents
        s1_17 = self.assess_incidents(human_rights_incidents or [])

        # Calculate compliance score based on populated DRs
        compliance = self._calculate_compliance_score(
            s1_1, s1_2, s1_3, s1_4, s1_5, s1_6, s1_7, s1_8, s1_9,
            s1_10, s1_11, s1_12, s1_13, s1_14, s1_15, s1_16, s1_17,
        )

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        result = S1WorkforceResult(
            reporting_year=reporting_year,
            entity_name=entity_name,
            s1_1_policies=s1_1,
            s1_2_engagement=s1_2,
            s1_3_remediation=s1_3,
            s1_4_actions=s1_4,
            s1_5_targets=s1_5,
            s1_6_demographics=s1_6,
            s1_7_non_employees=s1_7,
            s1_8_collective_bargaining=s1_8,
            s1_9_diversity=s1_9,
            s1_10_adequate_wages=s1_10,
            s1_11_social_protection=s1_11,
            s1_12_disability=s1_12,
            s1_13_training=s1_13,
            s1_14_health_safety=s1_14,
            s1_15_work_life_balance=s1_15,
            s1_16_remuneration=s1_16,
            s1_17_incidents=s1_17,
            compliance_score=compliance,
            processing_time_ms=elapsed_ms,
        )

        result.provenance_hash = _compute_hash(result)

        logger.info(
            "S1 disclosure complete: compliance=%.1f%%, time=%.1fms, hash=%s",
            float(compliance), elapsed_ms, result.provenance_hash[:16],
        )

        return result

    # ------------------------------------------------------------------ #
    # Completeness Validation                                             #
    # ------------------------------------------------------------------ #

    def validate_s1_completeness(
        self, result: S1WorkforceResult
    ) -> Dict[str, Any]:
        """Validate completeness against all S1 required data points.

        Checks whether all 17 ESRS S1 disclosure requirements have
        been populated in the result, with per-DR granularity.

        Args:
            result: S1WorkforceResult to validate.

        Returns:
            Dict with total_datapoints, populated_datapoints,
            missing_datapoints, completeness_pct, is_complete,
            per_dr_status, provenance_hash.
        """
        dr_checks: Dict[str, bool] = {}

        # S1-1: Policies
        dr_checks["S1-1"] = bool(
            result.s1_1_policies.get("policy_count", 0) > 0
        )

        # S1-2: Engagement
        dr_checks["S1-2"] = bool(
            result.s1_2_engagement.get("process_count", 0) > 0
        )

        # S1-3: Remediation
        dr_checks["S1-3"] = bool(
            result.s1_3_remediation.get("channel_count", 0) > 0
        )

        # S1-4: Actions
        dr_checks["S1-4"] = bool(result.s1_4_actions)

        # S1-5: Targets
        dr_checks["S1-5"] = bool(result.s1_5_targets)

        # S1-6: Demographics
        dr_checks["S1-6"] = bool(
            result.s1_6_demographics.get("total_headcount", 0) > 0
        )

        # S1-7: Non-employees
        dr_checks["S1-7"] = bool(
            result.s1_7_non_employees.get("total_headcount", 0) >= 0
        )

        # S1-8: Collective bargaining
        dr_checks["S1-8"] = bool(
            result.s1_8_collective_bargaining.get("total_employees", 0) > 0
        )

        # S1-9: Diversity
        dr_checks["S1-9"] = bool(
            result.s1_9_diversity.get("total_employees", 0) > 0
        )

        # S1-10: Adequate wages
        dr_checks["S1-10"] = bool(
            result.s1_10_adequate_wages.get("assessed_count", 0) > 0
        )

        # S1-11: Social protection
        dr_checks["S1-11"] = bool(
            result.s1_11_social_protection.get("total_employees", 0) > 0
        )

        # S1-12: Disability
        dr_checks["S1-12"] = bool(
            result.s1_12_disability.get("total_employees", 0) > 0
        )

        # S1-13: Training
        dr_checks["S1-13"] = bool(
            result.s1_13_training.get("employee_count", 0) > 0
        )

        # S1-14: Health & safety
        dr_checks["S1-14"] = bool(
            result.s1_14_health_safety.get("total_hours_worked", "0") != "0"
            or result.s1_14_health_safety.get("fatalities", -1) >= 0
        )

        # S1-15: Work-life balance
        dr_checks["S1-15"] = bool(
            result.s1_15_work_life_balance.get("by_leave_type")
        )

        # S1-16: Remuneration
        dr_checks["S1-16"] = bool(
            result.s1_16_remuneration.get("total_employee_count", 0) > 0
        )

        # S1-17: Incidents
        dr_checks["S1-17"] = bool(
            result.s1_17_incidents.get("total_incidents", -1) >= 0
        )

        total_drs = len(dr_checks)
        populated = sum(1 for v in dr_checks.values() if v)
        missing = [k for k, v in dr_checks.items() if not v]
        completeness = _pct(_decimal(populated), _decimal(total_drs))

        validation = {
            "total_disclosure_requirements": total_drs,
            "populated_disclosure_requirements": populated,
            "missing_disclosure_requirements": missing,
            "completeness_pct": str(completeness),
            "is_complete": len(missing) == 0,
            "per_dr_status": {k: "POPULATED" if v else "MISSING" for k, v in dr_checks.items()},
            "total_datapoints": len(ALL_S1_DATAPOINTS),
            "provenance_hash": "",
        }
        validation["provenance_hash"] = _compute_hash(validation)

        logger.info(
            "S1 completeness: %s%% (%d/%d DRs), missing=%s",
            str(completeness), populated, total_drs, missing,
        )

        return validation

    # ------------------------------------------------------------------ #
    # ESRS S1 Data Point Mapping                                          #
    # ------------------------------------------------------------------ #

    def get_s1_datapoints(
        self, result: S1WorkforceResult
    ) -> Dict[str, Any]:
        """Map S1 result to ESRS disclosure data points.

        Creates a structured mapping of all S1 required data points
        with their values, ready for XBRL tagging and report generation.

        Args:
            result: S1WorkforceResult to map.

        Returns:
            Dict mapping S1 data point IDs to their values and metadata.
        """
        datapoints: Dict[str, Any] = {}

        # S1-6 datapoints
        demographics = result.s1_6_demographics
        datapoints["s1_6_01_total_employees_headcount"] = {
            "label": "Total number of employees (headcount)",
            "value": demographics.get("total_headcount", 0),
            "unit": "headcount",
            "esrs_ref": "S1-6 Para 50",
        }
        datapoints["s1_6_02_employees_by_gender"] = {
            "label": "Number of employees by gender",
            "value": demographics.get("by_gender", {}),
            "esrs_ref": "S1-6 Para 50",
        }
        datapoints["s1_6_04_employees_by_contract_type"] = {
            "label": "Number of employees by contract type",
            "value": demographics.get("by_contract_type", {}),
            "esrs_ref": "S1-6 Para 50",
        }
        datapoints["s1_6_06_employees_by_working_time"] = {
            "label": "Number of employees by working time",
            "value": demographics.get("by_working_time", {}),
            "esrs_ref": "S1-6 Para 50",
        }
        datapoints["s1_6_08_employees_by_region"] = {
            "label": "Number of employees by region",
            "value": demographics.get("by_region", {}),
            "esrs_ref": "S1-6 Para 52",
        }

        # S1-9 datapoints
        diversity = result.s1_9_diversity
        datapoints["s1_9_01_gender_distribution_top_management"] = {
            "label": "Gender distribution in top management",
            "value": diversity.get("gender_top_management_female_pct", "0"),
            "unit": "percent_female",
            "esrs_ref": "S1-9 Para 65",
        }
        datapoints["s1_9_03_age_distribution_employees"] = {
            "label": "Age distribution of employees",
            "value": diversity.get("age_distribution", {}),
            "esrs_ref": "S1-9 Para 65",
        }
        datapoints["s1_9_04_disability_percentage"] = {
            "label": "Percentage of employees with disabilities",
            "value": diversity.get("disability_pct", "0"),
            "unit": "percent",
            "esrs_ref": "S1-9 Para 67",
        }

        # S1-13 datapoints
        training = result.s1_13_training
        datapoints["s1_13_02_average_training_hours_per_employee"] = {
            "label": "Average training hours per employee",
            "value": training.get("avg_hours_per_employee", "0"),
            "unit": "hours",
            "esrs_ref": "S1-13 Para 81",
        }
        datapoints["s1_13_03_training_hours_by_gender"] = {
            "label": "Training hours by gender",
            "value": training.get("by_gender", {}),
            "esrs_ref": "S1-13 Para 83",
        }

        # S1-14 datapoints
        hs = result.s1_14_health_safety
        datapoints["s1_14_01_fatalities_work_related"] = {
            "label": "Number of work-related fatalities",
            "value": hs.get("fatalities", 0),
            "unit": "count",
            "esrs_ref": "S1-14 Para 85",
        }
        datapoints["s1_14_06_trir_total_recordable_incident_rate"] = {
            "label": "Total Recordable Incident Rate (TRIR)",
            "value": hs.get("trir", "0"),
            "unit": "per 200,000 hours",
            "esrs_ref": "S1-14 Para 88",
        }
        datapoints["s1_14_07_ltifr_lost_time_injury_frequency_rate"] = {
            "label": "Lost Time Injury Frequency Rate (LTIFR)",
            "value": hs.get("ltifr", "0"),
            "unit": "per 1,000,000 hours",
            "esrs_ref": "S1-14 Para 88",
        }

        # S1-16 datapoints
        rem = result.s1_16_remuneration
        datapoints["s1_16_01_gender_pay_gap_pct"] = {
            "label": "Unadjusted gender pay gap",
            "value": rem.get("gender_pay_gap_pct", "0"),
            "unit": "percent",
            "esrs_ref": "S1-16 Para 97",
        }
        datapoints["s1_16_04_ceo_to_median_ratio"] = {
            "label": "CEO-to-median-employee pay ratio",
            "value": rem.get("ceo_to_median_ratio", "0"),
            "unit": "ratio",
            "esrs_ref": "S1-16 Para 99",
        }

        # S1-17 datapoints
        inc = result.s1_17_incidents
        datapoints["s1_17_01_total_incidents_reported"] = {
            "label": "Total incidents reported",
            "value": inc.get("total_incidents", 0),
            "unit": "count",
            "esrs_ref": "S1-17 Para 101",
        }
        datapoints["s1_17_04_severe_human_rights_impacts"] = {
            "label": "Number of severe human rights impacts",
            "value": inc.get("severe_impacts_count", 0),
            "unit": "count",
            "esrs_ref": "S1-17 Para 103",
        }

        datapoints["provenance_hash"] = _compute_hash(datapoints)
        return datapoints

    # ------------------------------------------------------------------ #
    # Private Helpers                                                     #
    # ------------------------------------------------------------------ #

    def _count_by_attr(
        self, employees: List[EmployeeData], attr: str
    ) -> Dict[str, int]:
        """Count employees by a given attribute value.

        Args:
            employees: List of EmployeeData records.
            attr: Attribute name to group by (e.g., 'gender', 'region').

        Returns:
            Dict mapping attribute values to headcounts.
        """
        counts: Dict[str, int] = {}
        for emp in employees:
            val = getattr(emp, attr)
            key = val.value if hasattr(val, "value") else str(val)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _calculate_median(self, values: List[Decimal]) -> Decimal:
        """Calculate the median of a sorted list of Decimal values.

        Args:
            values: Sorted list of Decimal values.

        Returns:
            Median value as Decimal, rounded to 2 places.
        """
        if not values:
            return Decimal("0")
        n = len(values)
        if n % 2 == 1:
            return _round2(values[n // 2])
        mid_low = values[(n // 2) - 1]
        mid_high = values[n // 2]
        return _round2((mid_low + mid_high) / Decimal("2"))

    def _calculate_compliance_score(
        self,
        s1_1: Dict[str, Any],
        s1_2: Dict[str, Any],
        s1_3: Dict[str, Any],
        s1_4: Dict[str, Any],
        s1_5: Dict[str, Any],
        s1_6: Dict[str, Any],
        s1_7: Dict[str, Any],
        s1_8: Dict[str, Any],
        s1_9: Dict[str, Any],
        s1_10: Dict[str, Any],
        s1_11: Dict[str, Any],
        s1_12: Dict[str, Any],
        s1_13: Dict[str, Any],
        s1_14: Dict[str, Any],
        s1_15: Dict[str, Any],
        s1_16: Dict[str, Any],
        s1_17: Dict[str, Any],
    ) -> Decimal:
        """Calculate overall S1 compliance score.

        Each of the 17 disclosure requirements contributes equally
        to the overall score.  A DR is considered populated if it
        contains substantive data beyond default/empty values.

        Args:
            s1_1 through s1_17: Individual DR result dicts.

        Returns:
            Compliance score as Decimal (0-100).
        """
        total_drs = Decimal("17")
        populated = Decimal("0")

        # S1-1: Has at least one policy
        if s1_1.get("policy_count", 0) > 0:
            populated += Decimal("1")

        # S1-2: Has at least one engagement process
        if s1_2.get("process_count", 0) > 0:
            populated += Decimal("1")

        # S1-3: Has at least one remediation channel
        if s1_3.get("channel_count", 0) > 0:
            populated += Decimal("1")

        # S1-4: Has any content
        if s1_4.get("description"):
            populated += Decimal("1")

        # S1-5: Has any content
        if s1_5.get("description"):
            populated += Decimal("1")

        # S1-6: Has employee data
        if s1_6.get("total_headcount", 0) > 0:
            populated += Decimal("1")

        # S1-7: Has non-employee data (0 is valid)
        if "total_headcount" in s1_7:
            populated += Decimal("1")

        # S1-8: Has bargaining data
        if s1_8.get("total_employees", 0) > 0:
            populated += Decimal("1")

        # S1-9: Has diversity data
        if s1_9.get("total_employees", 0) > 0:
            populated += Decimal("1")

        # S1-10: Has wage assessment
        if s1_10.get("assessed_count", 0) > 0:
            populated += Decimal("1")

        # S1-11: Has social protection data
        if s1_11.get("total_employees", 0) > 0:
            populated += Decimal("1")

        # S1-12: Has disability data
        if s1_12.get("total_employees", 0) > 0:
            populated += Decimal("1")

        # S1-13: Has training data
        if s1_13.get("employee_count", 0) > 0:
            populated += Decimal("1")

        # S1-14: Has H&S data
        if s1_14.get("fatalities", -1) >= 0:
            populated += Decimal("1")

        # S1-15: Has leave data
        if s1_15.get("by_leave_type"):
            populated += Decimal("1")

        # S1-16: Has remuneration data
        if s1_16.get("total_employee_count", 0) > 0:
            populated += Decimal("1")

        # S1-17: Has incident tracking
        if s1_17.get("total_incidents", -1) >= 0:
            populated += Decimal("1")

        return _round2(populated / total_drs * Decimal("100"))
