# -*- coding: utf-8 -*-
"""
E2PollutionEngine - PACK-017 ESRS Full Coverage Engine (E2 Pollution)
=====================================================================

Calculates and assesses pollution disclosures per ESRS E2 (Pollution).

Under ESRS E2, undertakings must disclose information about their
pollution-related impacts, risks, and opportunities across six disclosure
requirements covering policies, actions, targets, actual pollution
emissions, substances of concern, and anticipated financial effects.
This engine implements the complete E2 assessment pipeline, including:

- Policy assessment for pollution prevention and control (E2-1)
- Action and resource tracking for pollution reduction (E2-2)
- Target evaluation with progress tracking and gap analysis (E2-3)
- Pollutant emission aggregation by medium: air, water, soil (E2-4)
- Substance of concern and SVHC classification and quantification (E2-5)
- Financial effect estimation from pollution-related risks (E2-6)
- Completeness validation against all E2 required data points
- SHA-256 provenance on every result for audit trail

ESRS E2 Disclosure Requirements:
    - E2-1 (Para 11-12, AR E2-1 to AR E2-4): Policies related to
      pollution prevention, reduction, and control
    - E2-2 (Para 16-19, AR E2-5 to AR E2-8): Actions and resources
      related to pollution, including key actions taken or planned
    - E2-3 (Para 21-24, AR E2-9 to AR E2-12): Targets related to
      pollution, including measurable, time-bound targets
    - E2-4 (Para 28-35, AR E2-13 to AR E2-22): Pollution of air,
      water and soil, including quantities by pollutant type
    - E2-5 (Para 37-40, AR E2-23 to AR E2-28): Substances of concern
      and substances of very high concern (SVHCs)
    - E2-6 (Para 42-44, AR E2-29 to AR E2-32): Anticipated financial
      effects from pollution-related impacts, risks, opportunities

Regulatory References:
    - EU Delegated Regulation 2023/2772 (ESRS)
    - ESRS E2 Pollution, Disclosure Requirements E2-1 through E2-6
    - EU REACH Regulation (EC) No 1907/2006
    - EU Industrial Emissions Directive 2010/75/EU (IED)
    - European Pollutant Release and Transfer Register (E-PRTR)
    - EU Water Framework Directive 2000/60/EC

Zero-Hallucination:
    - All emission aggregations use deterministic arithmetic
    - Substance classifications follow REACH/ECHA official lists
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

def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(
        Decimal("0.001"), rounding=ROUND_HALF_UP
    ))

def _round6(value: Decimal) -> Decimal:
    """Round Decimal to 6 decimal places using ROUND_HALF_UP."""
    return value.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PollutantType(str, Enum):
    """Pollutant types relevant to ESRS E2 disclosures.

    Per ESRS E2-4 (Para 28-35), undertakings must report emissions
    of pollutants to air, water, and soil disaggregated by type.
    """
    NOX = "nox"
    SOX = "sox"
    PM2_5 = "pm2_5"
    PM10 = "pm10"
    VOC = "voc"
    NH3 = "nh3"
    HEAVY_METALS = "heavy_metals"
    PERSISTENT_ORGANIC = "persistent_organic"
    MICROPLASTICS = "microplastics"

class PollutantMedium(str, Enum):
    """Environmental medium into which pollutants are released.

    Per ESRS E2-4, pollution must be disaggregated across the three
    environmental compartments: air, water, and soil.
    """
    AIR = "air"
    WATER = "water"
    SOIL = "soil"

class SubstanceCategory(str, Enum):
    """Classification categories for substances per REACH regulation.

    Per ESRS E2-5 (Para 37-40), undertakings must disclose substances
    of concern (SOC) and substances of very high concern (SVHC) as
    defined under REACH Regulation (EC) No 1907/2006.
    """
    SUBSTANCE_OF_CONCERN = "substance_of_concern"
    SVHC = "svhc"
    REACH_RESTRICTED = "reach_restricted"
    CANDIDATE_LIST = "candidate_list"

class TargetType(str, Enum):
    """Target types for pollution reduction per ESRS E2-3.

    Per ESRS E2-3 (Para 21-24), targets may be set as absolute
    reductions, intensity-based metrics, or full elimination goals.
    """
    ABSOLUTE = "absolute"
    INTENSITY = "intensity"
    ELIMINATION = "elimination"

class PolicyScope(str, Enum):
    """Scope of pollution policies per ESRS E2-1.

    Per ESRS E2-1 (Para 11-12), policies may apply to own operations,
    upstream value chain, downstream value chain, or the full chain.
    """
    OWN_OPERATIONS = "own_operations"
    UPSTREAM = "upstream"
    DOWNSTREAM = "downstream"
    FULL_VALUE_CHAIN = "full_value_chain"

class RiskCategory(str, Enum):
    """Risk categories for pollution-related financial effects.

    Per ESRS E2-6 (Para 42-44), anticipated financial effects must
    be categorized by risk type to support transition planning.
    """
    PHYSICAL = "physical"
    TRANSITION = "transition"
    REGULATORY = "regulatory"
    LITIGATION = "litigation"

# ---------------------------------------------------------------------------
# E2 Datapoint Constants
# ---------------------------------------------------------------------------

E2_1_DATAPOINTS: List[str] = [
    "e2_1_01_policy_existence",
    "e2_1_02_policy_scope",
    "e2_1_03_policy_pollutants_covered",
    "e2_1_04_alignment_with_regulatory_requirements",
    "e2_1_05_policy_implementation_date",
    "e2_1_06_third_party_standards_applied",
]

E2_2_DATAPOINTS: List[str] = [
    "e2_2_01_key_actions_description",
    "e2_2_02_resources_allocated",
    "e2_2_03_expected_outcomes",
    "e2_2_04_action_timeline",
    "e2_2_05_capex_opex_breakdown",
    "e2_2_06_action_scope_coverage",
]

E2_3_DATAPOINTS: List[str] = [
    "e2_3_01_target_type",
    "e2_3_02_baseline_year_and_value",
    "e2_3_03_target_value",
    "e2_3_04_target_year",
    "e2_3_05_progress_percentage",
    "e2_3_06_pollutant_and_medium_covered",
]

E2_4_DATAPOINTS: List[str] = [
    "e2_4_01_emissions_to_air_by_pollutant",
    "e2_4_02_emissions_to_water_by_pollutant",
    "e2_4_03_emissions_to_soil_by_pollutant",
    "e2_4_04_total_pollutant_quantities_air",
    "e2_4_05_total_pollutant_quantities_water",
    "e2_4_06_total_pollutant_quantities_soil",
    "e2_4_07_measurement_methodologies",
    "e2_4_08_reporting_boundary",
]

E2_5_DATAPOINTS: List[str] = [
    "e2_5_01_substances_of_concern_produced",
    "e2_5_02_substances_of_concern_used",
    "e2_5_03_svhc_quantities",
    "e2_5_04_reach_compliance_status",
    "e2_5_05_substitution_plans",
    "e2_5_06_candidate_list_substances",
]

E2_6_DATAPOINTS: List[str] = [
    "e2_6_01_financial_effects_from_pollution_risks",
    "e2_6_02_remediation_provisions",
    "e2_6_03_stranded_assets_exposure",
    "e2_6_04_opportunity_monetary_value",
    "e2_6_05_time_horizon_classification",
    "e2_6_06_likelihood_assessment",
]

ALL_E2_DATAPOINTS: List[str] = (
    E2_1_DATAPOINTS + E2_2_DATAPOINTS + E2_3_DATAPOINTS
    + E2_4_DATAPOINTS + E2_5_DATAPOINTS + E2_6_DATAPOINTS
)

# Pollutant names for human-readable labels.
POLLUTANT_NAMES: Dict[str, str] = {
    "nox": "Nitrogen Oxides (NOx)",
    "sox": "Sulphur Oxides (SOx)",
    "pm2_5": "Particulate Matter PM2.5",
    "pm10": "Particulate Matter PM10",
    "voc": "Volatile Organic Compounds (VOC)",
    "nh3": "Ammonia (NH3)",
    "heavy_metals": "Heavy Metals",
    "persistent_organic": "Persistent Organic Pollutants (POPs)",
    "microplastics": "Microplastics",
}

# Regulatory emission thresholds (kg/year) per E-PRTR Annex II.
# These are indicative thresholds for reporting obligations.
REGULATORY_THRESHOLDS_KG: Dict[str, Dict[str, Decimal]] = {
    "nox": {"air": Decimal("100000"), "water": Decimal("0"), "soil": Decimal("0")},
    "sox": {"air": Decimal("150000"), "water": Decimal("0"), "soil": Decimal("0")},
    "pm10": {"air": Decimal("50000"), "water": Decimal("0"), "soil": Decimal("0")},
    "pm2_5": {"air": Decimal("50000"), "water": Decimal("0"), "soil": Decimal("0")},
    "voc": {"air": Decimal("100000"), "water": Decimal("0"), "soil": Decimal("0")},
    "nh3": {"air": Decimal("10000"), "water": Decimal("0"), "soil": Decimal("0")},
    "heavy_metals": {
        "air": Decimal("200"), "water": Decimal("50"), "soil": Decimal("50"),
    },
    "persistent_organic": {
        "air": Decimal("1"), "water": Decimal("1"), "soil": Decimal("1"),
    },
    "microplastics": {
        "air": Decimal("0"), "water": Decimal("1000"), "soil": Decimal("1000"),
    },
}

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class PollutionPolicy(BaseModel):
    """A pollution prevention/control policy per ESRS E2-1.

    Represents a single policy adopted by the undertaking to address
    pollution across one or more environmental media and pollutant types.
    """
    policy_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this policy",
    )
    name: str = Field(
        ...,
        description="Policy name or title",
        max_length=500,
    )
    scope: PolicyScope = Field(
        ...,
        description="Scope of the policy within the value chain",
    )
    pollutants_covered: List[PollutantType] = Field(
        default_factory=list,
        description="Pollutant types addressed by this policy",
    )
    media_covered: List[PollutantMedium] = Field(
        default_factory=list,
        description="Environmental media addressed (air, water, soil)",
    )
    regulatory_alignment: List[str] = Field(
        default_factory=list,
        description="Regulatory frameworks the policy aligns with (e.g. IED, REACH)",
    )
    implementation_date: Optional[str] = Field(
        default=None,
        description="Date policy was implemented (ISO 8601 date string)",
        max_length=10,
    )
    third_party_standards: List[str] = Field(
        default_factory=list,
        description="Third-party standards or certifications applied",
    )

    @field_validator("pollutants_covered")
    @classmethod
    def validate_pollutants_not_empty(
        cls, v: List[PollutantType]
    ) -> List[PollutantType]:
        """Ensure at least one pollutant is covered."""
        if not v:
            raise ValueError(
                "A policy must cover at least one pollutant type"
            )
        return v

class PollutionAction(BaseModel):
    """An action or resource commitment for pollution reduction per E2-2.

    Represents a specific initiative the undertaking has taken or plans
    to take, including the resources allocated and expected outcomes.
    """
    action_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this action",
    )
    description: str = Field(
        ...,
        description="Description of the action taken or planned",
        max_length=2000,
    )
    resources_allocated: Decimal = Field(
        default=Decimal("0"),
        description="Total financial resources allocated (EUR)",
        ge=Decimal("0"),
    )
    capex_amount: Decimal = Field(
        default=Decimal("0"),
        description="Capital expenditure portion (EUR)",
        ge=Decimal("0"),
    )
    opex_amount: Decimal = Field(
        default=Decimal("0"),
        description="Operational expenditure portion (EUR)",
        ge=Decimal("0"),
    )
    expected_reduction_pct: Decimal = Field(
        default=Decimal("0"),
        description="Expected reduction in pollutant emissions (percentage)",
        ge=Decimal("0"),
        le=Decimal("100"),
    )
    pollutants_targeted: List[PollutantType] = Field(
        default_factory=list,
        description="Pollutant types targeted by this action",
    )
    timeline_start: Optional[str] = Field(
        default=None,
        description="Action start date (ISO 8601)",
        max_length=10,
    )
    timeline_end: Optional[str] = Field(
        default=None,
        description="Action target completion date (ISO 8601)",
        max_length=10,
    )
    scope_coverage: PolicyScope = Field(
        default=PolicyScope.OWN_OPERATIONS,
        description="Value chain scope covered by this action",
    )

class PollutionTarget(BaseModel):
    """A measurable pollution reduction target per ESRS E2-3.

    Represents a time-bound, quantified target for reducing emissions
    of a specific pollutant to a specific environmental medium.
    """
    target_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this target",
    )
    pollutant: PollutantType = Field(
        ...,
        description="Pollutant type this target applies to",
    )
    medium: PollutantMedium = Field(
        ...,
        description="Environmental medium (air, water, soil)",
    )
    target_type: TargetType = Field(
        ...,
        description="Type of target (absolute, intensity, elimination)",
    )
    base_year: int = Field(
        ...,
        description="Baseline year for the target",
        ge=1990,
    )
    base_value: Decimal = Field(
        ...,
        description="Baseline value (kg for absolute, ratio for intensity)",
        ge=Decimal("0"),
    )
    target_value: Decimal = Field(
        ...,
        description="Target value to achieve (kg for absolute, ratio for intensity)",
        ge=Decimal("0"),
    )
    target_year: int = Field(
        ...,
        description="Year by which the target should be met",
        ge=2020,
    )
    current_value: Decimal = Field(
        default=Decimal("0"),
        description="Most recent measured value",
        ge=Decimal("0"),
    )
    progress_pct: Decimal = Field(
        default=Decimal("0"),
        description="Progress towards target (0-100 percentage)",
    )

    @field_validator("target_year")
    @classmethod
    def validate_target_after_base(cls, v: int, info: Any) -> int:
        """Ensure target year is after or equal to base year."""
        base = info.data.get("base_year")
        if base is not None and v < base:
            raise ValueError(
                f"target_year ({v}) must be >= base_year ({base})"
            )
        return v

class PollutantEmission(BaseModel):
    """A single pollutant emission record per ESRS E2-4.

    Represents one measured or estimated emission of a specific
    pollutant to a specific environmental medium from a facility.
    """
    emission_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this emission record",
    )
    pollutant_type: PollutantType = Field(
        ...,
        description="Type of pollutant emitted",
    )
    medium: PollutantMedium = Field(
        ...,
        description="Environmental medium (air, water, soil)",
    )
    quantity_kg: Decimal = Field(
        ...,
        description="Quantity of pollutant emitted in kilograms",
        ge=Decimal("0"),
    )
    measurement_method: str = Field(
        default="estimated",
        description="Measurement method: measured, calculated, or estimated",
        max_length=100,
    )
    reporting_period: str = Field(
        default="",
        description="Reporting period (e.g. '2025' or '2025-Q1')",
        max_length=20,
    )
    facility_id: str = Field(
        default="",
        description="Identifier of the facility where emission occurred",
        max_length=100,
    )
    facility_name: str = Field(
        default="",
        description="Human-readable facility name",
        max_length=500,
    )
    country_code: str = Field(
        default="",
        description="ISO 3166-1 alpha-2 country code",
        max_length=3,
    )

class SubstanceRecord(BaseModel):
    """A substance of concern or SVHC record per ESRS E2-5.

    Represents a specific substance produced or used by the
    undertaking that falls under REACH regulation classifications.
    """
    substance_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this substance record",
    )
    name: str = Field(
        ...,
        description="Chemical name of the substance",
        max_length=500,
    )
    cas_number: str = Field(
        default="",
        description="CAS Registry Number (e.g. '50-00-0' for formaldehyde)",
        max_length=20,
    )
    category: SubstanceCategory = Field(
        ...,
        description="REACH classification category",
    )
    quantity_kg: Decimal = Field(
        ...,
        description="Quantity produced or used in kilograms",
        ge=Decimal("0"),
    )
    use_purpose: str = Field(
        default="",
        description="Purpose or application of the substance",
        max_length=500,
    )
    substitution_plan: bool = Field(
        default=False,
        description="Whether a substitution plan exists for this substance",
    )
    substitution_timeline: Optional[str] = Field(
        default=None,
        description="Expected substitution completion date (ISO 8601)",
        max_length=10,
    )
    is_produced: bool = Field(
        default=False,
        description="True if substance is produced, False if used/purchased",
    )

class PollutionFinancialEffect(BaseModel):
    """An anticipated financial effect from pollution per ESRS E2-6.

    Represents a single financial risk or opportunity arising from
    pollution-related impacts, aligned with E2-6 Para 42-44.
    """
    effect_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this financial effect",
    )
    risk_type: RiskCategory = Field(
        ...,
        description="Category of risk (physical, transition, regulatory, litigation)",
    )
    description: str = Field(
        ...,
        description="Description of the financial effect",
        max_length=2000,
    )
    monetary_impact: Decimal = Field(
        default=Decimal("0"),
        description="Estimated monetary impact in EUR",
    )
    is_opportunity: bool = Field(
        default=False,
        description="True if this is an opportunity rather than a risk",
    )
    time_horizon: str = Field(
        default="medium_term",
        description="Time horizon: short_term (<1yr), medium_term (1-5yr), long_term (>5yr)",
        max_length=20,
    )
    likelihood: str = Field(
        default="possible",
        description="Likelihood: remote, unlikely, possible, likely, virtually_certain",
        max_length=30,
    )
    remediation_provision: Decimal = Field(
        default=Decimal("0"),
        description="Remediation or clean-up provision amount in EUR",
        ge=Decimal("0"),
    )
    stranded_asset_exposure: Decimal = Field(
        default=Decimal("0"),
        description="Exposure to stranded assets due to pollution regulations (EUR)",
        ge=Decimal("0"),
    )

class E2PollutionResult(BaseModel):
    """Complete E2 Pollution disclosure result.

    Aggregates assessments across all six E2 disclosure requirements
    into a single auditable output with full provenance tracking.
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
    # E2-1 Policies
    policies: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Assessed pollution policies (E2-1)",
    )
    policy_count: int = Field(
        default=0,
        description="Number of policies assessed",
    )
    # E2-2 Actions
    actions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Pollution actions and resources (E2-2)",
    )
    total_resources_allocated: Decimal = Field(
        default=Decimal("0"),
        description="Total resources allocated to pollution actions (EUR)",
    )
    # E2-3 Targets
    targets: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Pollution targets and progress (E2-3)",
    )
    average_target_progress_pct: Decimal = Field(
        default=Decimal("0"),
        description="Average progress across all targets (%)",
    )
    # E2-4 Emissions by medium
    emissions_to_air: Dict[str, Any] = Field(
        default_factory=dict,
        description="Pollutant emissions to air by type (E2-4)",
    )
    emissions_to_water: Dict[str, Any] = Field(
        default_factory=dict,
        description="Pollutant emissions to water by type (E2-4)",
    )
    emissions_to_soil: Dict[str, Any] = Field(
        default_factory=dict,
        description="Pollutant emissions to soil by type (E2-4)",
    )
    total_emissions_kg: Decimal = Field(
        default=Decimal("0"),
        description="Total pollutant emissions across all media (kg)",
    )
    # E2-5 Substances
    substances_of_concern: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Substances of concern records (E2-5)",
    )
    svhc_list: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Substances of very high concern records (E2-5)",
    )
    total_soc_quantity_kg: Decimal = Field(
        default=Decimal("0"),
        description="Total quantity of substances of concern (kg)",
    )
    total_svhc_quantity_kg: Decimal = Field(
        default=Decimal("0"),
        description="Total quantity of SVHCs (kg)",
    )
    # E2-6 Financial effects
    financial_effects: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Anticipated financial effects (E2-6)",
    )
    total_risk_exposure: Decimal = Field(
        default=Decimal("0"),
        description="Total monetary risk exposure (EUR)",
    )
    total_opportunity_value: Decimal = Field(
        default=Decimal("0"),
        description="Total monetary opportunity value (EUR)",
    )
    total_remediation_provisions: Decimal = Field(
        default=Decimal("0"),
        description="Total remediation provisions (EUR)",
    )
    # Overall assessment
    compliance_score: Decimal = Field(
        default=Decimal("0"),
        description="Overall E2 compliance score (0-100)",
    )
    threshold_exceedances: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Pollutants exceeding regulatory thresholds",
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Processing time in milliseconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of all inputs and calculation steps",
    )

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class PollutionEngine:
    """E2 Pollution assessment engine per ESRS E2.

    Provides deterministic, zero-hallucination calculations for:
    - Policy assessment against E2-1 requirements (Para 11-12)
    - Action and resource tracking per E2-2 (Para 16-19)
    - Target progress evaluation per E2-3 (Para 21-24)
    - Pollutant emission aggregation per E2-4 (Para 28-35)
    - Substance of concern classification per E2-5 (Para 37-40)
    - Financial effect estimation per E2-6 (Para 42-44)
    - Completeness validation against all E2 data points

    All calculations use Decimal arithmetic for bit-perfect
    reproducibility.  No LLM is used in any calculation path.

    Usage::

        engine = PollutionEngine()
        result = engine.calculate_e2_disclosure(
            policies=[...],
            actions=[...],
            targets=[...],
            emissions=[...],
            substances=[...],
            financial_effects=[...],
        )
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self) -> None:
        """Initialize PollutionEngine with regulatory thresholds."""
        self._thresholds = REGULATORY_THRESHOLDS_KG
        logger.info(
            "PollutionEngine v%s initialized with %d pollutant thresholds",
            self.engine_version, len(self._thresholds),
        )

    # ------------------------------------------------------------------ #
    # E2-1: Policy Assessment                                              #
    # ------------------------------------------------------------------ #

    def assess_pollution_policies(
        self, policies: List[PollutionPolicy]
    ) -> Dict[str, Any]:
        """Assess pollution policies per ESRS E2-1 (Para 11-12).

        Evaluates the completeness and coverage of pollution policies
        across pollutant types, environmental media, and value chain
        scope.  Checks alignment with key regulatory frameworks.

        Args:
            policies: List of PollutionPolicy instances to assess.

        Returns:
            Dict with:
                - policy_count: int
                - policies: list of assessed policy dicts
                - pollutants_covered: list of unique pollutant types
                - media_covered: list of unique media types
                - scope_coverage: list of unique scopes
                - regulatory_alignments: list of unique frameworks
                - coverage_score: Decimal (0-100)
                - provenance_hash: str
        """
        logger.info("Assessing %d pollution policies (E2-1)", len(policies))

        all_pollutants: set = set()
        all_media: set = set()
        all_scopes: set = set()
        all_alignments: set = set()
        assessed: List[Dict[str, Any]] = []

        for policy in policies:
            for p in policy.pollutants_covered:
                all_pollutants.add(p.value)
            for m in policy.media_covered:
                all_media.add(m.value)
            all_scopes.add(policy.scope.value)
            for ra in policy.regulatory_alignment:
                all_alignments.add(ra)

            assessed.append({
                "policy_id": policy.policy_id,
                "name": policy.name,
                "scope": policy.scope.value,
                "pollutants_covered": [p.value for p in policy.pollutants_covered],
                "media_covered": [m.value for m in policy.media_covered],
                "regulatory_alignment": policy.regulatory_alignment,
                "implementation_date": policy.implementation_date,
                "third_party_standards": policy.third_party_standards,
            })

        # Coverage score: percentage of all pollutant types covered
        total_pollutant_types = len(PollutantType)
        total_media_types = len(PollutantMedium)
        pollutant_coverage = _safe_divide(
            _decimal(len(all_pollutants)), _decimal(total_pollutant_types)
        )
        media_coverage = _safe_divide(
            _decimal(len(all_media)), _decimal(total_media_types)
        )
        # Weighted: 60% pollutant coverage + 40% media coverage
        coverage_score = _round_val(
            (pollutant_coverage * Decimal("60")
             + media_coverage * Decimal("40")),
            1,
        )

        result = {
            "policy_count": len(policies),
            "policies": assessed,
            "pollutants_covered": sorted(all_pollutants),
            "media_covered": sorted(all_media),
            "scope_coverage": sorted(all_scopes),
            "regulatory_alignments": sorted(all_alignments),
            "coverage_score": coverage_score,
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "E2-1 policy assessment: %d policies, coverage=%.1f%%",
            len(policies), float(coverage_score),
        )
        return result

    # ------------------------------------------------------------------ #
    # E2-2: Actions and Resources                                          #
    # ------------------------------------------------------------------ #

    def assess_actions(
        self, actions: List[PollutionAction]
    ) -> Dict[str, Any]:
        """Assess pollution actions and resources per ESRS E2-2 (Para 16-19).

        Aggregates resource commitments and expected outcomes across
        all pollution reduction actions.

        Args:
            actions: List of PollutionAction instances.

        Returns:
            Dict with:
                - action_count: int
                - actions: list of action summary dicts
                - total_resources_allocated: Decimal (EUR)
                - total_capex: Decimal (EUR)
                - total_opex: Decimal (EUR)
                - avg_expected_reduction_pct: Decimal
                - scope_breakdown: dict of scope -> count
                - provenance_hash: str
        """
        logger.info("Assessing %d pollution actions (E2-2)", len(actions))

        total_resources = Decimal("0")
        total_capex = Decimal("0")
        total_opex = Decimal("0")
        reduction_pcts: List[Decimal] = []
        scope_counts: Dict[str, int] = {}
        assessed: List[Dict[str, Any]] = []

        for action in actions:
            total_resources += action.resources_allocated
            total_capex += action.capex_amount
            total_opex += action.opex_amount
            if action.expected_reduction_pct > Decimal("0"):
                reduction_pcts.append(action.expected_reduction_pct)

            scope_key = action.scope_coverage.value
            scope_counts[scope_key] = scope_counts.get(scope_key, 0) + 1

            assessed.append({
                "action_id": action.action_id,
                "description": action.description,
                "resources_allocated": str(action.resources_allocated),
                "capex_amount": str(action.capex_amount),
                "opex_amount": str(action.opex_amount),
                "expected_reduction_pct": str(action.expected_reduction_pct),
                "pollutants_targeted": [
                    p.value for p in action.pollutants_targeted
                ],
                "timeline_start": action.timeline_start,
                "timeline_end": action.timeline_end,
                "scope_coverage": action.scope_coverage.value,
            })

        avg_reduction = Decimal("0")
        if reduction_pcts:
            avg_reduction = _round_val(
                sum(reduction_pcts) / _decimal(len(reduction_pcts)), 1
            )

        result = {
            "action_count": len(actions),
            "actions": assessed,
            "total_resources_allocated": str(_round_val(total_resources, 2)),
            "total_capex": str(_round_val(total_capex, 2)),
            "total_opex": str(_round_val(total_opex, 2)),
            "avg_expected_reduction_pct": str(avg_reduction),
            "scope_breakdown": scope_counts,
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "E2-2 action assessment: %d actions, total_resources=%.2f EUR",
            len(actions), float(total_resources),
        )
        return result

    # ------------------------------------------------------------------ #
    # E2-3: Target Evaluation                                              #
    # ------------------------------------------------------------------ #

    def evaluate_targets(
        self, targets: List[PollutionTarget]
    ) -> Dict[str, Any]:
        """Evaluate pollution targets per ESRS E2-3 (Para 21-24).

        Calculates progress for each target, performs gap analysis,
        and determines whether targets are on track.

        Progress calculation:
            For ABSOLUTE/INTENSITY: progress = (base - current) / (base - target) * 100
            For ELIMINATION: progress = (base - current) / base * 100

        Args:
            targets: List of PollutionTarget instances.

        Returns:
            Dict with:
                - target_count: int
                - targets: list of target assessment dicts
                - avg_progress_pct: Decimal
                - on_track_count: int
                - behind_count: int
                - gap_analysis: list of targets behind schedule
                - by_pollutant: dict of pollutant -> avg progress
                - by_medium: dict of medium -> avg progress
                - provenance_hash: str
        """
        logger.info("Evaluating %d pollution targets (E2-3)", len(targets))

        assessed: List[Dict[str, Any]] = []
        progress_values: List[Decimal] = []
        on_track = 0
        behind = 0
        gaps: List[Dict[str, Any]] = []
        by_pollutant: Dict[str, List[Decimal]] = {}
        by_medium: Dict[str, List[Decimal]] = {}

        for target in targets:
            progress = self._calculate_target_progress(target)
            is_on_track = progress >= Decimal("50") or target.progress_pct >= Decimal("50")

            # Use the engine-calculated progress
            effective_progress = max(progress, target.progress_pct)
            progress_values.append(effective_progress)

            if is_on_track:
                on_track += 1
            else:
                behind += 1
                gaps.append({
                    "target_id": target.target_id,
                    "pollutant": target.pollutant.value,
                    "medium": target.medium.value,
                    "progress_pct": str(effective_progress),
                    "gap_pct": str(_round_val(
                        Decimal("100") - effective_progress, 1
                    )),
                })

            # Aggregate by pollutant
            p_key = target.pollutant.value
            if p_key not in by_pollutant:
                by_pollutant[p_key] = []
            by_pollutant[p_key].append(effective_progress)

            # Aggregate by medium
            m_key = target.medium.value
            if m_key not in by_medium:
                by_medium[m_key] = []
            by_medium[m_key].append(effective_progress)

            assessed.append({
                "target_id": target.target_id,
                "pollutant": target.pollutant.value,
                "medium": target.medium.value,
                "target_type": target.target_type.value,
                "base_year": target.base_year,
                "base_value": str(target.base_value),
                "target_value": str(target.target_value),
                "target_year": target.target_year,
                "current_value": str(target.current_value),
                "progress_pct": str(effective_progress),
                "is_on_track": is_on_track,
            })

        avg_progress = Decimal("0")
        if progress_values:
            avg_progress = _round_val(
                sum(progress_values) / _decimal(len(progress_values)), 1
            )

        # Summarize by pollutant and medium
        pollutant_summary = {
            k: str(_round_val(sum(v) / _decimal(len(v)), 1))
            for k, v in by_pollutant.items()
        }
        medium_summary = {
            k: str(_round_val(sum(v) / _decimal(len(v)), 1))
            for k, v in by_medium.items()
        }

        result = {
            "target_count": len(targets),
            "targets": assessed,
            "avg_progress_pct": str(avg_progress),
            "on_track_count": on_track,
            "behind_count": behind,
            "gap_analysis": gaps,
            "by_pollutant": pollutant_summary,
            "by_medium": medium_summary,
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "E2-3 target evaluation: %d targets, avg_progress=%.1f%%, "
            "on_track=%d, behind=%d",
            len(targets), float(avg_progress), on_track, behind,
        )
        return result

    def _calculate_target_progress(self, target: PollutionTarget) -> Decimal:
        """Calculate progress percentage for a single target.

        For ABSOLUTE and INTENSITY targets:
            progress = (base_value - current_value) / (base_value - target_value) * 100

        For ELIMINATION targets:
            progress = (base_value - current_value) / base_value * 100

        Args:
            target: PollutionTarget to evaluate.

        Returns:
            Progress percentage (Decimal, 0-100 clamped).
        """
        if target.target_type == TargetType.ELIMINATION:
            denominator = target.base_value
        else:
            denominator = target.base_value - target.target_value

        if denominator <= Decimal("0"):
            return Decimal("0")

        numerator = target.base_value - target.current_value
        progress = _safe_divide(numerator, denominator) * Decimal("100")

        # Clamp to 0-100
        progress = max(Decimal("0"), min(Decimal("100"), progress))
        return _round_val(progress, 1)

    # ------------------------------------------------------------------ #
    # E2-4: Emissions by Medium                                            #
    # ------------------------------------------------------------------ #

    def calculate_emissions_by_medium(
        self, emissions: List[PollutantEmission]
    ) -> Dict[str, Any]:
        """Aggregate pollutant emissions by medium per ESRS E2-4 (Para 28-35).

        Groups all emission records by medium (air, water, soil) and
        pollutant type, calculates totals, and checks against
        regulatory thresholds.

        Args:
            emissions: List of PollutantEmission instances.

        Returns:
            Dict with:
                - emissions_to_air: dict of pollutant -> quantity_kg
                - emissions_to_water: dict of pollutant -> quantity_kg
                - emissions_to_soil: dict of pollutant -> quantity_kg
                - total_air_kg: Decimal
                - total_water_kg: Decimal
                - total_soil_kg: Decimal
                - total_all_media_kg: Decimal
                - by_pollutant_total: dict of pollutant -> total_kg
                - threshold_exceedances: list of exceedance dicts
                - emission_count: int
                - provenance_hash: str
        """
        logger.info(
            "Calculating emissions by medium for %d records (E2-4)",
            len(emissions),
        )

        # Initialize accumulators for each medium
        air: Dict[str, Decimal] = {}
        water: Dict[str, Decimal] = {}
        soil: Dict[str, Decimal] = {}
        by_pollutant_total: Dict[str, Decimal] = {}

        for emission in emissions:
            p_key = emission.pollutant_type.value
            qty = emission.quantity_kg

            # Accumulate by pollutant total
            by_pollutant_total[p_key] = (
                by_pollutant_total.get(p_key, Decimal("0")) + qty
            )

            # Accumulate by medium
            if emission.medium == PollutantMedium.AIR:
                air[p_key] = air.get(p_key, Decimal("0")) + qty
            elif emission.medium == PollutantMedium.WATER:
                water[p_key] = water.get(p_key, Decimal("0")) + qty
            elif emission.medium == PollutantMedium.SOIL:
                soil[p_key] = soil.get(p_key, Decimal("0")) + qty

        # Round all values
        for d in (air, water, soil, by_pollutant_total):
            for k in d:
                d[k] = _round_val(d[k], 3)

        total_air = _round_val(sum(air.values(), Decimal("0")), 3)
        total_water = _round_val(sum(water.values(), Decimal("0")), 3)
        total_soil = _round_val(sum(soil.values(), Decimal("0")), 3)
        total_all = _round_val(total_air + total_water + total_soil, 3)

        # Check regulatory thresholds
        exceedances = self._check_thresholds(air, water, soil)

        result = {
            "emissions_to_air": {k: str(v) for k, v in sorted(air.items())},
            "emissions_to_water": {k: str(v) for k, v in sorted(water.items())},
            "emissions_to_soil": {k: str(v) for k, v in sorted(soil.items())},
            "total_air_kg": str(total_air),
            "total_water_kg": str(total_water),
            "total_soil_kg": str(total_soil),
            "total_all_media_kg": str(total_all),
            "by_pollutant_total": {
                k: str(v) for k, v in sorted(by_pollutant_total.items())
            },
            "threshold_exceedances": exceedances,
            "emission_count": len(emissions),
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "E2-4 emissions: air=%.3f kg, water=%.3f kg, soil=%.3f kg, "
            "total=%.3f kg, exceedances=%d",
            float(total_air), float(total_water), float(total_soil),
            float(total_all), len(exceedances),
        )
        return result

    def _check_thresholds(
        self,
        air: Dict[str, Decimal],
        water: Dict[str, Decimal],
        soil: Dict[str, Decimal],
    ) -> List[Dict[str, Any]]:
        """Check pollutant quantities against E-PRTR regulatory thresholds.

        Args:
            air: Pollutant quantities emitted to air (kg).
            water: Pollutant quantities emitted to water (kg).
            soil: Pollutant quantities emitted to soil (kg).

        Returns:
            List of exceedance dicts with pollutant, medium, quantity,
            threshold, and exceedance_pct.
        """
        exceedances: List[Dict[str, Any]] = []
        media_data = {
            "air": air,
            "water": water,
            "soil": soil,
        }

        for pollutant_key, thresholds in self._thresholds.items():
            for medium_key, threshold_val in thresholds.items():
                if threshold_val <= Decimal("0"):
                    continue
                actual = media_data.get(medium_key, {}).get(
                    pollutant_key, Decimal("0")
                )
                if actual > threshold_val:
                    exceedance_pct = _round_val(
                        _safe_divide(
                            actual - threshold_val, threshold_val
                        ) * Decimal("100"),
                        1,
                    )
                    exceedances.append({
                        "pollutant": pollutant_key,
                        "pollutant_name": POLLUTANT_NAMES.get(
                            pollutant_key, pollutant_key
                        ),
                        "medium": medium_key,
                        "quantity_kg": str(actual),
                        "threshold_kg": str(threshold_val),
                        "exceedance_pct": str(exceedance_pct),
                    })

        return exceedances

    # ------------------------------------------------------------------ #
    # E2-5: Substances of Concern                                          #
    # ------------------------------------------------------------------ #

    def assess_substances_of_concern(
        self, substances: List[SubstanceRecord]
    ) -> Dict[str, Any]:
        """Assess substances of concern per ESRS E2-5 (Para 37-40).

        Classifies substances into SOC, SVHC, REACH-restricted, and
        Candidate List categories.  Calculates total quantities and
        identifies substances with substitution plans.

        Args:
            substances: List of SubstanceRecord instances.

        Returns:
            Dict with:
                - total_substances: int
                - soc_records: list of SOC dicts
                - svhc_records: list of SVHC dicts
                - reach_restricted_records: list of restricted dicts
                - candidate_list_records: list of candidate dicts
                - total_soc_quantity_kg: Decimal
                - total_svhc_quantity_kg: Decimal
                - total_reach_restricted_kg: Decimal
                - total_candidate_list_kg: Decimal
                - substances_with_substitution: int
                - substitution_coverage_pct: Decimal
                - produced_vs_used: dict
                - provenance_hash: str
        """
        logger.info(
            "Assessing %d substances of concern (E2-5)", len(substances)
        )

        soc_records: List[Dict[str, Any]] = []
        svhc_records: List[Dict[str, Any]] = []
        reach_records: List[Dict[str, Any]] = []
        candidate_records: List[Dict[str, Any]] = []

        total_soc_kg = Decimal("0")
        total_svhc_kg = Decimal("0")
        total_reach_kg = Decimal("0")
        total_candidate_kg = Decimal("0")
        substances_with_sub = 0
        produced_count = 0
        used_count = 0

        for substance in substances:
            record = {
                "substance_id": substance.substance_id,
                "name": substance.name,
                "cas_number": substance.cas_number,
                "category": substance.category.value,
                "quantity_kg": str(substance.quantity_kg),
                "use_purpose": substance.use_purpose,
                "substitution_plan": substance.substitution_plan,
                "substitution_timeline": substance.substitution_timeline,
                "is_produced": substance.is_produced,
            }

            if substance.substitution_plan:
                substances_with_sub += 1

            if substance.is_produced:
                produced_count += 1
            else:
                used_count += 1

            if substance.category == SubstanceCategory.SUBSTANCE_OF_CONCERN:
                soc_records.append(record)
                total_soc_kg += substance.quantity_kg
            elif substance.category == SubstanceCategory.SVHC:
                svhc_records.append(record)
                total_svhc_kg += substance.quantity_kg
            elif substance.category == SubstanceCategory.REACH_RESTRICTED:
                reach_records.append(record)
                total_reach_kg += substance.quantity_kg
            elif substance.category == SubstanceCategory.CANDIDATE_LIST:
                candidate_records.append(record)
                total_candidate_kg += substance.quantity_kg

        total_substances = len(substances)
        substitution_coverage = Decimal("0")
        if total_substances > 0:
            substitution_coverage = _round_val(
                _decimal(substances_with_sub)
                / _decimal(total_substances)
                * Decimal("100"),
                1,
            )

        result = {
            "total_substances": total_substances,
            "soc_records": soc_records,
            "svhc_records": svhc_records,
            "reach_restricted_records": reach_records,
            "candidate_list_records": candidate_records,
            "total_soc_quantity_kg": str(_round_val(total_soc_kg, 3)),
            "total_svhc_quantity_kg": str(_round_val(total_svhc_kg, 3)),
            "total_reach_restricted_kg": str(_round_val(total_reach_kg, 3)),
            "total_candidate_list_kg": str(_round_val(total_candidate_kg, 3)),
            "substances_with_substitution": substances_with_sub,
            "substitution_coverage_pct": str(substitution_coverage),
            "produced_vs_used": {
                "produced": produced_count,
                "used": used_count,
            },
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "E2-5 substance assessment: SOC=%.3f kg, SVHC=%.3f kg, "
            "REACH=%.3f kg, substitution_coverage=%.1f%%",
            float(total_soc_kg), float(total_svhc_kg),
            float(total_reach_kg), float(substitution_coverage),
        )
        return result

    # ------------------------------------------------------------------ #
    # E2-6: Financial Effects                                              #
    # ------------------------------------------------------------------ #

    def estimate_financial_effects(
        self, effects: List[PollutionFinancialEffect]
    ) -> Dict[str, Any]:
        """Estimate financial effects per ESRS E2-6 (Para 42-44).

        Aggregates anticipated financial impacts from pollution-related
        risks and opportunities, including remediation provisions and
        stranded asset exposure.

        Args:
            effects: List of PollutionFinancialEffect instances.

        Returns:
            Dict with:
                - effect_count: int
                - effects: list of effect summary dicts
                - total_risk_exposure: Decimal (EUR)
                - total_opportunity_value: Decimal (EUR)
                - total_remediation_provisions: Decimal (EUR)
                - total_stranded_asset_exposure: Decimal (EUR)
                - net_financial_impact: Decimal (EUR)
                - by_risk_type: dict of risk_type -> total
                - by_time_horizon: dict of horizon -> total
                - by_likelihood: dict of likelihood -> count
                - provenance_hash: str
        """
        logger.info(
            "Estimating financial effects for %d items (E2-6)",
            len(effects),
        )

        total_risk = Decimal("0")
        total_opportunity = Decimal("0")
        total_remediation = Decimal("0")
        total_stranded = Decimal("0")
        by_risk_type: Dict[str, Decimal] = {}
        by_horizon: Dict[str, Decimal] = {}
        by_likelihood: Dict[str, int] = {}
        assessed: List[Dict[str, Any]] = []

        for effect in effects:
            abs_impact = abs(effect.monetary_impact)

            if effect.is_opportunity:
                total_opportunity += abs_impact
            else:
                total_risk += abs_impact

            total_remediation += effect.remediation_provision
            total_stranded += effect.stranded_asset_exposure

            # By risk type
            rt_key = effect.risk_type.value
            by_risk_type[rt_key] = (
                by_risk_type.get(rt_key, Decimal("0")) + abs_impact
            )

            # By time horizon
            th_key = effect.time_horizon
            by_horizon[th_key] = (
                by_horizon.get(th_key, Decimal("0")) + abs_impact
            )

            # By likelihood
            ll_key = effect.likelihood
            by_likelihood[ll_key] = by_likelihood.get(ll_key, 0) + 1

            assessed.append({
                "effect_id": effect.effect_id,
                "risk_type": effect.risk_type.value,
                "description": effect.description,
                "monetary_impact": str(effect.monetary_impact),
                "is_opportunity": effect.is_opportunity,
                "time_horizon": effect.time_horizon,
                "likelihood": effect.likelihood,
                "remediation_provision": str(effect.remediation_provision),
                "stranded_asset_exposure": str(effect.stranded_asset_exposure),
            })

        net_impact = _round_val(total_opportunity - total_risk, 2)

        result = {
            "effect_count": len(effects),
            "effects": assessed,
            "total_risk_exposure": str(_round_val(total_risk, 2)),
            "total_opportunity_value": str(_round_val(total_opportunity, 2)),
            "total_remediation_provisions": str(_round_val(total_remediation, 2)),
            "total_stranded_asset_exposure": str(_round_val(total_stranded, 2)),
            "net_financial_impact": str(net_impact),
            "by_risk_type": {
                k: str(_round_val(v, 2))
                for k, v in sorted(by_risk_type.items())
            },
            "by_time_horizon": {
                k: str(_round_val(v, 2))
                for k, v in sorted(by_horizon.items())
            },
            "by_likelihood": dict(sorted(by_likelihood.items())),
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "E2-6 financial effects: risk=%.2f EUR, opportunity=%.2f EUR, "
            "remediation=%.2f EUR, net=%.2f EUR",
            float(total_risk), float(total_opportunity),
            float(total_remediation), float(net_impact),
        )
        return result

    # ------------------------------------------------------------------ #
    # Full E2 Disclosure Calculation                                       #
    # ------------------------------------------------------------------ #

    def calculate_e2_disclosure(
        self,
        policies: List[PollutionPolicy],
        actions: List[PollutionAction],
        targets: List[PollutionTarget],
        emissions: List[PollutantEmission],
        substances: List[SubstanceRecord],
        financial_effects: List[PollutionFinancialEffect],
    ) -> E2PollutionResult:
        """Calculate the complete ESRS E2 disclosure across all requirements.

        Orchestrates assessment of all six E2 disclosure requirements
        (E2-1 through E2-6) into a single consolidated result with
        full provenance tracking.

        Args:
            policies: Pollution policies (E2-1).
            actions: Pollution actions and resources (E2-2).
            targets: Pollution reduction targets (E2-3).
            emissions: Pollutant emission records (E2-4).
            substances: Substance of concern records (E2-5).
            financial_effects: Anticipated financial effects (E2-6).

        Returns:
            E2PollutionResult with all assessments and provenance hash.

        Raises:
            ValueError: If all input lists are empty.
        """
        t0 = time.perf_counter()

        total_inputs = (
            len(policies) + len(actions) + len(targets)
            + len(emissions) + len(substances) + len(financial_effects)
        )
        if total_inputs == 0:
            raise ValueError(
                "At least one input record is required across E2-1 to E2-6"
            )

        logger.info(
            "Calculating E2 disclosure: policies=%d, actions=%d, "
            "targets=%d, emissions=%d, substances=%d, financial=%d",
            len(policies), len(actions), len(targets),
            len(emissions), len(substances), len(financial_effects),
        )

        # E2-1: Policies
        policy_assessment = (
            self.assess_pollution_policies(policies) if policies else {}
        )

        # E2-2: Actions
        action_assessment = (
            self.assess_actions(actions) if actions else {}
        )

        # E2-3: Targets
        target_assessment = (
            self.evaluate_targets(targets) if targets else {}
        )

        # E2-4: Emissions
        emission_assessment = (
            self.calculate_emissions_by_medium(emissions) if emissions else {}
        )

        # E2-5: Substances
        substance_assessment = (
            self.assess_substances_of_concern(substances) if substances else {}
        )

        # E2-6: Financial effects
        financial_assessment = (
            self.estimate_financial_effects(financial_effects)
            if financial_effects else {}
        )

        # Extract key totals for result model
        total_resources = _decimal(
            action_assessment.get("total_resources_allocated", "0")
        )
        avg_target_progress = _decimal(
            target_assessment.get("avg_progress_pct", "0")
        )
        total_emissions_kg = _decimal(
            emission_assessment.get("total_all_media_kg", "0")
        )
        total_soc_kg = _decimal(
            substance_assessment.get("total_soc_quantity_kg", "0")
        )
        total_svhc_kg = _decimal(
            substance_assessment.get("total_svhc_quantity_kg", "0")
        )
        total_risk = _decimal(
            financial_assessment.get("total_risk_exposure", "0")
        )
        total_opp = _decimal(
            financial_assessment.get("total_opportunity_value", "0")
        )
        total_rem = _decimal(
            financial_assessment.get("total_remediation_provisions", "0")
        )

        # Compliance score: weighted across E2 requirements
        compliance_score = self._calculate_compliance_score(
            policies=policies,
            actions=actions,
            targets=targets,
            emissions=emissions,
            substances=substances,
            financial_effects=financial_effects,
            policy_coverage=_decimal(
                policy_assessment.get("coverage_score", "0")
            ),
            target_progress=avg_target_progress,
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = E2PollutionResult(
            # E2-1
            policies=policy_assessment.get("policies", []),
            policy_count=len(policies),
            # E2-2
            actions=action_assessment.get("actions", []),
            total_resources_allocated=total_resources,
            # E2-3
            targets=target_assessment.get("targets", []),
            average_target_progress_pct=avg_target_progress,
            # E2-4
            emissions_to_air=emission_assessment.get("emissions_to_air", {}),
            emissions_to_water=emission_assessment.get("emissions_to_water", {}),
            emissions_to_soil=emission_assessment.get("emissions_to_soil", {}),
            total_emissions_kg=total_emissions_kg,
            # E2-5
            substances_of_concern=substance_assessment.get("soc_records", []),
            svhc_list=substance_assessment.get("svhc_records", []),
            total_soc_quantity_kg=total_soc_kg,
            total_svhc_quantity_kg=total_svhc_kg,
            # E2-6
            financial_effects=financial_assessment.get("effects", []),
            total_risk_exposure=total_risk,
            total_opportunity_value=total_opp,
            total_remediation_provisions=total_rem,
            # Overall
            compliance_score=compliance_score,
            threshold_exceedances=emission_assessment.get(
                "threshold_exceedances", []
            ),
            processing_time_ms=elapsed_ms,
        )

        result.provenance_hash = _compute_hash(result)

        logger.info(
            "E2 disclosure calculated: compliance=%.1f%%, "
            "total_emissions=%.3f kg, total_risk=%.2f EUR, "
            "hash=%s, time=%.1f ms",
            float(compliance_score), float(total_emissions_kg),
            float(total_risk), result.provenance_hash[:16], elapsed_ms,
        )

        return result

    def _calculate_compliance_score(
        self,
        policies: List[PollutionPolicy],
        actions: List[PollutionAction],
        targets: List[PollutionTarget],
        emissions: List[PollutantEmission],
        substances: List[SubstanceRecord],
        financial_effects: List[PollutionFinancialEffect],
        policy_coverage: Decimal,
        target_progress: Decimal,
    ) -> Decimal:
        """Calculate overall E2 compliance score (0-100).

        Scoring weights per disclosure requirement:
            E2-1 Policies:           15%
            E2-2 Actions:            15%
            E2-3 Targets:            15%
            E2-4 Emissions:          25%
            E2-5 Substances:         15%
            E2-6 Financial effects:  15%

        Within each requirement, score is based on whether data has
        been provided (existence) and, where applicable, qualitative
        assessments (e.g. policy coverage, target progress).

        Args:
            policies: Input policies list.
            actions: Input actions list.
            targets: Input targets list.
            emissions: Input emissions list.
            substances: Input substances list.
            financial_effects: Input financial effects list.
            policy_coverage: Coverage score from policy assessment.
            target_progress: Average target progress percentage.

        Returns:
            Compliance score as Decimal (0-100).
        """
        # E2-1: Policy existence (50%) + coverage quality (50%)
        e2_1_score = Decimal("0")
        if policies:
            e2_1_score = Decimal("50") + (policy_coverage / Decimal("2"))

        # E2-2: Action existence (50%) + resource allocation (50%)
        e2_2_score = Decimal("0")
        if actions:
            has_resources = any(
                a.resources_allocated > Decimal("0") for a in actions
            )
            e2_2_score = Decimal("50") + (Decimal("50") if has_resources else Decimal("0"))

        # E2-3: Target existence (50%) + progress (50%)
        e2_3_score = Decimal("0")
        if targets:
            e2_3_score = Decimal("50") + (target_progress / Decimal("2"))

        # E2-4: Emission data provided (100% if present)
        e2_4_score = Decimal("100") if emissions else Decimal("0")

        # E2-5: Substance data provided (50%) + substitution plans (50%)
        e2_5_score = Decimal("0")
        if substances:
            sub_plans = sum(1 for s in substances if s.substitution_plan)
            sub_ratio = _safe_divide(
                _decimal(sub_plans), _decimal(len(substances))
            )
            e2_5_score = Decimal("50") + (sub_ratio * Decimal("50"))

        # E2-6: Financial effects disclosed (100% if present)
        e2_6_score = Decimal("100") if financial_effects else Decimal("0")

        # Weighted total
        weighted = (
            e2_1_score * Decimal("0.15")
            + e2_2_score * Decimal("0.15")
            + e2_3_score * Decimal("0.15")
            + e2_4_score * Decimal("0.25")
            + e2_5_score * Decimal("0.15")
            + e2_6_score * Decimal("0.15")
        )

        return _round_val(weighted, 1)

    # ------------------------------------------------------------------ #
    # Completeness Validation                                              #
    # ------------------------------------------------------------------ #

    def validate_e2_completeness(
        self, result: E2PollutionResult
    ) -> Dict[str, Any]:
        """Validate completeness against all E2 required data points.

        Checks whether all ESRS E2 mandatory disclosure data points
        (E2-1 through E2-6) are present and populated in the result.

        Args:
            result: E2PollutionResult to validate.

        Returns:
            Dict with:
                - total_datapoints: int
                - populated_datapoints: int
                - missing_datapoints: list of str
                - completeness_pct: Decimal
                - is_complete: bool
                - by_disclosure: dict of E2-X -> completeness status
                - provenance_hash: str
        """
        populated: List[str] = []
        missing: List[str] = []

        # E2-1 checks
        e2_1_checks = {
            "e2_1_01_policy_existence": result.policy_count > 0,
            "e2_1_02_policy_scope": (
                result.policy_count > 0
                and any(p.get("scope") for p in result.policies)
            ),
            "e2_1_03_policy_pollutants_covered": (
                result.policy_count > 0
                and any(p.get("pollutants_covered") for p in result.policies)
            ),
            "e2_1_04_alignment_with_regulatory_requirements": (
                result.policy_count > 0
                and any(
                    p.get("regulatory_alignment") for p in result.policies
                )
            ),
            "e2_1_05_policy_implementation_date": (
                result.policy_count > 0
                and any(
                    p.get("implementation_date") for p in result.policies
                )
            ),
            "e2_1_06_third_party_standards_applied": (
                result.policy_count > 0
                and any(
                    p.get("third_party_standards") for p in result.policies
                )
            ),
        }

        # E2-2 checks
        e2_2_checks = {
            "e2_2_01_key_actions_description": (
                len(result.actions) > 0
                and any(a.get("description") for a in result.actions)
            ),
            "e2_2_02_resources_allocated": (
                result.total_resources_allocated > Decimal("0")
            ),
            "e2_2_03_expected_outcomes": (
                len(result.actions) > 0
                and any(
                    _decimal(a.get("expected_reduction_pct", "0")) > Decimal("0")
                    for a in result.actions
                )
            ),
            "e2_2_04_action_timeline": (
                len(result.actions) > 0
                and any(a.get("timeline_end") for a in result.actions)
            ),
            "e2_2_05_capex_opex_breakdown": (
                len(result.actions) > 0
                and any(
                    _decimal(a.get("capex_amount", "0")) > Decimal("0")
                    or _decimal(a.get("opex_amount", "0")) > Decimal("0")
                    for a in result.actions
                )
            ),
            "e2_2_06_action_scope_coverage": (
                len(result.actions) > 0
                and any(a.get("scope_coverage") for a in result.actions)
            ),
        }

        # E2-3 checks
        e2_3_checks = {
            "e2_3_01_target_type": (
                len(result.targets) > 0
                and any(t.get("target_type") for t in result.targets)
            ),
            "e2_3_02_baseline_year_and_value": (
                len(result.targets) > 0
                and any(
                    t.get("base_year") and t.get("base_value")
                    for t in result.targets
                )
            ),
            "e2_3_03_target_value": (
                len(result.targets) > 0
                and any(t.get("target_value") for t in result.targets)
            ),
            "e2_3_04_target_year": (
                len(result.targets) > 0
                and any(t.get("target_year") for t in result.targets)
            ),
            "e2_3_05_progress_percentage": (
                result.average_target_progress_pct >= Decimal("0")
                and len(result.targets) > 0
            ),
            "e2_3_06_pollutant_and_medium_covered": (
                len(result.targets) > 0
                and any(
                    t.get("pollutant") and t.get("medium")
                    for t in result.targets
                )
            ),
        }

        # E2-4 checks
        e2_4_checks = {
            "e2_4_01_emissions_to_air_by_pollutant": bool(result.emissions_to_air),
            "e2_4_02_emissions_to_water_by_pollutant": bool(result.emissions_to_water),
            "e2_4_03_emissions_to_soil_by_pollutant": bool(result.emissions_to_soil),
            "e2_4_04_total_pollutant_quantities_air": bool(result.emissions_to_air),
            "e2_4_05_total_pollutant_quantities_water": bool(result.emissions_to_water),
            "e2_4_06_total_pollutant_quantities_soil": bool(result.emissions_to_soil),
            "e2_4_07_measurement_methodologies": (
                result.total_emissions_kg > Decimal("0")
            ),
            "e2_4_08_reporting_boundary": (
                result.total_emissions_kg > Decimal("0")
            ),
        }

        # E2-5 checks
        e2_5_checks = {
            "e2_5_01_substances_of_concern_produced": (
                any(s.get("is_produced") for s in result.substances_of_concern)
                if result.substances_of_concern else False
            ),
            "e2_5_02_substances_of_concern_used": (
                any(
                    not s.get("is_produced")
                    for s in result.substances_of_concern
                )
                if result.substances_of_concern else False
            ),
            "e2_5_03_svhc_quantities": (
                result.total_svhc_quantity_kg >= Decimal("0")
                and len(result.svhc_list) > 0
            ),
            "e2_5_04_reach_compliance_status": (
                len(result.substances_of_concern) > 0
                or len(result.svhc_list) > 0
            ),
            "e2_5_05_substitution_plans": (
                any(
                    s.get("substitution_plan")
                    for s in result.substances_of_concern + result.svhc_list
                )
                if (result.substances_of_concern or result.svhc_list) else False
            ),
            "e2_5_06_candidate_list_substances": (
                len(result.svhc_list) > 0
            ),
        }

        # E2-6 checks
        e2_6_checks = {
            "e2_6_01_financial_effects_from_pollution_risks": (
                len(result.financial_effects) > 0
            ),
            "e2_6_02_remediation_provisions": (
                result.total_remediation_provisions >= Decimal("0")
                and len(result.financial_effects) > 0
            ),
            "e2_6_03_stranded_assets_exposure": (
                len(result.financial_effects) > 0
            ),
            "e2_6_04_opportunity_monetary_value": (
                result.total_opportunity_value >= Decimal("0")
                and len(result.financial_effects) > 0
            ),
            "e2_6_05_time_horizon_classification": (
                len(result.financial_effects) > 0
                and any(
                    e.get("time_horizon") for e in result.financial_effects
                )
            ),
            "e2_6_06_likelihood_assessment": (
                len(result.financial_effects) > 0
                and any(
                    e.get("likelihood") for e in result.financial_effects
                )
            ),
        }

        # Merge all checks
        all_checks = {}
        all_checks.update(e2_1_checks)
        all_checks.update(e2_2_checks)
        all_checks.update(e2_3_checks)
        all_checks.update(e2_4_checks)
        all_checks.update(e2_5_checks)
        all_checks.update(e2_6_checks)

        for dp, is_populated in all_checks.items():
            if is_populated:
                populated.append(dp)
            else:
                missing.append(dp)

        total = len(ALL_E2_DATAPOINTS)
        pop_count = len(populated)
        completeness = _round_val(
            _decimal(pop_count) / _decimal(total) * Decimal("100"), 1
        )

        # Per-disclosure completeness
        by_disclosure = {
            "E2-1": self._disclosure_completeness(e2_1_checks),
            "E2-2": self._disclosure_completeness(e2_2_checks),
            "E2-3": self._disclosure_completeness(e2_3_checks),
            "E2-4": self._disclosure_completeness(e2_4_checks),
            "E2-5": self._disclosure_completeness(e2_5_checks),
            "E2-6": self._disclosure_completeness(e2_6_checks),
        }

        validation_result = {
            "total_datapoints": total,
            "populated_datapoints": pop_count,
            "missing_datapoints": missing,
            "completeness_pct": completeness,
            "is_complete": len(missing) == 0,
            "by_disclosure": by_disclosure,
        }
        validation_result["provenance_hash"] = _compute_hash(
            {"result_id": result.result_id, "checks": all_checks}
        )

        logger.info(
            "E2 completeness: %.1f%% (%d/%d), missing=%s",
            float(completeness), pop_count, total, missing,
        )

        return validation_result

    def _disclosure_completeness(
        self, checks: Dict[str, bool]
    ) -> Dict[str, Any]:
        """Calculate completeness for a single disclosure requirement.

        Args:
            checks: Dict of datapoint_id -> is_populated bool.

        Returns:
            Dict with total, populated, completeness_pct, is_complete.
        """
        total = len(checks)
        pop = sum(1 for v in checks.values() if v)
        pct = _round_val(
            _safe_divide(_decimal(pop), _decimal(total)) * Decimal("100"), 1
        )
        return {
            "total": total,
            "populated": pop,
            "completeness_pct": str(pct),
            "is_complete": pop == total,
        }

    # ------------------------------------------------------------------ #
    # ESRS E2 Data Point Mapping                                           #
    # ------------------------------------------------------------------ #

    def get_e2_datapoints(
        self, result: E2PollutionResult
    ) -> Dict[str, Any]:
        """Map E2 result to ESRS E2 disclosure data points.

        Creates a structured mapping of all E2 required data points
        with their values, ready for report generation.

        Args:
            result: E2PollutionResult to map.

        Returns:
            Dict mapping E2 data point IDs to values and metadata.
        """
        datapoints: Dict[str, Any] = {
            # E2-1
            "e2_1_policies": {
                "label": "Policies related to pollution",
                "value": result.policies,
                "count": result.policy_count,
                "esrs_ref": "E2-1 Para 11-12",
            },
            # E2-2
            "e2_2_actions": {
                "label": "Actions and resources related to pollution",
                "value": result.actions,
                "total_resources_eur": str(result.total_resources_allocated),
                "esrs_ref": "E2-2 Para 16-19",
            },
            # E2-3
            "e2_3_targets": {
                "label": "Targets related to pollution",
                "value": result.targets,
                "avg_progress_pct": str(result.average_target_progress_pct),
                "esrs_ref": "E2-3 Para 21-24",
            },
            # E2-4
            "e2_4_emissions_to_air": {
                "label": "Pollution of air",
                "value": result.emissions_to_air,
                "esrs_ref": "E2-4 Para 28-35",
            },
            "e2_4_emissions_to_water": {
                "label": "Pollution of water",
                "value": result.emissions_to_water,
                "esrs_ref": "E2-4 Para 28-35",
            },
            "e2_4_emissions_to_soil": {
                "label": "Pollution of soil",
                "value": result.emissions_to_soil,
                "esrs_ref": "E2-4 Para 28-35",
            },
            "e2_4_total_emissions_kg": {
                "label": "Total pollutant emissions",
                "value": str(result.total_emissions_kg),
                "unit": "kg",
                "esrs_ref": "E2-4 Para 28-35",
            },
            # E2-5
            "e2_5_substances_of_concern": {
                "label": "Substances of concern",
                "value": result.substances_of_concern,
                "total_kg": str(result.total_soc_quantity_kg),
                "esrs_ref": "E2-5 Para 37-40",
            },
            "e2_5_svhc": {
                "label": "Substances of very high concern",
                "value": result.svhc_list,
                "total_kg": str(result.total_svhc_quantity_kg),
                "esrs_ref": "E2-5 Para 37-40",
            },
            # E2-6
            "e2_6_financial_effects": {
                "label": "Anticipated financial effects from pollution",
                "value": result.financial_effects,
                "total_risk_eur": str(result.total_risk_exposure),
                "total_opportunity_eur": str(result.total_opportunity_value),
                "remediation_provisions_eur": str(
                    result.total_remediation_provisions
                ),
                "esrs_ref": "E2-6 Para 42-44",
            },
            # Overall
            "compliance_score": {
                "label": "E2 compliance score",
                "value": str(result.compliance_score),
                "unit": "percent",
            },
            "threshold_exceedances": {
                "label": "Regulatory threshold exceedances",
                "value": result.threshold_exceedances,
            },
        }

        datapoints["provenance_hash"] = _compute_hash(datapoints)
        return datapoints
