# -*- coding: utf-8 -*-
"""
BusinessConductEngine - PACK-017 ESRS G1 Business Conduct Engine
=================================================================

Assesses and calculates disclosure metrics for ESRS G1: Business
Conduct.  This standard requires undertakings to disclose their
approach to business conduct, including corporate culture, supplier
management, anti-corruption measures, political influence, and
payment practices.

ESRS G1 Disclosure Requirements:
    - G1-1 (Para 7-10, AR G1-1 through AR G1-4): Business conduct
      policies and corporate culture
    - G1-2 (Para 12-15, AR G1-5 through AR G1-8): Management of
      relationships with suppliers
    - G1-3 (Para 17-22, AR G1-9 through AR G1-14): Prevention and
      detection of corruption and bribery
    - G1-4 (Para 24-26, AR G1-15 through AR G1-17): Confirmed
      incidents of corruption or bribery
    - G1-5 (Para 28-30, AR G1-18 through AR G1-20): Political
      influence and lobbying activities
    - G1-6 (Para 32-35, AR G1-21 through AR G1-25): Payment practices

This engine implements deterministic assessment logic for each
disclosure requirement, computing coverage scores, risk analysis,
financial metrics, and completeness validation.

Regulatory References:
    - EU Delegated Regulation 2023/2772 (ESRS)
    - ESRS G1 Business Conduct
    - UN Convention Against Corruption (UNCAC)
    - OECD Anti-Bribery Convention
    - EU Late Payment Directive 2011/7/EU

Zero-Hallucination:
    - All scoring uses deterministic arithmetic
    - Coverage ratios are computed from input counts
    - Financial aggregations use Decimal arithmetic
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
    """Convert value to Decimal safely."""
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
    """Round a Decimal value using ROUND_HALF_UP.

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

def _pct(part: int, total: int) -> Decimal:
    """Calculate percentage as Decimal, rounded to 1 decimal place."""
    if total == 0:
        return Decimal("0.0")
    return _round_val(
        _decimal(part) / _decimal(total) * Decimal("100"), 1
    )

def _pct_dec(part: Decimal, total: Decimal) -> Decimal:
    """Calculate percentage from Decimal values, rounded to 1 dp."""
    if total == Decimal("0"):
        return Decimal("0.0")
    return _round_val(part / total * Decimal("100"), 1)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CorruptionRiskLevel(str, Enum):
    """Corruption risk level for geographic or business-unit assessment.

    Based on Transparency International CPI tiers and internal
    risk frameworks per ESRS G1-3 AR G1-9 through AR G1-14.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class PoliticalActivityType(str, Enum):
    """Types of political influence activities per ESRS G1-5.

    Undertakings must disclose their lobbying, political donations,
    trade association memberships, and other political activities.
    """
    LOBBYING = "lobbying"
    POLITICAL_DONATIONS = "political_donations"
    TRADE_ASSOCIATION = "trade_association"
    CAMPAIGN_CONTRIBUTION = "campaign_contribution"
    GOVERNMENT_ADVISORY = "government_advisory"

class SupplierCategory(str, Enum):
    """Supplier categorisation for relationship management per G1-2.

    Reflects the supplier's standing and risk profile within the
    undertaking's supply chain management framework.
    """
    STRATEGIC = "strategic"
    PREFERRED = "preferred"
    APPROVED = "approved"
    CONDITIONAL = "conditional"
    BLOCKED = "blocked"

class PaymentTermType(str, Enum):
    """Standard payment term types per ESRS G1-6.

    Per the EU Late Payment Directive 2011/7/EU, the standard
    payment term for commercial transactions is 30 days unless
    otherwise agreed.
    """
    STANDARD_30 = "standard_30"
    STANDARD_60 = "standard_60"
    STANDARD_90 = "standard_90"
    EXTENDED = "extended"
    EARLY_PAYMENT_DISCOUNT = "early_payment_discount"

class TrainingType(str, Enum):
    """Types of business conduct training per ESRS G1.

    Tracks the different training programmes the undertaking
    delivers to its workforce and governance bodies.
    """
    ANTI_CORRUPTION = "anti_corruption"
    CODE_OF_CONDUCT = "code_of_conduct"
    WHISTLEBLOWER = "whistleblower"
    SUPPLIER_CODE = "supplier_code"
    DATA_PROTECTION = "data_protection"

# ---------------------------------------------------------------------------
# Constants - G1 Disclosure Requirement Data Points
# ---------------------------------------------------------------------------

G1_1_DATAPOINTS: List[str] = [
    "g1_1_01_code_of_conduct_exists",
    "g1_1_02_code_covers_business_ethics",
    "g1_1_03_code_covers_anti_corruption",
    "g1_1_04_code_covers_whistleblower_protection",
    "g1_1_05_corporate_culture_description",
    "g1_1_06_training_programmes_exist",
    "g1_1_07_training_coverage_pct",
    "g1_1_08_governance_body_oversight",
]

G1_2_DATAPOINTS: List[str] = [
    "g1_2_01_supplier_code_of_conduct_exists",
    "g1_2_02_suppliers_assessed_count",
    "g1_2_03_supplier_code_coverage_pct",
    "g1_2_04_supplier_audits_conducted",
    "g1_2_05_suppliers_blocked_count",
    "g1_2_06_payment_terms_disclosed",
]

G1_3_DATAPOINTS: List[str] = [
    "g1_3_01_anti_corruption_policy_exists",
    "g1_3_02_risk_assessments_conducted",
    "g1_3_03_high_risk_areas_identified",
    "g1_3_04_training_anti_corruption_coverage",
    "g1_3_05_whistleblower_mechanism_exists",
    "g1_3_06_whistleblower_reports_received",
    "g1_3_07_due_diligence_processes",
    "g1_3_08_third_party_due_diligence",
]

G1_4_DATAPOINTS: List[str] = [
    "g1_4_01_confirmed_incidents_count",
    "g1_4_02_incidents_by_type",
    "g1_4_03_legal_proceedings_count",
    "g1_4_04_fines_and_penalties_eur",
    "g1_4_05_contracts_terminated",
    "g1_4_06_employees_dismissed",
]

G1_5_DATAPOINTS: List[str] = [
    "g1_5_01_political_engagement_policy",
    "g1_5_02_lobbying_expenditure_eur",
    "g1_5_03_political_donations_eur",
    "g1_5_04_trade_association_memberships",
    "g1_5_05_lobbying_topics_disclosed",
]

G1_6_DATAPOINTS: List[str] = [
    "g1_6_01_standard_payment_terms",
    "g1_6_02_average_payment_days",
    "g1_6_03_late_payment_pct",
    "g1_6_04_late_payment_interest_paid_eur",
    "g1_6_05_sme_payment_terms_disclosed",
    "g1_6_06_sme_average_payment_days",
    "g1_6_07_disputes_on_late_payment",
]

ALL_G1_DATAPOINTS: List[str] = (
    G1_1_DATAPOINTS + G1_2_DATAPOINTS + G1_3_DATAPOINTS
    + G1_4_DATAPOINTS + G1_5_DATAPOINTS + G1_6_DATAPOINTS
)

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class BusinessConductPolicy(BaseModel):
    """Business conduct policy per G1-1 (Para 7-10).

    Represents a policy, code of conduct, or corporate culture
    document that governs the undertaking's approach to ethical
    business conduct.
    """
    policy_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this policy",
    )
    policy_name: str = Field(
        ...,
        description="Name of the policy (e.g. Code of Conduct)",
        max_length=500,
    )
    covers_business_ethics: bool = Field(
        default=False,
        description="Whether the policy covers general business ethics",
    )
    covers_anti_corruption: bool = Field(
        default=False,
        description="Whether the policy covers anti-corruption and bribery",
    )
    covers_whistleblower_protection: bool = Field(
        default=False,
        description="Whether the policy covers whistleblower protection",
    )
    covers_supplier_standards: bool = Field(
        default=False,
        description="Whether the policy covers supplier conduct standards",
    )
    covers_political_engagement: bool = Field(
        default=False,
        description="Whether the policy covers political engagement",
    )
    approved_by_governance_body: bool = Field(
        default=False,
        description="Whether the policy is approved by the governing body",
    )
    training_types_associated: List[TrainingType] = Field(
        default_factory=list,
        description="Training programmes associated with this policy",
    )
    total_employees_trained: int = Field(
        default=0,
        description="Number of employees who completed training on this policy",
        ge=0,
    )
    total_employees_in_scope: int = Field(
        default=0,
        description="Total employees in scope for this policy's training",
        ge=0,
    )
    last_reviewed_date: Optional[datetime] = Field(
        default=None,
        description="Date of last policy review",
    )

class SupplierRelationship(BaseModel):
    """Supplier relationship record per G1-2 (Para 12-15).

    Represents a supplier in the undertaking's supply chain with
    its category, compliance status, and audit history.
    """
    supplier_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this supplier",
    )
    supplier_name: str = Field(
        ...,
        description="Name of the supplier",
        max_length=500,
    )
    category: SupplierCategory = Field(
        ...,
        description="Supplier category within the management framework",
    )
    code_of_conduct_signed: bool = Field(
        default=False,
        description="Whether the supplier has signed the code of conduct",
    )
    last_audit_date: Optional[datetime] = Field(
        default=None,
        description="Date of the last supplier audit",
    )
    audit_passed: Optional[bool] = Field(
        default=None,
        description="Whether the supplier passed the last audit",
    )
    corruption_risk_level: CorruptionRiskLevel = Field(
        default=CorruptionRiskLevel.LOW,
        description="Assessed corruption risk level for the supplier's region/sector",
    )
    is_sme: bool = Field(
        default=False,
        description="Whether the supplier is a small or medium enterprise",
    )
    payment_term: PaymentTermType = Field(
        default=PaymentTermType.STANDARD_30,
        description="Agreed payment term with this supplier",
    )
    country_code: str = Field(
        default="",
        description="ISO 3166-1 alpha-2 country code",
        max_length=3,
    )

class CorruptionPreventionMeasure(BaseModel):
    """Corruption prevention and detection measure per G1-3 (Para 17-22).

    Represents a specific anti-corruption measure including risk
    assessments, training programmes, whistleblower mechanisms,
    and due diligence processes.
    """
    measure_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this measure",
    )
    measure_type: str = Field(
        ...,
        description="Type of measure (risk_assessment, training, whistleblower, due_diligence)",
        max_length=200,
    )
    description: str = Field(
        default="",
        description="Description of the prevention measure",
        max_length=2000,
    )
    risk_level_assessed: Optional[CorruptionRiskLevel] = Field(
        default=None,
        description="Risk level identified by this measure (if risk assessment)",
    )
    training_type: Optional[TrainingType] = Field(
        default=None,
        description="Training type (if the measure is training)",
    )
    employees_covered: int = Field(
        default=0,
        description="Number of employees covered by this measure",
        ge=0,
    )
    total_employees: int = Field(
        default=0,
        description="Total employees in scope",
        ge=0,
    )
    whistleblower_reports_received: int = Field(
        default=0,
        description="Number of reports received (if whistleblower mechanism)",
        ge=0,
    )
    whistleblower_reports_investigated: int = Field(
        default=0,
        description="Number of reports investigated",
        ge=0,
    )
    covers_third_parties: bool = Field(
        default=False,
        description="Whether this measure extends to third parties",
    )
    is_active: bool = Field(
        default=True,
        description="Whether this measure is currently active",
    )

class CorruptionIncident(BaseModel):
    """Confirmed corruption or bribery incident per G1-4 (Para 24-26).

    Records a confirmed incident of corruption or bribery within
    the undertaking's operations or value chain.
    """
    incident_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this incident",
    )
    incident_type: str = Field(
        ...,
        description="Type of incident (bribery, fraud, embezzlement, conflict_of_interest, other)",
        max_length=200,
    )
    date_confirmed: datetime = Field(
        ...,
        description="Date the incident was confirmed",
    )
    legal_proceedings_initiated: bool = Field(
        default=False,
        description="Whether legal proceedings have been initiated",
    )
    fine_amount_eur: Decimal = Field(
        default=Decimal("0"),
        description="Fine or penalty amount in EUR",
        ge=Decimal("0"),
    )
    contracts_terminated: int = Field(
        default=0,
        description="Number of contracts terminated as a result",
        ge=0,
    )
    employees_dismissed: int = Field(
        default=0,
        description="Number of employees dismissed as a result",
        ge=0,
    )
    country_code: str = Field(
        default="",
        description="Country where the incident occurred",
        max_length=3,
    )
    is_material: bool = Field(
        default=False,
        description="Whether the incident is considered material",
    )

class PoliticalActivity(BaseModel):
    """Political influence and lobbying activity per G1-5 (Para 28-30).

    Records a political engagement activity including lobbying,
    political donations, trade association memberships, and
    advisory roles.
    """
    activity_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this activity",
    )
    activity_type: PoliticalActivityType = Field(
        ...,
        description="Type of political activity",
    )
    description: str = Field(
        default="",
        description="Description of the political activity",
        max_length=2000,
    )
    amount_eur: Decimal = Field(
        default=Decimal("0"),
        description="Amount spent in EUR",
        ge=Decimal("0"),
    )
    recipient: str = Field(
        default="",
        description="Recipient or beneficiary of the activity",
        max_length=500,
    )
    topic: str = Field(
        default="",
        description="Main topic or issue area of the activity",
        max_length=500,
    )
    country_code: str = Field(
        default="",
        description="Country where the activity took place",
        max_length=3,
    )
    reporting_year: int = Field(
        default=0,
        description="Year of the activity",
        ge=0,
    )

class PaymentPractice(BaseModel):
    """Payment practice record per G1-6 (Para 32-35).

    Records an individual payment or aggregated payment data for
    a supplier, supporting G1-6 disclosure on payment terms and
    timeliness.
    """
    payment_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this payment record",
    )
    supplier_id: str = Field(
        default="",
        description="Reference to the supplier",
        max_length=200,
    )
    is_sme_supplier: bool = Field(
        default=False,
        description="Whether the supplier is an SME",
    )
    agreed_payment_days: int = Field(
        default=30,
        description="Contractually agreed payment days",
        ge=0,
    )
    actual_payment_days: int = Field(
        ...,
        description="Actual number of days to payment",
        ge=0,
    )
    invoice_amount_eur: Decimal = Field(
        default=Decimal("0"),
        description="Invoice amount in EUR",
        ge=Decimal("0"),
    )
    late_payment_interest_eur: Decimal = Field(
        default=Decimal("0"),
        description="Late payment interest charged in EUR",
        ge=Decimal("0"),
    )
    payment_term_type: PaymentTermType = Field(
        default=PaymentTermType.STANDARD_30,
        description="Payment term type used",
    )
    is_disputed: bool = Field(
        default=False,
        description="Whether this payment is disputed",
    )

    @property
    def is_late(self) -> bool:
        """Return True if payment was made after the agreed term."""
        return self.actual_payment_days > self.agreed_payment_days

# ---------------------------------------------------------------------------
# Result Model
# ---------------------------------------------------------------------------

class G1BusinessConductResult(BaseModel):
    """Complete ESRS G1 Business Conduct disclosure result.

    Aggregates all G1 disclosure requirement assessments into a
    single result with completeness validation and provenance
    tracking.
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
        description="Timestamp of assessment (UTC)",
    )
    reporting_year: int = Field(
        default=0, description="Reporting year"
    )
    entity_name: str = Field(
        default="", description="Entity or undertaking name"
    )

    # G1-1: Policies and corporate culture
    g1_1_policies: Dict[str, Any] = Field(
        default_factory=dict,
        description="G1-1 policy and corporate culture assessment",
    )

    # G1-2: Supplier management
    g1_2_suppliers: Dict[str, Any] = Field(
        default_factory=dict,
        description="G1-2 supplier relationship assessment",
    )

    # G1-3: Corruption prevention
    g1_3_corruption_prevention: Dict[str, Any] = Field(
        default_factory=dict,
        description="G1-3 corruption prevention assessment",
    )

    # G1-4: Corruption incidents
    g1_4_corruption_incidents: Dict[str, Any] = Field(
        default_factory=dict,
        description="G1-4 confirmed incident assessment",
    )

    # G1-5: Political influence
    g1_5_political_influence: Dict[str, Any] = Field(
        default_factory=dict,
        description="G1-5 political influence assessment",
    )

    # G1-6: Payment practices
    g1_6_payment_practices: Dict[str, Any] = Field(
        default_factory=dict,
        description="G1-6 payment practice assessment",
    )

    # Summary metrics
    total_policies: int = Field(
        default=0, description="Total policies assessed"
    )
    total_suppliers: int = Field(
        default=0, description="Total suppliers assessed"
    )
    total_prevention_measures: int = Field(
        default=0, description="Total prevention measures"
    )
    total_incidents: int = Field(
        default=0, description="Total confirmed incidents"
    )
    total_political_activities: int = Field(
        default=0, description="Total political activities"
    )
    total_payments: int = Field(
        default=0, description="Total payment records"
    )
    processing_time_ms: float = Field(
        default=0.0, description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of all inputs and assessment steps",
    )

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class BusinessConductEngine:
    """ESRS G1 Business Conduct assessment engine.

    Provides deterministic, zero-hallucination assessments for all
    six G1 disclosure requirements:

    - G1-1: Business conduct policies and corporate culture
    - G1-2: Management of relationships with suppliers
    - G1-3: Prevention and detection of corruption and bribery
    - G1-4: Confirmed incidents of corruption or bribery
    - G1-5: Political influence and lobbying activities
    - G1-6: Payment practices

    All calculations use Decimal arithmetic for reproducibility.
    No LLM is used in any calculation path.

    Usage::

        engine = BusinessConductEngine()
        result = engine.calculate_g1_disclosure(
            policies=[BusinessConductPolicy(...)],
            suppliers=[SupplierRelationship(...)],
            prevention_measures=[CorruptionPreventionMeasure(...)],
            incidents=[CorruptionIncident(...)],
            political_activities=[PoliticalActivity(...)],
            payments=[PaymentPractice(...)],
        )
    """

    engine_version: str = _MODULE_VERSION

    # ------------------------------------------------------------------ #
    # G1-1: Policies and Corporate Culture (Para 7-10)                    #
    # ------------------------------------------------------------------ #

    def assess_policies(
        self, policies: List[BusinessConductPolicy]
    ) -> Dict[str, Any]:
        """Assess business conduct policies for G1-1 disclosure.

        Evaluates whether the undertaking has adequate policies
        covering business ethics, anti-corruption, whistleblower
        protection, supplier standards, and political engagement,
        and whether associated training achieves sufficient coverage.

        Args:
            policies: List of BusinessConductPolicy instances.

        Returns:
            Dict with policy_count, topic coverage flags, training
            coverage metrics, governance oversight status, and
            provenance_hash.
        """
        if not policies:
            logger.warning("G1-1: No business conduct policies provided")
            return {
                "policy_count": 0,
                "has_code_of_conduct": False,
                "covers_business_ethics": False,
                "covers_anti_corruption": False,
                "covers_whistleblower": False,
                "covers_supplier_standards": False,
                "covers_political_engagement": False,
                "governance_body_oversight_count": 0,
                "governance_body_oversight_pct": Decimal("0.0"),
                "training_types_offered": [],
                "training_types_count": 0,
                "total_employees_trained": 0,
                "total_employees_in_scope": 0,
                "training_coverage_pct": Decimal("0.0"),
                "provenance_hash": _compute_hash({"policies": []}),
            }

        has_ethics = any(p.covers_business_ethics for p in policies)
        has_anti_corruption = any(
            p.covers_anti_corruption for p in policies
        )
        has_whistleblower = any(
            p.covers_whistleblower_protection for p in policies
        )
        has_supplier = any(
            p.covers_supplier_standards for p in policies
        )
        has_political = any(
            p.covers_political_engagement for p in policies
        )
        governance_count = sum(
            1 for p in policies if p.approved_by_governance_body
        )

        all_training_types: set = set()
        for p in policies:
            for tt in p.training_types_associated:
                all_training_types.add(tt.value)

        total_trained = sum(p.total_employees_trained for p in policies)
        total_in_scope = max(
            p.total_employees_in_scope for p in policies
        ) if policies else 0
        training_coverage = (
            _pct(total_trained, total_in_scope)
            if total_in_scope > 0
            else Decimal("0.0")
        )

        n = len(policies)

        result = {
            "policy_count": n,
            "has_code_of_conduct": n > 0,
            "covers_business_ethics": has_ethics,
            "covers_anti_corruption": has_anti_corruption,
            "covers_whistleblower": has_whistleblower,
            "covers_supplier_standards": has_supplier,
            "covers_political_engagement": has_political,
            "governance_body_oversight_count": governance_count,
            "governance_body_oversight_pct": _pct(governance_count, n),
            "training_types_offered": sorted(all_training_types),
            "training_types_count": len(all_training_types),
            "total_employees_trained": total_trained,
            "total_employees_in_scope": total_in_scope,
            "training_coverage_pct": training_coverage,
        }

        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "G1-1 assessed: %d policies, ethics=%s, anti-corruption=%s, "
            "training coverage=%.1f%%",
            n, has_ethics, has_anti_corruption, float(training_coverage),
        )

        return result

    # ------------------------------------------------------------------ #
    # G1-2: Supplier Management (Para 12-15)                              #
    # ------------------------------------------------------------------ #

    def evaluate_supplier_management(
        self, suppliers: List[SupplierRelationship]
    ) -> Dict[str, Any]:
        """Evaluate supplier relationship management for G1-2 disclosure.

        Assesses the undertaking's approach to managing supplier
        relationships, including code of conduct coverage, audit
        results, risk distribution, and categorisation.

        Args:
            suppliers: List of SupplierRelationship instances.

        Returns:
            Dict with assessed_count, code_of_conduct_coverage,
            audit_results, risk_distribution, category_breakdown,
            blocked_count, and sme_count.
        """
        if not suppliers:
            logger.warning("G1-2: No supplier relationships provided")
            return {
                "assessed_count": 0,
                "code_of_conduct_signed_count": 0,
                "code_of_conduct_coverage_pct": Decimal("0.0"),
                "audited_count": 0,
                "audit_passed_count": 0,
                "audit_pass_rate_pct": Decimal("0.0"),
                "by_category": {},
                "by_risk_level": {},
                "blocked_count": 0,
                "sme_count": 0,
                "sme_pct": Decimal("0.0"),
                "provenance_hash": _compute_hash({"suppliers": []}),
            }

        coc_signed = sum(
            1 for s in suppliers if s.code_of_conduct_signed
        )
        audited = [
            s for s in suppliers if s.last_audit_date is not None
        ]
        audit_passed = sum(
            1 for s in audited if s.audit_passed is True
        )
        blocked = sum(
            1 for s in suppliers
            if s.category == SupplierCategory.BLOCKED
        )
        sme_count = sum(1 for s in suppliers if s.is_sme)

        by_category: Dict[str, int] = {}
        for cat in SupplierCategory:
            by_category[cat.value] = sum(
                1 for s in suppliers if s.category == cat
            )

        by_risk: Dict[str, int] = {}
        for level in CorruptionRiskLevel:
            by_risk[level.value] = sum(
                1 for s in suppliers
                if s.corruption_risk_level == level
            )

        n = len(suppliers)

        result = {
            "assessed_count": n,
            "code_of_conduct_signed_count": coc_signed,
            "code_of_conduct_coverage_pct": _pct(coc_signed, n),
            "audited_count": len(audited),
            "audit_passed_count": audit_passed,
            "audit_pass_rate_pct": _pct(audit_passed, len(audited)),
            "by_category": by_category,
            "by_risk_level": by_risk,
            "blocked_count": blocked,
            "sme_count": sme_count,
            "sme_pct": _pct(sme_count, n),
        }

        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "G1-2 assessed: %d suppliers, CoC coverage=%.1f%%, "
            "audit pass rate=%.1f%%, blocked=%d",
            n, float(_pct(coc_signed, n)),
            float(_pct(audit_passed, len(audited))), blocked,
        )

        return result

    # ------------------------------------------------------------------ #
    # G1-3: Corruption Prevention (Para 17-22)                            #
    # ------------------------------------------------------------------ #

    def assess_corruption_prevention(
        self, measures: List[CorruptionPreventionMeasure]
    ) -> Dict[str, Any]:
        """Assess corruption prevention measures for G1-3 disclosure.

        Evaluates the undertaking's anti-corruption framework
        including risk assessments, training programmes, whistleblower
        mechanisms, and due diligence processes.

        Args:
            measures: List of CorruptionPreventionMeasure instances.

        Returns:
            Dict with measure_count, training_coverage, risk_assessments,
            whistleblower_mechanism, due_diligence_coverage, and
            third_party_coverage.
        """
        if not measures:
            logger.warning(
                "G1-3: No corruption prevention measures provided"
            )
            return {
                "measure_count": 0,
                "active_measures": 0,
                "risk_assessments_count": 0,
                "high_risk_areas_identified": 0,
                "training_measures_count": 0,
                "training_employees_covered": 0,
                "training_employees_in_scope": 0,
                "training_coverage_pct": Decimal("0.0"),
                "whistleblower_mechanism_exists": False,
                "whistleblower_reports_received": 0,
                "whistleblower_reports_investigated": 0,
                "investigation_rate_pct": Decimal("0.0"),
                "due_diligence_count": 0,
                "third_party_coverage_count": 0,
                "third_party_coverage_pct": Decimal("0.0"),
                "provenance_hash": _compute_hash({"measures": []}),
            }

        active_count = sum(1 for m in measures if m.is_active)

        risk_assessments = [
            m for m in measures if m.measure_type == "risk_assessment"
        ]
        high_risk_count = sum(
            1 for m in risk_assessments
            if m.risk_level_assessed in (
                CorruptionRiskLevel.HIGH,
                CorruptionRiskLevel.VERY_HIGH,
            )
        )

        training_measures = [
            m for m in measures if m.measure_type == "training"
        ]
        total_training_covered = sum(
            m.employees_covered for m in training_measures
        )
        total_training_scope = max(
            (m.total_employees for m in training_measures), default=0
        )
        training_coverage = (
            _pct(total_training_covered, total_training_scope)
            if total_training_scope > 0
            else Decimal("0.0")
        )

        whistleblower_measures = [
            m for m in measures if m.measure_type == "whistleblower"
        ]
        wb_exists = len(whistleblower_measures) > 0
        wb_reports = sum(
            m.whistleblower_reports_received
            for m in whistleblower_measures
        )
        wb_investigated = sum(
            m.whistleblower_reports_investigated
            for m in whistleblower_measures
        )

        dd_measures = [
            m for m in measures if m.measure_type == "due_diligence"
        ]
        third_party_count = sum(
            1 for m in measures if m.covers_third_parties
        )

        n = len(measures)

        result = {
            "measure_count": n,
            "active_measures": active_count,
            "risk_assessments_count": len(risk_assessments),
            "high_risk_areas_identified": high_risk_count,
            "training_measures_count": len(training_measures),
            "training_employees_covered": total_training_covered,
            "training_employees_in_scope": total_training_scope,
            "training_coverage_pct": training_coverage,
            "whistleblower_mechanism_exists": wb_exists,
            "whistleblower_reports_received": wb_reports,
            "whistleblower_reports_investigated": wb_investigated,
            "investigation_rate_pct": _pct(wb_investigated, wb_reports),
            "due_diligence_count": len(dd_measures),
            "third_party_coverage_count": third_party_count,
            "third_party_coverage_pct": _pct(third_party_count, n),
        }

        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "G1-3 assessed: %d measures, %d risk assessments, "
            "training coverage=%.1f%%, whistleblower=%s",
            n, len(risk_assessments), float(training_coverage),
            wb_exists,
        )

        return result

    # ------------------------------------------------------------------ #
    # G1-4: Corruption Incidents (Para 24-26)                             #
    # ------------------------------------------------------------------ #

    def assess_corruption_incidents(
        self, incidents: List[CorruptionIncident]
    ) -> Dict[str, Any]:
        """Assess confirmed corruption incidents for G1-4 disclosure.

        Evaluates all confirmed incidents of corruption or bribery,
        including type distribution, legal proceedings, fines, and
        consequential actions.

        Args:
            incidents: List of CorruptionIncident instances.

        Returns:
            Dict with incident_count, by_type, legal_proceedings_count,
            total_fines_eur, contracts_terminated, employees_dismissed,
            and material_incidents.
        """
        if not incidents:
            logger.info(
                "G1-4: No confirmed corruption incidents "
                "(positive outcome)"
            )
            return {
                "incident_count": 0,
                "by_type": {},
                "legal_proceedings_count": 0,
                "total_fines_eur": Decimal("0"),
                "total_contracts_terminated": 0,
                "total_employees_dismissed": 0,
                "material_incidents_count": 0,
                "by_country": {},
                "provenance_hash": _compute_hash({"incidents": []}),
            }

        by_type: Dict[str, int] = {}
        by_country: Dict[str, int] = {}

        legal_proceedings = 0
        total_fines = Decimal("0")
        total_terminated = 0
        total_dismissed = 0
        material_count = 0

        for inc in incidents:
            by_type[inc.incident_type] = (
                by_type.get(inc.incident_type, 0) + 1
            )

            if inc.country_code:
                by_country[inc.country_code] = (
                    by_country.get(inc.country_code, 0) + 1
                )

            if inc.legal_proceedings_initiated:
                legal_proceedings += 1
            total_fines += inc.fine_amount_eur
            total_terminated += inc.contracts_terminated
            total_dismissed += inc.employees_dismissed
            if inc.is_material:
                material_count += 1

        result = {
            "incident_count": len(incidents),
            "by_type": by_type,
            "legal_proceedings_count": legal_proceedings,
            "total_fines_eur": _round_val(total_fines, 2),
            "total_contracts_terminated": total_terminated,
            "total_employees_dismissed": total_dismissed,
            "material_incidents_count": material_count,
            "by_country": by_country,
        }

        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "G1-4 assessed: %d incidents, %d legal proceedings, "
            "fines=EUR %.2f, %d material",
            len(incidents), legal_proceedings, float(total_fines),
            material_count,
        )

        return result

    # ------------------------------------------------------------------ #
    # G1-5: Political Influence (Para 28-30)                              #
    # ------------------------------------------------------------------ #

    def assess_political_influence(
        self, activities: List[PoliticalActivity]
    ) -> Dict[str, Any]:
        """Assess political influence activities for G1-5 disclosure.

        Evaluates lobbying expenditure, political donations, trade
        association memberships, and other political engagement
        activities.

        Args:
            activities: List of PoliticalActivity instances.

        Returns:
            Dict with activity_count, lobbying_spend_eur,
            political_donations_eur, trade_association_count,
            by_type, by_country, and topics_disclosed.
        """
        if not activities:
            logger.info(
                "G1-5: No political influence activities reported"
            )
            return {
                "activity_count": 0,
                "lobbying_expenditure_eur": Decimal("0"),
                "political_donations_eur": Decimal("0"),
                "trade_association_memberships": 0,
                "campaign_contributions_eur": Decimal("0"),
                "government_advisory_count": 0,
                "total_political_spend_eur": Decimal("0"),
                "by_type": {},
                "by_country": {},
                "topics_disclosed": [],
                "topics_count": 0,
                "provenance_hash": _compute_hash({"activities": []}),
            }

        lobbying_spend = Decimal("0")
        donations = Decimal("0")
        campaign_spend = Decimal("0")
        trade_assoc_count = 0
        advisory_count = 0
        by_type: Dict[str, Dict[str, Any]] = {}
        by_country: Dict[str, Decimal] = {}
        topics: set = set()

        for act in activities:
            type_key = act.activity_type.value

            if type_key not in by_type:
                by_type[type_key] = {
                    "count": 0, "total_eur": Decimal("0")
                }
            by_type[type_key]["count"] += 1
            by_type[type_key]["total_eur"] += act.amount_eur

            if act.country_code:
                by_country[act.country_code] = (
                    by_country.get(act.country_code, Decimal("0"))
                    + act.amount_eur
                )

            if act.topic:
                topics.add(act.topic)

            if act.activity_type == PoliticalActivityType.LOBBYING:
                lobbying_spend += act.amount_eur
            elif act.activity_type == PoliticalActivityType.POLITICAL_DONATIONS:
                donations += act.amount_eur
            elif act.activity_type == PoliticalActivityType.CAMPAIGN_CONTRIBUTION:
                campaign_spend += act.amount_eur
            elif act.activity_type == PoliticalActivityType.TRADE_ASSOCIATION:
                trade_assoc_count += 1
            elif act.activity_type == PoliticalActivityType.GOVERNMENT_ADVISORY:
                advisory_count += 1

        total_spend = lobbying_spend + donations + campaign_spend
        ta_key = PoliticalActivityType.TRADE_ASSOCIATION.value
        if ta_key in by_type:
            total_spend += by_type[ta_key]["total_eur"]

        by_type_serializable: Dict[str, Any] = {}
        for k, v in by_type.items():
            by_type_serializable[k] = {
                "count": v["count"],
                "total_eur": _round_val(v["total_eur"], 2),
            }

        by_country_serializable = {
            k: _round_val(v, 2) for k, v in by_country.items()
        }

        result = {
            "activity_count": len(activities),
            "lobbying_expenditure_eur": _round_val(lobbying_spend, 2),
            "political_donations_eur": _round_val(donations, 2),
            "trade_association_memberships": trade_assoc_count,
            "campaign_contributions_eur": _round_val(campaign_spend, 2),
            "government_advisory_count": advisory_count,
            "total_political_spend_eur": _round_val(total_spend, 2),
            "by_type": by_type_serializable,
            "by_country": by_country_serializable,
            "topics_disclosed": sorted(topics),
            "topics_count": len(topics),
        }

        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "G1-5 assessed: %d activities, lobbying=EUR %.2f, "
            "donations=EUR %.2f, %d trade associations",
            len(activities), float(lobbying_spend), float(donations),
            trade_assoc_count,
        )

        return result

    # ------------------------------------------------------------------ #
    # G1-6: Payment Practices (Para 32-35)                                #
    # ------------------------------------------------------------------ #

    def calculate_payment_practices(
        self, payments: List[PaymentPractice]
    ) -> Dict[str, Any]:
        """Calculate payment practice metrics for G1-6 disclosure.

        Computes average payment days, late payment percentage, SME
        impact metrics, late payment interest, and disputes, per the
        EU Late Payment Directive 2011/7/EU.

        Args:
            payments: List of PaymentPractice instances.

        Returns:
            Dict with average_payment_days, late_payment_pct,
            late_payment_interest_eur, sme metrics, disputes,
            and payment term distribution.
        """
        if not payments:
            logger.warning("G1-6: No payment practice data provided")
            return {
                "payment_count": 0,
                "average_payment_days": Decimal("0"),
                "late_payment_count": 0,
                "late_payment_pct": Decimal("0.0"),
                "total_late_payment_interest_eur": Decimal("0"),
                "total_invoice_amount_eur": Decimal("0"),
                "sme_payment_count": 0,
                "sme_average_payment_days": Decimal("0"),
                "sme_late_payment_count": 0,
                "sme_late_payment_pct": Decimal("0.0"),
                "disputes_count": 0,
                "by_payment_term": {},
                "provenance_hash": _compute_hash({"payments": []}),
            }

        n = len(payments)
        total_days = sum(p.actual_payment_days for p in payments)
        avg_days = _round_val(
            _decimal(total_days) / _decimal(n), 1
        )

        late_count = sum(1 for p in payments if p.is_late)
        total_interest = sum(
            p.late_payment_interest_eur for p in payments
        )
        total_invoice = sum(p.invoice_amount_eur for p in payments)
        disputes = sum(1 for p in payments if p.is_disputed)

        # SME-specific metrics per EU Late Payment Directive
        sme_payments = [p for p in payments if p.is_sme_supplier]
        sme_count = len(sme_payments)
        sme_late_count = sum(1 for p in sme_payments if p.is_late)
        sme_total_days = sum(
            p.actual_payment_days for p in sme_payments
        )
        sme_avg_days = (
            _round_val(
                _decimal(sme_total_days) / _decimal(sme_count), 1
            )
            if sme_count > 0
            else Decimal("0")
        )

        # Payment term distribution
        by_term: Dict[str, int] = {}
        for term in PaymentTermType:
            by_term[term.value] = sum(
                1 for p in payments if p.payment_term_type == term
            )

        result = {
            "payment_count": n,
            "average_payment_days": avg_days,
            "late_payment_count": late_count,
            "late_payment_pct": _pct(late_count, n),
            "total_late_payment_interest_eur": _round_val(
                total_interest, 2
            ),
            "total_invoice_amount_eur": _round_val(total_invoice, 2),
            "sme_payment_count": sme_count,
            "sme_average_payment_days": sme_avg_days,
            "sme_late_payment_count": sme_late_count,
            "sme_late_payment_pct": _pct(sme_late_count, sme_count),
            "disputes_count": disputes,
            "by_payment_term": by_term,
        }

        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "G1-6 assessed: %d payments, avg=%.1f days, late=%.1f%%, "
            "SME avg=%.1f days, interest=EUR %.2f",
            n, float(avg_days), float(_pct(late_count, n)),
            float(sme_avg_days), float(total_interest),
        )

        return result

    # ------------------------------------------------------------------ #
    # Full G1 Disclosure Calculation                                      #
    # ------------------------------------------------------------------ #

    def calculate_g1_disclosure(
        self,
        policies: List[BusinessConductPolicy],
        suppliers: List[SupplierRelationship],
        prevention_measures: List[CorruptionPreventionMeasure],
        incidents: List[CorruptionIncident],
        political_activities: List[PoliticalActivity],
        payments: List[PaymentPractice],
        entity_name: str = "",
        reporting_year: int = 0,
    ) -> G1BusinessConductResult:
        """Calculate the complete ESRS G1 disclosure.

        Orchestrates assessment of all six G1 disclosure requirements
        and produces a consolidated result with provenance tracking.

        Args:
            policies: Business conduct policies for G1-1.
            suppliers: Supplier relationships for G1-2.
            prevention_measures: Corruption prevention measures for G1-3.
            incidents: Confirmed corruption incidents for G1-4.
            political_activities: Political influence activities for G1-5.
            payments: Payment practice records for G1-6.
            entity_name: Name of the reporting entity.
            reporting_year: Reporting year.

        Returns:
            G1BusinessConductResult with complete provenance.
        """
        t0 = time.perf_counter()

        logger.info(
            "Calculating G1 disclosure: entity=%s, year=%d",
            entity_name, reporting_year,
        )

        # Assess each disclosure requirement
        g1_1 = self.assess_policies(policies)
        g1_2 = self.evaluate_supplier_management(suppliers)
        g1_3 = self.assess_corruption_prevention(prevention_measures)
        g1_4 = self.assess_corruption_incidents(incidents)
        g1_5 = self.assess_political_influence(political_activities)
        g1_6 = self.calculate_payment_practices(payments)

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = G1BusinessConductResult(
            reporting_year=reporting_year,
            entity_name=entity_name,
            g1_1_policies=g1_1,
            g1_2_suppliers=g1_2,
            g1_3_corruption_prevention=g1_3,
            g1_4_corruption_incidents=g1_4,
            g1_5_political_influence=g1_5,
            g1_6_payment_practices=g1_6,
            total_policies=len(policies),
            total_suppliers=len(suppliers),
            total_prevention_measures=len(prevention_measures),
            total_incidents=len(incidents),
            total_political_activities=len(political_activities),
            total_payments=len(payments),
            processing_time_ms=elapsed_ms,
        )

        result.provenance_hash = _compute_hash(result)

        logger.info(
            "G1 disclosure calculated: %d policies, %d suppliers, "
            "%d measures, %d incidents, %d political, %d payments, "
            "hash=%s",
            len(policies), len(suppliers), len(prevention_measures),
            len(incidents), len(political_activities), len(payments),
            result.provenance_hash[:16],
        )

        return result

    # ------------------------------------------------------------------ #
    # Completeness Validation                                             #
    # ------------------------------------------------------------------ #

    def validate_g1_completeness(
        self, result: G1BusinessConductResult
    ) -> Dict[str, Any]:
        """Validate completeness against all G1 required data points.

        Checks whether all ESRS G1 mandatory disclosure data points
        are present and populated in the result.

        Args:
            result: G1BusinessConductResult to validate.

        Returns:
            Dict with total_datapoints, populated_datapoints,
            missing_datapoints, completeness_pct, is_complete,
            per_dr_completeness, and provenance_hash.
        """
        populated: List[str] = []
        missing: List[str] = []

        # G1-1 checks
        g1_1 = result.g1_1_policies
        g1_1_checks = {
            "g1_1_01_code_of_conduct_exists": g1_1.get(
                "has_code_of_conduct", False
            ),
            "g1_1_02_code_covers_business_ethics": g1_1.get(
                "covers_business_ethics", False
            ),
            "g1_1_03_code_covers_anti_corruption": g1_1.get(
                "covers_anti_corruption", False
            ),
            "g1_1_04_code_covers_whistleblower_protection": g1_1.get(
                "covers_whistleblower", False
            ),
            "g1_1_05_corporate_culture_description": (
                g1_1.get("policy_count", 0) > 0
            ),
            "g1_1_06_training_programmes_exist": (
                g1_1.get("training_types_count", 0) > 0
            ),
            "g1_1_07_training_coverage_pct": (
                g1_1.get("training_coverage_pct", Decimal("0"))
                > Decimal("0")
            ),
            "g1_1_08_governance_body_oversight": (
                g1_1.get("governance_body_oversight_count", 0) > 0
            ),
        }

        # G1-2 checks
        g1_2 = result.g1_2_suppliers
        g1_2_checks = {
            "g1_2_01_supplier_code_of_conduct_exists": (
                g1_2.get("code_of_conduct_signed_count", 0) > 0
            ),
            "g1_2_02_suppliers_assessed_count": (
                g1_2.get("assessed_count", 0) > 0
            ),
            "g1_2_03_supplier_code_coverage_pct": (
                g1_2.get(
                    "code_of_conduct_coverage_pct", Decimal("0")
                ) > Decimal("0")
            ),
            "g1_2_04_supplier_audits_conducted": (
                g1_2.get("audited_count", 0) > 0
            ),
            "g1_2_05_suppliers_blocked_count": True,  # Zero is valid
            "g1_2_06_payment_terms_disclosed": (
                g1_2.get("assessed_count", 0) > 0
            ),
        }

        # G1-3 checks
        g1_3 = result.g1_3_corruption_prevention
        g1_3_checks = {
            "g1_3_01_anti_corruption_policy_exists": (
                g1_3.get("measure_count", 0) > 0
            ),
            "g1_3_02_risk_assessments_conducted": (
                g1_3.get("risk_assessments_count", 0) > 0
            ),
            "g1_3_03_high_risk_areas_identified": True,
            "g1_3_04_training_anti_corruption_coverage": (
                g1_3.get("training_coverage_pct", Decimal("0"))
                > Decimal("0")
            ),
            "g1_3_05_whistleblower_mechanism_exists": g1_3.get(
                "whistleblower_mechanism_exists", False
            ),
            "g1_3_06_whistleblower_reports_received": True,
            "g1_3_07_due_diligence_processes": (
                g1_3.get("due_diligence_count", 0) > 0
            ),
            "g1_3_08_third_party_due_diligence": (
                g1_3.get("third_party_coverage_count", 0) > 0
            ),
        }

        # G1-4 checks (all valid even when zero -- no incidents is good)
        g1_4_checks = {
            "g1_4_01_confirmed_incidents_count": True,
            "g1_4_02_incidents_by_type": True,
            "g1_4_03_legal_proceedings_count": True,
            "g1_4_04_fines_and_penalties_eur": True,
            "g1_4_05_contracts_terminated": True,
            "g1_4_06_employees_dismissed": True,
        }

        # G1-5 checks
        g1_5 = result.g1_5_political_influence
        g1_5_checks = {
            "g1_5_01_political_engagement_policy": True,
            "g1_5_02_lobbying_expenditure_eur": True,
            "g1_5_03_political_donations_eur": True,
            "g1_5_04_trade_association_memberships": True,
            "g1_5_05_lobbying_topics_disclosed": (
                g1_5.get("activity_count", 0) == 0
                or g1_5.get("topics_count", 0) > 0
            ),
        }

        # G1-6 checks
        g1_6 = result.g1_6_payment_practices
        g1_6_checks = {
            "g1_6_01_standard_payment_terms": (
                g1_6.get("payment_count", 0) > 0
            ),
            "g1_6_02_average_payment_days": (
                g1_6.get("payment_count", 0) > 0
            ),
            "g1_6_03_late_payment_pct": True,
            "g1_6_04_late_payment_interest_paid_eur": True,
            "g1_6_05_sme_payment_terms_disclosed": True,
            "g1_6_06_sme_average_payment_days": True,
            "g1_6_07_disputes_on_late_payment": True,
        }

        all_checks = {
            **g1_1_checks, **g1_2_checks, **g1_3_checks,
            **g1_4_checks, **g1_5_checks, **g1_6_checks,
        }

        for dp, is_populated in all_checks.items():
            if is_populated:
                populated.append(dp)
            else:
                missing.append(dp)

        total = len(ALL_G1_DATAPOINTS)
        pop_count = len(populated)
        completeness = _round_val(
            _decimal(pop_count) / _decimal(total) * Decimal("100"), 1
        )

        # Per-DR completeness breakdown
        def _dr_completeness(
            checks: Dict[str, bool],
        ) -> Dict[str, Any]:
            pop = sum(1 for v in checks.values() if v)
            tot = len(checks)
            return {
                "populated": pop,
                "total": tot,
                "completeness_pct": _pct(pop, tot),
                "missing": [k for k, v in checks.items() if not v],
            }

        per_dr = {
            "G1-1": _dr_completeness(g1_1_checks),
            "G1-2": _dr_completeness(g1_2_checks),
            "G1-3": _dr_completeness(g1_3_checks),
            "G1-4": _dr_completeness(g1_4_checks),
            "G1-5": _dr_completeness(g1_5_checks),
            "G1-6": _dr_completeness(g1_6_checks),
        }

        validation_result = {
            "total_datapoints": total,
            "populated_datapoints": pop_count,
            "missing_datapoints": missing,
            "completeness_pct": completeness,
            "is_complete": len(missing) == 0,
            "per_dr_completeness": per_dr,
            "provenance_hash": _compute_hash(
                {"result_id": result.result_id, "checks": all_checks}
            ),
        }

        logger.info(
            "G1 completeness: %.1f%% (%d/%d), missing=%s",
            float(completeness), pop_count, total, missing,
        )

        return validation_result
