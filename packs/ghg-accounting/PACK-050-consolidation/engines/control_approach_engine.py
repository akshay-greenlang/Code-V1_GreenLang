"""
PACK-050 GHG Consolidation Pack - Control Approach Engine
====================================================================

Implements the operational control and financial control
consolidation approaches per GHG Protocol Corporate Standard
Chapter 3. Provides binary inclusion logic (100% or 0%),
control assessment tests, franchise boundary decisions,
outsourcing boundary decisions, and leased asset boundary
decisions per GHG Protocol guidance.

Regulatory Basis:
    - GHG Protocol Corporate Standard (Chapter 3): Operational
      Control - "A company has operational control over an
      operation if it or one of its subsidiaries has the full
      authority to introduce and implement its operating policies."
    - GHG Protocol Corporate Standard (Chapter 3): Financial
      Control - "A company has financial control over an operation
      if it has the ability to direct the financial and operating
      policies of the operation with a view to gaining economic
      benefits from its activities."
    - GHG Protocol Corporate Standard (Appendix B): Franchise
      and outsourcing boundary decisions.
    - GHG Protocol Scope 2 Guidance: Leased asset boundary
      for electricity and energy purchases.
    - ISO 14064-1:2018 (Clause 5.1): Control-based approaches.

Calculation Methodology:
    Operational Control:
        inclusion_pct = 100 if has_operational_control else 0

    Financial Control:
        inclusion_pct = 100 if has_financial_control else 0

    Entity Contribution:
        contribution = entity_emissions * inclusion_pct / 100
        (either 100% or 0% of entity emissions)

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - No LLM involvement in any calculation or assessment path
    - SHA-256 provenance hash on every result object

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-050 GHG Consolidation
Engine:  5 of 5
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)
_MODULE_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC timestamp with second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 provenance hash, excluding volatile fields."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("created_at", "updated_at", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal:
    """Safely convert any value to Decimal."""
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
    """Divide safely, returning *default* when denominator is zero."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _round2(value: Any) -> Decimal:
    """Round a value to two decimal places using ROUND_HALF_UP."""
    return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _round4(value: Any) -> Decimal:
    """Round a value to four decimal places using ROUND_HALF_UP."""
    return Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ControlApproachType(str, Enum):
    """The two control-based consolidation approaches."""
    OPERATIONAL_CONTROL = "OPERATIONAL_CONTROL"
    FINANCIAL_CONTROL = "FINANCIAL_CONTROL"


class FranchiseRole(str, Enum):
    """Role in a franchise arrangement."""
    FRANCHISER = "FRANCHISER"
    FRANCHISEE = "FRANCHISEE"


class LeaseType(str, Enum):
    """Type of lease arrangement for GHG boundary decisions."""
    FINANCE_LEASE = "FINANCE_LEASE"
    OPERATING_LEASE = "OPERATING_LEASE"


class LeaseRole(str, Enum):
    """Role in a lease arrangement."""
    LESSEE = "LESSEE"
    LESSOR = "LESSOR"


class OutsourceType(str, Enum):
    """Type of outsourced operation."""
    MANUFACTURING = "MANUFACTURING"
    LOGISTICS = "LOGISTICS"
    DATA_PROCESSING = "DATA_PROCESSING"
    FACILITIES_MANAGEMENT = "FACILITIES_MANAGEMENT"
    WASTE_MANAGEMENT = "WASTE_MANAGEMENT"
    OTHER = "OTHER"


class ScopeType(str, Enum):
    """GHG emission scope categories."""
    SCOPE_1 = "SCOPE_1"
    SCOPE_2_LOCATION = "SCOPE_2_LOCATION"
    SCOPE_2_MARKET = "SCOPE_2_MARKET"
    SCOPE_3 = "SCOPE_3"


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class ControlInput(BaseModel):
    """Input data for control approach calculation.

    Contains the entity's 100% emissions and control indicators
    needed for both operational and financial control approaches.
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    entity_id: str = Field(
        ...,
        description="Entity identifier.",
    )
    entity_name: Optional[str] = Field(
        None,
        description="Human-readable entity name.",
    )
    has_operational_control: bool = Field(
        default=False,
        description="Whether reporting org has operational control.",
    )
    has_financial_control: bool = Field(
        default=False,
        description="Whether reporting org has financial control.",
    )
    manages_operations: bool = Field(
        default=False,
        description="Manages day-to-day operations.",
    )
    directs_policies: bool = Field(
        default=False,
        description="Directs financial/operating policies.",
    )
    has_board_majority: bool = Field(
        default=False,
        description="Holds majority of board seats.",
    )
    equity_pct: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Equity ownership percentage (for reference).",
    )
    scope1: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="100% Scope 1 emissions (tCO2e).",
    )
    scope2_location: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="100% Scope 2 location-based (tCO2e).",
    )
    scope2_market: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="100% Scope 2 market-based (tCO2e).",
    )
    scope3: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="100% Scope 3 (tCO2e).",
    )
    scope3_categories: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Scope 3 by category (tCO2e).",
    )

    @field_validator("equity_pct", "scope1", "scope2_location",
                     "scope2_market", "scope3", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Any:
        return Decimal(str(v))


class ControlResult(BaseModel):
    """Result of applying a control approach to a single entity.

    Shows the binary inclusion decision (100% or 0%) and the
    resulting emissions contribution.
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    entity_id: str = Field(
        ...,
        description="Entity identifier.",
    )
    entity_name: Optional[str] = Field(
        None,
        description="Entity name.",
    )
    approach: str = Field(
        ...,
        description="Control approach applied.",
    )
    has_control: bool = Field(
        ...,
        description="Whether control exists under this approach.",
    )
    inclusion_pct: Decimal = Field(
        ...,
        description="Inclusion percentage (100 or 0).",
    )
    scope1_contribution: Decimal = Field(
        default=Decimal("0"),
        description="Scope 1 contribution (tCO2e).",
    )
    scope2_location_contribution: Decimal = Field(
        default=Decimal("0"),
        description="Scope 2 location contribution (tCO2e).",
    )
    scope2_market_contribution: Decimal = Field(
        default=Decimal("0"),
        description="Scope 2 market contribution (tCO2e).",
    )
    scope3_contribution: Decimal = Field(
        default=Decimal("0"),
        description="Scope 3 contribution (tCO2e).",
    )
    total_contribution: Decimal = Field(
        default=Decimal("0"),
        description="Total contribution (S1+S2loc+S3).",
    )
    assessment_basis: List[str] = Field(
        default_factory=list,
        description="Basis for the control determination.",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash.",
    )

    @field_validator("inclusion_pct", "scope1_contribution",
                     "scope2_location_contribution",
                     "scope2_market_contribution",
                     "scope3_contribution", "total_contribution",
                     mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Any:
        return Decimal(str(v))


class ControlAssessmentDetail(BaseModel):
    """Detailed control assessment for documentation purposes.

    Provides structured evidence for the control determination,
    suitable for assurance and audit review.
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    assessment_id: str = Field(
        default_factory=_new_uuid,
        description="Unique assessment identifier.",
    )
    entity_id: str = Field(
        ...,
        description="Entity being assessed.",
    )
    operational_control_result: bool = Field(
        ...,
        description="Operational control determination.",
    )
    financial_control_result: bool = Field(
        ...,
        description="Financial control determination.",
    )
    operational_criteria: Dict[str, bool] = Field(
        default_factory=dict,
        description="Operational control criteria checklist.",
    )
    financial_criteria: Dict[str, bool] = Field(
        default_factory=dict,
        description="Financial control criteria checklist.",
    )
    justification: str = Field(
        default="",
        description="Narrative justification.",
    )
    evidence_references: List[str] = Field(
        default_factory=list,
        description="References to supporting evidence.",
    )
    assessor: Optional[str] = Field(
        None,
        description="Person who performed the assessment.",
    )
    assessment_date: date = Field(
        default_factory=date.today,
        description="Date of assessment.",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash.",
    )


class FranchiseBoundary(BaseModel):
    """Franchise boundary decision per GHG Protocol Appendix B.

    Documents whether franchise emissions should be reported
    by the franchiser or franchisee under the chosen approach.
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    decision_id: str = Field(
        default_factory=_new_uuid,
        description="Unique decision identifier.",
    )
    franchise_entity_id: str = Field(
        ...,
        description="Entity ID of the franchise operation.",
    )
    franchiser_entity_id: str = Field(
        ...,
        description="Entity ID of the franchiser.",
    )
    franchise_role: str = Field(
        ...,
        description="Reporting org's role (FRANCHISER or FRANCHISEE).",
    )
    approach: str = Field(
        ...,
        description="Consolidation approach used.",
    )
    franchiser_controls_operations: bool = Field(
        default=False,
        description="Whether franchiser controls franchise operations.",
    )
    include_in_scope1_2: bool = Field(
        ...,
        description="Whether to include in Scope 1/2.",
    )
    report_in_scope3: bool = Field(
        ...,
        description="Whether to report in Scope 3 Cat 14 (Franchises).",
    )
    rationale: str = Field(
        default="",
        description="Rationale for the boundary decision.",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash.",
    )


class LeaseBoundary(BaseModel):
    """Lease boundary decision per GHG Protocol guidance.

    Documents whether leased asset emissions should be reported
    by the lessee or lessor under the chosen approach.
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    decision_id: str = Field(
        default_factory=_new_uuid,
        description="Unique decision identifier.",
    )
    asset_entity_id: str = Field(
        ...,
        description="Entity/asset ID of the leased operation.",
    )
    lease_type: str = Field(
        ...,
        description="Type of lease (FINANCE_LEASE or OPERATING_LEASE).",
    )
    reporting_role: str = Field(
        ...,
        description="Reporting org's role (LESSEE or LESSOR).",
    )
    approach: str = Field(
        ...,
        description="Consolidation approach used.",
    )
    include_in_scope1_2: bool = Field(
        ...,
        description="Include in Scope 1/2 boundary.",
    )
    report_in_scope3: bool = Field(
        ...,
        description="Report in Scope 3 (Cat 8 or Cat 13).",
    )
    rationale: str = Field(
        default="",
        description="Rationale for the boundary decision.",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash.",
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ControlApproachEngine:
    """Implements control-based GHG consolidation approaches.

    Provides binary inclusion (100% or 0%) based on operational
    or financial control, handles franchise and lease boundary
    decisions, and produces consolidated results with full
    assessment documentation.

    Attributes:
        _results: Dict mapping entity_id to ControlResult.
        _assessments: List of ControlAssessmentDetail records.
        _franchise_decisions: List of FranchiseBoundary records.
        _lease_decisions: List of LeaseBoundary records.
        _change_log: Append-only audit log.

    Example:
        >>> engine = ControlApproachEngine()
        >>> result = engine.apply_control_approach(
        ...     ControlInput(
        ...         entity_id="E1",
        ...         has_operational_control=True,
        ...         scope1=Decimal("1000"),
        ...     ),
        ...     approach="OPERATIONAL_CONTROL",
        ... )
        >>> assert result.inclusion_pct == Decimal("100")
    """

    def __init__(self) -> None:
        """Initialise the ControlApproachEngine."""
        self._results: Dict[str, ControlResult] = {}
        self._assessments: List[ControlAssessmentDetail] = []
        self._franchise_decisions: List[FranchiseBoundary] = []
        self._lease_decisions: List[LeaseBoundary] = []
        self._change_log: List[Dict[str, Any]] = []
        logger.info("ControlApproachEngine v%s initialised.", _MODULE_VERSION)

    # ------------------------------------------------------------------
    # Core Control Application
    # ------------------------------------------------------------------

    def apply_control_approach(
        self,
        entity_input: ControlInput,
        approach: str,
    ) -> ControlResult:
        """Apply a control approach to a single entity.

        Determines binary inclusion (100% or 0%) and calculates
        the resulting emissions contribution.

        Formula:
            inclusion_pct = 100 if has_control else 0
            contribution = entity_emissions * inclusion_pct / 100

        Args:
            entity_input: Entity emissions and control indicators.
            approach: Control approach (OPERATIONAL_CONTROL or
                FINANCIAL_CONTROL).

        Returns:
            ControlResult with inclusion decision and contributions.

        Raises:
            ValueError: If approach is invalid.
        """
        approach_upper = approach.upper()
        valid = {ca.value for ca in ControlApproachType}
        if approach_upper not in valid:
            raise ValueError(
                f"Invalid control approach '{approach}'. "
                f"Must be one of {sorted(valid)}."
            )

        logger.info(
            "Applying %s to entity '%s'.",
            approach_upper, entity_input.entity_id,
        )

        # Determine control
        has_control = False
        basis: List[str] = []

        if approach_upper == ControlApproachType.OPERATIONAL_CONTROL.value:
            has_control, basis = self._assess_operational_control(entity_input)
        elif approach_upper == ControlApproachType.FINANCIAL_CONTROL.value:
            has_control, basis = self._assess_financial_control(entity_input)

        # Binary inclusion
        inclusion_pct = Decimal("100") if has_control else Decimal("0")
        multiplier = _safe_divide(inclusion_pct, Decimal("100"))

        s1_contrib = _round2(entity_input.scope1 * multiplier)
        s2_loc_contrib = _round2(entity_input.scope2_location * multiplier)
        s2_mkt_contrib = _round2(entity_input.scope2_market * multiplier)
        s3_contrib = _round2(entity_input.scope3 * multiplier)
        total_contrib = _round2(s1_contrib + s2_loc_contrib + s3_contrib)

        result = ControlResult(
            entity_id=entity_input.entity_id,
            entity_name=entity_input.entity_name,
            approach=approach_upper,
            has_control=has_control,
            inclusion_pct=inclusion_pct,
            scope1_contribution=s1_contrib,
            scope2_location_contribution=s2_loc_contrib,
            scope2_market_contribution=s2_mkt_contrib,
            scope3_contribution=s3_contrib,
            total_contribution=total_contrib,
            assessment_basis=basis,
        )
        result.provenance_hash = _compute_hash(result)
        self._results[entity_input.entity_id] = result

        self._change_log.append({
            "event": "CONTROL_APPLIED",
            "entity_id": entity_input.entity_id,
            "approach": approach_upper,
            "has_control": has_control,
            "inclusion_pct": str(inclusion_pct),
            "total_contribution": str(total_contrib),
            "timestamp": _utcnow().isoformat(),
        })

        logger.info(
            "Entity '%s' under %s: control=%s, inclusion=%s%%, "
            "contribution=%s tCO2e.",
            entity_input.entity_id, approach_upper,
            has_control, inclusion_pct, total_contrib,
        )
        return result

    # ------------------------------------------------------------------
    # Operational Control Assessment
    # ------------------------------------------------------------------

    def _assess_operational_control(
        self,
        entity_input: ControlInput,
    ) -> Tuple[bool, List[str]]:
        """Assess operational control for an entity.

        GHG Protocol defines operational control as: the reporting
        entity or one of its subsidiaries has the full authority
        to introduce and implement operating policies at the
        operation.

        Args:
            entity_input: Entity with control indicators.

        Returns:
            Tuple of (has_control, assessment_basis).
        """
        basis: List[str] = []

        if entity_input.has_operational_control:
            basis.append("Entity explicitly flagged as under operational control.")
            return True, basis

        if entity_input.manages_operations:
            basis.append(
                "Entity's operations are managed by the reporting "
                "organisation (manages day-to-day operations)."
            )
            return True, basis

        if entity_input.equity_pct >= Decimal("100"):
            basis.append(
                "Wholly-owned subsidiary (100% equity) implies "
                "operational control."
            )
            return True, basis

        basis.append(
            f"No operational control indicators found for entity "
            f"with {entity_input.equity_pct}% equity."
        )
        return False, basis

    def assess_operational_control(
        self,
        entity_input: ControlInput,
    ) -> ControlAssessmentDetail:
        """Perform detailed operational control assessment.

        Creates a structured assessment record suitable for
        assurance review and audit documentation.

        Args:
            entity_input: Entity with control indicators.

        Returns:
            ControlAssessmentDetail with full criteria checklist.
        """
        has_oper, oper_basis = self._assess_operational_control(entity_input)
        has_fin, fin_basis = self._assess_financial_control(entity_input)

        oper_criteria = {
            "has_operational_control_flag": entity_input.has_operational_control,
            "manages_operations": entity_input.manages_operations,
            "wholly_owned": entity_input.equity_pct >= Decimal("100"),
            "majority_owned": entity_input.equity_pct > Decimal("50"),
        }

        fin_criteria = {
            "has_financial_control_flag": entity_input.has_financial_control,
            "directs_policies": entity_input.directs_policies,
            "has_board_majority": entity_input.has_board_majority,
            "majority_equity": entity_input.equity_pct > Decimal("50"),
        }

        justification = " ".join(oper_basis)

        assessment = ControlAssessmentDetail(
            entity_id=entity_input.entity_id,
            operational_control_result=has_oper,
            financial_control_result=has_fin,
            operational_criteria=oper_criteria,
            financial_criteria=fin_criteria,
            justification=justification,
        )
        assessment.provenance_hash = _compute_hash(assessment)
        self._assessments.append(assessment)

        return assessment

    # ------------------------------------------------------------------
    # Financial Control Assessment
    # ------------------------------------------------------------------

    def _assess_financial_control(
        self,
        entity_input: ControlInput,
    ) -> Tuple[bool, List[str]]:
        """Assess financial control for an entity.

        GHG Protocol defines financial control as: the reporting
        entity has the ability to direct the financial and
        operating policies of the operation with a view to gaining
        economic benefits from its activities.

        Args:
            entity_input: Entity with control indicators.

        Returns:
            Tuple of (has_control, assessment_basis).
        """
        basis: List[str] = []

        if entity_input.has_financial_control:
            basis.append("Entity explicitly flagged as under financial control.")
            return True, basis

        if entity_input.directs_policies:
            basis.append(
                "Reporting org directs financial and operating policies "
                "of the entity."
            )
            return True, basis

        if entity_input.equity_pct > Decimal("50"):
            basis.append(
                f"Majority ownership ({entity_input.equity_pct}% > 50%) "
                f"implies financial control per IFRS 10."
            )
            return True, basis

        if entity_input.has_board_majority:
            basis.append(
                "Board majority provides ability to direct policies."
            )
            return True, basis

        basis.append(
            f"No financial control indicators found for entity "
            f"with {entity_input.equity_pct}% equity."
        )
        return False, basis

    def assess_financial_control(
        self,
        entity_input: ControlInput,
    ) -> ControlAssessmentDetail:
        """Perform detailed financial control assessment.

        Creates a structured assessment record suitable for
        assurance review and audit documentation.

        Args:
            entity_input: Entity with control indicators.

        Returns:
            ControlAssessmentDetail with full criteria checklist.
        """
        has_fin, fin_basis = self._assess_financial_control(entity_input)
        has_oper, oper_basis = self._assess_operational_control(entity_input)

        oper_criteria = {
            "has_operational_control_flag": entity_input.has_operational_control,
            "manages_operations": entity_input.manages_operations,
            "wholly_owned": entity_input.equity_pct >= Decimal("100"),
        }

        fin_criteria = {
            "has_financial_control_flag": entity_input.has_financial_control,
            "directs_policies": entity_input.directs_policies,
            "has_board_majority": entity_input.has_board_majority,
            "majority_equity": entity_input.equity_pct > Decimal("50"),
        }

        justification = " ".join(fin_basis)

        assessment = ControlAssessmentDetail(
            entity_id=entity_input.entity_id,
            operational_control_result=has_oper,
            financial_control_result=has_fin,
            operational_criteria=oper_criteria,
            financial_criteria=fin_criteria,
            justification=justification,
        )
        assessment.provenance_hash = _compute_hash(assessment)
        self._assessments.append(assessment)

        return assessment

    # ------------------------------------------------------------------
    # Franchise Boundary
    # ------------------------------------------------------------------

    def handle_franchise(
        self,
        franchise_entity_id: str,
        franchiser_entity_id: str,
        franchise_role: str,
        approach: str,
        franchiser_controls_operations: bool = False,
    ) -> FranchiseBoundary:
        """Determine franchise boundary per GHG Protocol Appendix B.

        Under operational control:
            - Franchiser includes if it controls franchise operations
            - Franchisee includes if it controls its own operations
        Under financial control:
            - Based on financial control indicators
        In all cases:
            - Non-included franchise emissions go to Scope 3 Cat 14

        Args:
            franchise_entity_id: The franchise operation entity.
            franchiser_entity_id: The franchiser entity.
            franchise_role: Reporting org's role.
            approach: Consolidation approach.
            franchiser_controls_operations: Whether franchiser
                controls franchise operations.

        Returns:
            FranchiseBoundary with inclusion decision.
        """
        approach_upper = approach.upper()
        role_upper = franchise_role.upper()

        logger.info(
            "Handling franchise boundary: role=%s, approach=%s.",
            role_upper, approach_upper,
        )

        include_scope1_2 = False
        report_scope3 = False
        rationale = ""

        if approach_upper == ControlApproachType.OPERATIONAL_CONTROL.value:
            if role_upper == FranchiseRole.FRANCHISER.value:
                if franchiser_controls_operations:
                    include_scope1_2 = True
                    rationale = (
                        "Franchiser controls franchise operations; "
                        "include 100% in Scope 1/2 under operational control."
                    )
                else:
                    report_scope3 = True
                    rationale = (
                        "Franchiser does not control franchise operations; "
                        "report in Scope 3 Category 14 (Franchises)."
                    )
            else:
                include_scope1_2 = True
                rationale = (
                    "Franchisee controls its own operations; "
                    "include 100% in Scope 1/2 under operational control."
                )

        elif approach_upper == ControlApproachType.FINANCIAL_CONTROL.value:
            if role_upper == FranchiseRole.FRANCHISER.value:
                report_scope3 = True
                rationale = (
                    "Under financial control, franchiser typically does not "
                    "have financial control over individual franchises; "
                    "report in Scope 3 Category 14."
                )
            else:
                include_scope1_2 = True
                rationale = (
                    "Franchisee has financial control over its operations; "
                    "include 100% in Scope 1/2."
                )

        decision = FranchiseBoundary(
            franchise_entity_id=franchise_entity_id,
            franchiser_entity_id=franchiser_entity_id,
            franchise_role=role_upper,
            approach=approach_upper,
            franchiser_controls_operations=franchiser_controls_operations,
            include_in_scope1_2=include_scope1_2,
            report_in_scope3=report_scope3,
            rationale=rationale,
        )
        decision.provenance_hash = _compute_hash(decision)
        self._franchise_decisions.append(decision)

        self._change_log.append({
            "event": "FRANCHISE_BOUNDARY_DECIDED",
            "franchise_entity_id": franchise_entity_id,
            "role": role_upper,
            "approach": approach_upper,
            "include_scope1_2": include_scope1_2,
            "timestamp": _utcnow().isoformat(),
        })

        logger.info(
            "Franchise decision: include_scope1_2=%s, scope3=%s.",
            include_scope1_2, report_scope3,
        )
        return decision

    # ------------------------------------------------------------------
    # Lease Boundary
    # ------------------------------------------------------------------

    def handle_lease(
        self,
        asset_entity_id: str,
        lease_type: str,
        reporting_role: str,
        approach: str,
    ) -> LeaseBoundary:
        """Determine lease boundary per GHG Protocol guidance.

        Under operational control:
            - Lessee includes if it operates the leased asset
            - Lessor includes if it operates the asset
        Under financial control:
            - Finance lease: lessee includes (economic ownership)
            - Operating lease: lessor includes (retains ownership)

        Args:
            asset_entity_id: The leased asset/operation entity.
            lease_type: Type of lease.
            reporting_role: Reporting org's role (LESSEE or LESSOR).
            approach: Consolidation approach.

        Returns:
            LeaseBoundary with inclusion decision.
        """
        approach_upper = approach.upper()
        lease_upper = lease_type.upper()
        role_upper = reporting_role.upper()

        logger.info(
            "Handling lease boundary: lease=%s, role=%s, approach=%s.",
            lease_upper, role_upper, approach_upper,
        )

        include_scope1_2 = False
        report_scope3 = False
        rationale = ""

        if approach_upper == ControlApproachType.OPERATIONAL_CONTROL.value:
            if role_upper == LeaseRole.LESSEE.value:
                include_scope1_2 = True
                rationale = (
                    "Under operational control, lessee typically operates "
                    "the leased asset; include 100% in Scope 1/2."
                )
            else:
                report_scope3 = True
                rationale = (
                    "Under operational control, lessor does not operate "
                    "the leased asset; report in Scope 3 (Cat 13: "
                    "Downstream Leased Assets)."
                )

        elif approach_upper == ControlApproachType.FINANCIAL_CONTROL.value:
            if lease_upper == LeaseType.FINANCE_LEASE.value:
                if role_upper == LeaseRole.LESSEE.value:
                    include_scope1_2 = True
                    rationale = (
                        "Finance lease under financial control: lessee "
                        "has economic ownership; include in Scope 1/2."
                    )
                else:
                    report_scope3 = True
                    rationale = (
                        "Finance lease under financial control: lessor "
                        "has transferred economic ownership; report in "
                        "Scope 3 (Cat 13)."
                    )
            elif lease_upper == LeaseType.OPERATING_LEASE.value:
                if role_upper == LeaseRole.LESSOR.value:
                    include_scope1_2 = True
                    rationale = (
                        "Operating lease under financial control: lessor "
                        "retains ownership and financial control; include "
                        "in Scope 1/2."
                    )
                else:
                    report_scope3 = True
                    rationale = (
                        "Operating lease under financial control: lessee "
                        "does not have financial control; report in "
                        "Scope 3 (Cat 8: Upstream Leased Assets)."
                    )

        decision = LeaseBoundary(
            asset_entity_id=asset_entity_id,
            lease_type=lease_upper,
            reporting_role=role_upper,
            approach=approach_upper,
            include_in_scope1_2=include_scope1_2,
            report_in_scope3=report_scope3,
            rationale=rationale,
        )
        decision.provenance_hash = _compute_hash(decision)
        self._lease_decisions.append(decision)

        self._change_log.append({
            "event": "LEASE_BOUNDARY_DECIDED",
            "asset_entity_id": asset_entity_id,
            "lease_type": lease_upper,
            "role": role_upper,
            "approach": approach_upper,
            "include_scope1_2": include_scope1_2,
            "timestamp": _utcnow().isoformat(),
        })

        logger.info(
            "Lease decision: include_scope1_2=%s, scope3=%s.",
            include_scope1_2, report_scope3,
        )
        return decision

    # ------------------------------------------------------------------
    # Batch Consolidation
    # ------------------------------------------------------------------

    def consolidate_control(
        self,
        entity_inputs: List[ControlInput],
        approach: str,
    ) -> Dict[str, Any]:
        """Consolidate emissions across entities using a control approach.

        Applies the control approach to each entity and aggregates
        the included entities' contributions into consolidated totals.

        Args:
            entity_inputs: List of entity inputs.
            approach: Control approach to apply.

        Returns:
            Dict with consolidated totals and per-entity results.
        """
        logger.info(
            "Consolidating %d entity(ies) using %s.",
            len(entity_inputs), approach,
        )

        results: List[ControlResult] = []
        total_s1 = Decimal("0")
        total_s2_loc = Decimal("0")
        total_s2_mkt = Decimal("0")
        total_s3 = Decimal("0")
        included_count = 0

        for entity_input in entity_inputs:
            result = self.apply_control_approach(entity_input, approach)
            results.append(result)

            if result.has_control:
                total_s1 += result.scope1_contribution
                total_s2_loc += result.scope2_location_contribution
                total_s2_mkt += result.scope2_market_contribution
                total_s3 += result.scope3_contribution
                included_count += 1

        consolidated_total = _round2(total_s1 + total_s2_loc + total_s3)

        consolidation = {
            "approach": approach.upper(),
            "entity_results": [r.model_dump(mode="json") for r in results],
            "total_entities": len(entity_inputs),
            "included_entities": included_count,
            "excluded_entities": len(entity_inputs) - included_count,
            "consolidated_scope1": str(_round2(total_s1)),
            "consolidated_scope2_location": str(_round2(total_s2_loc)),
            "consolidated_scope2_market": str(_round2(total_s2_mkt)),
            "consolidated_scope3": str(_round2(total_s3)),
            "consolidated_total": str(consolidated_total),
            "created_at": _utcnow().isoformat(),
        }
        consolidation["provenance_hash"] = _compute_hash(consolidation)

        logger.info(
            "Control consolidation complete: %d included, total=%s tCO2e.",
            included_count, consolidated_total,
        )
        return consolidation

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_result(self, entity_id: str) -> Optional[ControlResult]:
        """Retrieve a control result by entity ID.

        Args:
            entity_id: The entity ID.

        Returns:
            ControlResult if found, else None.
        """
        return self._results.get(entity_id)

    def get_all_results(self) -> List[ControlResult]:
        """Return all control results.

        Returns:
            List of all ControlResults.
        """
        return list(self._results.values())

    def get_assessments(self) -> List[ControlAssessmentDetail]:
        """Return all detailed assessments.

        Returns:
            List of ControlAssessmentDetail records.
        """
        return list(self._assessments)

    def get_franchise_decisions(self) -> List[FranchiseBoundary]:
        """Return all franchise boundary decisions.

        Returns:
            List of FranchiseBoundary records.
        """
        return list(self._franchise_decisions)

    def get_lease_decisions(self) -> List[LeaseBoundary]:
        """Return all lease boundary decisions.

        Returns:
            List of LeaseBoundary records.
        """
        return list(self._lease_decisions)

    def get_change_log(self) -> List[Dict[str, Any]]:
        """Return the complete change log.

        Returns:
            List of change log entries.
        """
        return list(self._change_log)
