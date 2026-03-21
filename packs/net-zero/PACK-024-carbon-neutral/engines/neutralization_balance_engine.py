# -*- coding: utf-8 -*-
"""
NeutralizationBalanceEngine - PACK-024 Carbon Neutral Engine 6
===============================================================

Footprint vs retirements reconciliation engine with temporal matching
(vintage must cover footprint year), over/under retirement tracking,
carryforward rules, multi-period balance management, and carbon neutral
declaration readiness assessment.

This engine performs the critical reconciliation between quantified
emissions (footprint) and retired carbon credits to determine whether
an organisation has achieved carbon neutrality for a given period,
per ISO 14068-1:2023 and PAS 2060:2014 requirements.

Calculation Methodology:
    Neutralization Balance:
        balance = total_retired_tco2e - total_footprint_tco2e
        neutral  if balance >= 0  (fully neutralised)
        deficit  if balance < 0   (under-retirement)
        surplus  if balance > 0   (over-retirement)

    Temporal Matching (ISO 14068-1:2023, Section 8.3):
        A credit vintage is valid for a footprint year if:
            footprint_year - max_look_back <= vintage_year <= footprint_year + 1
        Default max_look_back: 5 years

    Carryforward Rules (PAS 2060:2014, Section 5.5):
        Over-retirements may be carried forward to the next period:
            carryforward_max = over_retirement * carryforward_factor
            carryforward_factor: 1.0 (full carryforward allowed)
            carryforward_expires: 1 year (single period only)

    Declaration Readiness (ISO 14068-1:2023, Section 10):
        ready = (
            balance >= 0 AND
            all_vintages_valid AND
            all_retirements_confirmed AND
            footprint_verified AND
            management_plan_exists AND
            reduction_first_met
        )

Regulatory References:
    - ISO 14068-1:2023 - Section 10: Carbon neutrality declaration
    - PAS 2060:2014 - Section 5.5: Achievement of carbon neutrality
    - ISO 14068-1:2023 - Section 8.3: Temporal alignment of credits
    - PAS 2060:2014 - Section 5.4: Credit quality and timing
    - GHG Protocol Corporate Standard (2004) - Reporting boundaries

Zero-Hallucination:
    - Balance requirements from ISO 14068-1:2023
    - Temporal matching rules from published standards
    - No LLM involvement in any calculation path
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-024 Carbon Neutral
Engine:  6 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
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
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def _safe_divide(
    numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0"),
) -> Decimal:
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    return _safe_divide(part * Decimal("100"), whole)


def _round_val(value: Decimal, places: int = 6) -> Decimal:
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


def _round3(value: float) -> float:
    return float(
        Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    )


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class BalanceStatus(str, Enum):
    """Neutralization balance status.

    NEUTRAL: Footprint fully covered by retirements.
    SURPLUS: Over-retirement (more credits than footprint).
    DEFICIT: Under-retirement (footprint exceeds retirements).
    NOT_ASSESSED: Balance not yet calculated.
    """
    NEUTRAL = "neutral"
    SURPLUS = "surplus"
    DEFICIT = "deficit"
    NOT_ASSESSED = "not_assessed"


class DeclarationReadiness(str, Enum):
    """Carbon neutral declaration readiness.

    READY: All requirements met, declaration can be made.
    CONDITIONALLY_READY: Minor items pending, conditional declaration.
    NOT_READY: Significant gaps, declaration not appropriate.
    """
    READY = "ready"
    CONDITIONALLY_READY = "conditionally_ready"
    NOT_READY = "not_ready"


class TemporalMatchStatus(str, Enum):
    """Temporal matching status for a vintage.

    MATCHED: Vintage within acceptable range for footprint year.
    TOO_OLD: Vintage exceeds maximum lookback.
    FUTURE: Vintage is from the future (post-footprint).
    INVALID: Vintage year is invalid.
    """
    MATCHED = "matched"
    TOO_OLD = "too_old"
    FUTURE = "future"
    INVALID = "invalid"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default maximum vintage lookback (years).
DEFAULT_MAX_LOOKBACK: int = 5

# Maximum future vintage (footprint_year + 1 is acceptable).
MAX_FUTURE_VINTAGE: int = 1

# Carryforward factor (1.0 = full carryforward allowed).
DEFAULT_CARRYFORWARD_FACTOR: Decimal = Decimal("1.0")

# Carryforward expiry (periods).
DEFAULT_CARRYFORWARD_EXPIRY: int = 1

# Minimum coverage for conditional declaration.
CONDITIONAL_COVERAGE_MIN: Decimal = Decimal("95")


# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------


class FootprintPeriod(BaseModel):
    """Footprint data for a single period.

    Attributes:
        period_year: Year of the footprint period.
        scope1_tco2e: Scope 1 emissions.
        scope2_tco2e: Scope 2 emissions.
        scope3_tco2e: Scope 3 emissions (included in boundary).
        total_tco2e: Total footprint.
        is_verified: Whether footprint has been third-party verified.
        verifier: Name of third-party verifier.
        verification_date: Date of verification.
        scope_boundary: Scope boundary description.
    """
    period_year: int = Field(..., ge=2015, le=2060, description="Period year")
    scope1_tco2e: Decimal = Field(default=Decimal("0"), ge=0, description="Scope 1")
    scope2_tco2e: Decimal = Field(default=Decimal("0"), ge=0, description="Scope 2")
    scope3_tco2e: Decimal = Field(default=Decimal("0"), ge=0, description="Scope 3")
    total_tco2e: Decimal = Field(default=Decimal("0"), ge=0, description="Total")
    is_verified: bool = Field(default=False, description="Third-party verified")
    verifier: str = Field(default="", description="Verifier name")
    verification_date: Optional[str] = Field(default=None, description="Verification date")
    scope_boundary: str = Field(default="scope_1_2_3", description="Scope boundary")


class RetirementRecord(BaseModel):
    """A retirement record for reconciliation.

    Attributes:
        retirement_id: Unique retirement ID.
        registry: Registry name.
        project_name: Project name.
        vintage_year: Credit vintage year.
        quantity_tco2e: Quantity retired.
        retirement_confirmed: Whether retirement is confirmed.
        retirement_date: Date of retirement.
        allocated_to_year: Footprint year this is allocated to.
        is_removal: Whether this is a removal credit.
        quality_score: Credit quality score (0-100).
        serial_range: Serial number range.
    """
    retirement_id: str = Field(default_factory=_new_uuid, description="Retirement ID")
    registry: str = Field(default="", description="Registry")
    project_name: str = Field(default="", description="Project name")
    vintage_year: int = Field(default=0, ge=0, le=2060, description="Vintage year")
    quantity_tco2e: Decimal = Field(default=Decimal("0"), ge=0, description="Quantity")
    retirement_confirmed: bool = Field(default=False, description="Confirmed")
    retirement_date: Optional[str] = Field(default=None, description="Date")
    allocated_to_year: int = Field(default=0, ge=0, description="Allocated year")
    is_removal: bool = Field(default=False, description="Is removal")
    quality_score: Decimal = Field(default=Decimal("0"), ge=0, le=Decimal("100"))
    serial_range: str = Field(default="", description="Serial range")


class CarryforwardRecord(BaseModel):
    """Carryforward credits from a previous period.

    Attributes:
        source_year: Year the over-retirement occurred.
        quantity_tco2e: Carryforward amount.
        expiry_year: Year carryforward expires.
        is_expired: Whether carryforward has expired.
    """
    source_year: int = Field(default=0, description="Source year")
    quantity_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    expiry_year: int = Field(default=0, description="Expiry year")
    is_expired: bool = Field(default=False)


class NeutralizationBalanceInput(BaseModel):
    """Complete input for neutralization balance.

    Attributes:
        entity_name: Reporting entity name.
        assessment_year: Year being assessed.
        footprint_periods: Footprint data (one or more periods).
        retirements: Retirement records for reconciliation.
        carryforwards: Carryforward credits from previous periods.
        max_lookback_years: Maximum vintage lookback.
        carryforward_factor: Carryforward factor (0-1).
        management_plan_exists: Whether carbon management plan exists.
        reduction_first_met: Whether reduction-first hierarchy is met.
        target_standard: Target standard (iso_14068_1, pas_2060).
        include_multi_year: Whether to assess multi-year balance.
    """
    entity_name: str = Field(
        ..., min_length=1, max_length=300, description="Entity name"
    )
    assessment_year: int = Field(
        ..., ge=2015, le=2060, description="Assessment year"
    )
    footprint_periods: List[FootprintPeriod] = Field(
        default_factory=list, description="Footprint periods"
    )
    retirements: List[RetirementRecord] = Field(
        default_factory=list, description="Retirement records"
    )
    carryforwards: List[CarryforwardRecord] = Field(
        default_factory=list, description="Carryforward credits"
    )
    max_lookback_years: int = Field(
        default=DEFAULT_MAX_LOOKBACK, ge=1, le=10,
        description="Max vintage lookback"
    )
    carryforward_factor: Decimal = Field(
        default=DEFAULT_CARRYFORWARD_FACTOR,
        ge=0, le=Decimal("1"),
        description="Carryforward factor"
    )
    management_plan_exists: bool = Field(default=False, description="Plan exists")
    reduction_first_met: bool = Field(default=False, description="Reduction-first met")
    target_standard: str = Field(
        default="iso_14068_1", description="Target standard"
    )
    include_multi_year: bool = Field(default=False, description="Multi-year analysis")


# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------


class TemporalMatchResult(BaseModel):
    """Temporal matching result for a retirement.

    Attributes:
        retirement_id: Retirement ID.
        vintage_year: Vintage year.
        footprint_year: Footprint year allocated to.
        vintage_age: Age of vintage at footprint year.
        match_status: Temporal match status.
        is_valid: Whether vintage is valid for the footprint year.
        message: Human-readable message.
    """
    retirement_id: str = Field(default="")
    vintage_year: int = Field(default=0)
    footprint_year: int = Field(default=0)
    vintage_age: int = Field(default=0)
    match_status: str = Field(default=TemporalMatchStatus.INVALID.value)
    is_valid: bool = Field(default=False)
    message: str = Field(default="")


class PeriodBalance(BaseModel):
    """Balance for a single period.

    Attributes:
        period_year: Year.
        footprint_tco2e: Footprint for this period.
        retired_tco2e: Credits retired for this period.
        carryforward_applied_tco2e: Carryforward applied.
        total_offset_tco2e: Total offset (retired + carryforward).
        balance_tco2e: Balance (offset - footprint).
        balance_status: NEUTRAL/SURPLUS/DEFICIT.
        coverage_pct: Coverage percentage.
        surplus_tco2e: Surplus amount (if any).
        deficit_tco2e: Deficit amount (if any).
        carryforward_generated_tco2e: Carryforward to next period.
        is_neutral: Whether period is neutral.
        temporal_matches: Temporal match results for retirements.
        all_vintages_valid: Whether all vintages pass temporal check.
        footprint_verified: Whether footprint is verified.
    """
    period_year: int = Field(default=0)
    footprint_tco2e: Decimal = Field(default=Decimal("0"))
    retired_tco2e: Decimal = Field(default=Decimal("0"))
    carryforward_applied_tco2e: Decimal = Field(default=Decimal("0"))
    total_offset_tco2e: Decimal = Field(default=Decimal("0"))
    balance_tco2e: Decimal = Field(default=Decimal("0"))
    balance_status: str = Field(default=BalanceStatus.NOT_ASSESSED.value)
    coverage_pct: Decimal = Field(default=Decimal("0"))
    surplus_tco2e: Decimal = Field(default=Decimal("0"))
    deficit_tco2e: Decimal = Field(default=Decimal("0"))
    carryforward_generated_tco2e: Decimal = Field(default=Decimal("0"))
    is_neutral: bool = Field(default=False)
    temporal_matches: List[TemporalMatchResult] = Field(default_factory=list)
    all_vintages_valid: bool = Field(default=True)
    footprint_verified: bool = Field(default=False)


class DeclarationAssessment(BaseModel):
    """Carbon neutral declaration readiness assessment.

    Attributes:
        readiness: Overall readiness.
        criteria_met: List of criteria that are met.
        criteria_not_met: List of criteria not met.
        criteria_score: Number met / total criteria.
        can_declare: Whether declaration can be made.
        declaration_scope: Scope of the declaration.
        standard_reference: Standard section reference.
        conditions: Any conditions on the declaration.
        message: Human-readable assessment.
    """
    readiness: str = Field(default=DeclarationReadiness.NOT_READY.value)
    criteria_met: List[str] = Field(default_factory=list)
    criteria_not_met: List[str] = Field(default_factory=list)
    criteria_score: Decimal = Field(default=Decimal("0"))
    can_declare: bool = Field(default=False)
    declaration_scope: str = Field(default="")
    standard_reference: str = Field(default="")
    conditions: List[str] = Field(default_factory=list)
    message: str = Field(default="")


class NeutralizationBalanceResult(BaseModel):
    """Complete neutralization balance result.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Calculation timestamp.
        entity_name: Entity name.
        assessment_year: Assessment year.
        period_balances: Per-period balance results.
        total_footprint_tco2e: Total footprint across all periods.
        total_retired_tco2e: Total retired across all periods.
        total_carryforward_tco2e: Total carryforward applied.
        net_balance_tco2e: Net balance (offset - footprint).
        overall_status: Overall balance status.
        overall_coverage_pct: Overall coverage percentage.
        is_neutral: Whether carbon neutral is achieved.
        declaration_assessment: Declaration readiness.
        carryforward_to_next_tco2e: Carryforward to next period.
        all_vintages_valid: Whether all vintages pass temporal check.
        all_retirements_confirmed: Whether all retirements confirmed.
        weighted_quality_score: Weighted quality of retired credits.
        removal_pct: Percentage of removals in portfolio.
        warnings: Non-blocking warnings.
        errors: Blocking errors.
        processing_time_ms: Processing duration.
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=_utcnow)
    entity_name: str = Field(default="")
    assessment_year: int = Field(default=0)
    period_balances: List[PeriodBalance] = Field(default_factory=list)
    total_footprint_tco2e: Decimal = Field(default=Decimal("0"))
    total_retired_tco2e: Decimal = Field(default=Decimal("0"))
    total_carryforward_tco2e: Decimal = Field(default=Decimal("0"))
    net_balance_tco2e: Decimal = Field(default=Decimal("0"))
    overall_status: str = Field(default=BalanceStatus.NOT_ASSESSED.value)
    overall_coverage_pct: Decimal = Field(default=Decimal("0"))
    is_neutral: bool = Field(default=False)
    declaration_assessment: Optional[DeclarationAssessment] = Field(default=None)
    carryforward_to_next_tco2e: Decimal = Field(default=Decimal("0"))
    all_vintages_valid: bool = Field(default=False)
    all_retirements_confirmed: bool = Field(default=False)
    weighted_quality_score: Decimal = Field(default=Decimal("0"))
    removal_pct: Decimal = Field(default=Decimal("0"))
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class NeutralizationBalanceEngine:
    """Footprint vs retirements reconciliation engine.

    Performs the critical balance calculation between quantified
    emissions and retired credits to determine carbon neutrality status.

    Usage::

        engine = NeutralizationBalanceEngine()
        result = engine.reconcile(input_data)
        print(f"Status: {result.overall_status}")
        print(f"Neutral: {result.is_neutral}")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self._max_lookback = int(
            self.config.get("max_lookback_years", DEFAULT_MAX_LOOKBACK)
        )
        self._cf_factor = _decimal(
            self.config.get("carryforward_factor", DEFAULT_CARRYFORWARD_FACTOR)
        )
        logger.info("NeutralizationBalanceEngine v%s initialised", self.engine_version)

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def reconcile(
        self, data: NeutralizationBalanceInput,
    ) -> NeutralizationBalanceResult:
        """Reconcile footprint against retirements.

        Args:
            data: Validated balance input.

        Returns:
            NeutralizationBalanceResult with full reconciliation.
        """
        t0 = time.perf_counter()
        logger.info(
            "Neutralization balance: entity=%s, year=%d, periods=%d, retirements=%d",
            data.entity_name, data.assessment_year,
            len(data.footprint_periods), len(data.retirements),
        )

        warnings: List[str] = []
        errors: List[str] = []

        # Step 1: Process each period
        period_balances: List[PeriodBalance] = []
        available_cf = self._get_valid_carryforwards(
            data.carryforwards, data.assessment_year
        )

        for fp in data.footprint_periods:
            pb = self._calculate_period_balance(
                fp, data.retirements, available_cf,
                data.max_lookback_years, data.carryforward_factor,
            )
            period_balances.append(pb)

            # Update carryforward for next period
            if pb.carryforward_generated_tco2e > Decimal("0"):
                available_cf += pb.carryforward_generated_tco2e

        # Step 2: Aggregate totals
        total_fp = sum((pb.footprint_tco2e for pb in period_balances), Decimal("0"))
        total_ret = sum((pb.retired_tco2e for pb in period_balances), Decimal("0"))
        total_cf = sum((pb.carryforward_applied_tco2e for pb in period_balances), Decimal("0"))
        total_offset = total_ret + total_cf
        net_balance = total_offset - total_fp

        # Step 3: Overall status
        if net_balance >= Decimal("0"):
            status = BalanceStatus.NEUTRAL.value if net_balance == Decimal("0") else BalanceStatus.SURPLUS.value
            is_neutral = True
        else:
            status = BalanceStatus.DEFICIT.value
            is_neutral = False

        coverage = _safe_pct(total_offset, total_fp)

        # Step 4: Vintage compliance
        all_vintages = all(pb.all_vintages_valid for pb in period_balances)

        # Step 5: Retirement confirmation
        all_confirmed = all(r.retirement_confirmed for r in data.retirements) if data.retirements else False

        # Step 6: Weighted quality
        weighted_q = Decimal("0")
        if total_ret > Decimal("0"):
            weighted_q = sum(
                (r.quality_score * r.quantity_tco2e for r in data.retirements),
                Decimal("0"),
            ) / total_ret

        # Step 7: Removal percentage
        removal_total = sum(
            (r.quantity_tco2e for r in data.retirements if r.is_removal), Decimal("0")
        )
        removal_pct = _safe_pct(removal_total, total_ret)

        # Step 8: Carryforward to next
        cf_to_next = Decimal("0")
        if net_balance > Decimal("0"):
            cf_to_next = _round_val(net_balance * data.carryforward_factor)

        # Step 9: Declaration assessment
        declaration = self._assess_declaration_readiness(
            data, is_neutral, all_vintages, all_confirmed,
            coverage, warnings
        )

        # Warnings
        if not all_vintages:
            warnings.append("Some credit vintages fail temporal matching.")
        if not all_confirmed:
            warnings.append("Some retirements are not yet confirmed by registries.")

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = NeutralizationBalanceResult(
            entity_name=data.entity_name,
            assessment_year=data.assessment_year,
            period_balances=period_balances,
            total_footprint_tco2e=_round_val(total_fp),
            total_retired_tco2e=_round_val(total_ret),
            total_carryforward_tco2e=_round_val(total_cf),
            net_balance_tco2e=_round_val(net_balance),
            overall_status=status,
            overall_coverage_pct=_round_val(coverage, 2),
            is_neutral=is_neutral,
            declaration_assessment=declaration,
            carryforward_to_next_tco2e=cf_to_next,
            all_vintages_valid=all_vintages,
            all_retirements_confirmed=all_confirmed,
            weighted_quality_score=_round_val(weighted_q, 2),
            removal_pct=_round_val(removal_pct, 2),
            warnings=warnings,
            errors=errors,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Balance reconciliation complete: fp=%.2f, ret=%.2f, balance=%.2f, "
            "neutral=%s, hash=%s",
            float(total_fp), float(total_ret), float(net_balance),
            is_neutral, result.provenance_hash[:16],
        )
        return result

    # ------------------------------------------------------------------ #
    # Internal Methods                                                     #
    # ------------------------------------------------------------------ #

    def _calculate_period_balance(
        self,
        fp: FootprintPeriod,
        retirements: List[RetirementRecord],
        carryforward_available: Decimal,
        max_lookback: int,
        cf_factor: Decimal,
    ) -> PeriodBalance:
        """Calculate balance for a single period."""
        year = fp.period_year
        footprint = fp.total_tco2e
        if footprint <= Decimal("0"):
            footprint = fp.scope1_tco2e + fp.scope2_tco2e + fp.scope3_tco2e

        # Find retirements allocated to this year
        period_rets = [
            r for r in retirements
            if r.allocated_to_year == year or r.allocated_to_year == 0
        ]

        # Temporal matching
        temporal_matches: List[TemporalMatchResult] = []
        valid_retired = Decimal("0")
        for r in period_rets:
            tm = self._check_temporal_match(r, year, max_lookback)
            temporal_matches.append(tm)
            if tm.is_valid:
                valid_retired += r.quantity_tco2e

        # Apply carryforward
        cf_applied = Decimal("0")
        if valid_retired < footprint and carryforward_available > Decimal("0"):
            needed = footprint - valid_retired
            cf_applied = min(needed, carryforward_available)

        total_offset = valid_retired + cf_applied
        balance = total_offset - footprint

        if balance >= Decimal("0"):
            status = BalanceStatus.NEUTRAL.value if balance == Decimal("0") else BalanceStatus.SURPLUS.value
            is_neutral = True
        else:
            status = BalanceStatus.DEFICIT.value
            is_neutral = False

        coverage = _safe_pct(total_offset, footprint)
        surplus = max(Decimal("0"), balance)
        deficit = max(Decimal("0"), -balance)

        cf_gen = Decimal("0")
        if surplus > Decimal("0"):
            cf_gen = _round_val(surplus * cf_factor)

        all_valid = all(tm.is_valid for tm in temporal_matches) if temporal_matches else True

        return PeriodBalance(
            period_year=year,
            footprint_tco2e=_round_val(footprint),
            retired_tco2e=_round_val(valid_retired),
            carryforward_applied_tco2e=_round_val(cf_applied),
            total_offset_tco2e=_round_val(total_offset),
            balance_tco2e=_round_val(balance),
            balance_status=status,
            coverage_pct=_round_val(coverage, 2),
            surplus_tco2e=_round_val(surplus),
            deficit_tco2e=_round_val(deficit),
            carryforward_generated_tco2e=cf_gen,
            is_neutral=is_neutral,
            temporal_matches=temporal_matches,
            all_vintages_valid=all_valid,
            footprint_verified=fp.is_verified,
        )

    def _check_temporal_match(
        self,
        retirement: RetirementRecord,
        footprint_year: int,
        max_lookback: int,
    ) -> TemporalMatchResult:
        """Check temporal matching of a vintage against footprint year."""
        vintage = retirement.vintage_year
        if vintage <= 0:
            return TemporalMatchResult(
                retirement_id=retirement.retirement_id,
                vintage_year=vintage,
                footprint_year=footprint_year,
                match_status=TemporalMatchStatus.INVALID.value,
                message="Invalid vintage year.",
            )

        age = footprint_year - vintage
        min_year = footprint_year - max_lookback
        max_year = footprint_year + MAX_FUTURE_VINTAGE

        if vintage < min_year:
            return TemporalMatchResult(
                retirement_id=retirement.retirement_id,
                vintage_year=vintage,
                footprint_year=footprint_year,
                vintage_age=age,
                match_status=TemporalMatchStatus.TOO_OLD.value,
                is_valid=False,
                message=(
                    f"Vintage {vintage} is {age} years old, exceeding "
                    f"maximum lookback of {max_lookback} years."
                ),
            )
        elif vintage > max_year:
            return TemporalMatchResult(
                retirement_id=retirement.retirement_id,
                vintage_year=vintage,
                footprint_year=footprint_year,
                vintage_age=age,
                match_status=TemporalMatchStatus.FUTURE.value,
                is_valid=False,
                message=f"Vintage {vintage} is too far in the future.",
            )
        else:
            return TemporalMatchResult(
                retirement_id=retirement.retirement_id,
                vintage_year=vintage,
                footprint_year=footprint_year,
                vintage_age=age,
                match_status=TemporalMatchStatus.MATCHED.value,
                is_valid=True,
                message=f"Vintage {vintage} is valid for footprint year {footprint_year}.",
            )

    def _get_valid_carryforwards(
        self,
        carryforwards: List[CarryforwardRecord],
        assessment_year: int,
    ) -> Decimal:
        """Sum valid (non-expired) carryforward credits."""
        total = Decimal("0")
        for cf in carryforwards:
            if cf.is_expired:
                continue
            if cf.expiry_year > 0 and cf.expiry_year < assessment_year:
                continue
            total += cf.quantity_tco2e
        return total

    def _assess_declaration_readiness(
        self,
        data: NeutralizationBalanceInput,
        is_neutral: bool,
        all_vintages: bool,
        all_confirmed: bool,
        coverage: Decimal,
        warnings: List[str],
    ) -> DeclarationAssessment:
        """Assess readiness for carbon neutral declaration.

        ISO 14068-1:2023 Section 10 requirements for declaration:
        1. Carbon footprint quantified and verified
        2. Carbon management plan exists
        3. Reduction-first hierarchy followed
        4. Residual emissions offset with quality credits
        5. All credits properly retired
        6. Temporal alignment verified
        """
        met: List[str] = []
        not_met: List[str] = []

        # 1. Balance achieved
        if is_neutral:
            met.append("Carbon neutrality balance achieved (footprint fully covered)")
        else:
            not_met.append(f"Carbon neutrality balance NOT achieved (coverage: {_round_val(coverage, 1)}%)")

        # 2. Footprint verified
        all_verified = all(fp.is_verified for fp in data.footprint_periods) if data.footprint_periods else False
        if all_verified:
            met.append("All footprint periods third-party verified")
        else:
            not_met.append("Not all footprint periods are third-party verified")

        # 3. Management plan
        if data.management_plan_exists:
            met.append("Carbon management plan exists")
        else:
            not_met.append("Carbon management plan does not exist (required by ISO 14068-1 Section 9)")

        # 4. Reduction-first
        if data.reduction_first_met:
            met.append("Reduction-first mitigation hierarchy followed")
        else:
            not_met.append("Reduction-first mitigation hierarchy not demonstrated")

        # 5. All retirements confirmed
        if all_confirmed:
            met.append("All credit retirements confirmed by registries")
        else:
            not_met.append("Not all retirements confirmed by registries")

        # 6. Temporal alignment
        if all_vintages:
            met.append("All credit vintages pass temporal matching")
        else:
            not_met.append("Some credit vintages fail temporal matching requirements")

        total_criteria = len(met) + len(not_met)
        score = _safe_pct(_decimal(len(met)), _decimal(total_criteria))

        if len(not_met) == 0:
            readiness = DeclarationReadiness.READY.value
            can_declare = True
            msg = "All criteria met. Carbon neutral declaration can be made."
        elif is_neutral and coverage >= CONDITIONAL_COVERAGE_MIN:
            readiness = DeclarationReadiness.CONDITIONALLY_READY.value
            can_declare = True
            msg = (
                f"Conditional declaration possible. {len(not_met)} criteria pending: "
                f"{'; '.join(not_met[:3])}"
            )
        else:
            readiness = DeclarationReadiness.NOT_READY.value
            can_declare = False
            msg = (
                f"Declaration not appropriate. {len(not_met)} criteria not met: "
                f"{'; '.join(not_met[:3])}"
            )

        std_ref = (
            "ISO 14068-1:2023, Section 10"
            if data.target_standard == "iso_14068_1"
            else "PAS 2060:2014, Section 5.5"
        )

        return DeclarationAssessment(
            readiness=readiness,
            criteria_met=met,
            criteria_not_met=not_met,
            criteria_score=_round_val(score, 2),
            can_declare=can_declare,
            declaration_scope=data.footprint_periods[0].scope_boundary if data.footprint_periods else "",
            standard_reference=std_ref,
            conditions=not_met if readiness == DeclarationReadiness.CONDITIONALLY_READY.value else [],
            message=msg,
        )
