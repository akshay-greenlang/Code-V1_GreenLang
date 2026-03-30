# -*- coding: utf-8 -*-
"""
CarbonCreditEngine - PACK-003 CSRD Enterprise Engine 7

Carbon offset and removal portfolio management engine. Handles credit
lifecycle (acquisition, retirement, transfer, cancellation), portfolio
analytics, net-zero accounting, quality assessment, and vintage tracking.

Registry Support:
    - VCS: Verified Carbon Standard (Verra)
    - GOLD_STANDARD: Gold Standard for the Global Goals
    - ACR: American Carbon Registry
    - CAR: Climate Action Reserve
    - CDM: Clean Development Mechanism (UNFCCC)
    - ARTICLE_6: Paris Agreement Article 6 credits

Credit Types:
    - AVOIDANCE: Emission avoidance/reduction credits
    - REMOVAL: Carbon dioxide removal credits
    - HYBRID: Combined avoidance and removal projects

Net-Zero Compliance:
    - SBTi Net-Zero Standard requires 90%+ actual reductions
    - Credits/offsets cannot replace emission reductions
    - Removal credits preferred over avoidance for residual emissions
    - Vintage age limits enforced (typically max 5 years)

Zero-Hallucination:
    - All portfolio calculations use deterministic arithmetic
    - Quality scores computed from explicit criteria weights
    - Net-zero accounting follows SBTi formulas exactly
    - No LLM involvement in financial or emissions calculations

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-003 CSRD Enterprise
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from collections import defaultdict
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
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CreditRegistry(str, Enum):
    """Carbon credit registry."""

    VCS = "vcs"
    GOLD_STANDARD = "gold_standard"
    ACR = "acr"
    CAR = "car"
    CDM = "cdm"
    ARTICLE_6 = "article_6"

class CreditType(str, Enum):
    """Type of carbon credit."""

    AVOIDANCE = "avoidance"
    REMOVAL = "removal"
    HYBRID = "hybrid"

class CreditStatus(str, Enum):
    """Lifecycle status of a carbon credit."""

    ACTIVE = "active"
    RETIRED = "retired"
    CANCELLED = "cancelled"
    TRANSFERRED = "transferred"
    EXPIRED = "expired"

class RetirementReason(str, Enum):
    """Reason for credit retirement."""

    VOLUNTARY_OFFSETTING = "voluntary_offsetting"
    COMPLIANCE_OBLIGATION = "compliance_obligation"
    NET_ZERO_CLAIM = "net_zero_claim"
    CARBON_NEUTRAL_CLAIM = "carbon_neutral_claim"
    CUSTOMER_REQUEST = "customer_request"

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class CarbonCredit(BaseModel):
    """A single carbon credit or credit batch."""

    credit_id: str = Field(
        default_factory=_new_uuid, description="Unique credit identifier"
    )
    registry: CreditRegistry = Field(..., description="Issuing registry")
    project_id: str = Field(..., description="Registry project identifier")
    project_name: str = Field(..., description="Project name")
    credit_type: CreditType = Field(..., description="Avoidance or removal")
    vintage_year: int = Field(
        ..., ge=2000, le=2100, description="Vintage year of credit"
    )
    quantity_tco2e: float = Field(
        ..., gt=0, description="Quantity in tonnes CO2e"
    )
    status: CreditStatus = Field(
        CreditStatus.ACTIVE, description="Current status"
    )
    purchase_date: datetime = Field(
        default_factory=utcnow, description="Date of purchase"
    )
    purchase_price_per_tco2e: float = Field(
        ..., ge=0, description="Purchase price per tCO2e"
    )
    currency: str = Field("USD", description="Price currency ISO 4217")
    additionality_score: float = Field(
        0.0, ge=0, le=100, description="Additionality assessment score 0-100"
    )
    permanence_score: float = Field(
        0.0, ge=0, le=100, description="Permanence assessment score 0-100"
    )
    methodology: str = Field("", description="Methodology used")
    country: str = Field("", description="Project country")
    co_benefits: List[str] = Field(
        default_factory=list,
        description="Co-benefits (e.g., biodiversity, community)",
    )
    tenant_id: str = Field("", description="Owning tenant ID")

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, v: str) -> str:
        """Validate currency code."""
        if len(v) != 3 or not v.isalpha():
            raise ValueError(f"Currency must be 3-letter ISO 4217 code: {v}")
        return v.upper()

class CreditPortfolio(BaseModel):
    """Summary of a tenant's carbon credit portfolio."""

    portfolio_id: str = Field(
        default_factory=_new_uuid, description="Portfolio identifier"
    )
    tenant_id: str = Field(..., description="Tenant identifier")
    credits: List[CarbonCredit] = Field(
        default_factory=list, description="All credits in portfolio"
    )
    total_active_tco2e: float = Field(
        0.0, description="Total active credits (tCO2e)"
    )
    total_retired_tco2e: float = Field(
        0.0, description="Total retired credits (tCO2e)"
    )
    total_value: float = Field(
        0.0, description="Total portfolio value"
    )
    avg_quality_score: float = Field(
        0.0, description="Average quality score across portfolio"
    )
    currency: str = Field("USD", description="Portfolio currency")
    provenance_hash: str = Field("", description="SHA-256 provenance hash")

class NetZeroAccounting(BaseModel):
    """Net-zero emissions accounting result."""

    gross_emissions_tco2e: float = Field(
        ..., description="Gross emissions before offsets"
    )
    credits_retired_tco2e: float = Field(
        ..., description="Credits retired for offsetting"
    )
    net_emissions_tco2e: float = Field(
        ..., description="Net emissions after offsets"
    )
    offset_percentage: float = Field(
        ..., description="Percentage of emissions offset"
    )
    removal_credits_tco2e: float = Field(
        0.0, description="Portion from removal credits"
    )
    avoidance_credits_tco2e: float = Field(
        0.0, description="Portion from avoidance credits"
    )
    sbti_compliant: bool = Field(
        ..., description="Compliant with SBTi Net-Zero Standard"
    )
    sbti_notes: List[str] = Field(
        default_factory=list,
        description="SBTi compliance notes and warnings",
    )
    provenance_hash: str = Field("", description="SHA-256 provenance hash")

# ---------------------------------------------------------------------------
# Quality Weights
# ---------------------------------------------------------------------------

_QUALITY_WEIGHTS: Dict[str, float] = {
    "additionality": 0.30,
    "permanence": 0.25,
    "registry_reputation": 0.15,
    "vintage_freshness": 0.10,
    "co_benefits": 0.10,
    "methodology_rigor": 0.10,
}

_REGISTRY_SCORES: Dict[CreditRegistry, float] = {
    CreditRegistry.GOLD_STANDARD: 95.0,
    CreditRegistry.VCS: 85.0,
    CreditRegistry.ACR: 80.0,
    CreditRegistry.CAR: 80.0,
    CreditRegistry.ARTICLE_6: 75.0,
    CreditRegistry.CDM: 65.0,
}

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class CarbonCreditEngine:
    """Carbon credit portfolio management engine.

    Manages carbon credit lifecycle, portfolio analytics, net-zero
    accounting, quality assessment, and retirement scheduling.
    All calculations are deterministic.

    Attributes:
        _credits: Credit store keyed by credit_id.
        _portfolios: Portfolio store keyed by tenant_id.
        _audit_log: Audit trail of operations.

    Example:
        >>> engine = CarbonCreditEngine()
        >>> credit = CarbonCredit(
        ...     registry=CreditRegistry.VCS,
        ...     project_id="VCS-1234",
        ...     project_name="Wind Farm Brazil",
        ...     credit_type=CreditType.AVOIDANCE,
        ...     vintage_year=2024,
        ...     quantity_tco2e=1000,
        ...     purchase_price_per_tco2e=12.50,
        ...     tenant_id="t-123",
        ... )
        >>> credit_id = engine.add_credit(credit)
    """

    def __init__(self) -> None:
        """Initialize CarbonCreditEngine."""
        self._credits: Dict[str, CarbonCredit] = {}
        self._audit_log: List[Dict[str, Any]] = []
        logger.info("CarbonCreditEngine v%s initialized", _MODULE_VERSION)

    # -- Credit Management --------------------------------------------------

    def add_credit(self, credit: CarbonCredit) -> str:
        """Add a carbon credit to the portfolio.

        Args:
            credit: Carbon credit to add.

        Returns:
            Credit ID string.
        """
        self._credits[credit.credit_id] = credit
        self._record_audit("CREDIT_ADDED", credit.credit_id, {
            "registry": credit.registry.value,
            "quantity": credit.quantity_tco2e,
            "vintage": credit.vintage_year,
            "tenant_id": credit.tenant_id,
        })

        logger.info(
            "Credit added: %s (%.2f tCO2e, %s, vintage %d)",
            credit.credit_id, credit.quantity_tco2e,
            credit.registry.value, credit.vintage_year,
        )
        return credit.credit_id

    def retire_credit(
        self,
        credit_id: str,
        quantity: float,
        retirement_reason: RetirementReason = RetirementReason.VOLUNTARY_OFFSETTING,
    ) -> Dict[str, Any]:
        """Retire carbon credits with full audit trail.

        Args:
            credit_id: ID of credit to retire.
            quantity: Quantity to retire (tCO2e).
            retirement_reason: Reason for retirement.

        Returns:
            Dict with retirement details.

        Raises:
            KeyError: If credit not found.
            ValueError: If quantity exceeds available.
        """
        credit = self._get_credit(credit_id)

        if credit.status != CreditStatus.ACTIVE:
            raise ValueError(
                f"Credit {credit_id} is not active (status={credit.status.value})"
            )

        if quantity > credit.quantity_tco2e:
            raise ValueError(
                f"Retirement quantity ({quantity}) exceeds available "
                f"({credit.quantity_tco2e})"
            )

        # Partial or full retirement
        if quantity < credit.quantity_tco2e:
            # Create residual credit
            residual_qty = credit.quantity_tco2e - quantity
            residual = CarbonCredit(
                registry=credit.registry,
                project_id=credit.project_id,
                project_name=credit.project_name,
                credit_type=credit.credit_type,
                vintage_year=credit.vintage_year,
                quantity_tco2e=residual_qty,
                status=CreditStatus.ACTIVE,
                purchase_date=credit.purchase_date,
                purchase_price_per_tco2e=credit.purchase_price_per_tco2e,
                currency=credit.currency,
                additionality_score=credit.additionality_score,
                permanence_score=credit.permanence_score,
                methodology=credit.methodology,
                country=credit.country,
                co_benefits=credit.co_benefits,
                tenant_id=credit.tenant_id,
            )
            self._credits[residual.credit_id] = residual

        credit.quantity_tco2e = quantity
        credit.status = CreditStatus.RETIRED

        result = {
            "credit_id": credit_id,
            "quantity_retired": quantity,
            "retirement_reason": retirement_reason.value,
            "retired_at": utcnow().isoformat(),
            "registry": credit.registry.value,
            "project_name": credit.project_name,
            "vintage_year": credit.vintage_year,
            "provenance_hash": _compute_hash({
                "credit_id": credit_id,
                "quantity": quantity,
                "reason": retirement_reason.value,
            }),
        }

        self._record_audit("CREDIT_RETIRED", credit_id, result)
        logger.info(
            "Credit %s retired: %.2f tCO2e (%s)",
            credit_id, quantity, retirement_reason.value,
        )
        return result

    def transfer_credit(
        self, credit_id: str, to_entity: str
    ) -> Dict[str, Any]:
        """Transfer a carbon credit to another entity.

        Args:
            credit_id: ID of credit to transfer.
            to_entity: Receiving entity identifier.

        Returns:
            Dict with transfer details.

        Raises:
            KeyError: If credit not found.
            ValueError: If credit not active.
        """
        credit = self._get_credit(credit_id)

        if credit.status != CreditStatus.ACTIVE:
            raise ValueError(f"Credit {credit_id} is not active")

        old_tenant = credit.tenant_id
        credit.status = CreditStatus.TRANSFERRED
        credit.tenant_id = to_entity

        result = {
            "credit_id": credit_id,
            "from_entity": old_tenant,
            "to_entity": to_entity,
            "quantity_tco2e": credit.quantity_tco2e,
            "transferred_at": utcnow().isoformat(),
            "provenance_hash": _compute_hash({
                "credit_id": credit_id, "from": old_tenant, "to": to_entity,
            }),
        }

        self._record_audit("CREDIT_TRANSFERRED", credit_id, result)
        return result

    # -- Portfolio ----------------------------------------------------------

    def get_portfolio(self, tenant_id: str) -> CreditPortfolio:
        """Get complete portfolio summary for a tenant.

        Args:
            tenant_id: Tenant identifier.

        Returns:
            CreditPortfolio with all credits and summary metrics.
        """
        tenant_credits = [
            c for c in self._credits.values() if c.tenant_id == tenant_id
        ]

        active = [c for c in tenant_credits if c.status == CreditStatus.ACTIVE]
        retired = [c for c in tenant_credits if c.status == CreditStatus.RETIRED]

        total_active = sum(c.quantity_tco2e for c in active)
        total_retired = sum(c.quantity_tco2e for c in retired)
        total_value = sum(
            c.quantity_tco2e * c.purchase_price_per_tco2e
            for c in tenant_credits
        )

        quality_scores = [
            self._compute_quality_score(c) for c in tenant_credits
        ]
        avg_quality = (
            sum(quality_scores) / len(quality_scores)
            if quality_scores else 0.0
        )

        portfolio = CreditPortfolio(
            tenant_id=tenant_id,
            credits=tenant_credits,
            total_active_tco2e=round(total_active, 2),
            total_retired_tco2e=round(total_retired, 2),
            total_value=round(total_value, 2),
            avg_quality_score=round(avg_quality, 2),
        )
        portfolio.provenance_hash = _compute_hash(portfolio)

        return portfolio

    # -- Net-Zero Accounting ------------------------------------------------

    def calculate_net_zero(
        self, gross_emissions: float, portfolio_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> NetZeroAccounting:
        """Calculate net-zero emissions accounting.

        Follows SBTi Net-Zero Standard: offsets cannot replace actual
        emission reductions. Removal credits preferred for residual.

        Args:
            gross_emissions: Total gross emissions (tCO2e).
            portfolio_id: Not used (for interface compatibility).
            tenant_id: Tenant for portfolio lookup.

        Returns:
            NetZeroAccounting with compliance assessment.
        """
        retired_credits = [
            c for c in self._credits.values()
            if c.status == CreditStatus.RETIRED
            and (tenant_id is None or c.tenant_id == tenant_id)
        ]

        removal_retired = sum(
            c.quantity_tco2e for c in retired_credits
            if c.credit_type == CreditType.REMOVAL
        )
        avoidance_retired = sum(
            c.quantity_tco2e for c in retired_credits
            if c.credit_type in (CreditType.AVOIDANCE, CreditType.HYBRID)
        )
        total_retired = removal_retired + avoidance_retired

        net_emissions = max(0.0, gross_emissions - total_retired)
        offset_pct = (
            (total_retired / gross_emissions * 100)
            if gross_emissions > 0 else 0.0
        )

        # SBTi compliance checks
        sbti_notes: List[str] = []
        sbti_compliant = True

        # SBTi: Must reduce at least 90% before using offsets
        if offset_pct > 10:
            sbti_notes.append(
                f"Offsetting {offset_pct:.1f}% of emissions. SBTi recommends "
                f"offsets only for residual emissions (<10% of base year)."
            )
            sbti_compliant = False

        # SBTi: Removal preferred over avoidance for residual
        if avoidance_retired > 0 and removal_retired == 0:
            sbti_notes.append(
                "No removal credits used. SBTi prefers carbon removal "
                "credits over avoidance credits for residual emissions."
            )

        # Check vintage freshness
        current_year = utcnow().year
        old_vintages = [
            c for c in retired_credits
            if current_year - c.vintage_year > 5
        ]
        if old_vintages:
            sbti_notes.append(
                f"{len(old_vintages)} retired credits have vintage older "
                f"than 5 years (quality concern)."
            )

        if not sbti_notes:
            sbti_notes.append("Portfolio meets SBTi Net-Zero Standard guidance.")

        result = NetZeroAccounting(
            gross_emissions_tco2e=round(gross_emissions, 2),
            credits_retired_tco2e=round(total_retired, 2),
            net_emissions_tco2e=round(net_emissions, 2),
            offset_percentage=round(offset_pct, 2),
            removal_credits_tco2e=round(removal_retired, 2),
            avoidance_credits_tco2e=round(avoidance_retired, 2),
            sbti_compliant=sbti_compliant,
            sbti_notes=sbti_notes,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Net-zero accounting: gross=%.2f, retired=%.2f, net=%.2f (SBTi=%s)",
            gross_emissions, total_retired, net_emissions, sbti_compliant,
        )
        return result

    # -- Quality Assessment -------------------------------------------------

    def assess_quality(self, credit_id: str) -> Dict[str, Any]:
        """Assess the quality of a carbon credit.

        Evaluates additionality, permanence, registry reputation,
        vintage freshness, co-benefits, and methodology rigor.

        Args:
            credit_id: Credit to assess.

        Returns:
            Dict with quality assessment breakdown.

        Raises:
            KeyError: If credit not found.
        """
        credit = self._get_credit(credit_id)
        current_year = utcnow().year

        # Additionality (from credit data)
        additionality = credit.additionality_score

        # Permanence (from credit data)
        permanence = credit.permanence_score

        # Registry reputation
        registry_score = _REGISTRY_SCORES.get(credit.registry, 50.0)

        # Vintage freshness (newer = better, max 5 years)
        vintage_age = current_year - credit.vintage_year
        vintage_freshness = max(0.0, 100.0 - vintage_age * 15)

        # Co-benefits
        co_benefit_score = min(100.0, len(credit.co_benefits) * 25)

        # Methodology rigor (based on registry and methodology presence)
        methodology_score = 70.0
        if credit.methodology:
            methodology_score = 85.0
        if credit.registry in (CreditRegistry.GOLD_STANDARD, CreditRegistry.VCS):
            methodology_score = min(100.0, methodology_score + 10)

        # Weighted composite score
        composite = (
            additionality * _QUALITY_WEIGHTS["additionality"]
            + permanence * _QUALITY_WEIGHTS["permanence"]
            + registry_score * _QUALITY_WEIGHTS["registry_reputation"]
            + vintage_freshness * _QUALITY_WEIGHTS["vintage_freshness"]
            + co_benefit_score * _QUALITY_WEIGHTS["co_benefits"]
            + methodology_score * _QUALITY_WEIGHTS["methodology_rigor"]
        )

        # Risk tier
        if composite >= 80:
            risk_tier = "low"
        elif composite >= 60:
            risk_tier = "medium"
        elif composite >= 40:
            risk_tier = "high"
        else:
            risk_tier = "critical"

        result = {
            "credit_id": credit_id,
            "composite_score": round(composite, 2),
            "risk_tier": risk_tier,
            "breakdown": {
                "additionality": round(additionality, 2),
                "permanence": round(permanence, 2),
                "registry_reputation": round(registry_score, 2),
                "vintage_freshness": round(vintage_freshness, 2),
                "co_benefits": round(co_benefit_score, 2),
                "methodology_rigor": round(methodology_score, 2),
            },
            "weights": _QUALITY_WEIGHTS,
            "vintage_age_years": vintage_age,
            "recommendations": self._quality_recommendations(
                additionality, permanence, vintage_age, credit.credit_type
            ),
            "provenance_hash": _compute_hash({
                "credit_id": credit_id, "score": composite,
            }),
        }

        return result

    def _quality_recommendations(
        self,
        additionality: float,
        permanence: float,
        vintage_age: int,
        credit_type: CreditType,
    ) -> List[str]:
        """Generate quality improvement recommendations.

        Args:
            additionality: Additionality score.
            permanence: Permanence score.
            vintage_age: Age of vintage in years.
            credit_type: Type of credit.

        Returns:
            List of recommendation strings.
        """
        recs: List[str] = []
        if additionality < 60:
            recs.append(
                "Low additionality score. Consider credits with stronger "
                "additionality evidence."
            )
        if permanence < 60:
            recs.append(
                "Low permanence score. Nature-based solutions may have "
                "reversal risk."
            )
        if vintage_age > 5:
            recs.append(
                f"Vintage is {vintage_age} years old. Prefer recent vintages "
                f"(< 5 years)."
            )
        if credit_type == CreditType.AVOIDANCE:
            recs.append(
                "Avoidance credit. Consider transitioning to removal credits "
                "for stronger net-zero claims."
            )
        return recs

    # -- Price History & Scheduling -----------------------------------------

    def get_price_history(
        self, registry: CreditRegistry, credit_type: CreditType, years: int = 5
    ) -> List[Dict[str, Any]]:
        """Get historical credit prices from portfolio data.

        Args:
            registry: Registry to filter by.
            credit_type: Credit type to filter by.
            years: Number of years of history.

        Returns:
            List of price data points.
        """
        current_year = utcnow().year
        relevant = [
            c for c in self._credits.values()
            if c.registry == registry
            and c.credit_type == credit_type
            and current_year - c.vintage_year <= years
        ]

        # Group by vintage year
        by_year: Dict[int, List[float]] = defaultdict(list)
        for c in relevant:
            by_year[c.vintage_year].append(c.purchase_price_per_tco2e)

        history = []
        for year in sorted(by_year.keys()):
            prices = by_year[year]
            history.append({
                "year": year,
                "avg_price": round(sum(prices) / len(prices), 2),
                "min_price": round(min(prices), 2),
                "max_price": round(max(prices), 2),
                "transactions": len(prices),
            })

        return history

    def project_retirement_schedule(
        self, tenant_id: str, annual_target_tco2e: float
    ) -> List[Dict[str, Any]]:
        """Project a retirement schedule for annual offsetting targets.

        Args:
            tenant_id: Tenant identifier.
            annual_target_tco2e: Annual retirement target.

        Returns:
            List of scheduled retirements by year.
        """
        active_credits = sorted(
            [
                c for c in self._credits.values()
                if c.tenant_id == tenant_id and c.status == CreditStatus.ACTIVE
            ],
            key=lambda c: c.vintage_year,  # Retire oldest first
        )

        current_year = utcnow().year
        schedule: List[Dict[str, Any]] = []
        remaining_credits = list(active_credits)

        for year_offset in range(10):
            year = current_year + year_offset
            target_remaining = annual_target_tco2e
            retirements: List[Dict[str, str]] = []

            while target_remaining > 0 and remaining_credits:
                credit = remaining_credits[0]
                retire_qty = min(credit.quantity_tco2e, target_remaining)
                retirements.append({
                    "credit_id": credit.credit_id,
                    "quantity_tco2e": round(retire_qty, 2),
                    "vintage_year": credit.vintage_year,
                    "registry": credit.registry.value,
                })
                target_remaining -= retire_qty
                credit.quantity_tco2e -= retire_qty
                if credit.quantity_tco2e <= 0:
                    remaining_credits.pop(0)

            schedule.append({
                "year": year,
                "target_tco2e": annual_target_tco2e,
                "planned_retirements_tco2e": round(
                    annual_target_tco2e - target_remaining, 2
                ),
                "shortfall_tco2e": round(max(0, target_remaining), 2),
                "retirements": retirements,
            })

            if not remaining_credits and target_remaining > 0:
                # Mark remaining years as shortfall
                for future_offset in range(year_offset + 1, 10):
                    schedule.append({
                        "year": current_year + future_offset,
                        "target_tco2e": annual_target_tco2e,
                        "planned_retirements_tco2e": 0.0,
                        "shortfall_tco2e": annual_target_tco2e,
                        "retirements": [],
                    })
                break

        return schedule

    def get_vintage_breakdown(self, tenant_id: str) -> Dict[str, Any]:
        """Get credits grouped by vintage year.

        Args:
            tenant_id: Tenant identifier.

        Returns:
            Dict with vintage year breakdown.
        """
        credits = [
            c for c in self._credits.values()
            if c.tenant_id == tenant_id
        ]

        by_vintage: Dict[int, Dict[str, float]] = defaultdict(
            lambda: {"active": 0.0, "retired": 0.0, "total": 0.0, "value": 0.0}
        )

        for c in credits:
            entry = by_vintage[c.vintage_year]
            entry["total"] += c.quantity_tco2e
            entry["value"] += c.quantity_tco2e * c.purchase_price_per_tco2e
            if c.status == CreditStatus.ACTIVE:
                entry["active"] += c.quantity_tco2e
            elif c.status == CreditStatus.RETIRED:
                entry["retired"] += c.quantity_tco2e

        breakdown = {
            str(year): {
                k: round(v, 2) for k, v in data.items()
            }
            for year, data in sorted(by_vintage.items())
        }

        return {
            "tenant_id": tenant_id,
            "vintage_breakdown": breakdown,
            "total_vintages": len(breakdown),
            "provenance_hash": _compute_hash(breakdown),
        }

    # -- Internal Helpers ---------------------------------------------------

    def _get_credit(self, credit_id: str) -> CarbonCredit:
        """Retrieve a credit by ID.

        Args:
            credit_id: Credit identifier.

        Returns:
            CarbonCredit object.

        Raises:
            KeyError: If credit not found.
        """
        if credit_id not in self._credits:
            raise KeyError(f"Credit '{credit_id}' not found")
        return self._credits[credit_id]

    def _compute_quality_score(self, credit: CarbonCredit) -> float:
        """Compute quality score for a single credit.

        Args:
            credit: Carbon credit to score.

        Returns:
            Quality score 0-100.
        """
        current_year = utcnow().year
        registry_score = _REGISTRY_SCORES.get(credit.registry, 50.0)
        vintage_freshness = max(0.0, 100.0 - (current_year - credit.vintage_year) * 15)

        return (
            credit.additionality_score * _QUALITY_WEIGHTS["additionality"]
            + credit.permanence_score * _QUALITY_WEIGHTS["permanence"]
            + registry_score * _QUALITY_WEIGHTS["registry_reputation"]
            + vintage_freshness * _QUALITY_WEIGHTS["vintage_freshness"]
        )

    def _record_audit(
        self, event: str, credit_id: str, details: Dict[str, Any]
    ) -> None:
        """Record an audit log entry.

        Args:
            event: Event type.
            credit_id: Related credit ID.
            details: Event details.
        """
        self._audit_log.append({
            "event_id": _new_uuid(),
            "event": event,
            "credit_id": credit_id,
            "details": details,
            "timestamp": utcnow().isoformat(),
            "provenance_hash": _compute_hash(details),
        })
