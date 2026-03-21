# -*- coding: utf-8 -*-
"""
SBTiOffsetBridge - Carbon Credit Management for Net-Zero Neutralization (PACK-023)
=====================================================================================

This module bridges the SBTi Alignment Pack to carbon credit and offset
management for net-zero neutralization. It enforces SBTi Net-Zero Standard
requirements where offsets are ONLY permitted for residual emissions
neutralization after achieving 90%+ abatement in value chain.

SBTi Net-Zero Standard Compliance:
    - Offsets CANNOT count toward near-term Scope 1/2/3 reduction targets
    - Only high-quality carbon removals count for net-zero neutralization
    - Residual emissions must be <= 10% of base year emissions
    - Beyond Value Chain Mitigation (BVCM) recommended but separate
    - Permanence requirement: minimum 100-year storage
    - Additionality verification required

Credit Quality Standards:
    ICVCM Core Carbon Principles (CCP)
    VCMI Claims Code
    Verra VCS
    Gold Standard
    ACR, CAR, Puro.earth

Functions:
    - get_offset_strategy()     -- Design SBTi-compliant offset strategy
    - value_credits()           -- Value and price carbon credits
    - track_credits()           -- Track credit purchases and retirements
    - verify_quality()          -- Verify credit quality
    - check_sbti_compliance()   -- Check offset use against SBTi requirements
    - assess_bvcm()             -- Assess Beyond Value Chain Mitigation

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-023 SBTi Alignment Pack
Status: Production Ready
"""

import hashlib
import importlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Agent Stubs
# ---------------------------------------------------------------------------


class _AgentStub:
    """Stub for unavailable offset agent modules."""

    def __init__(self, agent_name: str) -> None:
        self._agent_name = agent_name
        self._available = False

    def __getattr__(self, name: str) -> Any:
        def _stub_method(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            return {
                "agent": self._agent_name,
                "method": name,
                "status": "degraded",
                "message": f"{self._agent_name} not available, using stub",
            }
        return _stub_method


def _try_import_agent(agent_id: str, module_path: str) -> Any:
    """Try to import an agent with graceful fallback."""
    try:
        return importlib.import_module(module_path)
    except ImportError:
        logger.debug("Agent %s not available, using stub", agent_id)
        return _AgentStub(agent_id)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class CreditType(str, Enum):
    """Carbon credit types."""

    NATURE_BASED_AVOIDANCE = "nature_based_avoidance"
    NATURE_BASED_REMOVAL = "nature_based_removal"
    TECHNOLOGY_BASED_REMOVAL = "technology_based_removal"
    RENEWABLE_ENERGY = "renewable_energy"
    METHANE_AVOIDANCE = "methane_avoidance"
    DIRECT_AIR_CAPTURE = "direct_air_capture"
    BIOCHAR = "biochar"
    ENHANCED_WEATHERING = "enhanced_weathering"
    BIOENERGY_CCS = "bioenergy_ccs"
    SOIL_CARBON = "soil_carbon"


class CreditStandard(str, Enum):
    """Carbon credit verification standards."""

    VERRA_VCS = "verra_vcs"
    GOLD_STANDARD = "gold_standard"
    ACR = "acr"
    CAR = "car"
    PURO_EARTH = "puro_earth"
    ISOMETRIC = "isometric"


class QualityTier(str, Enum):
    """Credit quality tier per ICVCM."""

    PLATINUM = "platinum"
    GOLD = "gold"
    SILVER = "silver"
    BRONZE = "bronze"
    UNRATED = "unrated"


class SBTiOffsetRole(str, Enum):
    """Role of offsets per SBTi Net-Zero Standard."""

    NEUTRALIZATION = "neutralization"
    BVCM = "beyond_value_chain_mitigation"
    NOT_PERMITTED = "not_permitted"


class VCMIClaimTier(str, Enum):
    """VCMI Claims Code claim tiers."""

    GOLD = "gold"
    SILVER = "silver"
    BRONZE = "bronze"
    NOT_ELIGIBLE = "not_eligible"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class OffsetBridgeConfig(BaseModel):
    """Configuration for the SBTi Offset Bridge."""

    pack_id: str = Field(default="PACK-023")
    enable_provenance: bool = Field(default=True)
    base_year_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    residual_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    sbti_max_residual_pct: float = Field(default=10.0, ge=0.0, le=100.0)
    require_removals_only: bool = Field(
        default=True,
        description="SBTi requires removal-based credits for neutralization",
    )
    min_permanence_years: int = Field(default=100, ge=1)


class OffsetStrategyResult(BaseModel):
    """SBTi-compliant offset strategy design result."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    residual_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    residual_pct_of_base: float = Field(default=0.0, ge=0.0, le=100.0)
    within_sbti_residual_limit: bool = Field(default=False)
    neutralization_required_tco2e: float = Field(default=0.0, ge=0.0)
    removal_credits_required_tco2e: float = Field(default=0.0, ge=0.0)
    bvcm_budget_tco2e: float = Field(default=0.0, ge=0.0)
    portfolio_allocation: Dict[str, float] = Field(default_factory=dict)
    estimated_cost_eur: float = Field(default=0.0, ge=0.0)
    credit_types_recommended: List[str] = Field(default_factory=list)
    sbti_compliant: bool = Field(default=False)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class CreditValuationResult(BaseModel):
    """Credit valuation result."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    credit_type: str = Field(default="")
    standard: str = Field(default="")
    vintage_year: int = Field(default=2025)
    volume_tco2e: float = Field(default=0.0, ge=0.0)
    price_eur_per_tco2e: float = Field(default=0.0, ge=0.0)
    total_value_eur: float = Field(default=0.0, ge=0.0)
    quality_tier: str = Field(default="unrated")
    sbti_eligible: bool = Field(default=False)
    sbti_role: str = Field(default="not_permitted")
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class CreditTrackingResult(BaseModel):
    """Credit tracking result."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    total_purchased_tco2e: float = Field(default=0.0, ge=0.0)
    total_retired_tco2e: float = Field(default=0.0, ge=0.0)
    total_available_tco2e: float = Field(default=0.0, ge=0.0)
    by_type: Dict[str, float] = Field(default_factory=dict)
    by_standard: Dict[str, float] = Field(default_factory=dict)
    by_vintage: Dict[str, float] = Field(default_factory=dict)
    neutralization_coverage_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    removal_credits_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class QualityVerificationResult(BaseModel):
    """Credit quality verification result."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    credits_assessed: int = Field(default=0)
    additionality_verified: bool = Field(default=False)
    permanence_years: int = Field(default=0)
    permanence_adequate: bool = Field(default=False)
    no_double_counting: bool = Field(default=False)
    leakage_assessed: bool = Field(default=False)
    social_environmental_safeguards: bool = Field(default=False)
    icvcm_ccp_aligned: bool = Field(default=False)
    quality_tier: str = Field(default="unrated")
    issues: List[str] = Field(default_factory=list)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class SBTiComplianceResult(BaseModel):
    """SBTi offset compliance check result."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    offsets_used_for_near_term: bool = Field(default=False)
    offsets_used_for_neutralization: bool = Field(default=False)
    offsets_used_for_bvcm: bool = Field(default=False)
    near_term_compliant: bool = Field(default=False)
    neutralization_compliant: bool = Field(default=False)
    removal_only_requirement_met: bool = Field(default=False)
    residual_within_limit: bool = Field(default=False)
    overall_compliant: bool = Field(default=False)
    violations: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class BVCMResult(BaseModel):
    """Beyond Value Chain Mitigation assessment result."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    bvcm_budget_tco2e: float = Field(default=0.0, ge=0.0)
    bvcm_budget_pct_of_base: float = Field(default=0.0, ge=0.0, le=100.0)
    credits_allocated_tco2e: float = Field(default=0.0, ge=0.0)
    credit_types: List[str] = Field(default_factory=list)
    estimated_cost_eur: float = Field(default=0.0, ge=0.0)
    sbti_recommended: bool = Field(default=True)
    vcmi_claim_eligible: bool = Field(default=False)
    vcmi_claim_tier: str = Field(default="not_eligible")
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Credit pricing reference
# ---------------------------------------------------------------------------

CREDIT_PRICE_REFERENCE: Dict[str, Dict[str, float]] = {
    "nature_based_avoidance": {"min_eur": 5, "avg_eur": 15, "max_eur": 40},
    "nature_based_removal": {"min_eur": 20, "avg_eur": 45, "max_eur": 100},
    "technology_based_removal": {"min_eur": 100, "avg_eur": 300, "max_eur": 800},
    "direct_air_capture": {"min_eur": 250, "avg_eur": 600, "max_eur": 1200},
    "biochar": {"min_eur": 80, "avg_eur": 150, "max_eur": 300},
    "enhanced_weathering": {"min_eur": 60, "avg_eur": 120, "max_eur": 250},
    "bioenergy_ccs": {"min_eur": 100, "avg_eur": 200, "max_eur": 400},
    "soil_carbon": {"min_eur": 15, "avg_eur": 35, "max_eur": 80},
}

# SBTi-eligible credit types for neutralization (removals only)
SBTI_NEUTRALIZATION_ELIGIBLE: set = {
    CreditType.NATURE_BASED_REMOVAL,
    CreditType.TECHNOLOGY_BASED_REMOVAL,
    CreditType.DIRECT_AIR_CAPTURE,
    CreditType.BIOCHAR,
    CreditType.ENHANCED_WEATHERING,
    CreditType.BIOENERGY_CCS,
    CreditType.SOIL_CARBON,
}


# ---------------------------------------------------------------------------
# SBTiOffsetBridge
# ---------------------------------------------------------------------------


class SBTiOffsetBridge:
    """Carbon credit management bridge for SBTi net-zero neutralization.

    Enforces SBTi Net-Zero Standard requirements where offsets are ONLY
    permitted for residual emissions neutralization and Beyond Value Chain
    Mitigation (BVCM). Near-term targets must be met through direct
    abatement only.

    Example:
        >>> bridge = SBTiOffsetBridge(OffsetBridgeConfig(
        ...     base_year_emissions_tco2e=100000,
        ...     residual_emissions_tco2e=8000
        ... ))
        >>> strategy = bridge.get_offset_strategy()
        >>> assert strategy.within_sbti_residual_limit
    """

    def __init__(self, config: Optional[OffsetBridgeConfig] = None) -> None:
        """Initialize the SBTi Offset Bridge."""
        self.config = config or OffsetBridgeConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(
            "SBTiOffsetBridge initialized: base_year=%s, residual=%s tCO2e",
            self.config.base_year_emissions_tco2e,
            self.config.residual_emissions_tco2e,
        )

    def get_offset_strategy(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> OffsetStrategyResult:
        """Design SBTi-compliant offset strategy for net-zero neutralization.

        Args:
            context: Optional context with override data.

        Returns:
            OffsetStrategyResult with neutralization and BVCM allocation.
        """
        start = time.monotonic()
        context = context or {}

        residual = context.get("residual_emissions_tco2e", self.config.residual_emissions_tco2e)
        base = context.get("base_year_emissions_tco2e", self.config.base_year_emissions_tco2e)
        residual_pct = round(residual / base * 100.0, 2) if base > 0 else 0.0
        within_limit = residual_pct <= self.config.sbti_max_residual_pct

        # BVCM budget: 5-10% of base year emissions recommended
        bvcm_budget = round(base * 0.05, 2)

        # Portfolio allocation
        portfolio = {
            "direct_air_capture": round(residual * 0.3, 2),
            "nature_based_removal": round(residual * 0.3, 2),
            "biochar": round(residual * 0.15, 2),
            "enhanced_weathering": round(residual * 0.15, 2),
            "bioenergy_ccs": round(residual * 0.1, 2),
        }

        # Cost estimate based on portfolio
        total_cost = sum(
            vol * CREDIT_PRICE_REFERENCE.get(ct, {}).get("avg_eur", 100)
            for ct, vol in portfolio.items()
        )

        recommended_types = [
            "direct_air_capture", "nature_based_removal", "biochar",
            "enhanced_weathering", "bioenergy_ccs",
        ]

        sbti_compliant = within_limit and self.config.require_removals_only

        result = OffsetStrategyResult(
            status="completed",
            residual_emissions_tco2e=round(residual, 2),
            residual_pct_of_base=residual_pct,
            within_sbti_residual_limit=within_limit,
            neutralization_required_tco2e=round(residual, 2),
            removal_credits_required_tco2e=round(residual, 2),
            bvcm_budget_tco2e=bvcm_budget,
            portfolio_allocation=portfolio,
            estimated_cost_eur=round(total_cost, 2),
            credit_types_recommended=recommended_types,
            sbti_compliant=sbti_compliant,
        )
        result.duration_ms = (time.monotonic() - start) * 1000
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    def value_credits(
        self,
        credit_type: CreditType,
        volume_tco2e: float = 0.0,
        standard: CreditStandard = CreditStandard.VERRA_VCS,
        context: Optional[Dict[str, Any]] = None,
    ) -> CreditValuationResult:
        """Value and price carbon credits with SBTi eligibility check.

        Args:
            credit_type: Type of carbon credit.
            volume_tco2e: Volume in tCO2e.
            standard: Verification standard.
            context: Optional context with pricing data.

        Returns:
            CreditValuationResult with pricing and eligibility.
        """
        start = time.monotonic()
        context = context or {}

        pricing = CREDIT_PRICE_REFERENCE.get(credit_type.value, {"min_eur": 50, "avg_eur": 100, "max_eur": 200})
        price = context.get("price_eur_per_tco2e", pricing["avg_eur"])
        total_value = round(volume_tco2e * price, 2)

        sbti_eligible = credit_type in SBTI_NEUTRALIZATION_ELIGIBLE
        role = SBTiOffsetRole.NEUTRALIZATION.value if sbti_eligible else SBTiOffsetRole.NOT_PERMITTED.value

        # BVCM: avoidance credits can be used for BVCM but not neutralization
        if not sbti_eligible and credit_type in (CreditType.NATURE_BASED_AVOIDANCE, CreditType.METHANE_AVOIDANCE, CreditType.RENEWABLE_ENERGY):
            role = SBTiOffsetRole.BVCM.value

        quality_tier = context.get("quality_tier", "silver")

        result = CreditValuationResult(
            status="completed",
            credit_type=credit_type.value,
            standard=standard.value,
            vintage_year=context.get("vintage_year", 2025),
            volume_tco2e=round(volume_tco2e, 2),
            price_eur_per_tco2e=round(price, 2),
            total_value_eur=total_value,
            quality_tier=quality_tier,
            sbti_eligible=sbti_eligible,
            sbti_role=role,
        )
        result.duration_ms = (time.monotonic() - start) * 1000
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    def track_credits(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> CreditTrackingResult:
        """Track credit purchases and retirements.

        Args:
            context: Optional context with tracking data.

        Returns:
            CreditTrackingResult with portfolio status.
        """
        start = time.monotonic()
        context = context or {}

        purchased = context.get("total_purchased_tco2e", 0.0)
        retired = context.get("total_retired_tco2e", 0.0)
        available = purchased - retired

        by_type = context.get("by_type", {})
        by_standard = context.get("by_standard", {})
        by_vintage = context.get("by_vintage", {})

        # Calculate neutralization coverage
        residual = self.config.residual_emissions_tco2e
        coverage = round(retired / residual * 100.0, 1) if residual > 0 else 0.0

        # Calculate removal % (SBTi requires removals for neutralization)
        removal_types = {"nature_based_removal", "technology_based_removal", "direct_air_capture", "biochar", "enhanced_weathering", "bioenergy_ccs", "soil_carbon"}
        removal_total = sum(v for k, v in by_type.items() if k in removal_types)
        removal_pct = round(removal_total / purchased * 100.0, 1) if purchased > 0 else 0.0

        result = CreditTrackingResult(
            status="completed",
            total_purchased_tco2e=round(purchased, 2),
            total_retired_tco2e=round(retired, 2),
            total_available_tco2e=round(available, 2),
            by_type=by_type,
            by_standard=by_standard,
            by_vintage=by_vintage,
            neutralization_coverage_pct=min(coverage, 100.0),
            removal_credits_pct=removal_pct,
        )
        result.duration_ms = (time.monotonic() - start) * 1000
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    def verify_quality(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> QualityVerificationResult:
        """Verify credit quality against ICVCM Core Carbon Principles.

        Args:
            context: Optional context with verification data.

        Returns:
            QualityVerificationResult with quality assessment.
        """
        start = time.monotonic()
        context = context or {}

        additionality = context.get("additionality_verified", False)
        permanence = context.get("permanence_years", 0)
        permanence_ok = permanence >= self.config.min_permanence_years
        no_double = context.get("no_double_counting", False)
        leakage = context.get("leakage_assessed", False)
        safeguards = context.get("social_environmental_safeguards", False)

        ccp_aligned = all([additionality, permanence_ok, no_double, leakage, safeguards])

        issues: List[str] = []
        if not additionality:
            issues.append("Additionality not verified")
        if not permanence_ok:
            issues.append(f"Permanence insufficient ({permanence} years, need {self.config.min_permanence_years})")
        if not no_double:
            issues.append("Double counting risk not addressed")
        if not leakage:
            issues.append("Leakage assessment missing")
        if not safeguards:
            issues.append("Social/environmental safeguards not confirmed")

        if ccp_aligned:
            tier = "gold"
        elif sum([additionality, permanence_ok, no_double]) >= 2:
            tier = "silver"
        elif additionality:
            tier = "bronze"
        else:
            tier = "unrated"

        result = QualityVerificationResult(
            status="completed",
            credits_assessed=context.get("credits_assessed", 0),
            additionality_verified=additionality,
            permanence_years=permanence,
            permanence_adequate=permanence_ok,
            no_double_counting=no_double,
            leakage_assessed=leakage,
            social_environmental_safeguards=safeguards,
            icvcm_ccp_aligned=ccp_aligned,
            quality_tier=tier,
            issues=issues,
        )
        result.duration_ms = (time.monotonic() - start) * 1000
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    def check_sbti_compliance(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> SBTiComplianceResult:
        """Check offset use against SBTi Net-Zero Standard requirements.

        Args:
            context: Optional context with compliance data.

        Returns:
            SBTiComplianceResult with violation assessment.
        """
        start = time.monotonic()
        context = context or {}

        used_for_near_term = context.get("offsets_used_for_near_term", False)
        used_for_neutralization = context.get("offsets_used_for_neutralization", False)
        used_for_bvcm = context.get("offsets_used_for_bvcm", False)
        removal_only = context.get("removal_only_credits", True)

        residual = self.config.residual_emissions_tco2e
        base = self.config.base_year_emissions_tco2e
        residual_pct = residual / base * 100.0 if base > 0 else 0.0
        within_limit = residual_pct <= self.config.sbti_max_residual_pct

        violations: List[str] = []
        recommendations: List[str] = []

        # Near-term compliance: offsets must NOT be used
        near_term_ok = not used_for_near_term
        if not near_term_ok:
            violations.append("VIOLATION: Offsets used for near-term target reduction (not permitted by SBTi)")
            recommendations.append("Remove offsets from near-term target accounting; achieve through direct abatement")

        # Neutralization compliance
        neutralization_ok = True
        if used_for_neutralization and not removal_only:
            neutralization_ok = False
            violations.append("VIOLATION: Non-removal credits used for neutralization (SBTi requires removals)")
            recommendations.append("Replace avoidance credits with removal-based credits for neutralization")
        if not within_limit:
            violations.append(f"WARNING: Residual emissions ({residual_pct:.1f}%) exceed SBTi limit ({self.config.sbti_max_residual_pct}%)")
            recommendations.append("Increase abatement to reduce residual emissions below threshold")

        overall = near_term_ok and neutralization_ok and within_limit

        if overall:
            recommendations.append("Offset strategy is SBTi-compliant")

        result = SBTiComplianceResult(
            status="completed",
            offsets_used_for_near_term=used_for_near_term,
            offsets_used_for_neutralization=used_for_neutralization,
            offsets_used_for_bvcm=used_for_bvcm,
            near_term_compliant=near_term_ok,
            neutralization_compliant=neutralization_ok,
            removal_only_requirement_met=removal_only,
            residual_within_limit=within_limit,
            overall_compliant=overall,
            violations=violations,
            recommendations=recommendations,
        )
        result.duration_ms = (time.monotonic() - start) * 1000
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    def assess_bvcm(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> BVCMResult:
        """Assess Beyond Value Chain Mitigation opportunity.

        Args:
            context: Optional context with BVCM data.

        Returns:
            BVCMResult with BVCM budget and VCMI eligibility.
        """
        start = time.monotonic()
        context = context or {}

        base = self.config.base_year_emissions_tco2e
        bvcm_pct = context.get("bvcm_pct_of_base", 5.0)
        bvcm_budget = round(base * bvcm_pct / 100.0, 2)
        allocated = context.get("credits_allocated_tco2e", 0.0)

        credit_types = context.get("credit_types", ["nature_based_avoidance", "nature_based_removal"])
        cost = sum(
            CREDIT_PRICE_REFERENCE.get(ct, {}).get("avg_eur", 50) * (allocated / len(credit_types) if credit_types else 0)
            for ct in credit_types
        )

        # VCMI Claims Code eligibility
        vcmi_eligible = allocated >= bvcm_budget * 0.5
        if vcmi_eligible and allocated >= bvcm_budget:
            vcmi_tier = "gold"
        elif vcmi_eligible:
            vcmi_tier = "silver"
        else:
            vcmi_tier = "not_eligible"

        result = BVCMResult(
            status="completed",
            bvcm_budget_tco2e=bvcm_budget,
            bvcm_budget_pct_of_base=bvcm_pct,
            credits_allocated_tco2e=round(allocated, 2),
            credit_types=credit_types,
            estimated_cost_eur=round(cost, 2),
            sbti_recommended=True,
            vcmi_claim_eligible=vcmi_eligible,
            vcmi_claim_tier=vcmi_tier,
        )
        result.duration_ms = (time.monotonic() - start) * 1000
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    def get_bridge_status(self) -> Dict[str, Any]:
        """Get current bridge status."""
        return {
            "pack_id": self.config.pack_id,
            "module_version": _MODULE_VERSION,
            "base_year_emissions_tco2e": self.config.base_year_emissions_tco2e,
            "residual_emissions_tco2e": self.config.residual_emissions_tco2e,
            "sbti_max_residual_pct": self.config.sbti_max_residual_pct,
            "require_removals_only": self.config.require_removals_only,
            "min_permanence_years": self.config.min_permanence_years,
        }
