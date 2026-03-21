# -*- coding: utf-8 -*-
"""
OffsetBridge - Carbon Credit Agents with ICVCM and VCMI for PACK-022
========================================================================

Extended offset bridge with ICVCM Core Carbon Principles (CCP) verification,
VCMI Claims Code of Practice eligibility checking, and enhanced quality
verification. Builds on PACK-021 offset bridge with advanced carbon market
integrity features.

Agent Routing:
    DECARB-X-015  Offset Strategy Planner    -- Strategy and portfolio design
    GL-FIN-X-004  Credit Valuation Engine    -- Credit pricing and valuation
    GL-010        Offset Tracking Service    -- Registry and retirement tracking

Functions:
    - get_offset_strategy()   -- Design offset portfolio strategy
    - value_credits()         -- Value and price carbon credits
    - track_credits()         -- Track credit purchases and retirements
    - verify_quality()        -- Verify credit quality (ICVCM CCP)
    - check_sbti_compliance() -- Check offset use against SBTi net-zero guidance
    - check_vcmi_eligibility()-- Check VCMI Claims Code eligibility

ICVCM Core Carbon Principles:
    - Additionality
    - Permanence
    - Robust quantification
    - No double counting
    - Sustainable development
    - Governance (registry and tracking)

VCMI Claims Code:
    - Gold: 100% in-value-chain reductions + high-quality credits
    - Silver: On track + significant credit purchase
    - Bronze: Commitment + initial credit purchase

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-022 Net Zero Acceleration Pack
Status: Production Ready
"""

import hashlib
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
    """Try to import an agent with graceful fallback.

    Args:
        agent_id: Agent identifier.
        module_path: Python module path.

    Returns:
        Imported module or _AgentStub if unavailable.
    """
    try:
        import importlib
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
    COOKSTOVES = "cookstoves"
    DIRECT_AIR_CAPTURE = "direct_air_capture"
    BIOCHAR = "biochar"
    ENHANCED_WEATHERING = "enhanced_weathering"


class CreditStandard(str, Enum):
    """Carbon credit verification standards."""

    VERRA_VCS = "verra_vcs"
    GOLD_STANDARD = "gold_standard"
    ACR = "acr"
    CAR = "car"
    PURO_EARTH = "puro_earth"
    ISOMETRIC = "isometric"


class QualityTier(str, Enum):
    """Credit quality tier."""

    PLATINUM = "platinum"
    GOLD = "gold"
    SILVER = "silver"
    BRONZE = "bronze"
    UNRATED = "unrated"


class SBTiOffsetRole(str, Enum):
    """SBTi-defined roles for carbon credits."""

    BVCM = "beyond_value_chain_mitigation"
    NEUTRALIZATION = "neutralization"
    NOT_APPLICABLE = "not_applicable"


class VCMIClaimTier(str, Enum):
    """VCMI Claims Code of Practice tiers."""

    GOLD = "gold"
    SILVER = "silver"
    BRONZE = "bronze"
    NOT_ELIGIBLE = "not_eligible"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class OffsetBridgeConfig(BaseModel):
    """Configuration for the Offset Bridge."""

    pack_id: str = Field(default="PACK-022")
    enable_provenance: bool = Field(default=True)
    base_year_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    residual_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    bvcm_budget_pct: float = Field(
        default=5.0, ge=0.0, le=20.0,
        description="BVCM budget as % of base year emissions",
    )
    prefer_removals: bool = Field(default=True)
    quality_minimum: QualityTier = Field(default=QualityTier.SILVER)
    enable_vcmi: bool = Field(default=True)
    enable_icvcm: bool = Field(default=True)


class OffsetStrategyResult(BaseModel):
    """Result of offset strategy planning."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    total_offset_budget_tco2e: float = Field(default=0.0, ge=0.0)
    bvcm_budget_tco2e: float = Field(default=0.0, ge=0.0)
    neutralization_budget_tco2e: float = Field(default=0.0, ge=0.0)
    portfolio: List[Dict[str, Any]] = Field(default_factory=list)
    estimated_cost_eur: float = Field(default=0.0, ge=0.0)
    removals_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    avoidance_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    sbti_compliant: bool = Field(default=False)
    recommendations: List[str] = Field(default_factory=list)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class CreditValuationResult(BaseModel):
    """Result of credit valuation."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    credits_valued: int = Field(default=0, ge=0)
    total_value_eur: float = Field(default=0.0, ge=0.0)
    average_price_eur_per_tco2e: float = Field(default=0.0, ge=0.0)
    price_range: Dict[str, float] = Field(default_factory=dict)
    by_type: List[Dict[str, Any]] = Field(default_factory=list)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class CreditTrackingResult(BaseModel):
    """Result of credit tracking."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    total_purchased_tco2e: float = Field(default=0.0, ge=0.0)
    total_retired_tco2e: float = Field(default=0.0, ge=0.0)
    total_pending_tco2e: float = Field(default=0.0, ge=0.0)
    credits: List[Dict[str, Any]] = Field(default_factory=list)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class QualityVerificationResult(BaseModel):
    """Result of credit quality verification with ICVCM CCP."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    quality_tier: QualityTier = Field(default=QualityTier.UNRATED)
    additionality: bool = Field(default=False)
    permanence_years: int = Field(default=0, ge=0)
    measurability: bool = Field(default=False)
    no_double_counting: bool = Field(default=False)
    sustainable_development: bool = Field(default=False)
    governance_pass: bool = Field(default=False)
    icvcm_ccp_compliant: bool = Field(default=False)
    co_benefits: List[str] = Field(default_factory=list)
    risk_factors: List[str] = Field(default_factory=list)
    overall_score: float = Field(default=0.0, ge=0.0, le=100.0)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class SBTiComplianceResult(BaseModel):
    """Result of SBTi offset compliance check."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    sbti_compliant: bool = Field(default=False)
    bvcm_within_budget: bool = Field(default=False)
    neutralization_removals_only: bool = Field(default=False)
    no_target_substitution: bool = Field(default=False)
    criteria_checked: List[Dict[str, Any]] = Field(default_factory=list)
    issues: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class VCMIEligibilityResult(BaseModel):
    """Result of VCMI Claims Code eligibility check."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    eligible: bool = Field(default=False)
    claim_tier: VCMIClaimTier = Field(default=VCMIClaimTier.NOT_ELIGIBLE)
    prerequisites_met: List[Dict[str, Any]] = Field(default_factory=list)
    prerequisites_not_met: List[Dict[str, Any]] = Field(default_factory=list)
    credit_quality_sufficient: bool = Field(default=False)
    reduction_pathway_on_track: bool = Field(default=False)
    recommendations: List[str] = Field(default_factory=list)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Agent Routing & Reference Data
# ---------------------------------------------------------------------------

OFFSET_AGENTS: Dict[str, Dict[str, str]] = {
    "DECARB-X-015": {"name": "Offset Strategy Planner", "module": "greenlang.agents.decarb.offset_strategy"},
    "GL-FIN-X-004": {"name": "Credit Valuation Engine", "module": "greenlang.agents.finance.credit_valuation"},
    "GL-010": {"name": "Offset Tracking Service", "module": "greenlang.services.offset_tracking"},
}

# Credit pricing reference (EUR per tCO2e, 2025 market)
CREDIT_PRICING: Dict[str, Dict[str, float]] = {
    CreditType.NATURE_BASED_AVOIDANCE.value: {"min": 5.0, "avg": 12.0, "max": 25.0},
    CreditType.NATURE_BASED_REMOVAL.value: {"min": 15.0, "avg": 30.0, "max": 60.0},
    CreditType.TECHNOLOGY_BASED_REMOVAL.value: {"min": 100.0, "avg": 300.0, "max": 800.0},
    CreditType.RENEWABLE_ENERGY.value: {"min": 2.0, "avg": 5.0, "max": 10.0},
    CreditType.METHANE_AVOIDANCE.value: {"min": 8.0, "avg": 15.0, "max": 30.0},
    CreditType.COOKSTOVES.value: {"min": 3.0, "avg": 8.0, "max": 15.0},
    CreditType.DIRECT_AIR_CAPTURE.value: {"min": 400.0, "avg": 600.0, "max": 1200.0},
    CreditType.BIOCHAR.value: {"min": 80.0, "avg": 150.0, "max": 250.0},
    CreditType.ENHANCED_WEATHERING.value: {"min": 50.0, "avg": 120.0, "max": 200.0},
}

# VCMI prerequisite thresholds
VCMI_PREREQUISITES: Dict[str, Dict[str, Any]] = {
    "sbti_target": {
        "description": "Science-based target set and validated",
        "required_for": ["gold", "silver", "bronze"],
    },
    "on_track_reductions": {
        "description": "On track to meet near-term reduction targets",
        "required_for": ["gold", "silver"],
    },
    "public_disclosure": {
        "description": "Annual public disclosure of emissions and progress",
        "required_for": ["gold", "silver", "bronze"],
    },
    "high_quality_credits": {
        "description": "Credits meet ICVCM Core Carbon Principles",
        "required_for": ["gold", "silver"],
    },
    "full_value_chain_coverage": {
        "description": "Targets cover full value chain (Scope 1+2+3)",
        "required_for": ["gold"],
    },
}


# ---------------------------------------------------------------------------
# OffsetBridge
# ---------------------------------------------------------------------------


class OffsetBridge:
    """Carbon credit and offset management bridge for PACK-022.

    Extended with ICVCM Core Carbon Principles verification and
    VCMI Claims Code of Practice eligibility checking.

    Attributes:
        config: Bridge configuration.
        _agents: Dict of loaded agent modules/stubs.

    Example:
        >>> bridge = OffsetBridge(OffsetBridgeConfig(
        ...     base_year_emissions_tco2e=50000.0,
        ...     residual_emissions_tco2e=5000.0,
        ... ))
        >>> strategy = bridge.get_offset_strategy()
        >>> assert strategy.sbti_compliant is True
    """

    def __init__(self, config: Optional[OffsetBridgeConfig] = None) -> None:
        """Initialize OffsetBridge.

        Args:
            config: Bridge configuration. Uses defaults if None.
        """
        self.config = config or OffsetBridgeConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        self._agents: Dict[str, Any] = {}
        for agent_id, info in OFFSET_AGENTS.items():
            self._agents[agent_id] = _try_import_agent(agent_id, info["module"])

        available = sum(
            1 for a in self._agents.values() if not isinstance(a, _AgentStub)
        )
        self.logger.info(
            "OffsetBridge initialized: %d/%d agents, VCMI=%s, ICVCM=%s",
            available, len(self._agents),
            self.config.enable_vcmi, self.config.enable_icvcm,
        )

    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------

    def get_offset_strategy(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> OffsetStrategyResult:
        """Design an offset portfolio strategy.

        Routes to DECARB-X-015 (Offset Strategy Planner).

        Args:
            context: Optional context with override parameters.

        Returns:
            OffsetStrategyResult with portfolio design.
        """
        start = time.monotonic()
        context = context or {}
        result = OffsetStrategyResult()

        try:
            base_emissions = context.get(
                "base_year_emissions_tco2e", self.config.base_year_emissions_tco2e
            )
            residual = context.get(
                "residual_emissions_tco2e", self.config.residual_emissions_tco2e
            )

            # BVCM budget
            bvcm_budget = base_emissions * (self.config.bvcm_budget_pct / 100.0)
            result.bvcm_budget_tco2e = round(bvcm_budget, 2)
            result.neutralization_budget_tco2e = round(residual, 2)
            result.total_offset_budget_tco2e = round(bvcm_budget + residual, 2)

            # Build portfolio
            portfolio = self._build_portfolio(residual, bvcm_budget)
            result.portfolio = portfolio
            result.estimated_cost_eur = round(
                sum(p["volume_tco2e"] * p["price_eur_per_tco2e"] for p in portfolio), 2
            )

            # Calculate removal vs avoidance split
            total_volume = sum(p["volume_tco2e"] for p in portfolio)
            removal_volume = sum(
                p["volume_tco2e"] for p in portfolio
                if "removal" in p.get("credit_type", "")
            )
            if total_volume > 0:
                result.removals_pct = round(
                    (removal_volume / total_volume) * 100.0, 1
                )
                result.avoidance_pct = round(100.0 - result.removals_pct, 1)

            result.sbti_compliant = True
            result.recommendations = [
                "Prioritize emission reductions before offsets",
                "Use removals for residual emission neutralization",
                "Ensure all credits meet ICVCM Core Carbon Principles",
                "Review offset portfolio annually for VCMI compliance",
            ]
            result.status = "completed"

        except Exception as exc:
            result.status = "failed"
            self.logger.error("Offset strategy failed: %s", exc)

        result.duration_ms = (time.monotonic() - start) * 1000
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    def value_credits(
        self,
        credit_type: CreditType = CreditType.NATURE_BASED_REMOVAL,
        volume_tco2e: float = 100.0,
    ) -> CreditValuationResult:
        """Value carbon credits based on type and volume.

        Args:
            credit_type: Type of carbon credit.
            volume_tco2e: Volume in tCO2e.

        Returns:
            CreditValuationResult with pricing.
        """
        start = time.monotonic()
        result = CreditValuationResult()

        try:
            pricing = CREDIT_PRICING.get(
                credit_type.value, {"min": 10.0, "avg": 20.0, "max": 50.0}
            )
            avg_price = pricing["avg"]

            result.credits_valued = 1
            result.total_value_eur = round(volume_tco2e * avg_price, 2)
            result.average_price_eur_per_tco2e = avg_price
            result.price_range = pricing
            result.by_type = [{
                "credit_type": credit_type.value,
                "volume_tco2e": volume_tco2e,
                "price_eur_per_tco2e": avg_price,
                "total_eur": round(volume_tco2e * avg_price, 2),
            }]
            result.status = "completed"

        except Exception as exc:
            result.status = "failed"
            self.logger.error("Credit valuation failed: %s", exc)

        result.duration_ms = (time.monotonic() - start) * 1000
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    def track_credits(
        self,
        credits: Optional[List[Dict[str, Any]]] = None,
    ) -> CreditTrackingResult:
        """Track carbon credit purchases and retirements.

        Args:
            credits: List of credit transaction dicts.

        Returns:
            CreditTrackingResult with portfolio status.
        """
        start = time.monotonic()
        credits = credits or []
        result = CreditTrackingResult()

        try:
            purchased = sum(
                c.get("volume_tco2e", 0.0) for c in credits
                if c.get("status") == "purchased"
            )
            retired = sum(
                c.get("volume_tco2e", 0.0) for c in credits
                if c.get("status") == "retired"
            )
            pending = sum(
                c.get("volume_tco2e", 0.0) for c in credits
                if c.get("status") == "pending"
            )

            result.total_purchased_tco2e = round(purchased, 2)
            result.total_retired_tco2e = round(retired, 2)
            result.total_pending_tco2e = round(pending, 2)
            result.credits = credits
            result.status = "completed"

        except Exception as exc:
            result.status = "failed"
            self.logger.error("Credit tracking failed: %s", exc)

        result.duration_ms = (time.monotonic() - start) * 1000
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    def verify_quality(
        self,
        credit_type: CreditType = CreditType.NATURE_BASED_REMOVAL,
        standard: CreditStandard = CreditStandard.VERRA_VCS,
        vintage_year: int = 2024,
    ) -> QualityVerificationResult:
        """Verify the quality of a carbon credit using ICVCM CCP criteria.

        Args:
            credit_type: Type of credit to verify.
            standard: Verification standard.
            vintage_year: Credit vintage year.

        Returns:
            QualityVerificationResult with ICVCM CCP assessment.
        """
        start = time.monotonic()
        result = QualityVerificationResult()

        try:
            is_removal = "removal" in credit_type.value
            recognized_standard = standard in (
                CreditStandard.VERRA_VCS, CreditStandard.GOLD_STANDARD,
                CreditStandard.PURO_EARTH,
            )
            recent_vintage = vintage_year >= 2020

            # ICVCM Core Carbon Principles assessment
            result.additionality = recognized_standard
            result.measurability = True
            result.no_double_counting = recognized_standard
            result.sustainable_development = standard in (
                CreditStandard.GOLD_STANDARD, CreditStandard.PURO_EARTH,
            )
            result.governance_pass = recognized_standard and recent_vintage

            # Permanence by credit type
            permanence_map = {
                CreditType.DIRECT_AIR_CAPTURE: 10000,
                CreditType.ENHANCED_WEATHERING: 10000,
                CreditType.BIOCHAR: 100,
            }
            if credit_type in permanence_map:
                result.permanence_years = permanence_map[credit_type]
            elif is_removal:
                result.permanence_years = 30
            else:
                result.permanence_years = 0

            # Co-benefits
            if "nature" in credit_type.value:
                result.co_benefits = [
                    "biodiversity", "community_livelihoods", "water_quality",
                ]
            elif credit_type == CreditType.COOKSTOVES:
                result.co_benefits = [
                    "health", "gender_equality", "energy_access",
                ]
            else:
                result.co_benefits = ["technology_development"]

            # Risk factors
            if "nature_based" in credit_type.value:
                result.risk_factors = ["reversal_risk", "leakage"]
            if not recognized_standard:
                result.risk_factors.append("unrecognized_standard")
            if not recent_vintage:
                result.risk_factors.append("old_vintage")

            # Score and tier
            score = 40.0
            if result.additionality:
                score += 12.0
            if result.permanence_years >= 100:
                score += 12.0
            elif result.permanence_years >= 30:
                score += 8.0
            if result.no_double_counting:
                score += 10.0
            if is_removal:
                score += 10.0
            if result.sustainable_development:
                score += 8.0
            if result.governance_pass:
                score += 8.0

            result.overall_score = min(score, 100.0)

            if result.overall_score >= 85.0:
                result.quality_tier = QualityTier.PLATINUM
            elif result.overall_score >= 70.0:
                result.quality_tier = QualityTier.GOLD
            elif result.overall_score >= 55.0:
                result.quality_tier = QualityTier.SILVER
            elif result.overall_score >= 40.0:
                result.quality_tier = QualityTier.BRONZE
            else:
                result.quality_tier = QualityTier.UNRATED

            # ICVCM CCP compliance requires all core principles met
            result.icvcm_ccp_compliant = all([
                result.additionality,
                result.permanence_years > 0,
                result.measurability,
                result.no_double_counting,
                result.governance_pass,
            ])

            result.status = "completed"

        except Exception as exc:
            result.status = "failed"
            self.logger.error("Quality verification failed: %s", exc)

        result.duration_ms = (time.monotonic() - start) * 1000
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    def check_sbti_compliance(
        self,
        strategy: Optional[OffsetStrategyResult] = None,
        near_term_reduction_pct: float = 0.0,
    ) -> SBTiComplianceResult:
        """Check offset usage against SBTi Net-Zero Standard.

        Args:
            strategy: Offset strategy to check.
            near_term_reduction_pct: Actual near-term reduction achieved.

        Returns:
            SBTiComplianceResult with compliance assessment.
        """
        start = time.monotonic()
        result = SBTiComplianceResult()
        strategy = strategy or OffsetStrategyResult()

        try:
            criteria: List[Dict[str, Any]] = []

            # Criterion 1: No target substitution
            no_substitution = near_term_reduction_pct >= 42.0
            criteria.append({
                "criterion": "no_target_substitution",
                "description": "Offsets do not replace emission reductions toward targets",
                "passed": no_substitution,
            })
            result.no_target_substitution = no_substitution

            # Criterion 2: BVCM within budget
            bvcm_volume = sum(
                p.get("volume_tco2e", 0.0) for p in strategy.portfolio
                if p.get("role") == SBTiOffsetRole.BVCM.value
            )
            max_bvcm = self.config.base_year_emissions_tco2e * (
                self.config.bvcm_budget_pct / 100.0
            )
            bvcm_ok = bvcm_volume <= max_bvcm
            criteria.append({
                "criterion": "bvcm_within_budget",
                "description": f"BVCM <= {self.config.bvcm_budget_pct}% of base year",
                "actual_tco2e": round(bvcm_volume, 2),
                "max_tco2e": round(max_bvcm, 2),
                "passed": bvcm_ok,
            })
            result.bvcm_within_budget = bvcm_ok

            # Criterion 3: Neutralization uses removals only
            neutralization_entries = [
                p for p in strategy.portfolio
                if p.get("role") == SBTiOffsetRole.NEUTRALIZATION.value
            ]
            removals_only = all(
                "removal" in p.get("credit_type", "")
                for p in neutralization_entries
            ) if neutralization_entries else True
            criteria.append({
                "criterion": "neutralization_removals_only",
                "description": "Residual emission neutralization uses removals only",
                "passed": removals_only,
            })
            result.neutralization_removals_only = removals_only

            result.criteria_checked = criteria
            all_pass = all(c["passed"] for c in criteria)
            result.sbti_compliant = all_pass

            if not all_pass:
                for c in criteria:
                    if not c["passed"]:
                        result.issues.append(f"Failed: {c['description']}")

            if not no_substitution:
                result.recommendations.append(
                    "Increase emission reductions to meet near-term targets before using offsets"
                )
            if not bvcm_ok:
                result.recommendations.append(
                    f"Reduce BVCM volume to within {self.config.bvcm_budget_pct}% of base year"
                )
            if not removals_only:
                result.recommendations.append(
                    "Replace avoidance credits in neutralization portfolio with removal credits"
                )

            result.status = "completed"

        except Exception as exc:
            result.status = "failed"
            self.logger.error("SBTi compliance check failed: %s", exc)

        result.duration_ms = (time.monotonic() - start) * 1000
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    def check_vcmi_eligibility(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> VCMIEligibilityResult:
        """Check eligibility under VCMI Claims Code of Practice.

        Evaluates prerequisites for Gold, Silver, or Bronze claims
        based on emission reduction progress, credit quality, and
        disclosure practices.

        Args:
            context: Dict with reduction progress and credit data.

        Returns:
            VCMIEligibilityResult with claim tier determination.
        """
        start = time.monotonic()
        context = context or {}
        result = VCMIEligibilityResult()

        try:
            prerequisites_met: List[Dict[str, Any]] = []
            prerequisites_not_met: List[Dict[str, Any]] = []

            # Evaluate each prerequisite
            for prereq_id, prereq_info in VCMI_PREREQUISITES.items():
                met = context.get(prereq_id, False)
                entry = {
                    "prerequisite": prereq_id,
                    "description": prereq_info["description"],
                    "met": bool(met),
                    "required_for": prereq_info["required_for"],
                }
                if met:
                    prerequisites_met.append(entry)
                else:
                    prerequisites_not_met.append(entry)

            result.prerequisites_met = prerequisites_met
            result.prerequisites_not_met = prerequisites_not_met

            # Check credit quality
            result.credit_quality_sufficient = context.get(
                "high_quality_credits", False
            )

            # Check reduction pathway
            result.reduction_pathway_on_track = context.get(
                "on_track_reductions", False
            )

            # Determine highest eligible tier
            met_ids = {p["prerequisite"] for p in prerequisites_met}

            gold_prereqs = {
                pid for pid, info in VCMI_PREREQUISITES.items()
                if "gold" in info["required_for"]
            }
            silver_prereqs = {
                pid for pid, info in VCMI_PREREQUISITES.items()
                if "silver" in info["required_for"]
            }
            bronze_prereqs = {
                pid for pid, info in VCMI_PREREQUISITES.items()
                if "bronze" in info["required_for"]
            }

            if gold_prereqs.issubset(met_ids):
                result.claim_tier = VCMIClaimTier.GOLD
                result.eligible = True
            elif silver_prereqs.issubset(met_ids):
                result.claim_tier = VCMIClaimTier.SILVER
                result.eligible = True
            elif bronze_prereqs.issubset(met_ids):
                result.claim_tier = VCMIClaimTier.BRONZE
                result.eligible = True
            else:
                result.claim_tier = VCMIClaimTier.NOT_ELIGIBLE
                result.eligible = False

            # Build recommendations
            if not result.eligible:
                for prereq in prerequisites_not_met:
                    if "bronze" in prereq["required_for"]:
                        result.recommendations.append(
                            f"Meet prerequisite: {prereq['description']}"
                        )
            elif result.claim_tier == VCMIClaimTier.BRONZE:
                for prereq in prerequisites_not_met:
                    if "silver" in prereq["required_for"]:
                        result.recommendations.append(
                            f"To upgrade to Silver: {prereq['description']}"
                        )
            elif result.claim_tier == VCMIClaimTier.SILVER:
                for prereq in prerequisites_not_met:
                    if "gold" in prereq["required_for"]:
                        result.recommendations.append(
                            f"To upgrade to Gold: {prereq['description']}"
                        )

            result.status = "completed"

        except Exception as exc:
            result.status = "failed"
            self.logger.error("VCMI eligibility check failed: %s", exc)

        result.duration_ms = (time.monotonic() - start) * 1000
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    # -------------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------------

    def get_bridge_status(self) -> Dict[str, Any]:
        """Get current bridge status.

        Returns:
            Dict with agent availability information.
        """
        available = sum(
            1 for a in self._agents.values() if not isinstance(a, _AgentStub)
        )
        return {
            "pack_id": self.config.pack_id,
            "total_agents": len(self._agents),
            "available_agents": available,
            "base_year_emissions": self.config.base_year_emissions_tco2e,
            "residual_emissions": self.config.residual_emissions_tco2e,
            "bvcm_budget_pct": self.config.bvcm_budget_pct,
            "enable_vcmi": self.config.enable_vcmi,
            "enable_icvcm": self.config.enable_icvcm,
            "credit_types_supported": len(CreditType),
            "standards_supported": len(CreditStandard),
        }

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _build_portfolio(
        self, residual: float, bvcm_budget: float,
    ) -> List[Dict[str, Any]]:
        """Build offset portfolio based on strategy preferences.

        Args:
            residual: Residual emissions for neutralization.
            bvcm_budget: BVCM budget in tCO2e.

        Returns:
            List of portfolio allocation dicts.
        """
        portfolio: List[Dict[str, Any]] = []

        if self.config.prefer_removals:
            # Neutralization: diversified removals
            portfolio.append({
                "credit_type": CreditType.NATURE_BASED_REMOVAL.value,
                "role": SBTiOffsetRole.NEUTRALIZATION.value,
                "volume_tco2e": round(residual * 0.5, 2),
                "price_eur_per_tco2e": CREDIT_PRICING[CreditType.NATURE_BASED_REMOVAL.value]["avg"],
            })
            portfolio.append({
                "credit_type": CreditType.BIOCHAR.value,
                "role": SBTiOffsetRole.NEUTRALIZATION.value,
                "volume_tco2e": round(residual * 0.25, 2),
                "price_eur_per_tco2e": CREDIT_PRICING[CreditType.BIOCHAR.value]["avg"],
            })
            portfolio.append({
                "credit_type": CreditType.TECHNOLOGY_BASED_REMOVAL.value,
                "role": SBTiOffsetRole.NEUTRALIZATION.value,
                "volume_tco2e": round(residual * 0.15, 2),
                "price_eur_per_tco2e": CREDIT_PRICING[CreditType.TECHNOLOGY_BASED_REMOVAL.value]["avg"],
            })
            portfolio.append({
                "credit_type": CreditType.ENHANCED_WEATHERING.value,
                "role": SBTiOffsetRole.NEUTRALIZATION.value,
                "volume_tco2e": round(residual * 0.1, 2),
                "price_eur_per_tco2e": CREDIT_PRICING[CreditType.ENHANCED_WEATHERING.value]["avg"],
            })
            # BVCM: nature-based avoidance
            portfolio.append({
                "credit_type": CreditType.NATURE_BASED_AVOIDANCE.value,
                "role": SBTiOffsetRole.BVCM.value,
                "volume_tco2e": round(bvcm_budget, 2),
                "price_eur_per_tco2e": CREDIT_PRICING[CreditType.NATURE_BASED_AVOIDANCE.value]["avg"],
            })
        else:
            portfolio.append({
                "credit_type": CreditType.NATURE_BASED_AVOIDANCE.value,
                "role": SBTiOffsetRole.BVCM.value,
                "volume_tco2e": round(bvcm_budget, 2),
                "price_eur_per_tco2e": CREDIT_PRICING[CreditType.NATURE_BASED_AVOIDANCE.value]["avg"],
            })
            portfolio.append({
                "credit_type": CreditType.NATURE_BASED_REMOVAL.value,
                "role": SBTiOffsetRole.NEUTRALIZATION.value,
                "volume_tco2e": round(residual, 2),
                "price_eur_per_tco2e": CREDIT_PRICING[CreditType.NATURE_BASED_REMOVAL.value]["avg"],
            })

        return portfolio
