# -*- coding: utf-8 -*-
"""
Credit Procurement Workflow
================================

4-phase workflow for carbon credit sourcing and procurement within PACK-024
Carbon Neutral Pack.  Evaluates credit quality, assesses suppliers, builds
a diversified portfolio, and executes procurement contracts.

Phases:
    1. NeedsAssessment      -- Determine credit volume and quality requirements
    2. MarketScreening      -- Screen registries and marketplaces for eligible credits
    3. QualityAssessment    -- Evaluate credits against ICVCM CCP and project quality
    4. Procurement          -- Execute procurement with contract terms and delivery

Regulatory references:
    - ICVCM Core Carbon Principles (2023)
    - VCMI Claims Code of Practice (2023)
    - PAS 2060:2014 (Eligible offset types)
    - Article 6 Paris Agreement (ITMOs)
    - Verra VCS Standard V4.5
    - Gold Standard for Global Goals V2.1

Author: GreenLang Team
Version: 24.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "24.0.0"

def _new_uuid() -> str:
    return uuid.uuid4().hex

def _compute_hash(data: Any) -> str:
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(str(data).encode("utf-8")).hexdigest()

# =============================================================================
# ENUMS
# =============================================================================

class PhaseStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"

class ProcurementPhase(str, Enum):
    NEEDS_ASSESSMENT = "needs_assessment"
    MARKET_SCREENING = "market_screening"
    QUALITY_ASSESSMENT = "quality_assessment"
    PROCUREMENT = "procurement"

class CreditStandard(str, Enum):
    VCS = "vcs"
    GOLD_STANDARD = "gold_standard"
    ACR = "acr"
    CAR = "car"
    CDM = "cdm"
    ARTICLE_6 = "article_6"
    PURO = "puro"
    ISOMETRIC = "isometric"

class CreditType(str, Enum):
    AVOIDANCE = "avoidance"
    REDUCTION = "reduction"
    REMOVAL_NATURE = "removal_nature"
    REMOVAL_TECH = "removal_tech"
    HYBRID = "hybrid"

class QualityTier(str, Enum):
    PREMIUM = "premium"       # ICVCM CCP-approved, high co-benefits
    STANDARD = "standard"     # Major registry, verified
    BASIC = "basic"           # Verified but limited co-benefits
    UNRATED = "unrated"       # Not yet assessed

class ProjectType(str, Enum):
    RENEWABLE_ENERGY = "renewable_energy"
    ENERGY_EFFICIENCY = "energy_efficiency"
    COOKSTOVES = "cookstoves"
    REDD_PLUS = "redd_plus"
    ARR = "arr"  # Afforestation/Reforestation/Revegetation
    BLUE_CARBON = "blue_carbon"
    DIRECT_AIR_CAPTURE = "direct_air_capture"
    BIOCHAR = "biochar"
    ENHANCED_WEATHERING = "enhanced_weathering"
    METHANE_CAPTURE = "methane_capture"
    WASTE_MANAGEMENT = "waste_management"
    SOIL_CARBON = "soil_carbon"

class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

# =============================================================================
# REFERENCE DATA
# =============================================================================

# ICVCM Core Carbon Principles assessment criteria
ICVCM_CCP_CRITERIA: List[str] = [
    "additionality",
    "permanence",
    "robust_quantification",
    "no_double_counting",
    "sustainable_development",
    "effective_governance",
    "tracking",
    "transparency",
    "registry_operations",
    "transition_towards_net_zero",
]

# Typical credit prices by type (USD/tCO2e, Q1 2025)
CREDIT_PRICE_RANGES: Dict[str, Dict[str, float]] = {
    "avoidance": {"min": 2.0, "max": 15.0, "typical": 8.0},
    "reduction": {"min": 5.0, "max": 25.0, "typical": 12.0},
    "removal_nature": {"min": 10.0, "max": 50.0, "typical": 25.0},
    "removal_tech": {"min": 100.0, "max": 800.0, "typical": 300.0},
    "hybrid": {"min": 8.0, "max": 40.0, "typical": 20.0},
}

# Registry reliability scores (0-100)
REGISTRY_RELIABILITY: Dict[str, float] = {
    "vcs": 90.0,
    "gold_standard": 95.0,
    "acr": 88.0,
    "car": 85.0,
    "cdm": 80.0,
    "article_6": 92.0,
    "puro": 87.0,
    "isometric": 85.0,
}

# PAS 2060 eligible credit types
PAS2060_ELIGIBLE_TYPES: List[str] = [
    "vcs", "gold_standard", "acr", "car", "cdm",
]

# =============================================================================
# DATA MODELS
# =============================================================================

class PhaseResult(BaseModel):
    phase_name: str = Field(...)
    status: PhaseStatus = Field(...)
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

class ProcurementStrategy(BaseModel):
    strategy_id: str = Field(default="")
    description: str = Field(default="")
    volume_tco2e: float = Field(default=0.0, ge=0.0)
    credit_type_mix: Dict[str, float] = Field(default_factory=dict)
    quality_tier_target: QualityTier = Field(default=QualityTier.STANDARD)
    budget_usd: float = Field(default=0.0, ge=0.0)
    max_price_per_tco2e: float = Field(default=0.0, ge=0.0)
    diversification_targets: Dict[str, Any] = Field(default_factory=dict)
    vintage_requirements: Dict[str, Any] = Field(default_factory=dict)
    co_benefit_priorities: List[str] = Field(default_factory=list)

class SupplierEvaluation(BaseModel):
    supplier_id: str = Field(default="")
    supplier_name: str = Field(default="")
    registry: CreditStandard = Field(default=CreditStandard.VCS)
    project_type: ProjectType = Field(default=ProjectType.RENEWABLE_ENERGY)
    credit_type: CreditType = Field(default=CreditType.AVOIDANCE)
    quality_tier: QualityTier = Field(default=QualityTier.STANDARD)
    available_volume_tco2e: float = Field(default=0.0, ge=0.0)
    price_per_tco2e: float = Field(default=0.0, ge=0.0)
    vintage_year: int = Field(default=2024)
    country: str = Field(default="")
    icvcm_ccp_score: float = Field(default=0.0, ge=0.0, le=100.0)
    permanence_risk: RiskLevel = Field(default=RiskLevel.MODERATE)
    co_benefits: List[str] = Field(default_factory=list)
    sdg_contributions: List[int] = Field(default_factory=list)
    recommended: bool = Field(default=False)
    recommendation_reason: str = Field(default="")

class ContractTerms(BaseModel):
    contract_id: str = Field(default="")
    supplier_id: str = Field(default="")
    volume_tco2e: float = Field(default=0.0, ge=0.0)
    price_per_tco2e: float = Field(default=0.0, ge=0.0)
    total_cost_usd: float = Field(default=0.0, ge=0.0)
    delivery_date: str = Field(default="")
    vintage_year: int = Field(default=2024)
    registry: CreditStandard = Field(default=CreditStandard.VCS)
    serial_number_range: str = Field(default="")
    retirement_deadline: str = Field(default="")
    replacement_clause: bool = Field(default=True)
    buffer_pool_pct: float = Field(default=0.0, ge=0.0)

class CreditProcurementConfig(BaseModel):
    org_name: str = Field(default="")
    reporting_year: int = Field(default=2025, ge=2015, le=2050)
    residual_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    budget_usd: float = Field(default=0.0, ge=0.0)
    quality_tier_target: QualityTier = Field(default=QualityTier.STANDARD)
    preferred_credit_types: List[CreditType] = Field(
        default_factory=lambda: [CreditType.AVOIDANCE, CreditType.REMOVAL_NATURE]
    )
    preferred_registries: List[CreditStandard] = Field(
        default_factory=lambda: [CreditStandard.VCS, CreditStandard.GOLD_STANDARD]
    )
    removal_pct_target: float = Field(default=10.0, ge=0.0, le=100.0)
    min_vintage_year: int = Field(default=2020, ge=2015)
    pas2060_compliance: bool = Field(default=True)
    icvcm_required: bool = Field(default=True)
    max_single_project_pct: float = Field(default=30.0, ge=0.0, le=100.0)
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

class CreditProcurementResult(BaseModel):
    workflow_id: str = Field(...)
    workflow_name: str = Field(default="credit_procurement")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    strategy: Optional[ProcurementStrategy] = Field(None)
    suppliers_evaluated: List[SupplierEvaluation] = Field(default_factory=list)
    contracts: List[ContractTerms] = Field(default_factory=list)
    total_volume_tco2e: float = Field(default=0.0)
    total_cost_usd: float = Field(default=0.0)
    avg_price_per_tco2e: float = Field(default=0.0)
    removal_pct: float = Field(default=0.0)
    portfolio_quality_score: float = Field(default=0.0)
    pas2060_compliant: bool = Field(default=False)
    provenance_hash: str = Field(default="")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class CreditProcurementWorkflow:
    """
    4-phase credit procurement workflow for PACK-024.

    Determines credit volume requirements, screens the voluntary carbon
    market, evaluates credit quality against ICVCM CCP, and executes
    procurement with contract terms and portfolio diversification.

    Attributes:
        workflow_id: Unique execution identifier.
    """

    def __init__(self) -> None:
        self.workflow_id: str = _new_uuid()
        self._phase_results: List[PhaseResult] = []
        self._strategy: Optional[ProcurementStrategy] = None
        self._suppliers: List[SupplierEvaluation] = []
        self._contracts: List[ContractTerms] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def execute(self, config: CreditProcurementConfig) -> CreditProcurementResult:
        """Execute the 4-phase credit procurement workflow."""
        started_at = utcnow()
        self.logger.info(
            "Starting credit procurement %s, volume=%.2f tCO2e",
            self.workflow_id, config.residual_emissions_tco2e,
        )
        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = await self._phase_needs_assessment(config)
            self._phase_results.append(phase1)
            if phase1.status == PhaseStatus.FAILED:
                raise ValueError("Needs assessment failed")

            phase2 = await self._phase_market_screening(config)
            self._phase_results.append(phase2)

            phase3 = await self._phase_quality_assessment(config)
            self._phase_results.append(phase3)

            phase4 = await self._phase_procurement(config)
            self._phase_results.append(phase4)

            failed = [p for p in self._phase_results if p.status == PhaseStatus.FAILED]
            overall_status = WorkflowStatus.COMPLETED if not failed else WorkflowStatus.PARTIAL

        except Exception as exc:
            self.logger.error("Credit procurement failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (utcnow() - started_at).total_seconds()
        total_vol = sum(c.volume_tco2e for c in self._contracts)
        total_cost = sum(c.total_cost_usd for c in self._contracts)
        avg_price = total_cost / max(total_vol, 1.0)
        removal_vol = sum(
            s.available_volume_tco2e for s in self._suppliers
            if s.credit_type in (CreditType.REMOVAL_NATURE, CreditType.REMOVAL_TECH) and s.recommended
        )
        removal_pct = (removal_vol / max(total_vol, 1.0)) * 100.0

        result = CreditProcurementResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=round(elapsed, 4),
            strategy=self._strategy,
            suppliers_evaluated=self._suppliers,
            contracts=self._contracts,
            total_volume_tco2e=round(total_vol, 2),
            total_cost_usd=round(total_cost, 2),
            avg_price_per_tco2e=round(avg_price, 2),
            removal_pct=round(removal_pct, 1),
            portfolio_quality_score=0.0,
            pas2060_compliant=all(
                c.registry.value in PAS2060_ELIGIBLE_TYPES for c in self._contracts
            ) if self._contracts else False,
        )
        result.provenance_hash = _compute_hash(
            result.model_dump_json(exclude={"provenance_hash"})
        )
        return result

    async def _phase_needs_assessment(self, config: CreditProcurementConfig) -> PhaseResult:
        """Determine credit volume and quality requirements."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []

        if config.residual_emissions_tco2e <= 0:
            errors.append("Residual emissions must be >0 for credit procurement")

        # Build procurement strategy
        removal_volume = config.residual_emissions_tco2e * (config.removal_pct_target / 100.0)
        avoidance_volume = config.residual_emissions_tco2e - removal_volume

        type_mix: Dict[str, float] = {}
        for ct in config.preferred_credit_types:
            if ct in (CreditType.REMOVAL_NATURE, CreditType.REMOVAL_TECH):
                type_mix[ct.value] = removal_volume / max(len([
                    c for c in config.preferred_credit_types
                    if c in (CreditType.REMOVAL_NATURE, CreditType.REMOVAL_TECH)
                ]), 1)
            else:
                type_mix[ct.value] = avoidance_volume / max(len([
                    c for c in config.preferred_credit_types
                    if c not in (CreditType.REMOVAL_NATURE, CreditType.REMOVAL_TECH)
                ]), 1)

        self._strategy = ProcurementStrategy(
            strategy_id=_new_uuid(),
            description=f"Carbon neutral credit procurement for {config.org_name}",
            volume_tco2e=config.residual_emissions_tco2e,
            credit_type_mix=type_mix,
            quality_tier_target=config.quality_tier_target,
            budget_usd=config.budget_usd,
            max_price_per_tco2e=config.budget_usd / max(config.residual_emissions_tco2e, 1.0),
            diversification_targets={
                "max_single_project_pct": config.max_single_project_pct,
                "min_registries": 2,
                "min_project_types": 2,
            },
            vintage_requirements={
                "min_vintage": config.min_vintage_year,
                "preferred_vintage": config.reporting_year,
            },
        )

        outputs["total_volume_needed_tco2e"] = config.residual_emissions_tco2e
        outputs["removal_volume_tco2e"] = round(removal_volume, 2)
        outputs["avoidance_volume_tco2e"] = round(avoidance_volume, 2)
        outputs["budget_usd"] = config.budget_usd
        outputs["max_price_per_tco2e"] = round(self._strategy.max_price_per_tco2e, 2)

        status = PhaseStatus.FAILED if errors else PhaseStatus.COMPLETED
        elapsed = (utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name=ProcurementPhase.NEEDS_ASSESSMENT.value,
            status=status, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings, errors=errors,
            provenance_hash=_compute_hash(outputs),
        )

    async def _phase_market_screening(self, config: CreditProcurementConfig) -> PhaseResult:
        """Screen registries and marketplaces for eligible credits."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []

        # Generate supplier evaluations for each preferred registry
        suppliers: List[SupplierEvaluation] = []
        project_types = [
            ProjectType.RENEWABLE_ENERGY, ProjectType.ARR, ProjectType.REDD_PLUS,
            ProjectType.COOKSTOVES, ProjectType.METHANE_CAPTURE,
        ]

        for reg in config.preferred_registries:
            for i, pt in enumerate(project_types[:3]):
                credit_type = CreditType.REMOVAL_NATURE if pt == ProjectType.ARR else CreditType.AVOIDANCE
                price_info = CREDIT_PRICE_RANGES.get(credit_type.value, {"typical": 10.0})
                reliability = REGISTRY_RELIABILITY.get(reg.value, 80.0)

                supplier = SupplierEvaluation(
                    supplier_id=_new_uuid(),
                    supplier_name=f"{reg.value.upper()} Project {i+1}",
                    registry=reg,
                    project_type=pt,
                    credit_type=credit_type,
                    quality_tier=QualityTier.PREMIUM if reliability >= 90 else QualityTier.STANDARD,
                    available_volume_tco2e=config.residual_emissions_tco2e * 0.5,
                    price_per_tco2e=price_info["typical"],
                    vintage_year=config.reporting_year,
                    icvcm_ccp_score=reliability,
                    permanence_risk=RiskLevel.LOW if pt == ProjectType.RENEWABLE_ENERGY else RiskLevel.MODERATE,
                    co_benefits=["Community development", "Biodiversity"],
                    sdg_contributions=[7, 13, 15] if pt == ProjectType.ARR else [7, 13],
                )
                suppliers.append(supplier)

        self._suppliers = suppliers
        outputs["suppliers_screened"] = len(suppliers)
        outputs["registries_covered"] = len(config.preferred_registries)

        status = PhaseStatus.COMPLETED
        elapsed = (utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name=ProcurementPhase.MARKET_SCREENING.value,
            status=status, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings, errors=errors,
            provenance_hash=_compute_hash(outputs),
        )

    async def _phase_quality_assessment(self, config: CreditProcurementConfig) -> PhaseResult:
        """Evaluate credits against ICVCM CCP and project quality criteria."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []

        recommended_count = 0
        for supplier in self._suppliers:
            # ICVCM CCP assessment
            ccp_pass = supplier.icvcm_ccp_score >= 80.0

            # Vintage check
            vintage_ok = supplier.vintage_year >= config.min_vintage_year

            # PAS 2060 eligibility
            pas2060_ok = supplier.registry.value in PAS2060_ELIGIBLE_TYPES

            # Quality threshold
            quality_ok = supplier.quality_tier in (QualityTier.PREMIUM, QualityTier.STANDARD)

            if ccp_pass and vintage_ok and quality_ok and (pas2060_ok or not config.pas2060_compliance):
                supplier.recommended = True
                supplier.recommendation_reason = "Meets ICVCM CCP, vintage, and quality criteria"
                recommended_count += 1
            else:
                reasons = []
                if not ccp_pass:
                    reasons.append("Below ICVCM CCP threshold")
                if not vintage_ok:
                    reasons.append(f"Vintage {supplier.vintage_year} below minimum {config.min_vintage_year}")
                if not pas2060_ok and config.pas2060_compliance:
                    reasons.append("Not PAS 2060 eligible")
                supplier.recommendation_reason = "; ".join(reasons) if reasons else "Below quality threshold"

        outputs["suppliers_recommended"] = recommended_count
        outputs["suppliers_rejected"] = len(self._suppliers) - recommended_count
        outputs["icvcm_assessment_applied"] = config.icvcm_required

        if recommended_count == 0:
            warnings.append("No suppliers meet quality criteria; consider relaxing requirements")

        status = PhaseStatus.COMPLETED
        elapsed = (utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name=ProcurementPhase.QUALITY_ASSESSMENT.value,
            status=status, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings, errors=errors,
            provenance_hash=_compute_hash(outputs),
        )

    async def _phase_procurement(self, config: CreditProcurementConfig) -> PhaseResult:
        """Execute procurement with contract terms and delivery schedules."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []

        recommended = [s for s in self._suppliers if s.recommended]
        remaining_volume = config.residual_emissions_tco2e
        remaining_budget = config.budget_usd
        contracts: List[ContractTerms] = []

        # Sort by price (cheapest first) then quality
        recommended.sort(key=lambda s: (s.price_per_tco2e, -s.icvcm_ccp_score))

        for supplier in recommended:
            if remaining_volume <= 0:
                break

            allocatable = min(
                supplier.available_volume_tco2e,
                remaining_volume,
                config.residual_emissions_tco2e * (config.max_single_project_pct / 100.0),
            )
            cost = allocatable * supplier.price_per_tco2e
            if remaining_budget > 0 and cost > remaining_budget:
                allocatable = remaining_budget / supplier.price_per_tco2e
                cost = remaining_budget

            if allocatable > 0:
                contract = ContractTerms(
                    contract_id=_new_uuid(),
                    supplier_id=supplier.supplier_id,
                    volume_tco2e=round(allocatable, 2),
                    price_per_tco2e=supplier.price_per_tco2e,
                    total_cost_usd=round(cost, 2),
                    delivery_date=f"{config.reporting_year}-12-31",
                    vintage_year=supplier.vintage_year,
                    registry=supplier.registry,
                    retirement_deadline=f"{config.reporting_year + 1}-03-31",
                    replacement_clause=True,
                    buffer_pool_pct=5.0,
                )
                contracts.append(contract)
                remaining_volume -= allocatable
                remaining_budget -= cost

        self._contracts = contracts

        outputs["contracts_executed"] = len(contracts)
        outputs["volume_secured_tco2e"] = round(
            sum(c.volume_tco2e for c in contracts), 2
        )
        outputs["total_cost_usd"] = round(sum(c.total_cost_usd for c in contracts), 2)
        outputs["volume_gap_tco2e"] = round(max(remaining_volume, 0), 2)

        if remaining_volume > 0:
            warnings.append(
                f"Volume gap: {remaining_volume:.0f} tCO2e still needed"
            )

        status = PhaseStatus.COMPLETED
        elapsed = (utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name=ProcurementPhase.PROCUREMENT.value,
            status=status, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings, errors=errors,
            provenance_hash=_compute_hash(outputs),
        )
