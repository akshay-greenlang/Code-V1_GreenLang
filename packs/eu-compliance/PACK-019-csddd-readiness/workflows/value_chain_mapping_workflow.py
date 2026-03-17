# -*- coding: utf-8 -*-
"""
CSDDD Value Chain Mapping Workflow
===============================================

4-phase workflow for mapping value chains under the EU Corporate Sustainability
Due Diligence Directive (CSDDD / CS3D). Maps suppliers across tiers, classifies
activities, and overlays country/sector risk data for due diligence scoping.

Phases:
    1. SupplierMapping         -- Catalogue and validate supplier relationships
    2. TierIdentification      -- Classify suppliers into direct/indirect tiers
    3. ActivityClassification  -- Classify activities by CSDDD-relevant categories
    4. RiskOverlay             -- Overlay geographic and sector risk indicators

Regulatory References:
    - Directive (EU) 2024/1760 (CSDDD / CS3D)
    - Art. 1(1)(a): Chain of activities (upstream/downstream)
    - Art. 3(b): Business relationships
    - Art. 3(g): Value chain definition
    - Art. 6: Identifying and assessing adverse impacts
    - Art. 8(3)(b-c): Contractual assurances from partners
    - Annex Part I & II: Adverse impacts on human rights and environment

Author: GreenLang Team
Version: 19.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"


# =============================================================================
# HELPERS
# =============================================================================


def _utcnow() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    """Generate a new UUID4 hex string."""
    return uuid.uuid4().hex


def _compute_hash(data: str) -> str:
    """Compute SHA-256 hex digest of *data*."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# =============================================================================
# ENUMS
# =============================================================================


class WorkflowPhase(str, Enum):
    """Phases of the value chain mapping workflow."""
    SUPPLIER_MAPPING = "supplier_mapping"
    TIER_IDENTIFICATION = "tier_identification"
    ACTIVITY_CLASSIFICATION = "activity_classification"
    RISK_OVERLAY = "risk_overlay"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PhaseStatus(str, Enum):
    """Status of a single phase."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class SupplierTier(str, Enum):
    """Supplier relationship tier classification."""
    TIER_1 = "tier_1"       # Direct suppliers / business partners
    TIER_2 = "tier_2"       # Indirect (suppliers of suppliers)
    TIER_3 = "tier_3"       # Further upstream
    DOWNSTREAM = "downstream"
    UNKNOWN = "unknown"


class ActivityCategory(str, Enum):
    """CSDDD activity classification categories."""
    PRODUCTION = "production"
    PROCESSING = "processing"
    MANUFACTURING = "manufacturing"
    EXTRACTION = "extraction"
    TRANSPORT = "transport"
    STORAGE = "storage"
    DISTRIBUTION = "distribution"
    RECYCLING = "recycling"
    DISPOSAL = "disposal"
    SERVICE = "service"
    OTHER = "other"


class RiskLevel(str, Enum):
    """Risk level classification for geographic/sector risk overlay."""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class SupplierRecord(BaseModel):
    """Individual supplier record for value chain mapping."""
    supplier_id: str = Field(default_factory=lambda: f"sup-{_new_uuid()[:8]}")
    supplier_name: str = Field(default="", description="Supplier legal name")
    country_code: str = Field(default="", description="ISO 3166-1 alpha-2 country code")
    sector: str = Field(default="", description="Sector of activity")
    nace_code: str = Field(default="", description="NACE Rev. 2 sector code")
    tier: SupplierTier = Field(default=SupplierTier.UNKNOWN, description="Tier classification")
    annual_spend_eur: float = Field(default=0.0, ge=0.0, description="Annual procurement spend")
    employee_count: int = Field(default=0, ge=0, description="Supplier employee count")
    is_sme: bool = Field(default=False, description="Whether supplier is an SME")
    activities: List[str] = Field(default_factory=list, description="Activities performed")
    parent_supplier_id: str = Field(default="", description="ID of tier-1 parent if applicable")
    has_code_of_conduct: bool = Field(default=False)
    contractual_assurances: bool = Field(default=False)


class CountryRiskData(BaseModel):
    """Country-level risk data for risk overlay."""
    country_code: str = Field(..., description="ISO 3166-1 alpha-2")
    country_name: str = Field(default="")
    human_rights_risk: RiskLevel = Field(default=RiskLevel.MEDIUM)
    environmental_risk: RiskLevel = Field(default=RiskLevel.MEDIUM)
    governance_risk: RiskLevel = Field(default=RiskLevel.MEDIUM)
    conflict_affected: bool = Field(default=False)
    rule_of_law_score: float = Field(default=50.0, ge=0.0, le=100.0)


class ValueChainMappingInput(BaseModel):
    """Input data model for ValueChainMappingWorkflow."""
    entity_id: str = Field(default="", description="Reporting entity ID")
    entity_name: str = Field(default="", description="Reporting entity name")
    reporting_year: int = Field(default=2026, ge=2024, le=2050)
    suppliers: List[SupplierRecord] = Field(
        default_factory=list, description="List of supplier records"
    )
    country_risk_data: List[CountryRiskData] = Field(
        default_factory=list, description="Country-level risk datasets"
    )
    high_risk_sectors: List[str] = Field(
        default_factory=lambda: [
            "mining", "agriculture", "textiles", "construction",
            "electronics", "food_processing", "forestry", "chemicals",
        ],
        description="Sectors deemed high-risk under CSDDD",
    )
    config: Dict[str, Any] = Field(default_factory=dict)


class SupplierRiskProfile(BaseModel):
    """Computed risk profile for a single supplier."""
    supplier_id: str = Field(...)
    supplier_name: str = Field(default="")
    tier: str = Field(default="unknown")
    country_code: str = Field(default="")
    sector: str = Field(default="")
    composite_risk_score: float = Field(default=0.0, ge=0.0, le=100.0)
    risk_level: RiskLevel = Field(default=RiskLevel.MEDIUM)
    risk_factors: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)


class ValueChainMappingResult(BaseModel):
    """Complete result from value chain mapping workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="value_chain_mapping")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    phases_completed: int = Field(default=0, ge=0)
    duration_ms: float = Field(default=0.0)
    total_duration_seconds: float = Field(default=0.0)
    # Supplier metrics
    total_suppliers: int = Field(default=0, ge=0)
    tier_1_count: int = Field(default=0, ge=0)
    tier_2_count: int = Field(default=0, ge=0)
    tier_3_count: int = Field(default=0, ge=0)
    downstream_count: int = Field(default=0, ge=0)
    unknown_tier_count: int = Field(default=0, ge=0)
    # Geographic
    countries_covered: int = Field(default=0, ge=0)
    high_risk_countries: int = Field(default=0, ge=0)
    # Risk overlay
    supplier_risk_profiles: List[SupplierRiskProfile] = Field(default_factory=list)
    very_high_risk_suppliers: int = Field(default=0, ge=0)
    high_risk_suppliers: int = Field(default=0, ge=0)
    # Activity
    activity_categories: Dict[str, int] = Field(default_factory=dict)
    total_annual_spend_eur: float = Field(default=0.0, ge=0.0)
    reporting_year: int = Field(default=2026)
    executed_at: str = Field(default="")
    provenance_hash: str = Field(default="")


# =============================================================================
# DEFAULT COUNTRY RISK SCORES
# =============================================================================


DEFAULT_COUNTRY_RISK: Dict[str, Dict[str, Any]] = {
    "DE": {"hr": "low", "env": "low", "gov": "low"},
    "FR": {"hr": "low", "env": "low", "gov": "low"},
    "NL": {"hr": "very_low", "env": "low", "gov": "very_low"},
    "SE": {"hr": "very_low", "env": "very_low", "gov": "very_low"},
    "US": {"hr": "low", "env": "medium", "gov": "low"},
    "CN": {"hr": "high", "env": "high", "gov": "high"},
    "IN": {"hr": "high", "env": "high", "gov": "medium"},
    "BD": {"hr": "very_high", "env": "high", "gov": "high"},
    "MM": {"hr": "very_high", "env": "very_high", "gov": "very_high"},
    "CD": {"hr": "very_high", "env": "very_high", "gov": "very_high"},
    "BR": {"hr": "medium", "env": "high", "gov": "medium"},
    "VN": {"hr": "high", "env": "medium", "gov": "high"},
    "TR": {"hr": "medium", "env": "medium", "gov": "medium"},
    "PL": {"hr": "low", "env": "low", "gov": "low"},
    "RO": {"hr": "low", "env": "medium", "gov": "medium"},
}


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class ValueChainMappingWorkflow:
    """
    4-phase CSDDD value chain mapping workflow.

    Maps supplier relationships, classifies tiers and activities, and overlays
    country/sector risk for CSDDD due diligence scoping under Art. 6.

    Zero-hallucination: all risk scoring uses deterministic formulas.
    No LLM in numeric calculation paths.

    Example:
        >>> wf = ValueChainMappingWorkflow()
        >>> inp = ValueChainMappingInput(suppliers=[...])
        >>> result = await wf.execute(inp)
        >>> assert result.total_suppliers > 0
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize ValueChainMappingWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._risk_profiles: List[SupplierRiskProfile] = []
        self._activity_counts: Dict[str, int] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_phases(self) -> List[Dict[str, str]]:
        """Return phase definitions for this workflow."""
        return [
            {"name": WorkflowPhase.SUPPLIER_MAPPING.value, "description": "Catalogue supplier relationships"},
            {"name": WorkflowPhase.TIER_IDENTIFICATION.value, "description": "Classify suppliers into tiers"},
            {"name": WorkflowPhase.ACTIVITY_CLASSIFICATION.value, "description": "Classify by activity category"},
            {"name": WorkflowPhase.RISK_OVERLAY.value, "description": "Overlay country/sector risk data"},
        ]

    def validate_inputs(self, input_data: ValueChainMappingInput) -> List[str]:
        """Validate workflow inputs and return list of issues."""
        issues: List[str] = []
        if not input_data.suppliers:
            issues.append("No suppliers provided")
        for sup in input_data.suppliers:
            if not sup.supplier_name:
                issues.append(f"Supplier {sup.supplier_id} missing name")
            if not sup.country_code:
                issues.append(f"Supplier {sup.supplier_id} missing country code")
        return issues

    async def execute(
        self,
        input_data: Optional[ValueChainMappingInput] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> ValueChainMappingResult:
        """
        Execute the 4-phase value chain mapping workflow.

        Args:
            input_data: Full input model.
            config: Configuration overrides.

        Returns:
            ValueChainMappingResult with mapped chain and risk overlay.
        """
        if input_data is None:
            input_data = ValueChainMappingInput(config=config or {})

        started_at = _utcnow()
        self.logger.info("Starting value chain mapping workflow %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.IN_PROGRESS

        try:
            phase_results.append(await self._phase_supplier_mapping(input_data))
            phase_results.append(await self._phase_tier_identification(input_data))
            phase_results.append(await self._phase_activity_classification(input_data))
            phase_results.append(await self._phase_risk_overlay(input_data))
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Value chain mapping failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (_utcnow() - started_at).total_seconds()
        completed_count = sum(1 for p in phase_results if p.status == PhaseStatus.COMPLETED)

        # Tier counts
        suppliers = input_data.suppliers
        tier_1 = sum(1 for s in suppliers if s.tier == SupplierTier.TIER_1)
        tier_2 = sum(1 for s in suppliers if s.tier == SupplierTier.TIER_2)
        tier_3 = sum(1 for s in suppliers if s.tier == SupplierTier.TIER_3)
        downstream = sum(1 for s in suppliers if s.tier == SupplierTier.DOWNSTREAM)
        unknown = sum(1 for s in suppliers if s.tier == SupplierTier.UNKNOWN)

        countries = set(s.country_code for s in suppliers if s.country_code)
        hr_countries = sum(
            1 for c in countries
            if DEFAULT_COUNTRY_RISK.get(c, {}).get("hr", "medium") in ("high", "very_high")
        )

        very_high_risk = sum(1 for rp in self._risk_profiles if rp.risk_level == RiskLevel.VERY_HIGH)
        high_risk = sum(1 for rp in self._risk_profiles if rp.risk_level == RiskLevel.HIGH)

        result = ValueChainMappingResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            phases_completed=completed_count,
            duration_ms=round(elapsed * 1000, 2),
            total_duration_seconds=elapsed,
            total_suppliers=len(suppliers),
            tier_1_count=tier_1,
            tier_2_count=tier_2,
            tier_3_count=tier_3,
            downstream_count=downstream,
            unknown_tier_count=unknown,
            countries_covered=len(countries),
            high_risk_countries=hr_countries,
            supplier_risk_profiles=self._risk_profiles,
            very_high_risk_suppliers=very_high_risk,
            high_risk_suppliers=high_risk,
            activity_categories=self._activity_counts,
            total_annual_spend_eur=round(sum(s.annual_spend_eur for s in suppliers), 2),
            reporting_year=input_data.reporting_year,
            executed_at=_utcnow().isoformat(),
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "Value chain mapping %s completed in %.2fs: %d suppliers, %d countries",
            self.workflow_id, elapsed, len(suppliers), len(countries),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Supplier Mapping
    # -------------------------------------------------------------------------

    async def _phase_supplier_mapping(
        self, input_data: ValueChainMappingInput,
    ) -> PhaseResult:
        """Catalogue and validate supplier relationships."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        suppliers = input_data.suppliers

        # Basic stats
        countries = set(s.country_code for s in suppliers if s.country_code)
        sectors = set(s.sector for s in suppliers if s.sector)
        total_spend = sum(s.annual_spend_eur for s in suppliers)
        sme_count = sum(1 for s in suppliers if s.is_sme)

        outputs["total_suppliers"] = len(suppliers)
        outputs["unique_countries"] = len(countries)
        outputs["unique_sectors"] = len(sectors)
        outputs["total_annual_spend_eur"] = round(total_spend, 2)
        outputs["sme_count"] = sme_count
        outputs["sme_pct"] = round((sme_count / len(suppliers)) * 100, 1) if suppliers else 0.0
        outputs["with_code_of_conduct"] = sum(1 for s in suppliers if s.has_code_of_conduct)
        outputs["with_contractual_assurances"] = sum(1 for s in suppliers if s.contractual_assurances)

        # Data quality checks
        missing_country = sum(1 for s in suppliers if not s.country_code)
        missing_sector = sum(1 for s in suppliers if not s.sector)

        if missing_country > 0:
            warnings.append(f"{missing_country} suppliers missing country code")
        if missing_sector > 0:
            warnings.append(f"{missing_sector} suppliers missing sector classification")
        if not suppliers:
            warnings.append("No supplier records provided")

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 SupplierMapping: %d suppliers across %d countries",
            len(suppliers), len(countries),
        )
        return PhaseResult(
            phase_name=WorkflowPhase.SUPPLIER_MAPPING.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Tier Identification
    # -------------------------------------------------------------------------

    async def _phase_tier_identification(
        self, input_data: ValueChainMappingInput,
    ) -> PhaseResult:
        """Classify suppliers into tier levels based on relationship data."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        suppliers = input_data.suppliers

        # Auto-classify unknown tiers using heuristics
        for sup in suppliers:
            if sup.tier == SupplierTier.UNKNOWN:
                if not sup.parent_supplier_id:
                    sup.tier = SupplierTier.TIER_1
                elif any(
                    s.supplier_id == sup.parent_supplier_id and s.tier == SupplierTier.TIER_1
                    for s in suppliers
                ):
                    sup.tier = SupplierTier.TIER_2
                elif any(
                    s.supplier_id == sup.parent_supplier_id and s.tier == SupplierTier.TIER_2
                    for s in suppliers
                ):
                    sup.tier = SupplierTier.TIER_3

        tier_counts: Dict[str, int] = {}
        for sup in suppliers:
            tier_counts[sup.tier.value] = tier_counts.get(sup.tier.value, 0) + 1

        # Spend by tier
        tier_spend: Dict[str, float] = {}
        for sup in suppliers:
            tier_spend[sup.tier.value] = tier_spend.get(sup.tier.value, 0.0) + sup.annual_spend_eur

        outputs["tier_distribution"] = tier_counts
        outputs["tier_spend_eur"] = {k: round(v, 2) for k, v in tier_spend.items()}
        outputs["auto_classified_count"] = sum(
            1 for s in suppliers
            if s.tier != SupplierTier.UNKNOWN
        )
        outputs["still_unknown"] = tier_counts.get("unknown", 0)

        if tier_counts.get("unknown", 0) > 0:
            warnings.append(
                f"{tier_counts['unknown']} suppliers still unclassified after auto-tier"
            )

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 TierIdentification: %s",
            ", ".join(f"{k}={v}" for k, v in tier_counts.items()),
        )
        return PhaseResult(
            phase_name=WorkflowPhase.TIER_IDENTIFICATION.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Activity Classification
    # -------------------------------------------------------------------------

    async def _phase_activity_classification(
        self, input_data: ValueChainMappingInput,
    ) -> PhaseResult:
        """Classify supplier activities by CSDDD-relevant categories."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        suppliers = input_data.suppliers

        # Map activities to categories
        activity_keyword_map: Dict[str, ActivityCategory] = {
            "mining": ActivityCategory.EXTRACTION,
            "extraction": ActivityCategory.EXTRACTION,
            "drilling": ActivityCategory.EXTRACTION,
            "manufacturing": ActivityCategory.MANUFACTURING,
            "assembly": ActivityCategory.MANUFACTURING,
            "processing": ActivityCategory.PROCESSING,
            "refining": ActivityCategory.PROCESSING,
            "transport": ActivityCategory.TRANSPORT,
            "logistics": ActivityCategory.TRANSPORT,
            "shipping": ActivityCategory.TRANSPORT,
            "storage": ActivityCategory.STORAGE,
            "warehousing": ActivityCategory.STORAGE,
            "distribution": ActivityCategory.DISTRIBUTION,
            "retail": ActivityCategory.DISTRIBUTION,
            "recycling": ActivityCategory.RECYCLING,
            "waste": ActivityCategory.DISPOSAL,
            "disposal": ActivityCategory.DISPOSAL,
            "consulting": ActivityCategory.SERVICE,
            "service": ActivityCategory.SERVICE,
            "farming": ActivityCategory.PRODUCTION,
            "agriculture": ActivityCategory.PRODUCTION,
            "production": ActivityCategory.PRODUCTION,
        }

        self._activity_counts = {}
        classified = 0

        for sup in suppliers:
            if sup.activities:
                for act in sup.activities:
                    act_lower = act.lower()
                    matched = False
                    for keyword, category in activity_keyword_map.items():
                        if keyword in act_lower:
                            cat_val = category.value
                            self._activity_counts[cat_val] = self._activity_counts.get(cat_val, 0) + 1
                            matched = True
                            classified += 1
                            break
                    if not matched:
                        self._activity_counts["other"] = self._activity_counts.get("other", 0) + 1
                        classified += 1
            else:
                # Classify by sector if no activities listed
                sector_lower = sup.sector.lower() if sup.sector else ""
                matched_cat = None
                for keyword, category in activity_keyword_map.items():
                    if keyword in sector_lower:
                        matched_cat = category.value
                        break
                if matched_cat:
                    self._activity_counts[matched_cat] = self._activity_counts.get(matched_cat, 0) + 1
                    classified += 1

        outputs["activity_categories"] = self._activity_counts
        outputs["classified_count"] = classified
        outputs["suppliers_with_activities"] = sum(1 for s in suppliers if s.activities)
        outputs["suppliers_without_activities"] = sum(1 for s in suppliers if not s.activities)

        # Identify high-impact activities
        high_impact_cats = {
            ActivityCategory.EXTRACTION.value,
            ActivityCategory.PROCESSING.value,
            ActivityCategory.MANUFACTURING.value,
        }
        high_impact = sum(
            self._activity_counts.get(c, 0) for c in high_impact_cats
        )
        outputs["high_impact_activity_count"] = high_impact

        if sum(1 for s in suppliers if not s.activities) > len(suppliers) * 0.3:
            warnings.append("Over 30% of suppliers have no activity data -- classification may be incomplete")

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 ActivityClassification: %d activities classified across %d categories",
            classified, len(self._activity_counts),
        )
        return PhaseResult(
            phase_name=WorkflowPhase.ACTIVITY_CLASSIFICATION.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Risk Overlay
    # -------------------------------------------------------------------------

    async def _phase_risk_overlay(
        self, input_data: ValueChainMappingInput,
    ) -> PhaseResult:
        """Overlay geographic and sector risk indicators onto supplier map."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._risk_profiles = []

        suppliers = input_data.suppliers
        high_risk_sectors = set(s.lower() for s in input_data.high_risk_sectors)

        # Build country risk lookup from input or defaults
        country_lookup: Dict[str, CountryRiskData] = {}
        for crd in input_data.country_risk_data:
            country_lookup[crd.country_code] = crd

        risk_level_score: Dict[str, float] = {
            "very_high": 90.0,
            "high": 70.0,
            "medium": 45.0,
            "low": 20.0,
            "very_low": 5.0,
        }

        for sup in suppliers:
            risk_factors: List[str] = []
            actions: List[str] = []

            # Country risk component (50%)
            country_risk = 45.0  # default medium
            cc = sup.country_code
            if cc in country_lookup:
                crd = country_lookup[cc]
                hr_score = risk_level_score.get(crd.human_rights_risk.value, 45.0)
                env_score = risk_level_score.get(crd.environmental_risk.value, 45.0)
                country_risk = (hr_score + env_score) / 2
                if crd.conflict_affected:
                    country_risk = min(100.0, country_risk + 20.0)
                    risk_factors.append("conflict_affected_zone")
                if crd.human_rights_risk in (RiskLevel.HIGH, RiskLevel.VERY_HIGH):
                    risk_factors.append("high_human_rights_risk_country")
                if crd.environmental_risk in (RiskLevel.HIGH, RiskLevel.VERY_HIGH):
                    risk_factors.append("high_environmental_risk_country")
            elif cc in DEFAULT_COUNTRY_RISK:
                defaults = DEFAULT_COUNTRY_RISK[cc]
                hr_score = risk_level_score.get(defaults["hr"], 45.0)
                env_score = risk_level_score.get(defaults["env"], 45.0)
                country_risk = (hr_score + env_score) / 2
                if defaults["hr"] in ("high", "very_high"):
                    risk_factors.append("high_human_rights_risk_country")
                if defaults["env"] in ("high", "very_high"):
                    risk_factors.append("high_environmental_risk_country")

            # Sector risk component (30%)
            sector_risk = 45.0
            if sup.sector and sup.sector.lower() in high_risk_sectors:
                sector_risk = 75.0
                risk_factors.append("high_risk_sector")
            elif sup.sector:
                sector_risk = 30.0

            # Governance component (20%)
            governance_risk = 50.0
            if sup.has_code_of_conduct:
                governance_risk -= 15.0
            else:
                risk_factors.append("no_code_of_conduct")
                actions.append("Request code of conduct from supplier")
            if sup.contractual_assurances:
                governance_risk -= 15.0
            else:
                risk_factors.append("no_contractual_assurances")
                actions.append("Establish contractual assurances per Art. 8(3)")

            # Composite score
            composite = round(
                0.50 * country_risk + 0.30 * sector_risk + 0.20 * governance_risk,
                1,
            )

            # Map to risk level
            if composite >= 75:
                level = RiskLevel.VERY_HIGH
                actions.append("Conduct enhanced due diligence immediately")
            elif composite >= 55:
                level = RiskLevel.HIGH
                actions.append("Conduct targeted due diligence")
            elif composite >= 35:
                level = RiskLevel.MEDIUM
            elif composite >= 15:
                level = RiskLevel.LOW
            else:
                level = RiskLevel.VERY_LOW

            self._risk_profiles.append(SupplierRiskProfile(
                supplier_id=sup.supplier_id,
                supplier_name=sup.supplier_name,
                tier=sup.tier.value,
                country_code=sup.country_code,
                sector=sup.sector,
                composite_risk_score=composite,
                risk_level=level,
                risk_factors=risk_factors,
                recommended_actions=actions,
            ))

        # Sort by risk score descending
        self._risk_profiles.sort(key=lambda rp: rp.composite_risk_score, reverse=True)

        # Summary
        level_counts: Dict[str, int] = {}
        for rp in self._risk_profiles:
            level_counts[rp.risk_level.value] = level_counts.get(rp.risk_level.value, 0) + 1

        outputs["risk_level_distribution"] = level_counts
        outputs["avg_composite_risk"] = round(
            sum(rp.composite_risk_score for rp in self._risk_profiles) / len(self._risk_profiles), 1
        ) if self._risk_profiles else 0.0
        outputs["top_5_riskiest"] = [
            {"supplier": rp.supplier_name, "score": rp.composite_risk_score, "level": rp.risk_level.value}
            for rp in self._risk_profiles[:5]
        ]
        outputs["suppliers_needing_enhanced_dd"] = sum(
            1 for rp in self._risk_profiles if rp.risk_level in (RiskLevel.VERY_HIGH, RiskLevel.HIGH)
        )

        if level_counts.get("very_high", 0) > 0:
            warnings.append(
                f"{level_counts['very_high']} suppliers at very high risk -- immediate action required"
            )

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 RiskOverlay: avg risk=%.1f, %d very high, %d high",
            outputs["avg_composite_risk"],
            level_counts.get("very_high", 0),
            level_counts.get("high", 0),
        )
        return PhaseResult(
            phase_name=WorkflowPhase.RISK_OVERLAY.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: ValueChainMappingResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return _compute_hash(payload)

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return _compute_hash(raw)
