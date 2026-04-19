# -*- coding: utf-8 -*-
"""
SupplyChainBridge - Value Chain Due Diligence Bridge for PACK-019
====================================================================

This module links supply chain mapping agents to CSDDD value chain due
diligence obligations. It maps supplier tiers, assesses supplier-level
risk, identifies high-risk business relationships, and provides the
value chain structure needed for CSDDD Articles 6-10 compliance.

Legal References:
    - Directive (EU) 2024/1760 (CSDDD), Articles 6-10
    - Art 6: Identifying actual and potential adverse impacts in value chain
    - Art 7: Prioritisation of identified adverse impacts
    - Art 8: Preventing potential adverse impacts (including via contracts)
    - Art 9: Bringing actual adverse impacts to an end
    - Art 16: Model contractual clauses with business partners

CSDDD Value Chain Scope:
    - Upstream: Tier 1 direct suppliers + indirect suppliers where risk exists
    - Downstream: Distribution, transport, storage (excluding end consumers)
    - Art 2(1)(b): "Chain of activities" covering design, manufacture, transport,
      storage, distribution

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-019 CSDDD Readiness Pack
Status: Production Ready
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

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
# Enums
# ---------------------------------------------------------------------------

class SupplierTier(str, Enum):
    """Supplier tier classification in the value chain."""

    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"
    TIER_N = "tier_n"
    DOWNSTREAM = "downstream"

class SupplierRiskLevel(str, Enum):
    """Supplier-level due diligence risk classification."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NOT_ASSESSED = "not_assessed"

class RiskCategory(str, Enum):
    """Category of adverse impact risk in the supply chain."""

    HUMAN_RIGHTS = "human_rights"
    LABOUR_RIGHTS = "labour_rights"
    ENVIRONMENTAL = "environmental"
    CORRUPTION = "corruption"
    CHILD_LABOUR = "child_labour"
    FORCED_LABOUR = "forced_labour"
    HEALTH_SAFETY = "health_safety"
    LAND_RIGHTS = "land_rights"

class ValueChainDirection(str, Enum):
    """Direction in the value chain."""

    UPSTREAM = "upstream"
    DOWNSTREAM = "downstream"
    OWN_OPERATIONS = "own_operations"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class SupplyChainBridgeConfig(BaseModel):
    """Configuration for the Supply Chain Bridge."""

    pack_id: str = Field(default="PACK-019")
    enable_provenance: bool = Field(default=True)
    high_risk_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Risk score threshold for high-risk classification",
    )
    critical_risk_threshold: float = Field(
        default=0.9, ge=0.0, le=1.0,
        description="Risk score threshold for critical-risk classification",
    )
    max_tier_depth: int = Field(
        default=5, ge=1, le=10,
        description="Maximum supply chain tier depth to map",
    )

class SupplierProfile(BaseModel):
    """Profile of a supplier in the value chain."""

    supplier_id: str = Field(default_factory=_new_uuid)
    supplier_name: str = Field(default="")
    tier: SupplierTier = Field(default=SupplierTier.TIER_1)
    country: str = Field(default="")
    sector: str = Field(default="")
    risk_level: SupplierRiskLevel = Field(default=SupplierRiskLevel.NOT_ASSESSED)
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_categories: List[RiskCategory] = Field(default_factory=list)
    has_contract: bool = Field(default=False)
    has_dd_clause: bool = Field(default=False)
    has_code_of_conduct: bool = Field(default=False)
    annual_spend_eur: float = Field(default=0.0, ge=0.0)
    employee_count: Optional[int] = Field(None, ge=0)
    direction: ValueChainDirection = Field(default=ValueChainDirection.UPSTREAM)

class ValueChainMap(BaseModel):
    """Complete value chain map for CSDDD due diligence."""

    company_id: str = Field(default="")
    total_suppliers: int = Field(default=0)
    tier_breakdown: Dict[str, int] = Field(default_factory=dict)
    direction_breakdown: Dict[str, int] = Field(default_factory=dict)
    risk_breakdown: Dict[str, int] = Field(default_factory=dict)
    high_risk_suppliers: List[SupplierProfile] = Field(default_factory=list)
    countries_covered: List[str] = Field(default_factory=list)
    total_spend_eur: float = Field(default=0.0, ge=0.0)
    dd_clause_coverage_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    provenance_hash: str = Field(default="")

class SupplierRiskAssessment(BaseModel):
    """Detailed risk assessment for a single supplier."""

    supplier_id: str = Field(default="")
    supplier_name: str = Field(default="")
    overall_risk: SupplierRiskLevel = Field(default=SupplierRiskLevel.NOT_ASSESSED)
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_factors: List[Dict[str, Any]] = Field(default_factory=list)
    csddd_articles: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

class BridgeResult(BaseModel):
    """Result of a supply chain bridge operation."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_ms: float = Field(default=0.0)
    records_processed: int = Field(default=0)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Country and Sector Risk Factors
# ---------------------------------------------------------------------------

COUNTRY_HR_RISK: Dict[str, float] = {
    "CN": 0.7, "BD": 0.8, "MM": 0.9, "PK": 0.7, "IN": 0.6,
    "VN": 0.6, "KH": 0.7, "ET": 0.8, "TR": 0.6, "TH": 0.5,
    "NG": 0.7, "CD": 0.9, "BR": 0.5, "MX": 0.5, "ID": 0.6,
    "DE": 0.1, "FR": 0.1, "NL": 0.1, "SE": 0.1, "DK": 0.1,
    "US": 0.2, "GB": 0.1, "JP": 0.2, "KR": 0.2, "AU": 0.1,
}

SECTOR_RISK: Dict[str, float] = {
    "mining": 0.9, "agriculture": 0.8, "textiles": 0.8,
    "electronics": 0.7, "construction": 0.6, "food_processing": 0.6,
    "chemicals": 0.7, "metals": 0.7, "logistics": 0.4,
    "manufacturing": 0.6, "services": 0.3, "technology": 0.3,
    "financial": 0.2, "healthcare": 0.3, "retail": 0.5,
}

# ---------------------------------------------------------------------------
# SupplyChainBridge
# ---------------------------------------------------------------------------

class SupplyChainBridge:
    """Value chain due diligence bridge for PACK-019 CSDDD Readiness.

    Links supply chain mapping agents to CSDDD value chain obligations,
    maps supplier tiers, assesses risk, and identifies high-risk business
    relationships requiring enhanced due diligence under Articles 6-10.

    Attributes:
        config: Bridge configuration.

    Example:
        >>> bridge = SupplyChainBridge(SupplyChainBridgeConfig())
        >>> suppliers = [{"supplier_name": "SupplierA", "country": "BD", ...}]
        >>> risk = bridge.assess_supplier_risk(suppliers[0])
        >>> assert risk.overall_risk != SupplierRiskLevel.NOT_ASSESSED
    """

    def __init__(self, config: Optional[SupplyChainBridgeConfig] = None) -> None:
        """Initialize SupplyChainBridge."""
        self.config = config or SupplyChainBridgeConfig()
        logger.info("SupplyChainBridge initialized (pack=%s)", self.config.pack_id)

    def get_supplier_data(
        self,
        company_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> BridgeResult:
        """Get supplier data for a company from supply chain agents.

        Args:
            company_id: Company identifier.
            context: Optional context with pre-loaded supplier data.

        Returns:
            BridgeResult with status and records processed.
        """
        result = BridgeResult(started_at=utcnow())
        ctx = context or {}

        try:
            suppliers = ctx.get("suppliers", [])
            result.records_processed = len(suppliers)
            result.status = "completed"

            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash({
                    "company_id": company_id,
                    "supplier_count": len(suppliers),
                })

            logger.info(
                "Supplier data loaded for %s: %d suppliers",
                company_id,
                len(suppliers),
            )

        except Exception as exc:
            result.status = "failed"
            result.errors.append(str(exc))
            logger.error("Supplier data retrieval failed: %s", str(exc))

        result.completed_at = utcnow()
        if result.started_at:
            result.duration_ms = (
                result.completed_at - result.started_at
            ).total_seconds() * 1000
        return result

    def map_value_chain(
        self,
        suppliers: List[Dict[str, Any]],
    ) -> ValueChainMap:
        """Map the complete value chain for CSDDD due diligence scope.

        Args:
            suppliers: List of supplier dicts with keys:
                supplier_name, tier, country, sector, annual_spend_eur,
                has_contract, has_dd_clause, direction.

        Returns:
            ValueChainMap with tier/risk breakdowns and high-risk suppliers.
        """
        profiles: List[SupplierProfile] = []

        for s_data in suppliers:
            tier = SupplierTier(s_data.get("tier", "tier_1"))
            country = s_data.get("country", "")
            sector = s_data.get("sector", "")
            direction = ValueChainDirection(
                s_data.get("direction", "upstream")
            )

            risk_score = self._calculate_supplier_risk_score(country, sector, tier)
            risk_level = self._score_to_risk_level(risk_score)
            risk_categories = self._identify_risk_categories(country, sector)

            profiles.append(SupplierProfile(
                supplier_id=s_data.get("supplier_id", _new_uuid()),
                supplier_name=s_data.get("supplier_name", ""),
                tier=tier,
                country=country,
                sector=sector,
                risk_level=risk_level,
                risk_score=risk_score,
                risk_categories=risk_categories,
                has_contract=s_data.get("has_contract", False),
                has_dd_clause=s_data.get("has_dd_clause", False),
                has_code_of_conduct=s_data.get("has_code_of_conduct", False),
                annual_spend_eur=s_data.get("annual_spend_eur", 0.0),
                employee_count=s_data.get("employee_count"),
                direction=direction,
            ))

        # Compute breakdowns
        tier_breakdown: Dict[str, int] = {}
        direction_breakdown: Dict[str, int] = {}
        risk_breakdown: Dict[str, int] = {}

        for p in profiles:
            tier_breakdown[p.tier.value] = tier_breakdown.get(p.tier.value, 0) + 1
            direction_breakdown[p.direction.value] = (
                direction_breakdown.get(p.direction.value, 0) + 1
            )
            risk_breakdown[p.risk_level.value] = (
                risk_breakdown.get(p.risk_level.value, 0) + 1
            )

        high_risk = [
            p for p in profiles
            if p.risk_level in (SupplierRiskLevel.HIGH, SupplierRiskLevel.CRITICAL)
        ]

        countries = list(set(p.country for p in profiles if p.country))
        total_spend = round(sum(p.annual_spend_eur for p in profiles), 2)

        dd_clause_count = sum(1 for p in profiles if p.has_dd_clause)
        dd_coverage = (
            round(dd_clause_count / len(profiles) * 100, 1) if profiles else 0.0
        )

        chain_map = ValueChainMap(
            total_suppliers=len(profiles),
            tier_breakdown=tier_breakdown,
            direction_breakdown=direction_breakdown,
            risk_breakdown=risk_breakdown,
            high_risk_suppliers=high_risk,
            countries_covered=sorted(countries),
            total_spend_eur=total_spend,
            dd_clause_coverage_pct=dd_coverage,
        )
        chain_map.provenance_hash = _compute_hash(chain_map)

        logger.info(
            "Value chain mapped: %d suppliers, %d high-risk, DD coverage=%.1f%%",
            len(profiles),
            len(high_risk),
            dd_coverage,
        )
        return chain_map

    def assess_supplier_risk(
        self,
        supplier: Dict[str, Any],
    ) -> SupplierRiskAssessment:
        """Assess CSDDD due diligence risk for a single supplier.

        Uses deterministic risk scoring (zero-hallucination).

        Args:
            supplier: Supplier data dict with country, sector, tier.

        Returns:
            SupplierRiskAssessment with risk factors and recommendations.
        """
        country = supplier.get("country", "")
        sector = supplier.get("sector", "")
        tier = SupplierTier(supplier.get("tier", "tier_1"))

        risk_score = self._calculate_supplier_risk_score(country, sector, tier)
        risk_level = self._score_to_risk_level(risk_score)
        risk_categories = self._identify_risk_categories(country, sector)

        risk_factors: List[Dict[str, Any]] = []
        country_risk = COUNTRY_HR_RISK.get(country, 0.5)
        sector_risk = SECTOR_RISK.get(sector, 0.5)

        risk_factors.append({
            "factor": "country_risk",
            "country": country,
            "score": country_risk,
            "description": f"Country human rights risk: {country_risk:.1f}",
        })
        risk_factors.append({
            "factor": "sector_risk",
            "sector": sector,
            "score": sector_risk,
            "description": f"Sector adverse impact risk: {sector_risk:.1f}",
        })
        risk_factors.append({
            "factor": "tier_proximity",
            "tier": tier.value,
            "score": self._tier_risk_factor(tier),
            "description": f"Tier proximity factor: {tier.value}",
        })

        # Map to CSDDD articles based on risk
        csddd_articles = ["Art_6"]  # Identification always applies
        if risk_level in (SupplierRiskLevel.HIGH, SupplierRiskLevel.CRITICAL):
            csddd_articles.extend(["Art_7", "Art_8", "Art_9"])
        if risk_level == SupplierRiskLevel.CRITICAL:
            csddd_articles.append("Art_10")

        recommendations = self._generate_supplier_recommendations(
            risk_level, risk_categories, supplier
        )

        assessment = SupplierRiskAssessment(
            supplier_id=supplier.get("supplier_id", ""),
            supplier_name=supplier.get("supplier_name", ""),
            overall_risk=risk_level,
            overall_score=risk_score,
            risk_factors=risk_factors,
            csddd_articles=csddd_articles,
            recommended_actions=recommendations,
        )
        assessment.provenance_hash = _compute_hash(assessment)
        return assessment

    def get_tier_breakdown(
        self,
        suppliers: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Get supply chain tier breakdown analysis.

        Args:
            suppliers: List of supplier data dicts.

        Returns:
            Dict with tier counts, spend distribution, and risk overview.
        """
        tiers: Dict[str, List[Dict[str, Any]]] = {}
        for s in suppliers:
            tier = s.get("tier", "tier_1")
            tiers.setdefault(tier, []).append(s)

        breakdown: Dict[str, Any] = {}
        for tier_name, tier_suppliers in tiers.items():
            spend = sum(s.get("annual_spend_eur", 0.0) for s in tier_suppliers)
            countries = list(set(s.get("country", "") for s in tier_suppliers))
            breakdown[tier_name] = {
                "supplier_count": len(tier_suppliers),
                "total_spend_eur": round(spend, 2),
                "countries": sorted(c for c in countries if c),
                "country_count": len([c for c in countries if c]),
            }

        return {
            "tiers": breakdown,
            "total_suppliers": len(suppliers),
            "tier_count": len(tiers),
            "provenance_hash": _compute_hash(breakdown),
        }

    def identify_high_risk_suppliers(
        self,
        suppliers: List[Dict[str, Any]],
    ) -> List[SupplierRiskAssessment]:
        """Identify suppliers requiring enhanced CSDDD due diligence.

        Args:
            suppliers: List of supplier data dicts.

        Returns:
            List of SupplierRiskAssessment for high/critical risk suppliers.
        """
        high_risk_assessments: List[SupplierRiskAssessment] = []

        for s_data in suppliers:
            assessment = self.assess_supplier_risk(s_data)
            if assessment.overall_risk in (
                SupplierRiskLevel.HIGH,
                SupplierRiskLevel.CRITICAL,
            ):
                high_risk_assessments.append(assessment)

        # Sort by risk score descending
        high_risk_assessments.sort(key=lambda a: a.overall_score, reverse=True)

        logger.info(
            "Identified %d high-risk suppliers from %d total",
            len(high_risk_assessments),
            len(suppliers),
        )
        return high_risk_assessments

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _calculate_supplier_risk_score(
        self,
        country: str,
        sector: str,
        tier: SupplierTier,
    ) -> float:
        """Calculate composite supplier risk score (0.0-1.0)."""
        country_risk = COUNTRY_HR_RISK.get(country, 0.5)
        sector_risk = SECTOR_RISK.get(sector, 0.5)
        tier_factor = self._tier_risk_factor(tier)

        # Weighted: 40% country, 35% sector, 25% tier proximity
        score = 0.40 * country_risk + 0.35 * sector_risk + 0.25 * tier_factor
        return round(min(max(score, 0.0), 1.0), 3)

    def _tier_risk_factor(self, tier: SupplierTier) -> float:
        """Get risk factor for a supplier tier."""
        factors = {
            SupplierTier.TIER_1: 0.3,
            SupplierTier.TIER_2: 0.5,
            SupplierTier.TIER_3: 0.7,
            SupplierTier.TIER_N: 0.8,
            SupplierTier.DOWNSTREAM: 0.4,
        }
        return factors.get(tier, 0.5)

    def _score_to_risk_level(self, score: float) -> SupplierRiskLevel:
        """Map numeric risk score to risk level."""
        if score >= self.config.critical_risk_threshold:
            return SupplierRiskLevel.CRITICAL
        if score >= self.config.high_risk_threshold:
            return SupplierRiskLevel.HIGH
        if score >= 0.4:
            return SupplierRiskLevel.MEDIUM
        return SupplierRiskLevel.LOW

    def _identify_risk_categories(
        self,
        country: str,
        sector: str,
    ) -> List[RiskCategory]:
        """Identify applicable risk categories based on country and sector."""
        categories: List[RiskCategory] = []
        country_risk = COUNTRY_HR_RISK.get(country, 0.5)
        sector_risk = SECTOR_RISK.get(sector, 0.5)

        if country_risk >= 0.7:
            categories.append(RiskCategory.HUMAN_RIGHTS)
        if country_risk >= 0.8:
            categories.append(RiskCategory.FORCED_LABOUR)
            categories.append(RiskCategory.CHILD_LABOUR)
        if sector in ("mining", "agriculture", "chemicals"):
            categories.append(RiskCategory.ENVIRONMENTAL)
        if sector in ("textiles", "agriculture", "electronics"):
            categories.append(RiskCategory.LABOUR_RIGHTS)
        if sector in ("mining", "construction", "manufacturing"):
            categories.append(RiskCategory.HEALTH_SAFETY)
        if sector_risk >= 0.7:
            categories.append(RiskCategory.CORRUPTION)

        return list(set(categories))

    def _generate_supplier_recommendations(
        self,
        risk_level: SupplierRiskLevel,
        risk_categories: List[RiskCategory],
        supplier: Dict[str, Any],
    ) -> List[str]:
        """Generate CSDDD-specific recommendations for a supplier."""
        actions: List[str] = []
        has_contract = supplier.get("has_contract", False)
        has_dd_clause = supplier.get("has_dd_clause", False)

        if risk_level == SupplierRiskLevel.CRITICAL:
            actions.append(
                "Conduct immediate on-site due diligence assessment (Art 6)"
            )
            actions.append(
                "Develop action plan to bring adverse impacts to an end (Art 9)"
            )
        if risk_level in (SupplierRiskLevel.HIGH, SupplierRiskLevel.CRITICAL):
            actions.append(
                "Include enhanced due diligence contractual clauses (Art 16)"
            )

        if not has_contract:
            actions.append(
                "Formalise business relationship with contract including "
                "sustainability requirements"
            )
        elif not has_dd_clause:
            actions.append(
                "Add CSDDD due diligence clauses to existing contract (Art 16)"
            )

        if RiskCategory.FORCED_LABOUR in risk_categories:
            actions.append(
                "Implement forced labour risk monitoring with ILO indicators"
            )
        if RiskCategory.CHILD_LABOUR in risk_categories:
            actions.append(
                "Deploy child labour detection protocols per UN Guiding Principles"
            )
        if RiskCategory.ENVIRONMENTAL in risk_categories:
            actions.append(
                "Require environmental impact assessment from supplier (Art 8)"
            )

        return actions
